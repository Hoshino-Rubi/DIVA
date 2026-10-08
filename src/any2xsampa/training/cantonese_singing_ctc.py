"""Cantonese singing CTC-head training over cached acoustic features.

The acoustic backbone stays behind a replaceable feature-extractor adapter.
This module trains only a CTC projection head and records enough provenance for
the real recognizer to reject mismatched datasets, vocabularies, and models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence
import uuid

from any2xsampa.data.label_provenance import human_phone_truth
from any2xsampa.data.deterministic_split import SplitError, SplitRecord, split_records


CHECKPOINT_SCHEMA = "cantonese_singing_ctc_checkpoint_v1"
FEATURE_CACHE_SCHEMA = "cantonese_singing_ctc_feature_cache_v1"
REPORT_SCHEMA = "cantonese_singing_ctc_training_report_v1"
VERIFIED_TORCH_PREFIX = "2.13.0+cu126"
VERIFIED_CUDA_VERSION = "12.6"


@dataclass(frozen=True, slots=True)
class Metrics:
    loss: float
    token_error_rate: float
    token_similarity: float
    exact_sequence_ratio: float
    example_count: int
    reference_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CTC training requires PyTorch") from exc
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _feature_sha256(features: Mapping[str, Any], valid_frames: Mapping[str, int]) -> str:
    digest = hashlib.sha256()
    for example_id in sorted(features):
        tensor = features[example_id].detach().cpu().contiguous()
        digest.update(example_id.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(str(int(valid_frames[example_id])).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _digest(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _resolve_device(torch: Any, requested: str) -> Any:
    name = str(requested or "cuda").strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ""
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        if not str(torch.__version__).startswith(VERIFIED_TORCH_PREFIX) or str(torch.version.cuda) != VERIFIED_CUDA_VERSION:
            raise RuntimeError(
                "GPU training requires torch 2.13.0+cu126 with CUDA 12.6; "
                f"found torch {torch.__version__}, CUDA {torch.version.cuda}"
            )
        device = torch.device(name)
        torch.empty(1, device=device)
        return device
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be an explicit CUDA device or cpu")


def _ctc_collapse(sequence: Sequence[int], blank_id: int) -> list[int]:
    result: list[int] = []
    previous: int | None = None
    for value in sequence:
        current = int(value)
        if current != blank_id and current != previous:
            result.append(current)
        previous = current
    return result


def _required_ctc_frames(tokens: Sequence[str]) -> int:
    repeats = sum(left == right for left, right in zip(tokens, tokens[1:]))
    return len(tokens) + repeats


def _levenshtein(reference: Sequence[int], prediction: Sequence[int]) -> int:
    previous = list(range(len(prediction) + 1))
    for ref_index, ref_value in enumerate(reference, start=1):
        current = [ref_index]
        for pred_index, pred_value in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[pred_index] + 1,
                    previous[pred_index - 1] + int(ref_value != pred_value),
                )
            )
        previous = current
    return previous[-1]


def write_feature_cache(
    *,
    dataset_path: str | Path,
    features: Mapping[str, Any],
    output_path: str | Path,
    model_sha256: str,
    valid_frames: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Persist injected backbone features without coupling to Transformers."""

    torch = _torch()
    dataset = Path(dataset_path)
    examples = _examples(_load_json(dataset))
    expected_ids = {str(row["example_id"]) for row in examples}
    if set(features) != expected_ids:
        missing = sorted(expected_ids - set(features))
        extra = sorted(set(features) - expected_ids)
        raise ValueError(f"feature identities disagree with dataset; missing={missing}, extra={extra}")
    normalized: dict[str, Any] = {}
    hidden_size: int | None = None
    for example_id in sorted(features):
        tensor = torch.as_tensor(features[example_id], dtype=torch.float32).detach().cpu().contiguous()
        if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
            raise ValueError(f"features must be non-empty [frames, hidden] tensors: {example_id}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"features contain NaN or infinity: {example_id}")
        targets = next(row for row in examples if str(row["example_id"]) == example_id)["target_phone_tokens"]
        required_frames = _required_ctc_frames(targets)
        if tensor.shape[0] < required_frames:
            raise ValueError(f"feature frames are shorter than CTC target: {example_id}")
        if hidden_size is None:
            hidden_size = int(tensor.shape[1])
        elif hidden_size != int(tensor.shape[1]):
            raise ValueError("all cached features must share one hidden size")
        requested_frames = int((valid_frames or {}).get(example_id, int(tensor.shape[0])))
        if requested_frames <= 0 or requested_frames > int(tensor.shape[0]):
            raise ValueError(f"valid frame count is outside feature tensor: {example_id}")
        if requested_frames < required_frames:
            raise ValueError(f"valid feature frames are shorter than CTC target: {example_id}")
        cast = tensor[:requested_frames].to(dtype=torch.float16)
        if not bool(torch.isfinite(cast).all()):
            raise ValueError(f"features overflow float16 representation: {example_id}")
        normalized[example_id] = cast
    valid_frame_counts = {
        example_id: int(tensor.shape[0]) for example_id, tensor in normalized.items()
    }
    payload = {
        "schema_version": FEATURE_CACHE_SCHEMA,
        "dataset_sha256": _sha256(dataset),
        "model_sha256": _digest(model_sha256, label="model_sha256"),
        "hidden_size": hidden_size,
        "example_count": len(normalized),
        "features": normalized,
        "valid_frame_counts": valid_frame_counts,
        "features_sha256": _feature_sha256(normalized, valid_frame_counts),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"feature cache already exists: {destination}")
    _atomic_torch_save(torch, payload, destination)
    return {key: value for key, value in payload.items() if key != "features"}


def _validated_feature_cache(
    cache: Mapping[str, Any],
    *,
    examples: Sequence[Mapping[str, Any]],
    torch: Any,
) -> tuple[Mapping[str, Any], int, str]:
    features = cache.get("features")
    valid_frames = cache.get("valid_frame_counts")
    if not isinstance(features, Mapping) or not isinstance(valid_frames, Mapping):
        raise ValueError("feature cache has no features/valid-frame mapping")
    expected_ids = {str(row["example_id"]) for row in examples}
    if set(features) != expected_ids or set(valid_frames) != expected_ids:
        raise ValueError("feature cache identities do not match dataset")
    if cache.get("example_count") != len(expected_ids):
        raise ValueError("feature cache example count does not match dataset")
    hidden_size = int(cache.get("hidden_size") or 0)
    if hidden_size <= 0:
        raise ValueError("feature cache hidden_size must be positive")
    targets_by_id = {str(row["example_id"]): row["target_phone_tokens"] for row in examples}
    normalized: dict[str, Any] = {}
    normalized_frames: dict[str, int] = {}
    for example_id in sorted(expected_ids):
        tensor = torch.as_tensor(features[example_id]).detach().cpu().contiguous()
        if tensor.ndim != 2 or int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) != hidden_size:
            raise ValueError(f"feature cache tensor shape is invalid: {example_id}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"feature cache contains NaN or infinity: {example_id}")
        frames = int(valid_frames[example_id])
        if frames != int(tensor.shape[0]):
            raise ValueError(f"feature cache valid-frame count is inconsistent: {example_id}")
        if frames < _required_ctc_frames(targets_by_id[example_id]):
            raise ValueError(f"feature cache is shorter than CTC target: {example_id}")
        normalized[example_id] = tensor
        normalized_frames[example_id] = frames
    expected_hash = _digest(cache.get("features_sha256"), label="feature cache features_sha256")
    if _feature_sha256(normalized, normalized_frames) != expected_hash:
        raise ValueError("feature cache tensor SHA-256 mismatch")
    return normalized, hidden_size, expected_hash


def _examples(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = dataset.get("examples")
    if not isinstance(raw, list) or not raw:
        raise ValueError("CTC dataset requires a non-empty examples list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("CTC examples must be objects")
        row = dict(value)
        example_id = str(row.get("example_id") or "").strip()
        group_id = str(row.get("group_id") or "").strip()
        target = [str(token).strip() for token in row.get("target_phone_tokens", []) if str(token).strip()]
        if not example_id or example_id in seen or not group_id or not target:
            raise ValueError("each CTC example needs a unique ID, group ID, and target phones")
        if "<blank>" in target:
            raise ValueError(f"CTC target cannot contain reserved <blank>: {example_id}")
        human_label = row.get("human_authored_annotation_evidence") is True
        pseudo_label = row.get("pseudo_label") is True
        if human_label and pseudo_label:
            raise ValueError(f"CTC example cannot be both human and pseudo labeled: {example_id}")
        if human_label:
            truth = human_phone_truth(row)
            row.update(truth)
        elif pseudo_label:
            _validate_pseudo_label(row, example_id=example_id)
        else:
            raise ValueError(f"CTC example has no auditable label source: {example_id}")
        seen.add(example_id)
        row["example_id"] = example_id
        row["group_id"] = group_id
        row["target_phone_tokens"] = target
        result.append(row)
    return result


def prepare_ctc_dataset(
    examples: Sequence[Mapping[str, Any]],
    *,
    validation_ratio: float = 0.2,
    seed: int = 1337,
) -> dict[str, Any]:
    """Normalize audited human/pseudo examples and assign leakage-safe splits."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between zero and one")
    normalized = _examples({"examples": [dict(row) for row in examples]})
    records: list[SplitRecord] = []
    for row in normalized:
        source_hashes = sorted(_source_evidence_keys(row))
        records.append(
            SplitRecord(
                example_id=str(row["example_id"]),
                singer_id=str(row.get("singer_id") or "").strip(),
                song_id=str(row.get("song_id") or "").strip(),
                phrase_id=str(row.get("phrase_id") or "").strip(),
                content_hash=source_hashes[0] if source_hashes else None,
                metadata={"source_hashes": source_hashes},
            )
        )
    try:
        result = split_records(
            records,
            ratios=(1.0 - validation_ratio, validation_ratio, 0.0),
            seed=seed,
        )
    except SplitError as exc:
        raise ValueError(f"CTC identity split failed: {exc}") from exc
    group_by_id = {
        example_id: group.group_id
        for group in result.groups
        for example_id in group.record_ids
    }
    prepared: list[dict[str, Any]] = []
    for row in normalized:
        example_id = str(row["example_id"])
        value = dict(row)
        value["group_id"] = group_by_id[example_id]
        value["split"] = "val" if result.assignments[example_id] == "validation" else result.assignments[example_id]
        value["split_manifest_sha256"] = result.manifest["manifest_hash"]
        prepared.append(value)
    prepared.sort(key=lambda row: str(row["example_id"]))
    unsigned = {
        "schema_version": "cantonese_singing_ctc_dataset_v1",
        "examples": prepared,
        "split_manifest": dict(result.manifest),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "truth_policy": {
            "phones_ipa_articulatory_are_truth": True,
            "human_and_pseudo_labels_are_separate": True,
            "asr_transcript_is_not_phone_truth": True,
        },
    }
    payload = {**unsigned, "dataset_hash": _canonical_sha256(unsigned)}
    _split(payload, prepared, 0)
    return payload


def _split(dataset: Mapping[str, Any], examples: Sequence[Mapping[str, Any]], fold_index: int) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], str]:
    folds = dataset.get("folds")
    if isinstance(folds, list) and folds:
        if fold_index < 0 or fold_index >= len(folds) or not isinstance(folds[fold_index], Mapping):
            raise ValueError("fold_index is outside dataset folds")
        fold = folds[fold_index]
        validation_group = str(fold.get("validation_group") or "").strip()
        raw_validation_groups = fold.get("validation_groups")
        validation_groups = (
            {str(value) for value in raw_validation_groups}
            if isinstance(raw_validation_groups, list)
            else {validation_group}
        )
        train_groups = {str(value) for value in fold.get("train_groups", [])}
        if not validation_group or not validation_groups or validation_groups & train_groups:
            raise ValueError("fold has an invalid validation group")
        all_groups = {str(row["group_id"]) for row in examples}
        assigned_groups = train_groups | validation_groups
        if assigned_groups != all_groups:
            missing = sorted(all_groups - assigned_groups)
            unknown = sorted(assigned_groups - all_groups)
            raise ValueError(
                "fold must assign every dataset group exactly once: "
                f"missing={missing} unknown={unknown}"
            )
        train = [row for row in examples if str(row["group_id"]) in train_groups]
        validation = [row for row in examples if str(row["group_id"]) in validation_groups]
        identity = "fold:" + _canonical_sha256(
            {
                "fold_index": fold_index,
                "train_groups": sorted(train_groups),
                "validation_groups": sorted(validation_groups),
                "train_examples": sorted(str(row["example_id"]) for row in train),
                "validation_examples": sorted(str(row["example_id"]) for row in validation),
                "dataset_identity": dataset.get("dataset_hash")
                or dataset.get("manifest_sha256")
                or _canonical_sha256(dataset),
            }
        )
    else:
        train = [row for row in examples if row.get("split") == "train"]
        validation = [row for row in examples if row.get("split") in {"val", "validation"}]
        identity = f"manifest:{_canonical_sha256({'train': [r['example_id'] for r in train], 'validation': [r['example_id'] for r in validation]})}"
    if not train or not validation:
        raise ValueError("CTC training requires non-empty train and validation sets")
    train_groups = {str(row["group_id"]) for row in train}
    validation_groups = {str(row["group_id"]) for row in validation}
    if train_groups & validation_groups:
        raise ValueError("CTC train and validation groups overlap")
    train_evidence = {key for row in train for key in _source_evidence_keys(row)}
    validation_evidence = {key for row in validation for key in _source_evidence_keys(row)}
    overlap = sorted(train_evidence & validation_evidence)
    if overlap:
        raise ValueError("CTC train and validation source evidence overlaps: " + ", ".join(overlap))
    return train, validation, identity


def _validate_pseudo_label(row: Mapping[str, Any], *, example_id: str) -> None:
    if row.get("schema_version") != "any2xsampa_pseudo_phone_label_v1":
        raise ValueError(f"CTC pseudo label has unsupported schema: {example_id}")
    if (
        str(row.get("label_source") or "").casefold() != "pseudo"
        or row.get("model_output") is not True
        or row.get("human_truth_overwritten") is not False
    ):
        raise ValueError(f"CTC pseudo label source is invalid: {example_id}")
    phones = [str(value).strip() for value in row.get("target_phone_tokens", [])]
    aliases = [str(value).strip() for value in row.get("phone_tokens", [])]
    ipa = [str(value).strip() for value in row.get("internal_ipa_tokens", [])]
    normalized_ipa = [str(value).strip() for value in row.get("normalized_ipa", [])]
    features = row.get("articulatory_features")
    if (
        not phones
        or aliases != phones
        or not ipa
        or normalized_ipa != ipa
        or len(ipa) != len(phones)
        or not isinstance(features, list)
        or len(features) != len(phones)
        or any(not isinstance(value, Mapping) or not value for value in features)
    ):
        raise ValueError(f"CTC pseudo label phone/IPA/articulatory truth is incomplete: {example_id}")
    provenance = row.get("model_provenance")
    required = {
        "model_id",
        "revision",
        "source",
        "checkpoint_sha256",
        "inference_config_sha256",
        "input_audio_sha256",
        "vocabulary_sha256",
        "calibration_sha256",
        "policy_version",
    }
    if not isinstance(provenance, Mapping) or any(not str(provenance.get(key) or "").strip() for key in required):
        raise ValueError(f"CTC pseudo label provenance is incomplete: {example_id}")
    for key in (
        "checkpoint_sha256",
        "inference_config_sha256",
        "input_audio_sha256",
        "vocabulary_sha256",
        "calibration_sha256",
    ):
        _digest(provenance.get(key), label=key)
    _digest(row.get("source_candidate_sha256"), label="source_candidate_sha256")
    _digest(row.get("pseudo_label_sha256"), label="pseudo_label_sha256")


def _source_evidence_keys(row: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in (
        "audio_sha256",
        "original_audio_sha256",
        "source_vsqx_sha256",
        "source_corpus_sha256",
    ):
        value = str(row.get(field) or "").strip().lower()
        if value:
            keys.add(f"{field}:{value}")
    for field in (
        "audio_path",
        "original_audio_path",
        "source_vsqx",
        "source_corpus",
    ):
        value = str(row.get(field) or "").strip().replace("\\", "/").casefold()
        if value:
            keys.add(f"{field}:{value}")
    return keys


class _FeatureDataset:
    def __init__(self, examples: Sequence[Mapping[str, Any]], features: Mapping[str, Any], token_to_id: Mapping[str, int], torch: Any) -> None:
        self.rows: list[dict[str, Any]] = []
        for example in examples:
            example_id = str(example["example_id"])
            if example_id not in features:
                raise ValueError(f"feature cache missing example: {example_id}")
            targets = [token_to_id[str(token)] for token in example["target_phone_tokens"]]
            self.rows.append({"features": features[example_id].float(), "targets": targets})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _collate(rows: Sequence[Mapping[str, Any]], torch: Any) -> dict[str, Any]:
    max_frames = max(int(row["features"].shape[0]) for row in rows)
    hidden_size = int(rows[0]["features"].shape[1])
    features = torch.zeros((len(rows), max_frames, hidden_size), dtype=torch.float32)
    input_lengths: list[int] = []
    targets: list[int] = []
    target_lengths: list[int] = []
    references: list[list[int]] = []
    for index, row in enumerate(rows):
        item = row["features"]
        length = int(item.shape[0])
        features[index, :length] = item
        input_lengths.append(length)
        reference = [int(value) for value in row["targets"]]
        targets.extend(reference)
        target_lengths.append(len(reference))
        references.append(reference)
    return {
        "features": features,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "references": references,
    }


def _evaluate(head: Any, loader: Any, *, loss_fn: Any, blank_id: int, device: Any, torch: Any) -> Metrics:
    head.eval()
    total_loss = 0.0
    batches = edits = reference_count = exact = count = 0
    with torch.inference_mode():
        for batch in loader:
            logits = head(batch["features"].to(device))
            loss = loss_fn(
                torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1),
                batch["targets"].to(device),
                batch["input_lengths"].to(device),
                batch["target_lengths"].to(device),
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("CTC evaluation produced a non-finite loss")
            total_loss += float(loss.detach().cpu())
            batches += 1
            winners = logits.argmax(dim=-1).detach().cpu()
            for index, reference in enumerate(batch["references"]):
                prediction = _ctc_collapse(winners[index, : int(batch["input_lengths"][index])].tolist(), blank_id)
                edits += _levenshtein(reference, prediction)
                reference_count += len(reference)
                exact += int(reference == prediction)
                count += 1
    rate = edits / max(reference_count, 1)
    return Metrics(total_loss / max(batches, 1), rate, max(0.0, 1.0 - rate), exact / max(count, 1), count, reference_count)


def train_ctc_head(
    *,
    dataset_path: str | Path,
    feature_cache: str | Path,
    output_root: str | Path,
    fold_index: int = 0,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    patience: int = 2,
    seed: int = 1337,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or patience <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, patience, and learning_rate must be positive")
    output = Path(output_root)
    checkpoint_path = output / "model.pt"
    metrics_path = output / "metrics.json"
    if checkpoint_path.exists() or metrics_path.exists():
        raise FileExistsError(f"CTC training output already exists: {output}")
    torch = _torch()
    device = _resolve_device(torch, device_name)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    dataset_file = Path(dataset_path)
    cache_file = Path(feature_cache)
    dataset = _load_json(dataset_file)
    if dataset.get("schema_version") != "cantonese_singing_ctc_dataset_v1":
        raise ValueError("unsupported CTC dataset schema")
    examples = _examples(dataset)
    train_examples, validation_examples, split_identity = _split(dataset, examples, fold_index)
    cache = torch.load(cache_file, map_location="cpu", weights_only=True)
    if not isinstance(cache, Mapping) or cache.get("schema_version") != FEATURE_CACHE_SCHEMA:
        raise ValueError("unsupported CTC feature-cache schema")
    dataset_sha256 = _sha256(dataset_file)
    if cache.get("dataset_sha256") != dataset_sha256:
        raise ValueError("feature cache dataset SHA-256 mismatch")
    backbone_sha256 = _digest(cache.get("model_sha256"), label="feature cache model_sha256")
    features, hidden_size, feature_tensor_sha256 = _validated_feature_cache(
        cache,
        examples=examples,
        torch=torch,
    )
    phones = ["<blank>"] + sorted({token for row in examples for token in row["target_phone_tokens"]})
    token_to_id = {phone: index for index, phone in enumerate(phones)}
    train_data = _FeatureDataset(train_examples, features, token_to_id, torch)
    validation_data = _FeatureDataset(validation_examples, features, token_to_id, torch)
    generator = torch.Generator().manual_seed(seed)
    collate = lambda rows: _collate(rows, torch)  # noqa: E731
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collate)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate)
    head = torch.nn.Linear(hidden_size, len(phones)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_rate = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        head.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch["features"].to(device))
            loss = loss_fn(
                torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1),
                batch["targets"].to(device),
                batch["input_lengths"].to(device),
                batch["target_lengths"].to(device),
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("CTC training produced a non-finite loss")
            loss.backward()
            optimizer.step()
        validation = _evaluate(head, validation_loader, loss_fn=loss_fn, blank_id=0, device=device, torch=torch)
        history.append({"epoch": epoch, "validation": validation.to_dict()})
        if validation.token_error_rate < best_rate:
            best_rate = validation.token_error_rate
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("CTC training produced no checkpoint state")
    head.load_state_dict(best_state)
    final_train = _evaluate(head, train_loader, loss_fn=loss_fn, blank_id=0, device=device, torch=torch)
    final_validation = _evaluate(head, validation_loader, loss_fn=loss_fn, blank_id=0, device=device, torch=torch)
    model_sha256 = _state_sha256(best_state)
    config = {
        "hidden_size": hidden_size,
        "vocab_size": len(phones),
        "blank_id": 0,
        "backbone_frozen": True,
        "backbone_sha256": backbone_sha256,
        "fold_index": fold_index,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "ctc_loss": {"blank_id": 0, "zero_infinity": False},
    }
    vocabulary_sha256 = _canonical_sha256(phones)
    configuration_sha256 = _canonical_sha256(config)
    split_sha256 = _canonical_sha256(
        {
            "split_identity": split_identity,
            "train_examples": sorted(str(row["example_id"]) for row in train_examples),
            "validation_examples": sorted(
                str(row["example_id"]) for row in validation_examples
            ),
        }
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "vocabulary": phones,
        "phones": phones,
        "vocab": {phone: index for index, phone in enumerate(phones) if index != 0},
        "blank_id": 0,
        "dataset_sha256": dataset_sha256,
        "feature_cache_sha256": _sha256(cache_file),
        "feature_tensor_sha256": feature_tensor_sha256,
        "model_sha256": model_sha256,
        "split_identity": split_identity,
        "split_sha256": split_sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "configuration_sha256": configuration_sha256,
        "seed": seed,
        "configuration": config,
        "model_config": config,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or "none"),
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda or "none"), "device": str(device)},
        "hidden_size": hidden_size,
        "backbone_state_dict": {},
        "head_state_dict": best_state,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "history": history,
        "train_metrics": final_train.to_dict(),
        "validation_metrics": final_validation.to_dict(),
        "manual_vsqx_override_allowed": False,
        "lyrics_used_as_recognizer_input": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(torch, checkpoint, checkpoint_path)
    report = {
        "schema_version": REPORT_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_sha256": dataset_sha256,
        "feature_cache_sha256": _sha256(cache_file),
        "feature_tensor_sha256": feature_tensor_sha256,
        "model_sha256": model_sha256,
        "split_identity": split_identity,
        "split_sha256": split_sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "configuration_sha256": configuration_sha256,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train": final_train.to_dict(),
        "validation": final_validation.to_dict(),
        "configuration": config,
        "vocabulary": phones,
        "model_output": True,
        "training_data": False,
    }
    _atomic_text_write(metrics_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def load_ctc_head(
    checkpoint_path: str | Path,
    *,
    device_name: str = "cpu",
    expected_dataset_sha256: str | None = None,
    expected_backbone_sha256: str | None = None,
    expected_vocabulary_sha256: str | None = None,
    expected_configuration_sha256: str | None = None,
    expected_split_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    device = _resolve_device(torch, device_name)
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported CTC checkpoint schema")
    required = (
        "vocabulary",
        "dataset_sha256",
        "feature_cache_sha256",
        "feature_tensor_sha256",
        "model_sha256",
        "split_identity",
        "split_sha256",
        "vocabulary_sha256",
        "configuration_sha256",
        "seed",
        "configuration",
        "torch_version",
        "cuda_version",
        "head_state_dict",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError("CTC checkpoint is missing: " + ", ".join(missing))
    dataset_sha256 = _digest(payload["dataset_sha256"], label="dataset_sha256")
    _digest(payload["feature_cache_sha256"], label="feature_cache_sha256")
    _digest(payload["feature_tensor_sha256"], label="feature_tensor_sha256")
    expected_model = _digest(payload["model_sha256"], label="model_sha256")
    state = payload["head_state_dict"]
    if not isinstance(state, Mapping) or _state_sha256(state) != expected_model:
        raise ValueError("CTC checkpoint model-state SHA-256 mismatch")
    config = payload["configuration"]
    if not isinstance(config, Mapping):
        raise ValueError("CTC checkpoint configuration must be an object")
    vocabulary_sha256 = _digest(payload["vocabulary_sha256"], label="vocabulary_sha256")
    configuration_sha256 = _digest(
        payload["configuration_sha256"], label="configuration_sha256"
    )
    split_sha256 = _digest(payload["split_sha256"], label="split_sha256")
    if _canonical_sha256(payload["vocabulary"]) != vocabulary_sha256:
        raise ValueError("CTC checkpoint vocabulary SHA-256 mismatch")
    if _canonical_sha256(config) != configuration_sha256:
        raise ValueError("CTC checkpoint configuration SHA-256 mismatch")
    expected_values = {
        "dataset": (expected_dataset_sha256, dataset_sha256),
        "backbone": (expected_backbone_sha256, config.get("backbone_sha256")),
        "vocabulary": (expected_vocabulary_sha256, vocabulary_sha256),
        "configuration": (expected_configuration_sha256, configuration_sha256),
        "split": (expected_split_sha256, split_sha256),
    }
    for label, (expected, observed) in expected_values.items():
        if expected is not None and _digest(expected, label=f"expected {label}_sha256") != observed:
            raise ValueError(f"CTC checkpoint {label} SHA-256 does not match the expected identity")
    if payload.get("blank_id") != 0:
        raise ValueError("CTC checkpoint blank_id must be zero")
    if isinstance(payload.get("seed"), bool) or not isinstance(payload.get("seed"), int):
        raise ValueError("CTC checkpoint seed must be an integer")
    if not str(payload.get("split_identity") or "").strip():
        raise ValueError("CTC checkpoint split_identity must be non-empty")
    if not str(payload.get("torch_version") or "").startswith(VERIFIED_TORCH_PREFIX):
        raise ValueError("CTC checkpoint Torch version is incompatible with the verified runtime")
    if str(payload.get("cuda_version") or "") != VERIFIED_CUDA_VERSION:
        raise ValueError("CTC checkpoint CUDA version is incompatible with the verified runtime")
    hidden_size = int(config.get("hidden_size") or 0)
    vocabulary = [str(value) for value in payload["vocabulary"]]
    if hidden_size <= 0 or not vocabulary or vocabulary[0] != "<blank>" or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("CTC checkpoint has an invalid model/vocabulary contract")
    head = torch.nn.Linear(hidden_size, len(vocabulary)).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    return head, payload


def _atomic_torch_save(torch: Any, payload: Mapping[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text_write(destination: Path, content: str) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "CHECKPOINT_SCHEMA",
    "FEATURE_CACHE_SCHEMA",
    "Metrics",
    "prepare_ctc_dataset",
    "REPORT_SCHEMA",
    "_ctc_collapse",
    "_levenshtein",
    "load_ctc_head",
    "train_ctc_head",
    "write_feature_cache",
]

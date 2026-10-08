"""Small, phone-level Miku V4C candidate ranker.

This module is a functional reconstruction.  It intentionally keeps the
public surface used by the recovered tests while making device selection and
checkpoint provenance explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence
import uuid

try:  # Torch stays optional for fixture-only users of the package.
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - exercised only without torch installed.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

    class _MissingModule:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PyTorch is unavailable")

    class _MissingNN:
        Module = _MissingModule

    nn = _MissingNN()  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[misc,assignment]
    Dataset = object  # type: ignore[misc,assignment]


SCHEMA_VERSION = "miku_v4c_phoneme_ranker_v1"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
VERIFIED_TORCH_PREFIX = "2.13.0+cu126"
VERIFIED_CUDA_VERSION = "12.6"
RANKER_DATASET_SCHEMA = "miku-v4c-ranker-dataset-v2"
RANKER_VOCAB_SCHEMA = "miku-v4c-ranker-vocab-v2"


class TorchUnavailableError(RuntimeError):
    """Raised when model training is requested without PyTorch."""


def require_torch() -> Any:
    """Return the imported torch module or fail with an actionable message."""

    if torch is None:
        raise TorchUnavailableError("PyTorch is required for Miku V4C ranker training")
    return torch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _normalise_tokens(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = tuple(token for token in value.split() if token)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        tokens = tuple(str(token) for token in value if str(token))
    else:
        raise ValueError(f"{field_name} must be a string or sequence of tokens")
    return tokens


def _digest(value: object, *, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return digest


def _validate_training_row(row: Mapping[str, Any], *, expected_split: str | None) -> None:
    candidate_id = str(row.get("candidate_id") or "").strip()
    identity_group = str(row.get("identity_group") or row.get("group_id") or "").strip()
    split = str(row.get("split") or "").strip()
    lyrics = str(row.get("lyrics") or "").strip()
    tokens = _normalise_tokens(row.get("phoneme_tokens", ()), field_name="phoneme_tokens")
    if not candidate_id or not identity_group or not split or not lyrics or not tokens:
        raise ValueError("ranker training row requires candidate, identity, split, lyrics, and phone tokens")
    if expected_split is not None and split != expected_split:
        raise ValueError(f"ranker training row split must be {expected_split!r}: {candidate_id}")
    if row.get("training_admissible") is not True:
        raise ValueError(f"ranker training row is quarantined or not admissible: {candidate_id}")
    if row.get("truth_layer") != "phones-ipa-articulatory":
        raise ValueError(f"ranker training row has an unsupported truth layer: {candidate_id}")
    if row.get("truth_ipa_complete") is not True or row.get("truth_features_complete") is not True:
        raise ValueError(f"ranker training row has incomplete IPA/articulatory truth: {candidate_id}")
    phones = row.get("phones")
    if not isinstance(phones, list) or len(phones) != len(tokens):
        raise ValueError(f"ranker training row phone evidence length differs from tokens: {candidate_id}")
    for token, evidence in zip(tokens, phones):
        if (
            not isinstance(evidence, Mapping)
            or str(evidence.get("phone") or "").strip() != token
            or not str(evidence.get("ipa") or "").strip()
            or not isinstance(evidence.get("features"), Mapping)
            or not evidence["features"]
        ):
            raise ValueError(f"ranker training row has incomplete phone evidence: {candidate_id}")
    if row.get("label_provenance_status") != "complete" or not isinstance(row.get("label_provenance"), Mapping):
        raise ValueError(f"ranker training row has incomplete label provenance: {candidate_id}")
    source = row["label_provenance"].get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"ranker training row has no label source provenance: {candidate_id}")
    _digest(source.get("sha256"), field_name=f"ranker label source SHA-256 for {candidate_id}")
    _digest(row.get("split_manifest_sha256"), field_name=f"ranker split manifest SHA-256 for {candidate_id}")


@dataclass(frozen=True)
class PhonemeRankerExample:
    """One candidate row and its vocabulary-encoded sequences."""

    lyrics: str
    phoneme_tokens: tuple[str, ...]
    label: float
    score: float | None
    char_ids: tuple[int, ...] = ()
    phoneme_ids: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RankerVocab:
    """Stable character and phone token vocabulary used by the ranker."""

    def __init__(self, chars: Sequence[str], phoneme_tokens: Sequence[str]) -> None:
        self.chars = self._ordered_unique(chars)
        self.phoneme_tokens = self._ordered_unique(phoneme_tokens)
        self.char_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.phone_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for token in self.chars:
            if token not in self.char_to_id:
                self.char_to_id[token] = len(self.char_to_id)
        for token in self.phoneme_tokens:
            if token not in self.phone_to_id:
                self.phone_to_id[token] = len(self.phone_to_id)
        # Both names appeared in historical fixtures; keep aliases stable.
        self.phoneme_to_id = self.phone_to_id

    @staticmethod
    def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            token = str(value)
            if token and token not in seen and token not in {PAD_TOKEN, UNK_TOKEN}:
                seen.add(token)
                result.append(token)
        return tuple(result)

    @property
    def char_vocab_size(self) -> int:
        return len(self.char_to_id)

    @property
    def phoneme_vocab_size(self) -> int:
        return len(self.phone_to_id)

    @classmethod
    def from_json(cls, path: str | Path) -> "RankerVocab":
        source = _as_path(path)
        with source.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("vocab JSON must contain an object")
        if payload.get("schema_version") != RANKER_VOCAB_SCHEMA:
            raise ValueError("ranker vocabulary schema is unsupported")
        chars = payload.get("chars")
        phones = payload.get("phoneme_tokens", payload.get("phonemes"))
        if not isinstance(chars, Sequence) or isinstance(chars, (str, bytes, bytearray)):
            raise ValueError("vocab JSON chars must be a sequence")
        if not isinstance(phones, Sequence) or isinstance(phones, (str, bytes, bytearray)):
            raise ValueError("vocab JSON phoneme_tokens must be a sequence")
        return cls([str(item) for item in chars], [str(item) for item in phones])

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "RankerVocab":
        chars = payload.get("chars")
        phones = payload.get("phoneme_tokens")
        if chars is None and isinstance(payload.get("char_to_id"), Mapping):
            chars = [key for key in payload["char_to_id"] if key not in {PAD_TOKEN, UNK_TOKEN}]
        if phones is None and isinstance(payload.get("phone_to_id"), Mapping):
            phones = [key for key in payload["phone_to_id"] if key not in {PAD_TOKEN, UNK_TOKEN}]
        if not isinstance(chars, Sequence) or isinstance(chars, (str, bytes, bytearray)):
            raise ValueError("checkpoint vocab chars are missing")
        if not isinstance(phones, Sequence) or isinstance(phones, (str, bytes, bytearray)):
            raise ValueError("checkpoint vocab phoneme_tokens are missing")
        return cls([str(item) for item in chars], [str(item) for item in phones])

    def encode_chars(self, lyrics: str) -> tuple[int, ...]:
        ids = tuple(self.char_to_id.get(char, UNK_ID) for char in str(lyrics))
        return ids or (UNK_ID,)

    def encode_phonemes(self, tokens: Sequence[str]) -> tuple[int, ...]:
        ids = tuple(self.phone_to_id.get(str(token), UNK_ID) for token in tokens)
        return ids or (UNK_ID,)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "chars": list(self.chars),
            "phoneme_tokens": list(self.phoneme_tokens),
            "char_to_id": dict(self.char_to_id),
            "phone_to_id": dict(self.phone_to_id),
        }

    def fingerprint(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_json_dict()))


def _row_label(row: Mapping[str, Any]) -> float:
    value = row.get("label")
    if value is None:
        value = row.get("target")
    if value is None:
        value = row.get("positive")
    if value is None:
        score = row.get("score")
        value = float(score) >= 0.5 if isinstance(score, (float, int)) else 0.0
    if isinstance(value, bool):
        return float(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"label must be numeric, got {value!r}") from exc
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"label must be in [0, 1], got {numeric}")
    return numeric


def _example_from_row(row: Mapping[str, Any], vocab: RankerVocab) -> PhonemeRankerExample:
    lyrics = str(row.get("lyrics", row.get("text", "")))
    tokens = _normalise_tokens(row.get("phoneme_tokens", row.get("phonemes", ())), field_name="phoneme_tokens")
    score_value = row.get("score")
    score = None if score_value is None else float(score_value)
    return PhonemeRankerExample(
        lyrics=lyrics,
        phoneme_tokens=tokens,
        label=_row_label(row),
        score=score,
        char_ids=vocab.encode_chars(lyrics),
        phoneme_ids=vocab.encode_phonemes(tokens),
        metadata=dict(row),
    )


def read_ranker_jsonl(
    path: str | Path,
    vocab: RankerVocab | None = None,
    *,
    expected_split: str | None = None,
) -> list[PhonemeRankerExample] | list[dict[str, Any]]:
    source = _as_path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object")
            normalized = dict(row)
            _validate_training_row(normalized, expected_split=expected_split)
            rows.append(normalized)
    if not rows:
        raise ValueError(f"ranker dataset is empty: {source}")
    if vocab is None:
        return rows
    return [_example_from_row(row, vocab) for row in rows]


class PhonemeRankerJsonlDataset(Dataset):
    """Torch dataset that preserves row metadata and encoded phone sequences."""

    def __init__(self, path: str | Path, vocab: RankerVocab, *, expected_split: str | None = None) -> None:
        self.path = _as_path(path)
        self.vocab = vocab
        self.examples = read_ranker_jsonl(self.path, vocab, expected_split=expected_split)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PhonemeRankerExample:
        return self.examples[index]


def phoneme_ranker_collate(examples: Sequence[PhonemeRankerExample]) -> dict[str, Tensor]:
    """Pad a batch while retaining explicit boolean masks for both sequences."""

    torch_module = require_torch()
    if not examples:
        raise ValueError("cannot collate an empty ranker batch")
    max_chars = max(1, max(len(example.char_ids) for example in examples))
    max_phones = max(1, max(len(example.phoneme_ids) for example in examples))
    char_ids = torch_module.full((len(examples), max_chars), PAD_ID, dtype=torch_module.long)
    phone_ids = torch_module.full((len(examples), max_phones), PAD_ID, dtype=torch_module.long)
    char_mask = torch_module.zeros((len(examples), max_chars), dtype=torch_module.bool)
    phone_mask = torch_module.zeros((len(examples), max_phones), dtype=torch_module.bool)
    labels = torch_module.zeros((len(examples),), dtype=torch_module.float32)
    scores = torch_module.full((len(examples),), float("nan"), dtype=torch_module.float32)
    for row_index, example in enumerate(examples):
        char_values = example.char_ids or (UNK_ID,)
        phone_values = example.phoneme_ids or (UNK_ID,)
        char_length = len(char_values)
        phone_length = len(phone_values)
        char_ids[row_index, :char_length] = torch_module.tensor(char_values, dtype=torch_module.long)
        phone_ids[row_index, :phone_length] = torch_module.tensor(phone_values, dtype=torch_module.long)
        char_mask[row_index, :char_length] = True
        phone_mask[row_index, :phone_length] = True
        labels[row_index] = example.label
        if example.score is not None:
            scores[row_index] = example.score
    return {
        "char_ids": char_ids,
        "phoneme_ids": phone_ids,
        "char_mask": char_mask,
        "phoneme_mask": phone_mask,
        "labels": labels,
        "scores": scores,
    }


class MikuV4CPhonemeRanker(nn.Module):
    """Masked mean-pooling ranker over lyrics characters and phone tokens."""

    def __init__(
        self,
        vocab: RankerVocab,
        *,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        super().__init__()
        self.vocab = vocab
        self.char_embedding = nn.Embedding(vocab.char_vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.phoneme_embedding = nn.Embedding(vocab.phoneme_vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.config: dict[str, Any] = {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "char_vocab_size": vocab.char_vocab_size,
            "phoneme_vocab_size": vocab.phoneme_vocab_size,
            "vocab_hash": vocab.fingerprint(),
        }

    @staticmethod
    def _masked_mean(embeddings: Tensor, mask: Tensor) -> Tensor:
        mask_float = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        denominator = mask_float.sum(dim=1).clamp_min(1.0)
        return (embeddings * mask_float).sum(dim=1) / denominator

    def forward(
        self,
        char_ids: Tensor,
        phoneme_ids: Tensor,
        char_mask: Tensor | None = None,
        phoneme_mask: Tensor | None = None,
    ) -> Tensor:
        if char_mask is None:
            char_mask = char_ids.ne(PAD_ID)
        if phoneme_mask is None:
            phoneme_mask = phoneme_ids.ne(PAD_ID)
        parameter_device = next(self.parameters()).device
        for name, tensor in (("char_ids", char_ids), ("phoneme_ids", phoneme_ids), ("char_mask", char_mask), ("phoneme_mask", phoneme_mask)):
            if tensor.device != parameter_device:
                raise RuntimeError(f"{name} is on {tensor.device}, model is on {parameter_device}")
        chars = self._masked_mean(self.char_embedding(char_ids), char_mask)
        phones = self._masked_mean(self.phoneme_embedding(phoneme_ids), phoneme_mask)
        return self.head(torch.cat((chars, phones), dim=-1)).squeeze(-1)


def build_ranker_model(
    vocab: RankerVocab,
    *,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    dropout: float = 0.1,
) -> MikuV4CPhonemeRanker:
    return MikuV4CPhonemeRanker(
        vocab,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def _verified_device(device: str | Any) -> Any:
    torch_module = require_torch()
    device_obj = torch_module.device(device)
    if device_obj.type == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch_version = str(torch_module.__version__)
        cuda_version = str(torch_module.version.cuda)
        if not torch_version.startswith(VERIFIED_TORCH_PREFIX) or cuda_version != VERIFIED_CUDA_VERSION:
            raise RuntimeError(
                "GPU training requires torch 2.13.0+cu126 with CUDA 12.6; "
                f"found torch {torch_version}, CUDA {cuda_version}"
            )
        if device_obj.index is None:
            device_obj = torch_module.device("cuda", torch_module.cuda.current_device())
        if device_obj.index is not None and device_obj.index >= torch_module.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {device_obj}")
    return device_obj


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch_module = require_torch()
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _move_batch(batch: Mapping[str, Tensor], device: Any) -> dict[str, Tensor]:
    moved: dict[str, Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device=device, non_blocking=device.type == "cuda")
        if moved[key].device != device:
            raise RuntimeError(f"batch tensor {key} did not move to requested device {device}")
    return moved


def _run_epoch(
    model: MikuV4CPhonemeRanker,
    loader: Iterable[Mapping[str, Tensor]] | None,
    *,
    device: Any,
    optimizer: Any = None,
) -> float | None:
    if loader is None:
        return None
    torch_module = require_torch()
    training = optimizer is not None
    model.train(training)
    criterion = torch_module.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_rows = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(batch["char_ids"], batch["phoneme_ids"], batch["char_mask"], batch["phoneme_mask"])
        loss = criterion(logits, batch["labels"])
        if not torch_module.isfinite(loss):
            raise RuntimeError("ranker loss became non-finite")
        if training:
            loss.backward()
            torch_module.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            if not all(torch_module.isfinite(parameter).all() for parameter in model.parameters()):
                raise RuntimeError("ranker parameters became non-finite")
        rows = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows
    if total_rows == 0:
        raise ValueError("ranker loader produced no rows")
    return total_loss / total_rows


def _runtime_metadata(device: Any) -> dict[str, Any]:
    torch_module = require_torch()
    payload: dict[str, Any] = {
        "torch_version": str(torch_module.__version__),
        "cuda_version": None if torch_module.version.cuda is None else str(torch_module.version.cuda),
        "device": str(device),
        "device_type": str(device.type),
        "device_index": device.index,
        "cuda_available": bool(torch_module.cuda.is_available()),
        "device_name": None,
    }
    if device.type == "cuda":
        payload["device_name"] = str(torch_module.cuda.get_device_name(device))
    return payload


def _source_metadata() -> dict[str, str]:
    source_path = Path(inspect.getfile(MikuV4CPhonemeRanker)).resolve()
    return {
        "path": str(source_path),
        "sha256": _sha256_file(source_path),
        "classification": "functional_reconstruction",
    }


def _metadata(
    *,
    train_jsonl: Path,
    val_jsonl: Path | None,
    vocab: RankerVocab,
    model: MikuV4CPhonemeRanker,
    device: Any,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    split_identity: str | None,
    split_hash: str | None,
    split_manifest_path: Path,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    train_hash = _sha256_file(train_jsonl)
    val_hash = None if val_jsonl is None else _sha256_file(val_jsonl)
    derived_split_hash = _sha256_bytes(_canonical_json({"train": train_hash, "val": val_hash}))
    config = {
        "embedding_dim": model.config["embedding_dim"],
        "hidden_dim": model.config["hidden_dim"],
        "dropout": model.config["dropout"],
        "char_vocab_size": model.config["char_vocab_size"],
        "phoneme_vocab_size": model.config["phoneme_vocab_size"],
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "phone_vocabulary": {
            "tokens": list(vocab.phoneme_tokens),
            "hash": vocab.fingerprint(),
        },
        "dataset": {
            "train_path": str(train_jsonl),
            "train_sha256": train_hash,
            "val_path": None if val_jsonl is None else str(val_jsonl),
            "val_sha256": val_hash,
            "combined_sha256": derived_split_hash,
        },
        "split": {
            "identity": split_identity or f"manifest:{split_manifest_sha256}",
            "hash": split_hash or split_manifest_sha256,
            "manifest_path": str(split_manifest_path),
            "manifest_file_sha256": _sha256_file(split_manifest_path),
        },
        "model": {
            "config": config,
            "config_sha256": _sha256_bytes(_canonical_json(config)),
            "state_sha256": _state_sha256(model.state_dict()),
        },
        "seed": seed,
        "runtime": _runtime_metadata(device),
        "source": _source_metadata(),
    }


def _load_split_manifest(
    path: Path,
    *,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"ranker split manifest is unreadable: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != RANKER_DATASET_SCHEMA:
        raise ValueError("ranker split manifest schema is unsupported")
    declared = _digest(payload.get("manifest_sha256"), field_name="ranker split manifest SHA-256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if _sha256_bytes(_canonical_json(unsigned)) != declared:
        raise ValueError("ranker split manifest SHA-256 mismatch")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("ranker split manifest candidates must be a list")
    assignments: dict[str, str] = {}
    for value in candidates:
        if not isinstance(value, Mapping):
            raise ValueError("ranker split manifest candidate must be an object")
        candidate_id = str(value.get("candidate_id") or "").strip()
        split = str(value.get("split") or "").strip()
        if not candidate_id or split not in {"train", "val", "test"} or candidate_id in assignments:
            raise ValueError("ranker split manifest has invalid or duplicate candidate assignments")
        assignments[candidate_id] = split
    seen_groups: dict[str, str] = {}
    for expected_split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            candidate_id = str(row["candidate_id"])
            if assignments.get(candidate_id) != expected_split:
                raise ValueError(f"ranker row does not match split manifest: {candidate_id}")
            if row.get("split_manifest_sha256") != declared:
                raise ValueError(f"ranker row split hash does not match manifest: {candidate_id}")
            group = str(row.get("identity_group") or row.get("group_id"))
            previous = seen_groups.setdefault(group, expected_split)
            if previous != expected_split:
                raise ValueError(f"ranker identity group crosses train/validation: {group}")
    return payload, declared


def train_ranker(
    *,
    train_jsonl: str | Path,
    vocab: str | Path | RankerVocab,
    output_dir: str | Path,
    val_jsonl: str | Path | None = None,
    epochs: int = 1,
    batch_size: int = 8,
    device: str = "cpu",
    seed: int = 0,
    learning_rate: float = 1e-3,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    split_identity: str | None = None,
    split_hash: str | None = None,
    split_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Train and save one ranker checkpoint without implicit CPU fallback."""

    torch_module = require_torch()
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    device_obj = _verified_device(device)
    _set_seed(seed)
    ranker_vocab = vocab if isinstance(vocab, RankerVocab) else RankerVocab.from_json(vocab)
    train_path = _as_path(train_jsonl)
    val_path = None if val_jsonl is None else _as_path(val_jsonl)
    train_dataset = PhonemeRankerJsonlDataset(train_path, ranker_vocab, expected_split="train")
    val_dataset = None if val_path is None else PhonemeRankerJsonlDataset(
        val_path,
        ranker_vocab,
        expected_split="val",
    )
    manifest_path = _as_path(split_manifest) if split_manifest is not None else train_path.parent / "split_manifest.json"
    _, manifest_sha256 = _load_split_manifest(
        manifest_path,
        train_rows=[example.metadata for example in train_dataset.examples],
        val_rows=[] if val_dataset is None else [example.metadata for example in val_dataset.examples],
    )
    if split_hash is not None and split_hash != manifest_sha256:
        raise ValueError("caller-provided split hash does not match split manifest")
    if split_identity is not None and split_identity != f"manifest:{manifest_sha256}":
        raise ValueError("caller-provided split identity does not match split manifest")
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=phoneme_ranker_collate,
        generator=generator,
        pin_memory=device_obj.type == "cuda",
    )
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=phoneme_ranker_collate,
        pin_memory=device_obj.type == "cuda",
    )
    model = build_ranker_model(
        ranker_vocab,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device_obj)
    if any(parameter.device != device_obj for parameter in model.parameters()):
        raise RuntimeError(f"model parameters did not move to requested device {device_obj}")
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=learning_rate)

    history: list[dict[str, float | int | None]] = []
    last_train_loss = 0.0
    last_val_loss: float | None = None
    for epoch in range(1, epochs + 1):
        last_train_loss = float(_run_epoch(model, train_loader, device=device_obj, optimizer=optimizer))
        last_val_loss = _run_epoch(model, val_loader, device=device_obj)
        history.append({"epoch": epoch, "train_loss": last_train_loss, "val_loss": last_val_loss})
    if device_obj.type == "cuda":
        torch_module.cuda.synchronize(device_obj)

    output_path = _as_path(output_dir)
    model_path = output_path / "model.pt"
    metrics_path = output_path / "metrics.json"
    if model_path.exists() or metrics_path.exists():
        raise FileExistsError(f"ranker training output already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        train_jsonl=train_path,
        val_jsonl=val_path,
        vocab=ranker_vocab,
        model=model,
        device=device_obj,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        split_identity=split_identity,
        split_hash=split_hash,
        split_manifest_path=manifest_path,
        split_manifest_sha256=manifest_sha256,
    )
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "train_jsonl": str(train_path),
        "val_jsonl": None if val_path is None else str(val_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device_obj),
        "seed": seed,
        "learning_rate": learning_rate,
        "train_size": len(train_dataset),
        "val_size": 0 if val_dataset is None else len(val_dataset),
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
        "history": history,
        "model_path": str(model_path),
        "checkpoint_metadata": metadata,
    }
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "model_sha256": metadata["model"]["state_sha256"],
        "model_config": model.config,
        "vocab": ranker_vocab.to_json_dict(),
        "metrics": metrics,
        "metadata": metadata,
    }
    _atomic_torch_save(torch_module, checkpoint, model_path)
    _atomic_text_write(metrics_path, json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    return metrics


def load_ranker_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    expected_vocab_hash: str | None = None,
    expected_split_hash: str | None = None,
) -> tuple[MikuV4CPhonemeRanker, dict[str, Any]]:
    """Load a checkpoint and reject missing or incompatible provenance."""

    torch_module = require_torch()
    device_obj = _verified_device(device)
    payload = torch_module.load(_as_path(checkpoint_path), map_location=device_obj, weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing ranker checkpoint schema_version")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("ranker checkpoint is missing required metadata")
    vocabulary = metadata.get("phone_vocabulary")
    split = metadata.get("split")
    runtime = metadata.get("runtime")
    dataset = metadata.get("dataset")
    model_metadata = metadata.get("model")
    for name, value in (
        ("phone_vocabulary", vocabulary),
        ("dataset", dataset),
        ("split", split),
        ("model", model_metadata),
        ("runtime", runtime),
        ("source", metadata.get("source")),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"ranker checkpoint metadata is missing {name}")
    if not dataset.get("train_sha256") or not dataset.get("combined_sha256"):
        raise ValueError("ranker checkpoint dataset metadata is incomplete")
    if not split.get("identity") or not split.get("hash"):
        raise ValueError("ranker checkpoint split metadata is incomplete")
    if expected_vocab_hash is not None and vocabulary.get("hash") != expected_vocab_hash:
        raise ValueError("ranker checkpoint vocabulary hash does not match expected hash")
    if expected_split_hash is not None and split.get("hash") != expected_split_hash:
        raise ValueError("ranker checkpoint split hash does not match expected hash")
    vocab_payload = payload.get("vocab")
    config = payload.get("model_config")
    if not isinstance(vocab_payload, Mapping) or not isinstance(config, Mapping):
        raise ValueError("ranker checkpoint is missing model_config or vocab")
    ranker_vocab = RankerVocab.from_json_dict(vocab_payload)
    if ranker_vocab.fingerprint() != vocabulary.get("hash"):
        raise ValueError("ranker checkpoint embedded vocabulary hash is invalid")
    metadata_config = model_metadata.get("config")
    if not isinstance(metadata_config, Mapping):
        raise ValueError("ranker checkpoint model metadata is missing config")
    config_hash = _sha256_bytes(_canonical_json(dict(metadata_config)))
    if config_hash != model_metadata.get("config_sha256"):
        raise ValueError("ranker checkpoint model config hash is invalid")
    architecture_keys = (
        "embedding_dim",
        "hidden_dim",
        "dropout",
        "char_vocab_size",
        "phoneme_vocab_size",
    )
    for key in architecture_keys:
        value = config.get(key)
        if metadata_config.get(key) != value:
            raise ValueError(f"ranker checkpoint model config disagrees for {key}")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("ranker checkpoint is missing model_state_dict")
    if vocabulary.get("hash") != ranker_vocab.fingerprint():
        raise ValueError("ranker checkpoint vocabulary metadata does not match vocab payload")
    model_metadata = metadata.get("model")
    if not isinstance(model_metadata, Mapping):
        raise ValueError("ranker checkpoint metadata is missing model")
    expected_state = str(model_metadata.get("state_sha256") or "").strip().lower()
    if len(expected_state) != 64 or any(char not in "0123456789abcdef" for char in expected_state):
        raise ValueError("ranker checkpoint model state SHA-256 metadata is missing or invalid")
    if _state_sha256(model_state) != expected_state or payload.get("model_sha256") != expected_state:
        raise ValueError("ranker checkpoint model-state SHA-256 mismatch")
    model_config_metadata = model_metadata.get("config")
    if not isinstance(model_config_metadata, Mapping):
        raise ValueError("ranker checkpoint metadata is missing model config")
    config_for_hash = {
        "embedding_dim": config.get("embedding_dim"),
        "hidden_dim": config.get("hidden_dim"),
        "dropout": config.get("dropout"),
        "char_vocab_size": config.get("char_vocab_size"),
        "phoneme_vocab_size": config.get("phoneme_vocab_size"),
        "learning_rate": model_config_metadata.get("learning_rate"),
        "epochs": model_config_metadata.get("epochs"),
        "batch_size": model_config_metadata.get("batch_size"),
    }
    if model_metadata.get("config_sha256") != _sha256_bytes(_canonical_json(config_for_hash)):
        raise ValueError("ranker checkpoint model configuration hash does not match payload")
    model = build_ranker_model(
        ranker_vocab,
        embedding_dim=int(config.get("embedding_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device_obj)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, dict(payload)


def _atomic_torch_save(torch_module: Any, payload: Mapping[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch_module.save(payload, temporary)
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
    "PAD_ID",
    "PAD_TOKEN",
    "SCHEMA_VERSION",
    "UNK_ID",
    "UNK_TOKEN",
    "MikuV4CPhonemeRanker",
    "PhonemeRankerExample",
    "PhonemeRankerJsonlDataset",
    "RankerVocab",
    "TorchUnavailableError",
    "build_ranker_model",
    "load_ranker_checkpoint",
    "phoneme_ranker_collate",
    "read_ranker_jsonl",
    "require_torch",
    "train_ranker",
]

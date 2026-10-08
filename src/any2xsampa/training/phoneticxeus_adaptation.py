"""Bounded residual adaptation of cached pretrained phone evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256, ctc_collapse
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.limited_data_reference import load_vocabulary, training_gate, validate_dataset, validate_dataset_source_roles
from any2xsampa.training.phone_error_metrics import phone_metrics


def make_residual_adapter(torch: Any, width: int = 1024, bottleneck: int = 64) -> Any:
    class ResidualAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = torch.nn.Linear(width, bottleneck)
            self.up = torch.nn.Linear(bottleneck, width)
            torch.nn.init.zeros_(self.up.weight)
            torch.nn.init.zeros_(self.up.bias)

        def forward(self, features: Any) -> Any:
            return features + self.up(torch.nn.functional.gelu(self.down(features)))

    return ResidualAdapter()


def required_ctc_frames(ids: Sequence[int]) -> int:
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def validate_feature_index(index: Mapping[str, Any], dataset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    validate_dataset(dataset)
    if index.get("schema_version") != "phoneticxeus_feature_index_v1" or index.get("purpose") != "rendered-features":
        raise ValueError("training requires the rendered-only acoustic cache")
    if index.get("source_dataset_hash") != dataset["dataset_hash"] or index.get("split_sha256") != dataset["split_sha256"]:
        raise ValueError("stale or differently split feature cache")
    model = index.get("model_metadata", {})
    if model.get("model_id") != MODEL_ID or model.get("revision") != REVISION or model.get("weights_sha256") != WEIGHT_SHA256:
        raise ValueError("feature cache is not from the authorized pretrained model")
    expected = {row["example_id"]: row for row in dataset["examples"] if row["eligible"] and row["split"] in {"train", "dev"}}
    entries = index["entries"]
    if len(entries) != len(expected) or {row["example_id"] for row in entries} != set(expected):
        raise ValueError("feature cache must cover the fixed train/dev subset exactly")
    for entry in entries:
        row = expected[entry["example_id"]]
        for field in ("song_id", "split", "audio_role", "audio_sha256", "start_sec", "end_sec"):
            if entry[field] != row[field]:
                raise ValueError(f"feature binding mismatch: {entry['example_id']} {field}")
        if entry["test_outputs_sealed"] or entry["split"] == "test":
            raise ValueError("sealed test evidence cannot enter adaptation")
        if entry["vocabulary_sha256"] != dataset["bindings"]["vocabulary"]["sha256"]:
            raise ValueError("feature vocabulary differs from reference projection")
        if entry["model_snapshot_lock_sha256"] != index["model_metadata"]["snapshot_lock_sha256"]:
            raise ValueError("mixed model snapshots in feature cache")
        if entry["frames"] < required_ctc_frames(row["model_target_ids"]):
            raise ValueError(f"insufficient CTC frames for intact target: {entry['example_id']}")
    return expected


def _load_cache(index_path: Path, dataset: Mapping[str, Any], torch: Any) -> tuple[list[dict], Any, dict]:
    from safetensors.torch import load_file

    index = read_json(index_path)
    expected = validate_feature_index(index, dataset)
    validate_dataset_source_roles(dataset)
    head_path = Path(index["head"]["path"])
    if not head_path.resolve().is_relative_to(index_path.parent.resolve()):
        raise ValueError("pretrained head must remain inside its cache")
    verify_file(head_path, index["head"]["sha256"])
    state = load_file(str(head_path), device="cpu")
    if set(state) != {"weight", "bias"} or tuple(state["weight"].shape) != (428, 1024) or tuple(state["bias"].shape) != (428,):
        raise ValueError("invalid pretrained phone head")
    head = torch.nn.Linear(1024, 428)
    head.load_state_dict(state, strict=True)
    head.requires_grad_(False).eval()
    rows = []
    source_hashes: dict[str, str] = {}
    vocab_path = Path(dataset["bindings"]["vocabulary"]["path"])
    verify_file(vocab_path, dataset["bindings"]["vocabulary"]["sha256"])
    vocabulary = load_vocabulary(vocab_path)
    for entry in index["entries"]:
        reference = expected[entry["example_id"]]
        expected_ids = [vocabulary.get(symbol) for symbol in reference["model_target_ipa"]]
        if expected_ids != reference["model_target_ids"] or any(
            isinstance(value, bool) or not isinstance(value, int) or not 4 <= value < 428
            for value in expected_ids
        ):
            raise ValueError("reference IDs do not equal the pinned non-special phone vocabulary")
        source_hashes[reference["audio_path"]] = reference["audio_sha256"]
        path = Path(entry["evidence_path"])
        if not path.resolve().is_relative_to(index_path.parent.resolve()):
            raise ValueError("features must remain inside their cache")
        verify_file(path, entry["evidence_sha256"])
        values = load_file(str(path), device="cpu")
        features, logits = values["encoder_features"], values["frame_logits"]
        if tuple(features.shape) != (entry["frames"], 1024) or tuple(logits.shape) != (entry["frames"], 428):
            raise ValueError("invalid cached feature/logit dimensions")
        if features.dtype != torch.float32 or logits.dtype != torch.float32 or not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise ValueError("invalid feature dtype or non-finite evidence")
        with torch.no_grad():
            if not torch.allclose(head(features), logits, rtol=1e-4, atol=1e-4):
                raise ValueError("cached logits do not equal the frozen pretrained head(features)")
        rows.append({**reference, "features": features, "baseline_logits": logits})
    for path, digest in source_hashes.items():
        verify_file(Path(path), digest)
    return rows, head, index


def evaluate_rows(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], torch: Any, adapter: Any = None, head: Any = None) -> dict[str, Any]:
    pairs = []
    details = []
    with torch.no_grad():
        for row in rows:
            logits = row["baseline_logits"] if adapter is None else head(adapter(row["features"].to("cuda")))
            ids = ctc_collapse(logits.argmax(dim=-1).cpu().tolist())
            reference = [labels[index] for index in row["model_target_ids"]]
            prediction = [labels[index] for index in ids]
            pairs.append((reference, prediction))
            details.append({"example_id": row["example_id"], "song_id": row["song_id"], "reference": reference, "prediction": prediction, "metrics": phone_metrics([(reference, prediction)])})
    return {"metrics": phone_metrics(pairs), "examples": details}


def _batch(rows: Sequence[Mapping[str, Any]], torch: Any) -> tuple[Any, Any, Any, Any]:
    maximum = max(row["features"].shape[0] for row in rows)
    features = torch.zeros((len(rows), maximum, 1024), dtype=torch.float32)
    input_lengths, targets, target_lengths = [], [], []
    for index, row in enumerate(rows):
        frames = row["features"].shape[0]
        features[index, :frames] = row["features"]
        input_lengths.append(frames)
        targets.extend(row["model_target_ids"])
        target_lengths.append(len(row["model_target_ids"]))
    return features.to("cuda"), torch.tensor(input_lengths, dtype=torch.long), torch.tensor(targets, dtype=torch.long), torch.tensor(target_lengths, dtype=torch.long)


def run_adaptation(dataset_path: Path, index_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget, configure_cuda, enforce_offline
    from safetensors.torch import save_file

    if seed not in {17, 29, 43}:
        raise ValueError("only the three predeclared seeds are permitted")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a training run")
    dataset = read_json(dataset_path)
    if not training_gate(dataset)["passed"]:
        raise ValueError("reference training gate did not pass")
    output_dir.mkdir(parents=True)
    enforce_offline(output_dir / "runtime-cache")
    torch = configure_cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    before = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    rows, head, index = _load_cache(index_path, dataset, torch)
    vocabulary_path = Path(dataset["bindings"]["vocabulary"]["path"])
    verify_file(vocabulary_path, dataset["bindings"]["vocabulary"]["sha256"])
    vocab = load_vocabulary(vocabulary_path)
    labels = [""] * len(vocab)
    for label, value in vocab.items():
        labels[value] = label
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    baseline = evaluate_rows(dev, labels, torch)
    write_json_new(output_dir / "baseline-dev.json", baseline)
    head = head.to("cuda")
    adapter = make_residual_adapter(torch).to("cuda")
    probe = train[0]["features"].to("cuda")
    if not torch.equal(adapter(probe), probe):
        raise RuntimeError("zero-initialized adapter changes the baseline features")
    frozen_head = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    best_state, best_per, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    stop_reason = "maximum_epochs"
    rng = random.Random(seed)
    for epoch in range(1, 21):
        if time.monotonic() - before >= 1500:
            stop_reason = "wall_clock_limit"
            break
        adapter.train()
        order = list(train)
        rng.shuffle(order)
        losses = []
        for offset in range(0, len(order), 8):
            if time.monotonic() - before >= 1500:
                stop_reason = "wall_clock_limit"
                break
            features, lengths, targets, target_lengths = _batch(order[offset:offset + 8], torch)
            optimizer.zero_grad(set_to_none=True)
            logits = head(adapter(features))
            loss = loss_fn(logits.log_softmax(-1).transpose(0, 1), targets, lengths, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite CTC loss; no zero-infinity masking")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.item()))
        adapter.eval()
        evaluated = evaluate_rows(dev, labels, torch, adapter, head)
        score = evaluated["metrics"]["per"]
        memory = check_gpu_budget(torch)
        event = {"epoch": epoch, "mean_ctc_loss": sum(losses) / len(losses) if losses else None, "dev_per": score, "elapsed_seconds": time.monotonic() - before, "memory": memory}
        history.append(event)
        write_json_new(output_dir / f"epoch-{epoch:02d}.json", event)
        print(f"seed={seed}; epoch={epoch}; dev_per={score:.6f}; loss={event['mean_ctc_loss']}; elapsed={event['elapsed_seconds']:.1f}", flush=True)
        if score < best_per:
            best_per, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()}
        else:
            stale += 1
        if stop_reason == "wall_clock_limit" or stale >= 4:
            stop_reason = "wall_clock_limit" if stop_reason == "wall_clock_limit" else "early_stopping"
            break
    if best_state is None:
        raise RuntimeError("no complete development checkpoint was produced")
    adapter.load_state_dict(best_state)
    adapter.eval()
    if any(not torch.equal(value.cpu(), frozen_head[name]) for name, value in head.state_dict().items()):
        raise RuntimeError("pretrained output head changed during adaptation")
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("pretrained output head was not frozen")
    final = evaluate_rows(dev, labels, torch, adapter, head)
    final_hash = write_json_new(output_dir / "adapted-dev.json", final)
    checkpoint = output_dir / "adapter.safetensors"
    save_file(best_state, str(checkpoint))
    report = {
        "schema_version": "phoneticxeus_residual_adaptation_v1",
        "seed": seed, "mode": "rendered_proxy_only",
        "dataset_hash": dataset["dataset_hash"], "dataset_file_sha256": sha256_file(dataset_path),
        "feature_index_sha256": sha256_file(index_path), "model_metadata": index["model_metadata"],
        "head": index["head"], "adapter": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "architecture": {"width": 1024, "bottleneck": 64, "zero_initialized_up": True, "trainable_parameters": sum(p.numel() for p in adapter.parameters())},
        "hyperparameters": {"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 8, "max_epochs": 20, "patience": 4, "gradient_clip_norm": 1.0, "max_seconds": 1500},
        "best_epoch": best_epoch, "stop_reason": stop_reason, "epochs": history,
        "baseline_dev": baseline["metrics"], "adapted_dev": final["metrics"], "adapted_dev_sha256": final_hash,
        "head_unchanged": True, "backbone_loaded_during_training": False,
        "test_songs_loaded": False, "original_accuracy_claim_allowed": False, "qualified_recognizer": False,
        "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - before, "device": "cuda:0",
    }
    write_json_new(output_dir / "training-report.json", report)
    return report

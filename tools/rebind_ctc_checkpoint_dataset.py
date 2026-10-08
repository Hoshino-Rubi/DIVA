from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.training import cantonese_singing_ctc as ctc  # noqa: E402


IDENTITY_FIELDS = (
    "example_id",
    "group_id",
    "audio_path",
    "audio_sha256",
    "crop_start_sec",
    "crop_end_sec",
    "target_phone_tokens",
    "model_audio_sample_rate",
    "model_audio_channels",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _identity_rows(payload: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    raw = payload.get("examples")
    if not isinstance(raw, list) or not raw:
        raise ValueError("dataset must contain a non-empty examples list")
    result: dict[str, tuple[Any, ...]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("dataset examples must be objects")
        example_id = str(row.get("example_id") or "").strip()
        if not example_id or example_id in result:
            raise ValueError("dataset example IDs must be unique and non-empty")
        result[example_id] = tuple(row.get(field) for field in IDENTITY_FIELDS)
    return result


def _split_sha256(split_identity: str, train: list[Mapping[str, Any]], validation: list[Mapping[str, Any]]) -> str:
    return ctc._canonical_sha256(
        {
            "split_identity": split_identity,
            "train_examples": sorted(str(row["example_id"]) for row in train),
            "validation_examples": sorted(str(row["example_id"]) for row in validation),
        }
    )


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


def rebind_checkpoint(
    old_dataset_path: Path,
    new_dataset_path: Path,
    old_cache_path: Path,
    new_cache_path: Path,
    old_checkpoint_path: Path,
    old_metrics_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    import torch

    old_dataset = _json(old_dataset_path)
    new_dataset = _json(new_dataset_path)
    if old_dataset.get("schema_version") != "cantonese_singing_ctc_dataset_v1" or new_dataset.get("schema_version") != "cantonese_singing_ctc_dataset_v1":
        raise ValueError("checkpoint rebind requires cantonese_singing_ctc_dataset_v1 datasets")
    if _identity_rows(old_dataset) != _identity_rows(new_dataset):
        raise ValueError("dataset rebind changes acoustic identities")
    old_sha = _sha256(old_dataset_path)
    new_sha = _sha256(new_dataset_path)
    if old_sha == new_sha:
        raise ValueError("new dataset must have a distinct file identity")

    old_cache = torch.load(old_cache_path, map_location="cpu", weights_only=True)
    new_cache = torch.load(new_cache_path, map_location="cpu", weights_only=True)
    if not isinstance(old_cache, Mapping) or not isinstance(new_cache, Mapping):
        raise ValueError("feature caches must be mapping payloads")
    if old_cache.get("schema_version") != ctc.FEATURE_CACHE_SCHEMA or new_cache.get("schema_version") != ctc.FEATURE_CACHE_SCHEMA:
        raise ValueError("unsupported feature cache schema")
    if old_cache.get("dataset_sha256") != old_sha or new_cache.get("dataset_sha256") != new_sha:
        raise ValueError("feature cache dataset binding does not match the requested datasets")
    old_examples = ctc._examples(old_dataset)
    new_examples = ctc._examples(new_dataset)
    old_features, old_hidden, old_tensor_hash = ctc._validated_feature_cache(old_cache, examples=old_examples, torch=torch)
    new_features, new_hidden, new_new_tensor_hash = ctc._validated_feature_cache(new_cache, examples=new_examples, torch=torch)
    if old_hidden != new_hidden or old_tensor_hash != new_new_tensor_hash or set(old_features) != set(new_features):
        raise ValueError("feature tensor identity changed during checkpoint rebind")

    checkpoint = torch.load(old_checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != ctc.CHECKPOINT_SCHEMA:
        raise ValueError("unsupported CTC checkpoint schema")
    if checkpoint.get("dataset_sha256") != old_sha or checkpoint.get("feature_cache_sha256") != _sha256(old_cache_path):
        raise ValueError("checkpoint is not bound to the supplied old dataset/cache")
    state = checkpoint.get("head_state_dict")
    if not isinstance(state, Mapping) or ctc._state_sha256(state) != checkpoint.get("model_sha256"):
        raise ValueError("checkpoint head state hash is invalid")
    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("checkpoint configuration is missing")
    fold_index = int(configuration.get("fold_index"))
    train, validation, split_identity = ctc._split(new_dataset, new_examples, fold_index)
    split_sha = _split_sha256(split_identity, train, validation)

    rebound = dict(checkpoint)
    rebound["dataset_sha256"] = new_sha
    rebound["feature_cache_sha256"] = _sha256(new_cache_path)
    rebound["split_identity"] = split_identity
    rebound["split_sha256"] = split_sha
    rebound["checkpoint_rebind"] = {
        "schema_version": "ctc_checkpoint_dataset_rebind_v1",
        "from_checkpoint_path": str(old_checkpoint_path.resolve()),
        "from_checkpoint_sha256": _sha256(old_checkpoint_path),
        "from_dataset_sha256": old_sha,
        "to_dataset_sha256": new_sha,
        "from_feature_cache_sha256": _sha256(old_cache_path),
        "to_feature_cache_sha256": _sha256(new_cache_path),
        "feature_tensor_sha256_unchanged": old_tensor_hash == new_new_tensor_hash,
        "weights_unchanged": True,
        "split_recomputed": True,
        "verified": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_root / "model.pt"
    output_metrics = output_root / "metrics.json"
    if output_checkpoint.exists() or output_metrics.exists():
        raise FileExistsError(f"checkpoint rebind output already exists: {output_root}")
    _atomic_torch_save(torch, rebound, output_checkpoint)

    metrics = _json(old_metrics_path)
    if metrics.get("checkpoint_sha256") != _sha256(old_checkpoint_path) or metrics.get("dataset_sha256") != old_sha:
        raise ValueError("metrics are not bound to the supplied old checkpoint/dataset")
    updated_metrics = dict(metrics)
    updated_metrics["checkpoint"] = str(output_checkpoint.resolve())
    updated_metrics["checkpoint_sha256"] = _sha256(output_checkpoint)
    updated_metrics["dataset_sha256"] = new_sha
    updated_metrics["feature_cache_sha256"] = _sha256(new_cache_path)
    updated_metrics["split_identity"] = split_identity
    updated_metrics["split_sha256"] = split_sha
    updated_metrics["provenance_rebound"] = rebound["checkpoint_rebind"]
    output_metrics.write_text(json.dumps(updated_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "schema_version": "ctc_checkpoint_dataset_rebind_v1",
        "fold_index": fold_index,
        "dataset_sha256": new_sha,
        "feature_cache_sha256": _sha256(new_cache_path),
        "feature_tensor_sha256": old_tensor_hash,
        "split_identity": split_identity,
        "split_sha256": split_sha,
        "old_checkpoint_sha256": _sha256(old_checkpoint_path),
        "new_checkpoint_sha256": _sha256(output_checkpoint),
        "validation_similarity": updated_metrics.get("validation", {}).get("token_similarity"),
        "weights_unchanged": True,
        "output_checkpoint": str(output_checkpoint.resolve()),
        "output_metrics": str(output_metrics.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebind a CTC checkpoint to a provenance-only equivalent dataset.")
    parser.add_argument("old_dataset", type=Path)
    parser.add_argument("new_dataset", type=Path)
    parser.add_argument("old_cache", type=Path)
    parser.add_argument("new_cache", type=Path)
    parser.add_argument("old_checkpoint", type=Path)
    parser.add_argument("old_metrics", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = rebind_checkpoint(
            args.old_dataset,
            args.new_dataset,
            args.old_cache,
            args.new_cache,
            args.old_checkpoint,
            args.old_metrics,
            args.output_root,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

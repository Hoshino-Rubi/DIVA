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
        values = tuple(row.get(field) for field in IDENTITY_FIELDS)
        example_id = str(row.get("example_id") or "").strip()
        if not example_id or example_id in result:
            raise ValueError("dataset example IDs must be unique and non-empty")
        result[example_id] = values
    return result


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


def rebind_feature_cache(
    old_dataset_path: Path,
    new_dataset_path: Path,
    cache_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    import torch

    old_dataset = _json(old_dataset_path)
    new_dataset = _json(new_dataset_path)
    if old_dataset.get("schema_version") != "cantonese_singing_ctc_dataset_v1":
        raise ValueError("old dataset has an unsupported schema")
    if new_dataset.get("schema_version") != "cantonese_singing_ctc_dataset_v1":
        raise ValueError("new dataset has an unsupported schema")
    old_rows = _identity_rows(old_dataset)
    new_rows = _identity_rows(new_dataset)
    if old_rows != new_rows:
        missing = sorted(set(old_rows) - set(new_rows))
        extra = sorted(set(new_rows) - set(old_rows))
        changed = sorted(example_id for example_id in set(old_rows) & set(new_rows) if old_rows[example_id] != new_rows[example_id])
        raise ValueError(
            "dataset rebind changes acoustic identities; "
            f"missing={missing[:5]} extra={extra[:5]} changed={changed[:5]}"
        )
    old_sha = _sha256(old_dataset_path)
    new_sha = _sha256(new_dataset_path)
    if old_sha == new_sha:
        raise ValueError("new dataset must have a distinct file identity")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cache, Mapping) or cache.get("schema_version") != ctc.FEATURE_CACHE_SCHEMA:
        raise ValueError("unsupported feature cache schema")
    if str(cache.get("dataset_sha256") or "") != old_sha:
        raise ValueError("feature cache is not bound to the old dataset")
    old_examples = ctc._examples(old_dataset)
    new_examples = ctc._examples(new_dataset)
    old_features, old_hidden, old_feature_hash = ctc._validated_feature_cache(
        cache,
        examples=old_examples,
        torch=torch,
    )
    new_features, new_hidden, new_new_hash = ctc._validated_feature_cache(
        cache,
        examples=new_examples,
        torch=torch,
    )
    if old_hidden != new_hidden or old_feature_hash != new_new_hash or set(old_features) != set(new_features):
        raise ValueError("feature cache validation differs between old and new dataset")
    if output_path.exists():
        raise FileExistsError(f"feature cache output already exists: {output_path}")
    rebound = dict(cache)
    rebound["dataset_sha256"] = new_sha
    rebound["dataset_rebind"] = {
        "schema_version": "feature_cache_dataset_rebind_v1",
        "from_dataset_path": str(old_dataset_path.resolve()),
        "from_dataset_sha256": old_sha,
        "to_dataset_path": str(new_dataset_path.resolve()),
        "to_dataset_sha256": new_sha,
        "identity_fields": list(IDENTITY_FIELDS),
        "feature_tensor_sha256_unchanged": True,
        "verified": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(torch, rebound, output_path)
    return {
        "schema_version": "feature_cache_dataset_rebind_v1",
        "from_dataset_sha256": old_sha,
        "to_dataset_sha256": new_sha,
        "cache_sha256": _sha256(cache_path),
        "output_sha256": _sha256(output_path),
        "feature_tensor_sha256": rebound.get("features_sha256"),
        "hidden_size": rebound.get("hidden_size"),
        "example_count": rebound.get("example_count"),
        "output_path": str(output_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebind a feature cache to a provenance-only equivalent dataset.")
    parser.add_argument("old_dataset", type=Path)
    parser.add_argument("new_dataset", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = rebind_feature_cache(args.old_dataset, args.new_dataset, args.cache, args.output)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

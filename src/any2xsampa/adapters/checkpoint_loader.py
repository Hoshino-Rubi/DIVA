"""Validated loading for locally trained recognizer checkpoints.

The project keeps model files outside the source tree.  This module therefore
does not download or discover models: callers pass an explicit checkpoint path
and may inject a deserialiser for tests or alternate runtimes.  Metadata is
validated before a model object is handed to a real recognizer adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping


class CheckpointLoadError(RuntimeError):
    """Raised when a trained checkpoint is unavailable or contract-invalid."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Portable provenance fields required by the recovery training contract."""

    schema_version: str
    vocabulary: tuple[str, ...]
    dataset_sha256: str
    model_sha256: str
    split_identity: str
    seed: int
    configuration: Mapping[str, Any]
    torch_version: str
    cuda_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "vocabulary": list(self.vocabulary),
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "split_identity": self.split_identity,
            "seed": self.seed,
            "configuration": dict(self.configuration),
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
        }


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """A validated payload and its normalized provenance metadata."""

    path: Path
    payload: Mapping[str, Any]
    metadata: CheckpointMetadata
    state_dict: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(payload: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointLoadError(f"checkpoint {label} must be an object")
    return payload


def _text(payload: Mapping[str, Any], keys: tuple[str, ...], *, label: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise CheckpointLoadError(f"checkpoint is missing {label}")


def _digest(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CheckpointLoadError(f"checkpoint {label} must be a SHA-256 hex digest")
    return text


def _vocabulary(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = payload.get("vocabulary", payload.get("vocab", payload.get("phones")))
    if isinstance(raw, Mapping):
        ordered: list[tuple[int, str]] = []
        for token, index in raw.items():
            if isinstance(index, bool):
                raise CheckpointLoadError("checkpoint vocabulary IDs must be integers")
            try:
                ordered.append((int(index), str(token)))
            except (TypeError, ValueError) as exc:
                raise CheckpointLoadError("checkpoint vocabulary IDs must be integers") from exc
        ordered.sort(key=lambda item: item[0])
        values = [token for _, token in ordered]
        if [index for index, _ in ordered] != list(range(len(values))):
            raise CheckpointLoadError("checkpoint vocabulary IDs must be contiguous from zero")
    elif isinstance(raw, (list, tuple)):
        values = [str(item).strip() for item in raw]
    else:
        raise CheckpointLoadError("checkpoint requires a non-empty vocabulary")
    if not values or any(not item for item in values) or len(set(values)) != len(values):
        raise CheckpointLoadError("checkpoint vocabulary must contain unique non-empty tokens")
    return tuple(values)


def _metadata(payload: Mapping[str, Any], *, require_provenance: bool) -> CheckpointMetadata:
    runtime = payload.get("runtime")
    runtime_map = runtime if isinstance(runtime, Mapping) else {}
    config = payload.get("configuration", payload.get("config", payload.get("model_config", {})))
    if not isinstance(config, Mapping):
        raise CheckpointLoadError("checkpoint configuration must be an object")

    schema = _text(payload, ("schema_version", "checkpoint_schema"), label="schema_version")
    vocabulary = _vocabulary(payload)
    dataset_value = payload.get("dataset_sha256", payload.get("dataset_hash"))
    model_value = payload.get("model_sha256", payload.get("model_hash"))
    split_value = payload.get("split_identity", payload.get("split_id", payload.get("split")))
    seed_value = payload.get("seed")
    torch_value = payload.get("torch_version", runtime_map.get("torch_version"))
    cuda_value = payload.get("cuda_version", runtime_map.get("cuda_version"))

    if require_provenance:
        dataset_sha256 = _digest(dataset_value, label="dataset_sha256")
        model_sha256 = _digest(model_value, label="model_sha256")
        split_identity = str(split_value or "").strip()
        if not split_identity:
            raise CheckpointLoadError("checkpoint is missing split_identity")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise CheckpointLoadError("checkpoint seed must be an integer")
        torch_version = str(torch_value or "").strip()
        cuda_version = str(cuda_value or "").strip()
        if not torch_version or not cuda_version:
            raise CheckpointLoadError("checkpoint must record torch_version and cuda_version")
    else:
        dataset_sha256 = _digest(dataset_value, label="dataset_sha256") if dataset_value else ""
        model_sha256 = _digest(model_value, label="model_sha256") if model_value else ""
        split_identity = str(split_value or "").strip()
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            seed_value = 0
        torch_version = str(torch_value or "").strip()
        cuda_version = str(cuda_value or "").strip()

    return CheckpointMetadata(
        schema_version=schema,
        vocabulary=vocabulary,
        dataset_sha256=dataset_sha256,
        model_sha256=model_sha256,
        split_identity=split_identity,
        seed=int(seed_value),
        configuration=dict(config),
        torch_version=torch_version,
        cuda_version=cuda_version,
    )


def _deserialize(path: Path, deserializer: Callable[[Path], object] | None) -> Mapping[str, Any]:
    if deserializer is not None:
        return _mapping(deserializer(path), label="payload")
    if path.suffix.lower() == ".json":
        try:
            return _mapping(json.loads(path.read_text(encoding="utf-8-sig")), label="payload")
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointLoadError(f"checkpoint JSON could not be read: {path}") from exc
    try:
        import torch
    except ImportError as exc:
        raise CheckpointLoadError(
            "loading a binary checkpoint requires optional torch; inject deserializer for tests"
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise CheckpointLoadError(
            "installed PyTorch does not support safe weights_only checkpoint loading"
        ) from exc
    except Exception as exc:
        raise CheckpointLoadError(f"binary checkpoint could not be loaded: {path}") from exc
    return _mapping(payload, label="payload")


def load_trained_checkpoint(
    path: Path | str,
    *,
    expected_schema: str | None = None,
    expected_vocabulary: tuple[str, ...] | list[str] | None = None,
    expected_dataset_sha256: str | None = None,
    expected_model_sha256: str | None = None,
    require_provenance: bool = True,
    deserializer: Callable[[Path], object] | None = None,
) -> LoadedCheckpoint:
    """Load and validate a local trained checkpoint.

    The function deliberately keeps deserialization replaceable.  Production
    binary checkpoints use ``torch.load(..., weights_only=True)`` where
    available; tests and alternate runtimes can inject a reader without adding
    a hard dependency to the project package.
    """

    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"trained checkpoint not found: {checkpoint_path}")
    payload = _deserialize(checkpoint_path, deserializer)
    metadata = _metadata(payload, require_provenance=require_provenance)
    if expected_schema is not None and metadata.schema_version != expected_schema:
        raise CheckpointLoadError(
            f"checkpoint schema mismatch: {metadata.schema_version!r} != {expected_schema!r}"
        )
    if expected_vocabulary is not None and metadata.vocabulary != tuple(expected_vocabulary):
        raise CheckpointLoadError("checkpoint vocabulary does not match the recognizer bundle")
    if expected_dataset_sha256 is not None:
        expected = _digest(expected_dataset_sha256, label="expected_dataset_sha256")
        if metadata.dataset_sha256 != expected:
            raise CheckpointLoadError("checkpoint dataset SHA-256 does not match the expected hash")
    if expected_model_sha256 is not None:
        expected = _digest(expected_model_sha256, label="expected_model_sha256")
        if metadata.model_sha256 != expected:
            raise CheckpointLoadError("checkpoint model SHA-256 does not match the expected hash")

    raw_state = payload.get(
        "state_dict",
        payload.get("model_state_dict", payload.get("head_state_dict")),
    )
    state_dict = _mapping(raw_state, label="state_dict") if raw_state is not None else {}
    if not state_dict:
        raise CheckpointLoadError("checkpoint trained state_dict must be non-empty")
    return LoadedCheckpoint(
        path=checkpoint_path,
        payload=payload,
        metadata=metadata,
        state_dict=state_dict,
    )


__all__ = [
    "CheckpointLoadError",
    "CheckpointMetadata",
    "LoadedCheckpoint",
    "load_trained_checkpoint",
    "sha256_file",
]

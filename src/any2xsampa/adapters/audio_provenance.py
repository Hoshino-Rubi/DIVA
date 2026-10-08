"""Hash-bound, read-only audio provenance for external evidence artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping
import wave


class AudioProvenanceError(ValueError):
    """Raised when an audio artifact cannot be bound to its source WAV."""


def describe_audio_file(path: Path | str) -> dict[str, Any]:
    """Return stable WAV facts and a streaming SHA-256 digest."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioProvenanceError(f"Audio file not found: {audio_path}")
    digest = hashlib.sha256()
    with audio_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    try:
        with wave.open(str(audio_path), "rb") as stream:
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            frame_count = stream.getnframes()
    except (OSError, wave.Error) as exc:
        raise AudioProvenanceError(f"Audio file is not readable PCM WAV: {audio_path}") from exc
    if sample_rate < 1 or channels < 1 or sample_width < 1 or frame_count < 0:
        raise AudioProvenanceError(f"Audio WAV metadata is invalid: {audio_path}")
    return {
        "sha256": digest.hexdigest(),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_sec": frame_count / sample_rate,
    }


def validate_audio_binding(
    binding: Mapping[str, Any] | None,
    audio_path: Path | str,
    *,
    duration_tolerance_sec: float = 0.05,
) -> dict[str, Any]:
    """Validate a producer binding, preserving an explicit legacy status."""

    if binding is None:
        return {"status": "legacy_unbound", "verified": False}
    if not isinstance(binding, Mapping):
        raise AudioProvenanceError("producer.input_audio must be an object")
    declared_hash = str(binding.get("sha256", "")).strip().lower()
    if len(declared_hash) != 64 or any(char not in "0123456789abcdef" for char in declared_hash):
        raise AudioProvenanceError("producer.input_audio.sha256 must be a SHA-256 digest")
    actual = describe_audio_file(audio_path)
    if declared_hash != actual["sha256"]:
        raise AudioProvenanceError(
            "posterior audio hash mismatch: "
            f"declared={declared_hash} actual={actual['sha256']}"
        )
    declared_rate = binding.get("sample_rate")
    if declared_rate is not None and int(declared_rate) != actual["sample_rate"]:
        raise AudioProvenanceError("posterior audio sample_rate does not match the input WAV")
    declared_frames = binding.get("frame_count")
    if declared_frames is not None and int(declared_frames) != actual["frame_count"]:
        raise AudioProvenanceError("posterior audio frame_count does not match the input WAV")
    declared_duration = binding.get("duration_sec")
    if declared_duration is not None and abs(float(declared_duration) - actual["duration_sec"]) > duration_tolerance_sec:
        raise AudioProvenanceError("posterior audio duration does not match the input WAV")
    return {"status": "verified", "verified": True, "declared": dict(binding), "actual": actual}


__all__ = ["AudioProvenanceError", "describe_audio_file", "validate_audio_binding"]

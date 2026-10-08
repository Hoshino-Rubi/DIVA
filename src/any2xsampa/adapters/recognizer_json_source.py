from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from any2xsampa.core.io import load_json
from any2xsampa.core.models import PhoneSegment


JsonPayloadNormalizer = Callable[..., dict[str, Any]]
JsonSegmentBuilder = Callable[..., list[PhoneSegment]]
SegmentPostprocessor = Callable[[PhoneSegment], None]


def load_segments_from_json_source(
    *,
    audio_path: Path,
    json_path: Path,
    source_mode: str,
    backend_name: str,
    source_label: str,
    missing_path_label: str,
    lyrics: str | None,
    normalize_payload: JsonPayloadNormalizer,
    build_segments: JsonSegmentBuilder,
    source_override: str | None = None,
    postprocess_segment: SegmentPostprocessor | None = None,
) -> list[PhoneSegment]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Vocal audio file not found: {audio_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"{missing_path_label} not found: {json_path}")

    payload = load_json(json_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{source_label} must contain a JSON object payload.")

    try:
        normalized = normalize_payload(
            payload,
            source_mode=source_mode,
            backend_name=backend_name,
            source_label=source_label,
        )
    except ValueError as exc:
        raise ValueError(f"{source_label} replay schema error in {json_path}: {exc}") from exc
    segments = build_segments(normalized, source_mode=source_mode)
    for segment in segments:
        segment.metadata["source"] = source_override or source_label
        segment.metadata["lyrics_guided"] = bool(lyrics)
        if postprocess_segment is not None:
            postprocess_segment(segment)
    return segments

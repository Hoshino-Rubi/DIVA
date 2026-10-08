"""Rebase a verified review clip onto its immutable full-song vocal stem."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import wave
from typing import Any

from any2xsampa.lyrics.timeline import (
    LyricsTimeline,
    LyricsTimelineError,
    canonical_timeline_sha256,
    load_lyrics_timeline,
    sha256_file,
)


ALIGNMENT_SCHEMA_VERSION = "forced_phone_alignment_v1"
OUTPUT_SCHEMA_VERSION = "forced_phone_review_draft_v1"


class RebaseError(ValueError):
    """A clip cannot be proven to be an exact full-song review window."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RebaseError(f"missing JSON input: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RebaseError(f"invalid JSON input: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RebaseError(f"JSON input must be an object: {path}")
    return dict(payload)


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise RebaseError(f"{label} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RebaseError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise RebaseError(f"{label} must be finite")
    return result


def _pcm_crop_evidence(clip_path: Path, full_path: Path, offset_sec: float) -> dict[str, Any]:
    try:
        with wave.open(str(clip_path), "rb") as clip, wave.open(str(full_path), "rb") as full:
            clip_format = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(), clip.getcomptype())
            full_format = (full.getnchannels(), full.getsampwidth(), full.getframerate(), full.getcomptype())
            if clip_format != full_format or clip.getcomptype() != "NONE":
                raise RebaseError("clip and full-song audio must have identical uncompressed PCM formats")
            exact_start_frame = offset_sec * full.getframerate()
            start_frame = round(exact_start_frame)
            if abs(exact_start_frame - start_frame) > 1e-6:
                raise RebaseError("clip offset does not resolve to an exact PCM frame")
            end_frame = start_frame + clip.getnframes()
            if start_frame < 0 or end_frame > full.getnframes():
                raise RebaseError("clip range falls outside the full-song audio")
            full.setpos(start_frame)
            full_pcm = full.readframes(clip.getnframes())
            clip_pcm = clip.readframes(clip.getnframes())
            if full_pcm != clip_pcm:
                raise RebaseError("clip PCM does not exactly match the declared full-song range")
            pcm_hash = hashlib.sha256(clip_pcm).hexdigest()
            return {
                "sample_rate": full.getframerate(),
                "channels": full.getnchannels(),
                "sample_width_bytes": full.getsampwidth(),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": start_frame / full.getframerate(),
                "end_sec": end_frame / full.getframerate(),
                "frame_count": clip.getnframes(),
                "pcm_sha256": pcm_hash,
                "exact_pcm_match": True,
            }
    except RebaseError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise RebaseError("clip and full-song inputs must be readable PCM WAV files") from exc


def _validate_timeline_window(
    clip_timeline: LyricsTimeline,
    full_timeline: LyricsTimeline,
    *,
    offset_sec: float,
    lyric_index_offset: int,
) -> None:
    for clip_item in clip_timeline.items:
        full_index = clip_item.index + lyric_index_offset
        if full_index < 0 or full_index >= len(full_timeline.items):
            raise RebaseError(f"clip lyric item {clip_item.index} has no full-song counterpart")
        full_item = full_timeline.items[full_index]
        if clip_item.text != full_item.text or (clip_item.jyutping or "") != (full_item.jyutping or ""):
            raise RebaseError(f"clip lyric item {clip_item.index} text/Jyutping differs from the full song")
        if (
            abs(clip_item.start_sec + offset_sec - full_item.start_sec) > 1e-6
            or abs(clip_item.end_sec + offset_sec - full_item.end_sec) > 1e-6
        ):
            raise RebaseError(f"clip lyric item {clip_item.index} timing differs from the full song")


def _rebase_segments(
    raw_segments: object,
    *,
    offset_sec: float,
    lyric_index_offset: int,
    clip_duration: float,
) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)) or not raw_segments:
        raise RebaseError("clip alignment has no phone segments")
    result: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise RebaseError(f"clip phone segment {index} must be an object")
        start = _finite(raw.get("start_sec"), label=f"segment {index} start_sec")
        end = _finite(raw.get("end_sec"), label=f"segment {index} end_sec")
        metadata = raw.get("metadata")
        lyric_index = metadata.get("lyric_item_index") if isinstance(metadata, Mapping) else None
        candidates = raw.get("candidate_phones")
        if (
            start < 0.0
            or end <= start
            or start < previous_end
            or end > clip_duration + 1e-6
            or isinstance(lyric_index, bool)
            or not isinstance(lyric_index, int)
            or not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or not candidates
        ):
            raise RebaseError(f"clip phone segment {index} is invalid")
        rebased = dict(raw)
        rebased["start_sec"] = start + offset_sec
        rebased["end_sec"] = end + offset_sec
        rebased_metadata = dict(metadata)
        rebased_metadata.update(
            {
                "lyric_item_index": lyric_index + lyric_index_offset,
                "review_draft_time_offset_sec": offset_sec,
                "review_draft_source_segment_index": index,
            }
        )
        for key in ("human_reviewed", "reviewer_id", "reviewed_at", "review_status", "approved"):
            rebased_metadata.pop(key, None)
        rebased["metadata"] = rebased_metadata
        rebased.pop("reviewed", None)
        result.append(rebased)
        previous_end = end
    return result


def rebase_phone_review_clip(
    *,
    clip_audio: Path | str,
    full_audio: Path | str,
    clip_alignment: Path | str,
    clip_timeline: Path | str,
    full_timeline: Path | str,
    offset_sec: float,
    lyric_index_offset: int,
) -> dict[str, Any]:
    clip_audio_path = Path(clip_audio).expanduser().resolve()
    full_audio_path = Path(full_audio).expanduser().resolve()
    clip_alignment_path = Path(clip_alignment).expanduser().resolve()
    clip_timeline_path = Path(clip_timeline).expanduser().resolve()
    full_timeline_path = Path(full_timeline).expanduser().resolve()
    offset = _finite(offset_sec, label="offset_sec")
    if offset < 0.0 or isinstance(lyric_index_offset, bool) or not isinstance(lyric_index_offset, int):
        raise RebaseError("offsets must be non-negative and lyric_index_offset must be an integer")

    crop = _pcm_crop_evidence(clip_audio_path, full_audio_path, offset)
    try:
        clip_timeline_value = load_lyrics_timeline(clip_timeline_path)
        full_timeline_value = load_lyrics_timeline(full_timeline_path)
    except LyricsTimelineError as exc:
        raise RebaseError(str(exc)) from exc
    _validate_timeline_window(
        clip_timeline_value,
        full_timeline_value,
        offset_sec=offset,
        lyric_index_offset=lyric_index_offset,
    )

    alignment = _read_json(clip_alignment_path)
    if alignment.get("schema_version") != ALIGNMENT_SCHEMA_VERSION or alignment.get("model_output") is not True:
        raise RebaseError("clip alignment must be forced_phone_alignment_v1 model output")
    clip_audio_hash = sha256_file(clip_audio_path)
    if str(alignment.get("vocal_stem_sha256") or alignment.get("audio_sha256") or "").lower() != clip_audio_hash:
        raise RebaseError("clip alignment is bound to a different clip audio file")
    clip_timeline_hash = canonical_timeline_sha256(clip_timeline_value)
    if str(alignment.get("lyrics_timeline_sha256") or "").lower() != clip_timeline_hash:
        raise RebaseError("clip alignment is bound to a different clip lyrics timeline")

    rebased_segments = _rebase_segments(
        alignment.get("segments"),
        offset_sec=offset,
        lyric_index_offset=lyric_index_offset,
        clip_duration=crop["frame_count"] / crop["sample_rate"],
    )
    full_audio_hash = sha256_file(full_audio_path)
    full_timeline_hash = canonical_timeline_sha256(full_timeline_value)
    provenance = dict(alignment.get("provenance")) if isinstance(alignment.get("provenance"), Mapping) else {}
    provenance["review_draft_rebase"] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "machine_draft_pending_human_phone_review",
        "source_clip_alignment_path": str(clip_alignment_path),
        "source_clip_alignment_sha256": sha256_file(clip_alignment_path),
        "source_clip_audio_path": str(clip_audio_path),
        "source_clip_audio_sha256": clip_audio_hash,
        "full_vocal_stem_path": str(full_audio_path),
        "full_vocal_stem_sha256": full_audio_hash,
        "clip_timeline_path": str(clip_timeline_path),
        "clip_timeline_sha256": clip_timeline_hash,
        "full_timeline_path": str(full_timeline_path),
        "full_timeline_sha256": full_timeline_hash,
        "lyric_index_offset": lyric_index_offset,
        "pcm_crop": crop,
    }
    result = dict(alignment)
    result.update(
        {
            "schema_version": ALIGNMENT_SCHEMA_VERSION,
            "vocal_stem_sha256": full_audio_hash,
            "lyrics_timeline_sha256": full_timeline_hash,
            "alignment_source": f"{alignment.get('alignment_source', 'machine')}_rebased_review_draft",
            "model_output": True,
            "segments": rebased_segments,
            "provenance": provenance,
            "review": {
                "status": "pending_human_phone_review",
                "scope": "short_clip",
                "reviewed_range_sec": [crop["start_sec"], crop["end_sec"]],
            },
        }
    )
    result.pop("audio_sha256", None)
    return result


def write_rebased_review_draft(output_path: Path | str, payload: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RebaseError(f"refusing to overwrite existing output: {output}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    return {"path": str(output), "sha256": hashlib.sha256(encoded).hexdigest(), "segment_count": len(payload["segments"])}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove and rebase a phone-review clip onto its full vocal stem.")
    parser.add_argument("--clip-audio", required=True, type=Path)
    parser.add_argument("--full-audio", required=True, type=Path)
    parser.add_argument("--clip-alignment", required=True, type=Path)
    parser.add_argument("--clip-timeline", required=True, type=Path)
    parser.add_argument("--full-timeline", required=True, type=Path)
    parser.add_argument("--offset-sec", required=True, type=float)
    parser.add_argument("--lyric-index-offset", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = rebase_phone_review_clip(
            clip_audio=args.clip_audio,
            full_audio=args.full_audio,
            clip_alignment=args.clip_alignment,
            clip_timeline=args.clip_timeline,
            full_timeline=args.full_timeline,
            offset_sec=args.offset_sec,
            lyric_index_offset=args.lyric_index_offset,
        )
        report = write_rebased_review_draft(args.output, payload)
    except RebaseError as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

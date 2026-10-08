"""Local browser reviewer for forced_phone_alignment_v1 artifacts.

The source WAV and machine alignment are read-only.  A reviewer edits phone
rows and explicitly checks every row before this tool writes a new alignment
and a detached forced_phone_acceptance_v1 record.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import mimetypes
import os
import re
import threading
import wave
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from any2xsampa.adapters.forced_alignment_acceptance import (
    ForcedPhoneAcceptanceError,
    validate_short_clip_predecessor,
)
from any2xsampa.lyrics.timeline import (
    LyricsTimelineError,
    canonical_timeline_sha256,
    load_lyrics_timeline,
)

ALIGNMENT_SCHEMA_VERSION = "forced_phone_alignment_v1"
ACCEPTANCE_SCHEMA_VERSION = "forced_phone_acceptance_v1"
HUMAN_REVIEW_ROUTE = "human_reviewed_phone_alignment"
SHORT_CLIP_SCOPE = "short_clip"
FULL_SONG_SCOPE = "full_song"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_FULL_REVIEW_PORT = 8767
DEFAULT_REVIEWED_ALIGNMENT_NAME = "reviewed_forced_phone_alignment_v1.json"
DEFAULT_ACCEPTANCE_NAME = "forced_phone_acceptance_v1.json"
BROWSER_DRAFT_SCHEMA_VERSION = "phone_review_browser_draft_v3"
MAX_POST_BYTES = 4 * 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ReviewError(ValueError):
    """A review input or save request failed a hard contract check."""

    def __init__(self, message: str, *, error_kind: str = "invalid_review") -> None:
        super().__init__(message)
        self.error_kind = error_kind


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReviewError(f"cannot read file: {path}", error_kind="missing_input") from exc
    return digest.hexdigest()


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ReviewError(f"{label} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReviewError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ReviewError(f"{label} must be finite")
    return result


def wav_duration(path: Path) -> float:
    """Read a PCM WAV duration and reject unreadable audio."""

    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            if rate <= 0 or frames < 0:
                raise ReviewError("WAV sample rate/frame count is invalid", error_kind="invalid_audio")
            return frames / rate
    except ReviewError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise ReviewError("audio must be a readable PCM WAV", error_kind="invalid_audio") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ReviewError(f"missing alignment: {path}", error_kind="missing_input") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"invalid alignment JSON: {path}", error_kind="invalid_alignment") from exc
    if not isinstance(payload, Mapping):
        raise ReviewError("alignment JSON must be an object", error_kind="invalid_alignment")
    return dict(payload)


def _symbol_from_segment(raw: Mapping[str, Any], *, index: int) -> str:
    value = raw.get("normalized_ipa")
    if value is None or not str(value).strip():
        candidates = raw.get("candidate_phones")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)) and candidates:
            candidate = candidates[0]
            if isinstance(candidate, Mapping):
                value = candidate.get("symbol", candidate.get("phone", candidate.get("label")))
    symbol = str(value or "").strip()
    if not symbol:
        raise ReviewError(f"segment {index} has no phone symbol", error_kind="invalid_alignment")
    return symbol


def _lyric_index_from_segment(raw: Mapping[str, Any], *, index: int) -> int:
    metadata = raw.get("metadata")
    value = metadata.get("lyric_item_index") if isinstance(metadata, Mapping) else raw.get("lyric_item_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReviewError(f"segment {index} has an invalid lyric index", error_kind="invalid_alignment")
    return value


def _original_confidence(raw: Mapping[str, Any]) -> float:
    value = raw.get("confidence")
    if value is None:
        candidates = raw.get("candidate_phones")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)) and candidates:
            first = candidates[0]
            if isinstance(first, Mapping):
                value = first.get("confidence", first.get("score", 1.0))
    try:
        confidence = _finite_number(1.0 if value is None else value, label="segment confidence")
    except ReviewError:
        confidence = 1.0
    return min(1.0, max(0.0, confidence))


def _validate_raw_segments(payload: Mapping[str, Any], *, duration: float) -> list[dict[str, Any]]:
    if payload.get("schema_version") != ALIGNMENT_SCHEMA_VERSION:
        raise ReviewError(f"alignment schema must be {ALIGNMENT_SCHEMA_VERSION}", error_kind="invalid_alignment")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)) or not raw_segments:
        raise ReviewError("alignment contains no phone segments", error_kind="invalid_alignment")
    rows: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise ReviewError(f"segment {index} must be an object", error_kind="invalid_alignment")
        candidates = raw.get("candidate_phones")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
            raise ReviewError(f"segment {index} has no candidate_phones", error_kind="invalid_alignment")
        previous_confidence = float("inf")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ReviewError(f"segment {index} candidate {candidate_index} must be an object", error_kind="invalid_alignment")
            candidate_symbol = str(candidate.get("symbol", candidate.get("phone", candidate.get("label"))) or "").strip()
            if not candidate_symbol:
                raise ReviewError(f"segment {index} candidate {candidate_index} has no symbol", error_kind="invalid_alignment")
            candidate_confidence = _finite_number(candidate.get("confidence", candidate.get("score", 0.0)), label="candidate confidence")
            if not 0.0 <= candidate_confidence <= 1.0 or candidate_confidence > previous_confidence:
                raise ReviewError(f"segment {index} candidate confidences are invalid", error_kind="invalid_alignment")
            previous_confidence = candidate_confidence
        start = _finite_number(raw.get("start_sec"), label=f"segment {index} start_sec")
        end = _finite_number(raw.get("end_sec"), label=f"segment {index} end_sec")
        if start < 0.0 or end <= start or start < previous_end or end > duration + 1e-6:
            raise ReviewError(f"segment {index} has invalid, overlapping, or out-of-audio timing", error_kind="invalid_alignment")
        rows.append({
            "index": index,
            "symbol": _symbol_from_segment(raw, index=index),
            "start_sec": start,
            "end_sec": end,
            "lyric_item_index": _lyric_index_from_segment(raw, index=index),
            "confidence": _original_confidence(raw),
            "metadata": dict(raw.get("metadata")) if isinstance(raw.get("metadata"), Mapping) else {},
            "raw": dict(raw),
            "reviewed": False,
        })
        previous_end = end
    return rows


def load_review_inputs(
    audio_path: Path | str,
    alignment_path: Path | str,
    *,
    lyrics_timeline_path: Path | str | None = None,
    lyrics_timeline_audio_path: Path | str | None = None,
    review_scope: str = SHORT_CLIP_SCOPE,
    reviewed_range_sec: Sequence[float] | None = None,
    short_clip_acceptance_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load immutable review inputs and return browser-safe row data."""

    audio = Path(audio_path).expanduser().resolve()
    alignment = Path(alignment_path).expanduser().resolve()
    if not audio.is_file():
        raise ReviewError(f"missing audio: {audio}", error_kind="missing_input")
    duration = wav_duration(audio)
    payload = _read_json(alignment)
    audio_hash = sha256_file(audio)
    canonical_hash = str(payload.get("vocal_stem_sha256") or "").strip().lower()
    legacy_hash = str(payload.get("audio_sha256") or "").strip().lower()
    if canonical_hash and legacy_hash and canonical_hash != legacy_hash:
        raise ReviewError("alignment contains conflicting audio hashes", error_kind="audio_hash_mismatch")
    declared_audio_hash = canonical_hash or legacy_hash
    if declared_audio_hash != audio_hash:
        raise ReviewError("alignment is bound to a different vocal stem", error_kind="audio_hash_mismatch")
    if payload.get("model_output") is not True:
        raise ReviewError("alignment must explicitly declare model_output=true", error_kind="invalid_alignment")
    alignment_source = str(payload.get("alignment_source") or "").strip()
    if not alignment_source:
        raise ReviewError("alignment_source must be non-empty", error_kind="invalid_alignment")
    model_payload = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    direct_model_hash = str(payload.get("model_hash") or "").strip().lower()
    nested_model_hash = str(model_payload.get("hash") or "").strip().lower()
    if direct_model_hash and nested_model_hash and direct_model_hash != nested_model_hash:
        raise ReviewError("alignment contains conflicting model hashes", error_kind="invalid_alignment")
    model_hash = direct_model_hash or nested_model_hash
    if not _SHA256.fullmatch(model_hash):
        raise ReviewError("alignment model_hash must be SHA-256", error_kind="invalid_alignment")
    rows = _validate_raw_segments(payload, duration=duration)
    lyrics_timeline: dict[str, Any] | None = None
    timeline_path: Path | None = None
    timeline_audio_path: Path | None = None
    timeline_file_hash: str | None = None
    timeline_audio_hash: str | None = None
    declared_timeline_hash = str(payload.get("lyrics_timeline_sha256") or "").strip().lower()
    if declared_timeline_hash and not _SHA256.fullmatch(declared_timeline_hash):
        raise ReviewError(
            "alignment lyrics_timeline_sha256 must be SHA-256",
            error_kind="invalid_alignment",
        )
    if declared_timeline_hash and lyrics_timeline_path is None:
        raise ReviewError(
            "alignment declares lyrics_timeline_sha256 but no lyrics timeline was supplied",
            error_kind="missing_lyrics_timing_evidence",
        )
    if lyrics_timeline_path is not None and not declared_timeline_hash:
        raise ReviewError(
            "lyrics timeline review context requires alignment lyrics_timeline_sha256",
            error_kind="invalid_alignment",
        )
    if lyrics_timeline_path is not None:
        timeline_path = Path(lyrics_timeline_path).expanduser().resolve()
        try:
            timeline = load_lyrics_timeline(timeline_path)
            if lyrics_timeline_audio_path is None:
                if timeline.audio_sha256 != audio_hash:
                    raise ReviewError(
                        "lyrics timeline is bound to separate source audio; "
                        "supply --lyrics-timeline-audio",
                        error_kind="missing_lyrics_timing_evidence",
                    )
                timeline_audio_path = audio
            else:
                timeline_audio_path = Path(lyrics_timeline_audio_path).expanduser().resolve()
            timeline = load_lyrics_timeline(timeline_path, audio_path=timeline_audio_path)
        except LyricsTimelineError as exc:
            raise ReviewError(str(exc), error_kind=exc.error_kind) from exc
        actual_timeline_hash = canonical_timeline_sha256(timeline)
        if declared_timeline_hash != actual_timeline_hash:
            raise ReviewError(
                "alignment is bound to a different lyrics timeline",
                error_kind="lyrics_timeline_hash_mismatch",
            )
        lyric_items = {item.index: item for item in timeline.items}
        for row in rows:
            lyric_index = int(row["lyric_item_index"])
            lyric = lyric_items.get(lyric_index)
            if lyric is None:
                raise ReviewError(
                    f"segment {row['index']} references a missing lyric item",
                    error_kind="invalid_alignment",
                )
            row.update(
                {
                    "lyric_text": lyric.text,
                    "jyutping": lyric.jyutping or "",
                    "lyric_start_sec": lyric.start_sec,
                    "lyric_end_sec": lyric.end_sec,
                }
            )
        timeline_file_hash = sha256_file(timeline_path)
        timeline_audio_hash = sha256_file(timeline_audio_path)
        lyrics_timeline = timeline.to_dict()
    scope = str(review_scope or "").strip()
    if scope not in {SHORT_CLIP_SCOPE, FULL_SONG_SCOPE}:
        raise ReviewError(f"unsupported review scope: {scope}", error_kind="invalid_scope")
    if reviewed_range_sec is None:
        initial_start, initial_end = (0.0, duration) if scope == FULL_SONG_SCOPE else (0.0, min(duration, 20.0))
    else:
        if isinstance(reviewed_range_sec, (str, bytes)) or len(reviewed_range_sec) != 2:
            raise ReviewError("reviewed range must contain start and end", error_kind="invalid_scope")
        initial_start = _finite_number(reviewed_range_sec[0], label="review start")
        initial_end = _finite_number(reviewed_range_sec[1], label="review end")
    if initial_start < 0.0 or initial_end <= initial_start or initial_end > duration + 1e-6:
        raise ReviewError("configured review range is outside the vocal stem", error_kind="invalid_scope")
    if scope == SHORT_CLIP_SCOPE and not 10.0 <= initial_end - initial_start <= 20.0:
        raise ReviewError("short-clip review range must be 10-20 seconds", error_kind="invalid_scope")
    if scope == FULL_SONG_SCOPE and (initial_start > 0.05 or initial_end < duration - 0.05):
        raise ReviewError("full-song review range must cover the complete vocal stem", error_kind="invalid_scope")
    if scope == SHORT_CLIP_SCOPE:
        overlapping_rows = [
            row
            for row in rows
            if row["end_sec"] > initial_start + 1e-6
            and row["start_sec"] < initial_end - 1e-6
        ]
        if not overlapping_rows:
            raise ReviewError("review window contains no phone segments", error_kind="invalid_scope")
        if any(
            row["start_sec"] < initial_start - 1e-6
            or row["end_sec"] > initial_end + 1e-6
            for row in overlapping_rows
        ):
            raise ReviewError(
                "review window cuts through a phone segment; use complete phone boundaries",
                error_kind="invalid_scope",
            )
        rows = overlapping_rows
    elif rows[0]["start_sec"] < initial_start - 1e-6 or rows[-1]["end_sec"] > initial_end + 1e-6:
        raise ReviewError("phone segments fall outside the configured review range", error_kind="invalid_scope")
    predecessor: dict[str, str] | None = None
    if scope == FULL_SONG_SCOPE:
        if short_clip_acceptance_path is None:
            raise ReviewError("full-song review requires a short-clip acceptance", error_kind="missing_predecessor")
        predecessor_path = Path(short_clip_acceptance_path).expanduser().resolve()
        if not predecessor_path.is_file():
            raise ReviewError("short-clip acceptance is missing", error_kind="missing_predecessor")
        try:
            validate_short_clip_predecessor(
                predecessor_path,
                vocal_stem_sha256=audio_hash,
            )
        except ForcedPhoneAcceptanceError as exc:
            raise ReviewError(str(exc), error_kind="invalid_predecessor") from exc
        predecessor = {
            "acceptance_path": str(predecessor_path),
            "acceptance_sha256": sha256_file(predecessor_path),
        }
    for row in rows:
        row.pop("raw", None)
    result: dict[str, Any] = {
        "audio_path": str(audio),
        "alignment_path": str(alignment),
        "audio_sha256": audio_hash,
        "alignment_sha256": sha256_file(alignment),
        "duration_sec": duration,
        "reviewed_range_sec": [initial_start, initial_end],
        "review_scope": scope,
        "segments": rows,
        "alignment": payload,
    }
    if predecessor is not None:
        result["short_clip_predecessor"] = predecessor
    if (
        timeline_path is not None
        and timeline_audio_path is not None
        and lyrics_timeline is not None
        and timeline_file_hash is not None
        and timeline_audio_hash is not None
    ):
        result.update(
            {
                "lyrics_timeline_path": str(timeline_path),
                "lyrics_timeline_audio_path": str(timeline_audio_path),
                "lyrics_timeline_audio_sha256": timeline_audio_hash,
                "lyrics_timeline_file_sha256": timeline_file_hash,
                "lyrics_timeline_sha256": declared_timeline_hash,
                "lyrics_timeline": lyrics_timeline,
            }
        )
    return result


def _safe_output_name(name: str, *, label: str) -> str:
    value = str(name or "").strip()
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."} or Path(value).name != value:
        raise ReviewError(f"{label} is not a safe file name", error_kind="path_traversal")
    if not value.lower().endswith(".json"):
        raise ReviewError(f"{label} must be a JSON file name", error_kind="path_traversal")
    return value


def _canonical_segments_sha256(segments: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(list(segments), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_inventory_sha256(symbols: Sequence[str]) -> str:
    encoded = json.dumps(list(symbols), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phone_segment_dict(
    row: Mapping[str, Any],
    *,
    reviewer_id: str,
    reviewed_at: str,
    alignment_source: str,
    model_hash: str,
) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "").strip()
    if not symbol:
        raise ReviewError("phone symbol is required")
    start = _finite_number(row.get("start_sec"), label="phone start_sec")
    end = _finite_number(row.get("end_sec"), label="phone end_sec")
    lyric_index = row.get("lyric_item_index")
    if isinstance(lyric_index, bool) or not isinstance(lyric_index, int) or lyric_index < 0:
        raise ReviewError("lyric_item_index must be a non-negative integer")
    confidence = _original_confidence(row)
    metadata = dict(row.get("metadata")) if isinstance(row.get("metadata"), Mapping) else {}
    # Do not carry a machine-side approval flag into the human evidence.
    metadata.pop("reviewed", None)
    metadata.pop("approved", None)
    metadata.pop("review_status", None)
    metadata.update({
        "alignment_source": alignment_source,
        "lyric_item_index": lyric_index,
        "model_hash": model_hash,
        "source_mode": "real",
        "truth_layer": "phones_ipa_articulatory_features",
        "interpolation_reason": None,
        "alignment_confidence": confidence,
        "human_reviewed": True,
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
    })
    # This is the exact JSON shape emitted by PhoneSegment.to_dict(), with a
    # small explicit reviewed marker retained in the alignment artifact.
    return {
        "candidate_phones": [{"symbol": symbol, "confidence": confidence}],
        "confidence": confidence,
        "normalized_ipa": symbol,
        "xsampa": "",
        "engine_export": None,
        "dialect_result": None,
        "syllable_group": None,
        "metadata": metadata,
        "start_sec": start,
        "end_sec": end,
        "reviewed": True,
    }


def _validate_review_request(
    request: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, str, float, float, str]:
    reviewer_id = str(request.get("reviewer_id") or "").strip()
    if not reviewer_id:
        raise ReviewError("reviewer_id is required", error_kind="reviewer_required")
    review_method = str(request.get("review_method", request.get("method")) or "").strip()
    if not review_method:
        raise ReviewError("review_method is required", error_kind="reviewer_required")
    reviewed_at = str(request.get("reviewed_at") or datetime.now(timezone.utc).isoformat()).strip()
    try:
        parsed_timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewError("reviewed_at must be an ISO-8601 timestamp", error_kind="reviewer_required") from exc
    if parsed_timestamp.tzinfo is None:
        raise ReviewError("reviewed_at must include a timezone", error_kind="reviewer_required")

    raw_range = request.get("reviewed_range_sec")
    if raw_range is None:
        raw_range = [request.get("review_start_sec"), request.get("review_end_sec")]
    if not isinstance(raw_range, Sequence) or isinstance(raw_range, (str, bytes)) or len(raw_range) != 2:
        raise ReviewError("reviewed_range_sec must contain start and end")
    start = _finite_number(raw_range[0], label="reviewed range start")
    end = _finite_number(raw_range[1], label="reviewed range end")
    range_duration = end - start
    audio_duration = _finite_number(source.get("duration_sec"), label="audio duration")
    scope = str(source.get("review_scope") or SHORT_CLIP_SCOPE).strip()
    if start < 0.0 or end <= start or end > audio_duration + 1e-6:
        raise ReviewError("reviewed range exceeds the vocal stem")
    if scope == SHORT_CLIP_SCOPE and not 10.0 <= range_duration <= 20.0:
        raise ReviewError("short-clip review range must be 10-20 seconds")
    if scope == FULL_SONG_SCOPE and (start > 0.05 or end < audio_duration - 0.05):
        raise ReviewError("full-song review range must cover the complete vocal stem")
    if scope not in {SHORT_CLIP_SCOPE, FULL_SONG_SCOPE}:
        raise ReviewError(f"unsupported review scope: {scope}")

    raw_segments = request.get("segments")
    source_segments = source.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise ReviewError("segments must be a list")
    if not raw_segments:
        raise ReviewError("segments must contain at least one phone row")
    if (
        not isinstance(source_segments, Sequence)
        or isinstance(source_segments, (str, bytes))
        or not source_segments
    ):
        raise ReviewError("source alignment contains no phone rows")
    timeline_items: dict[int, Mapping[str, Any]] = {}
    source_timeline = source.get("lyrics_timeline")
    raw_timeline_items = source_timeline.get("items") if isinstance(source_timeline, Mapping) else None
    timeline_context = isinstance(raw_timeline_items, Sequence) and not isinstance(raw_timeline_items, (str, bytes))
    if timeline_context:
        for timeline_item in raw_timeline_items:
            if isinstance(timeline_item, Mapping):
                item_index = timeline_item.get("index")
                if isinstance(item_index, int) and not isinstance(item_index, bool):
                    timeline_items[item_index] = timeline_item
    normalized: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise ReviewError(f"segment {index} must be an object")
        if "index" in raw and raw.get("index") != index:
            raise ReviewError(f"segment {index} has an unexpected index")
        if raw.get("reviewed") is not True:
            raise ReviewError(f"segment {index} has not been explicitly reviewed", error_kind="unreviewed_phone")
        # The submitted rows are the human truth. Machine-row metadata is kept
        # only at artifact level so insertion/deletion cannot attach another
        # suggested phone's provenance or confidence to this reviewed phone.
        row = {
            key: raw[key]
            for key in ("symbol", "start_sec", "end_sec", "lyric_item_index", "confidence", "metadata")
            if key in raw
        }
        row["start_sec"] = _finite_number(row.get("start_sec"), label=f"segment {index} start_sec")
        row["end_sec"] = _finite_number(row.get("end_sec"), label=f"segment {index} end_sec")
        if row["start_sec"] < start - 1e-6 or row["end_sec"] > end + 1e-6:
            raise ReviewError(f"segment {index} falls outside the reviewed range")
        if row["end_sec"] <= row["start_sec"] or row["start_sec"] < previous_end:
            raise ReviewError(f"segment {index} has invalid or overlapping timing")
        row["symbol"] = str(row.get("symbol") or "").strip()
        if not row["symbol"]:
            raise ReviewError(f"segment {index} phone symbol is required")
        if "confidence" in raw:
            confidence = _finite_number(raw.get("confidence"), label=f"segment {index} confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ReviewError(f"segment {index} confidence is outside [0,1]")
        lyric_index = row.get("lyric_item_index")
        if isinstance(lyric_index, bool) or not isinstance(lyric_index, int) or lyric_index < 0:
            raise ReviewError(f"segment {index} lyric_item_index is invalid")
        if timeline_context:
            lyric_item = timeline_items.get(lyric_index)
            if lyric_item is None:
                raise ReviewError(f"segment {index} references a missing lyric item")
            lyric_start = _finite_number(
                lyric_item.get("start_sec"), label=f"lyric item {lyric_index} start_sec"
            )
            lyric_end = _finite_number(
                lyric_item.get("end_sec"), label=f"lyric item {lyric_index} end_sec"
            )
            overlap = max(
                0.0,
                min(float(row["end_sec"]), lyric_end)
                - max(float(row["start_sec"]), lyric_start),
            )
            if overlap <= 0.0:
                raise ReviewError(
                    f"segment {index} does not overlap lyric item {lyric_index}",
                    error_kind="phone_lyric_mismatch",
                )
            metadata = dict(row.get("metadata")) if isinstance(row.get("metadata"), Mapping) else {}
            metadata["direct_overlap_sec"] = overlap
            metadata["phone_overlap_ratio"] = overlap / (
                float(row["end_sec"]) - float(row["start_sec"])
            )
            row["metadata"] = metadata
        row["index"] = index
        normalized.append(row)
        previous_end = row["end_sec"]
    return normalized, reviewer_id, reviewed_at, start, end, scope


def _build_review_artifacts(
    *,
    source: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    reviewer_id: str,
    reviewed_at: str,
    review_method: str,
    range_start: float,
    range_end: float,
    audio_hash: str,
    scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_alignment = source.get("alignment")
    alignment = dict(original_alignment) if isinstance(original_alignment, Mapping) else {}
    alignment_source = str(alignment.get("alignment_source") or "").strip()
    model_payload = alignment.get("model") if isinstance(alignment.get("model"), Mapping) else {}
    model_hash = str(alignment.get("model_hash") or model_payload.get("hash") or "").strip().lower()
    normalized_segments = [
        _phone_segment_dict(
            row,
            reviewer_id=reviewer_id,
            reviewed_at=reviewed_at,
            alignment_source=alignment_source,
            model_hash=model_hash,
        )
        for row in segments
    ]
    source_rows = source.get("segments")
    machine_source_segment_count = (
        len(source_rows)
        if isinstance(source_rows, Sequence) and not isinstance(source_rows, (str, bytes))
        else 0
    )
    reviewed_segment_count = len(normalized_segments)
    row_edit_provenance = {
        "machine_source_segment_count": machine_source_segment_count,
        "reviewed_segment_count": reviewed_segment_count,
        "human_row_edits_permitted": True,
        "machine_source_role": "reference_only_not_human_truth",
        "machine_source_is_truth": False,
    }
    alignment.update({
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "vocal_stem_sha256": audio_hash,
        "model_output": True,
        "segments": normalized_segments,
        "review": {
            "status": "human_reviewed",
            "reviewer_id": reviewer_id,
            "reviewed_at": reviewed_at,
            "review_method": review_method,
            "reviewed_range_sec": [range_start, range_end],
        },
        "provenance": {
            **(dict(alignment.get("provenance")) if isinstance(alignment.get("provenance"), Mapping) else {}),
            "status": "human_reviewed_phone_alignment_pending_detached_acceptance_validation",
            "product_accepted": False,
            "machine_alignment_sha256": source.get("alignment_sha256"),
            "human_reviewed": True,
            "reviewer_id": reviewer_id,
            "reviewed_at": reviewed_at,
            "review_method": review_method,
            **row_edit_provenance,
        },
    })
    # ``reviewed`` is an artifact-only marker; PhoneSegment.to_dict() does not
    # emit it, so exclude it from the contract hash used by the validator.
    segment_hash = _canonical_segments_sha256(
        [{key: value for key, value in item.items() if key != "reviewed"} for item in normalized_segments]
    )
    symbols: list[str] = []
    for row in normalized_segments:
        symbol = str(row["normalized_ipa"])
        if symbol not in symbols:
            symbols.append(symbol)
    inventory_hash = _canonical_inventory_sha256(symbols)
    acceptance_review: dict[str, Any] = {
        "review_status": "approved",
        "reviewer_kind": "human",
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "review_method": review_method,
        "phone_boundaries_reviewed": True,
        "scope": scope,
        "reviewed_range_sec": [range_start, range_end],
        **row_edit_provenance,
    }
    if scope == FULL_SONG_SCOPE:
        predecessor = source.get("short_clip_predecessor")
        if not isinstance(predecessor, Mapping):
            raise ReviewError("full-song review requires a short-clip predecessor", error_kind="missing_predecessor")
        acceptance_review["short_clip_predecessor"] = dict(predecessor)
    acceptance = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "route": HUMAN_REVIEW_ROUTE,
        "alignment_artifact_sha256": "",
        "vocal_stem_sha256": audio_hash,
        "phone_segments_sha256": segment_hash,
        "phone_inventory": {
            "symbols": symbols,
            "sha256": inventory_hash,
            "source": "human_review_forced_phone_alignment_v1",
        },
        "review": acceptance_review,
        "provenance": dict(row_edit_provenance),
    }
    return alignment, acceptance


def _exclusive_write(path: Path, data: bytes) -> None:
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ReviewError(f"refusing to overwrite existing output: {path.name}", error_kind="output_exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _revalidate_short_clip_predecessor(
    source: Mapping[str, Any],
    *,
    vocal_stem_sha256: str,
) -> None:
    """Re-check the full-song predecessor against its startup snapshot."""

    raw_predecessor = source.get("short_clip_predecessor")
    if not isinstance(raw_predecessor, Mapping):
        raise ReviewError(
            "full-song review requires a short-clip predecessor",
            error_kind="missing_predecessor",
        )
    predecessor_path_text = str(raw_predecessor.get("acceptance_path") or "").strip()
    expected_hash = str(raw_predecessor.get("acceptance_sha256") or "").strip().lower()
    if not predecessor_path_text or not _SHA256.fullmatch(expected_hash):
        raise ReviewError(
            "short-clip predecessor binding is invalid",
            error_kind="invalid_predecessor",
        )
    predecessor_path = Path(predecessor_path_text).expanduser().resolve()
    try:
        actual_hash = sha256_file(predecessor_path)
    except ReviewError as exc:
        raise ReviewError(
            "short-clip predecessor changed during review",
            error_kind="input_changed",
        ) from exc
    if actual_hash != expected_hash:
        raise ReviewError(
            "short-clip predecessor changed during review",
            error_kind="input_changed",
        )
    try:
        validate_short_clip_predecessor(
            predecessor_path,
            vocal_stem_sha256=vocal_stem_sha256,
        )
    except ForcedPhoneAcceptanceError as exc:
        raise ReviewError(
            f"short-clip predecessor is no longer valid: {exc}",
            error_kind="invalid_predecessor",
        ) from exc


def save_review(
    output_dir: Path | str,
    source: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    reviewed_alignment_name: str = DEFAULT_REVIEWED_ALIGNMENT_NAME,
    acceptance_name: str = DEFAULT_ACCEPTANCE_NAME,
) -> dict[str, Any]:
    """Validate and exclusively create the reviewed alignment and acceptance."""

    output = Path(output_dir).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReviewError(f"cannot create output directory: {output}", error_kind="output_error") from exc
    if not output.is_dir():
        raise ReviewError("output directory is not a directory", error_kind="output_error")
    alignment_name = _safe_output_name(reviewed_alignment_name, label="reviewed alignment name")
    acceptance_file_name = _safe_output_name(acceptance_name, label="acceptance name")
    if alignment_name == acceptance_file_name:
        raise ReviewError("alignment and acceptance output names must differ", error_kind="path_traversal")
    alignment_path = output / alignment_name
    acceptance_path = output / acceptance_file_name
    if alignment_path.exists() or acceptance_path.exists():
        raise ReviewError("refusing to overwrite existing output", error_kind="output_exists")

    # The browser session is deliberately bound to the bytes it displayed.
    # Detect a source file replacement before creating acceptance evidence.
    source_audio_path = source.get("audio_path")
    if source_audio_path:
        current_audio_hash = sha256_file(Path(str(source_audio_path)).resolve())
        if current_audio_hash != str(source.get("audio_sha256") or "").lower():
            raise ReviewError("source audio changed during review", error_kind="input_changed")
    source_alignment_path = source.get("alignment_path")
    if source_alignment_path:
        current_alignment_hash = sha256_file(Path(str(source_alignment_path)).resolve())
        if current_alignment_hash != str(source.get("alignment_sha256") or "").lower():
            raise ReviewError("source alignment changed during review", error_kind="input_changed")
    source_timeline_hash = str(source.get("lyrics_timeline_sha256") or "").strip().lower()
    source_timeline_path = source.get("lyrics_timeline_path")
    if source_timeline_hash and not source_timeline_path:
        raise ReviewError(
            "alignment-bound lyrics timeline is missing from the review session",
            error_kind="missing_lyrics_timing_evidence",
        )
    if source_timeline_path:
        timeline_path = Path(str(source_timeline_path)).resolve()
        timeline_audio_path = source.get("lyrics_timeline_audio_path")
        if not timeline_audio_path:
            raise ReviewError(
                "alignment-bound lyrics timeline audio is missing from the review session",
                error_kind="missing_lyrics_timing_evidence",
            )
        resolved_timeline_audio = Path(str(timeline_audio_path)).resolve()
        try:
            current_timeline = load_lyrics_timeline(
                timeline_path,
                audio_path=resolved_timeline_audio,
            )
            current_timeline_hash = canonical_timeline_sha256(current_timeline)
            current_timeline_file_hash = sha256_file(timeline_path)
            current_timeline_audio_hash = sha256_file(resolved_timeline_audio)
        except (LyricsTimelineError, OSError, ReviewError) as exc:
            raise ReviewError("lyrics timeline changed during review", error_kind="input_changed") from exc
        if (
            current_timeline_hash != source_timeline_hash
            or current_timeline_file_hash
            != str(source.get("lyrics_timeline_file_sha256") or "").lower()
            or current_timeline_audio_hash
            != str(source.get("lyrics_timeline_audio_sha256") or "").lower()
        ):
            raise ReviewError("lyrics timeline changed during review", error_kind="input_changed")

    review_method = str(request.get("review_method", request.get("method")) or "").strip()
    audio_hash = str(source.get("audio_sha256") or "").lower()
    if not _SHA256.fullmatch(audio_hash):
        raise ReviewError("source audio hash is invalid", error_kind="invalid_input")
    if str(source.get("review_scope") or SHORT_CLIP_SCOPE).strip() == FULL_SONG_SCOPE:
        _revalidate_short_clip_predecessor(source, vocal_stem_sha256=audio_hash)
    normalized, reviewer_id, reviewed_at, range_start, range_end, scope = _validate_review_request(request, source=source)
    alignment, acceptance = _build_review_artifacts(
        source=source,
        segments=normalized,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        review_method=review_method,
        range_start=range_start,
        range_end=range_end,
        audio_hash=audio_hash,
        scope=scope,
    )
    alignment_bytes = (json.dumps(alignment, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    alignment_digest = hashlib.sha256(alignment_bytes).hexdigest()
    acceptance["alignment_artifact_sha256"] = alignment_digest
    acceptance["alignment_artifact_path"] = alignment_name
    acceptance_bytes = (json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    written: list[Path] = []
    try:
        _exclusive_write(alignment_path, alignment_bytes)
        written.append(alignment_path)
        _exclusive_write(acceptance_path, acceptance_bytes)
        written.append(acceptance_path)
    except Exception:
        for path in written:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return {
        "reviewed_alignment_path": str(alignment_path),
        "alignment_path": str(alignment_path),
        "acceptance_path": str(acceptance_path),
        "reviewed_alignment_sha256": alignment_digest,
        "alignment_sha256": alignment_digest,
        "acceptance_sha256": hashlib.sha256(acceptance_bytes).hexdigest(),
        "phone_segments_sha256": acceptance["phone_segments_sha256"],
        "phone_inventory_sha256": acceptance["phone_inventory"]["sha256"],
        "review": acceptance["review"],
    }


def _json_for_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def _review_session_id(
    source: Mapping[str, Any],
    *,
    range_start: float,
    range_end: float,
    row_count: int,
) -> str:
    identity = {
        "audio_sha256": source.get("audio_sha256"),
        "alignment_sha256": source.get("alignment_sha256"),
        "lyrics_timeline_sha256": source.get("lyrics_timeline_sha256"),
        "lyrics_timeline_file_sha256": source.get("lyrics_timeline_file_sha256"),
        "lyrics_timeline_audio_sha256": source.get("lyrics_timeline_audio_sha256"),
        "review_scope": source.get("review_scope"),
        "reviewed_range_sec": [range_start, range_end],
        "row_count": row_count,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_review_page(source: Mapping[str, Any]) -> bytes:
    """Render a restrained, responsive browser review page."""

    rows = source.get("segments") if isinstance(source.get("segments"), Sequence) else []
    title = html.escape(Path(str(source.get("audio_path") or "audio")).name)
    review_range = source.get("reviewed_range_sec", [0.0, 10.0])
    if not isinstance(review_range, Sequence) or len(review_range) != 2:
        review_range = [0.0, 10.0]
    start_value = _finite_number(review_range[0], label="review start")
    end_value = _finite_number(review_range[1], label="review end")
    evidence = (
        ("Scope", source.get("review_scope")),
        ("Phone rows", len(rows)),
        ("Vocal stem SHA-256", source.get("audio_sha256")),
        ("Alignment SHA-256", source.get("alignment_sha256")),
        ("Timeline SHA-256", source.get("lyrics_timeline_sha256")),
        ("Timeline audio SHA-256", source.get("lyrics_timeline_audio_sha256")),
    )
    evidence_html = "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(str(value or 'not supplied'))}</dd></div>"
        for label, value in evidence
    )
    review_session_id = _review_session_id(
        source,
        range_start=start_value,
        range_end=end_value,
        row_count=len(rows),
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phone alignment review</title>
<style>
:root {{ color-scheme:light; font-family:system-ui,-apple-system,Segoe UI,sans-serif; color:#20252b; background:#f5f6f7; }}
* {{ box-sizing:border-box; }} body {{ margin:0; }} main {{ max-width:1100px; margin:0 auto; padding:24px 18px 48px; }}
h1 {{ font-size:1.35rem; margin:0 0 4px; }} .muted {{ color:#66717c; font-size:.88rem; }}
section {{ background:#fff; border:1px solid #d8dde2; border-radius:6px; padding:16px; margin-top:14px; }}
.audio-row {{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; }} audio {{ min-width:260px; flex:1; height:38px; }}
canvas {{ display:block; width:100%; height:150px; border:1px solid #cbd2d8; border-radius:4px; background:#fbfcfd; cursor:crosshair; margin-top:12px; }}
.controls {{ display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:10px; }} label {{ display:flex; flex-direction:column; gap:4px; font-size:.82rem; color:#4b5661; }}
input {{ border:1px solid #bbc4cc; border-radius:4px; font:inherit; padding:7px 8px; min-width:0; color:#20252b; background:#fff; }}
details {{ margin-top:12px; }} summary {{ cursor:pointer; color:#4b5661; font-size:.86rem; }} .evidence-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 16px; margin:10px 0 0; }} .evidence-grid div {{ min-width:0; }} .evidence-grid dt {{ color:#66717c; font-size:.76rem; }} .evidence-grid dd {{ margin:2px 0 0; overflow-wrap:anywhere; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.76rem; }}
    .table-toolbar {{ position:sticky; top:0; z-index:3; display:flex; align-items:center; gap:6px; min-height:44px; padding:6px 0 10px; border-bottom:1px solid #e5e8eb; background:#fff; }} .table-toolbar .icon-button {{ width:34px; height:34px; }} #active-phone {{ min-width:190px; margin-left:4px; }} .table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; min-width:980px; }} th,td {{ border-bottom:1px solid #e5e8eb; padding:8px 6px; text-align:left; white-space:nowrap; }} th {{ position:sticky; top:0; z-index:1; background:#fff; color:#596570; font-size:.78rem; font-weight:600; }} td {{ font-size:.9rem; }} td input {{ width:100%; }} .target {{ width:178px; min-width:178px; vertical-align:top; border-right:1px solid #dfe4e8; background:#fafbfc; }} .target strong {{ display:block; font-size:1.1rem; }} .target .jyutping {{ display:block; color:#66717c; font-size:.78rem; }} .group-meta {{ display:block; color:#66717c; font-size:.72rem; margin-top:2px; }} .group-controls {{ display:flex; align-items:center; gap:8px; margin-top:9px; }} .group-review-label {{ display:flex; flex-direction:row; align-items:center; gap:5px; color:#34404a; font-size:.74rem; }} .group-reviewed {{ accent-color:#216e4a; width:18px; height:18px; }} .group-play.active {{ border-color:#216e4a; background:#e4f1ea; color:#185a3d; }} .group-start td {{ border-top:2px solid #aeb8c0; }} .group-start:first-child td {{ border-top:0; }} .active-group td {{ background:#f3f8f5; }} .phone-duration {{ min-width:64px; color:#596570; font-variant-numeric:tabular-nums; }} .play {{ width:32px; height:32px; border:1px solid #b8c1c9; background:#f8fafb; border-radius:4px; padding:0; cursor:pointer; }} .row-actions {{ display:flex; justify-content:center; gap:4px; }} .icon-button {{ width:30px; height:30px; border:1px solid #b8c1c9; background:#f8fafb; border-radius:4px; padding:0; cursor:pointer; font-size:1rem; line-height:1; }} .icon-button.delete-row {{ color:#a12622; }} .active-row td {{ background:#e8f3ed; }} .dirty-row td:first-child {{ box-shadow:inset 3px 0 #c47a13; }} th:last-child,td:last-child {{ text-align:center; }}
 .reviewed {{ accent-color:#216e4a; width:18px; height:18px; }} .actions {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }} button.primary {{ border:0; border-radius:4px; padding:9px 16px; background:#216e4a; color:#fff; font-weight:600; cursor:pointer; }} button:disabled {{ opacity:.55; cursor:not-allowed; }} #status {{ min-height:1.35em; font-size:.9rem; }} .error {{ color:#a12622; }} .ok {{ color:#216e4a; }}
@media (max-width:650px) {{ main {{ padding:16px 10px 36px; }} section {{ padding:12px; }} .controls {{ grid-template-columns:repeat(2,minmax(120px,1fr)); }} canvas {{ height:120px; }} }}
</style></head><body><main>
    <h1>Phone alignment review</h1><div class="muted">{title} &middot; <span id="review-progress">0 / {len(rows)} reviewed</span></div>
<section><div class="audio-row"><audio id="player" controls preload="metadata" src="/audio"></audio><span id="position" class="muted">0.00 s</span></div><canvas id="waveform" width="1200" height="180" aria-label="Audio waveform"></canvas></section>
<section><div class="controls">
<label>Reviewer ID<input id="reviewer" autocomplete="off" placeholder="required"></label>
<label>Review method<input id="method" value="auditory_acoustic_phone_boundary_review"></label>
<label>Range start (s)<input id="range-start" type="number" min="0" step="0.001" value="{start_value:.3f}"></label>
<label>Range end (s)<input id="range-end" type="number" min="0" step="0.001" value="{end_value:.3f}"></label>
</div><details><summary>Review evidence</summary><dl class="evidence-grid">{evidence_html}</dl></details></section>
    <section><div class="table-toolbar"><button id="first-unreviewed" class="icon-button" type="button" title="First unreviewed syllable" aria-label="First unreviewed syllable">&#8676;</button><button id="previous-unreviewed" class="icon-button" type="button" title="Previous unreviewed syllable" aria-label="Previous unreviewed syllable">&#8593;</button><button id="play-active" class="icon-button" type="button" title="Loop active syllable" aria-label="Loop active syllable">&#8635;</button><button id="next-unreviewed" class="icon-button" type="button" title="Next unreviewed syllable" aria-label="Next unreviewed syllable">&#8595;</button><button id="discard-draft" class="icon-button" type="button" title="Discard local draft" aria-label="Discard local draft" disabled>&#8634;</button><span id="active-phone" class="muted">Syllable - / - &middot; Phone - / {len(rows)}</span></div><div class="table-wrap"><table><thead><tr><th>#</th><th>Syllable / lyric item</th><th>Phone</th><th>Start</th><th>End</th><th>Duration</th><th>Item</th><th>Play phone</th><th>Reviewed</th><th>Actions</th></tr></thead><tbody id="segments"></tbody></table></div></section>
    <section class="actions"><div id="status" class="muted">0 / {len(rows)} reviewed</div><button id="save" class="primary" type="button" disabled>Save reviewed artifacts</button></section>
</main><script>
    const source = {_json_for_script({"duration_sec": source.get("duration_sec"), "reviewed_range_sec": [start_value, end_value], "review_session_id": review_session_id, "draft_schema_version": BROWSER_DRAFT_SCHEMA_VERSION, "segments": rows})};
const player=document.getElementById('player'),table=document.getElementById('segments'),statusNode=document.getElementById('status'),saveButton=document.getElementById('save'),reviewerInput=document.getElementById('reviewer'),methodInput=document.getElementById('method'),rangeStartInput=document.getElementById('range-start'),rangeEndInput=document.getElementById('range-end'),discardDraftButton=document.getElementById('discard-draft'); const rowRefs=[]; const lyricContext=new Map(source.segments.map(row=>[Number(row.lyric_item_index),{{lyric_text:row.lyric_text||'',jyutping:row.jyutping||''}}])); const draftKey='any2xsampa.phone-review.'+source.review_session_id; let activeRow=null; let restoredActiveIndex=-1; let playbackTimer=null; let playbackToken=0; let loopingGroupButton=null; let draftTimer=null; let draftRestored=false; let allowUnload=false;
function number(el) {{ return Number.parseFloat(el.value); }}
function formattedNumber(value) {{ const parsed=Number(value);return Number.isFinite(parsed)?parsed.toFixed(3):String(value??''); }}
function rowData(tr) {{ return {{symbol:tr.querySelector('.symbol').value,start_sec:number(tr.querySelector('.start')),end_sec:number(tr.querySelector('.end')),lyric_item_index:Number.parseInt(tr.querySelector('.lyric').value,10),lyric_text:tr.querySelector('.target strong').textContent,jyutping:tr.querySelector('.target .jyutping').textContent,reviewed:tr.querySelector('.reviewed').checked}}; }}
function draftRowData(tr) {{ return {{symbol:tr.querySelector('.symbol').value,start_sec:tr.querySelector('.start').value,end_sec:tr.querySelector('.end').value,lyric_item_index:tr.querySelector('.lyric').value,lyric_text:tr.querySelector('.target strong').textContent,jyutping:tr.querySelector('.target .jyutping').textContent,reviewed:tr.querySelector('.reviewed').checked,dirty:tr.classList.contains('dirty-row')}}; }}
function readLocalDraft() {{ try {{ const raw=window.localStorage.getItem(draftKey);if(!raw)return null;const draft=JSON.parse(raw);if(draft?.schema_version!==source.draft_schema_version||draft?.review_session_id!==source.review_session_id||!Array.isArray(draft?.segments)||draft.segments.length<1||draft.segments.length>10000||!draft.segments.every(row=>row&&typeof row==='object'))return null;return draft; }} catch(e) {{ return null; }} }}
function writeLocalDraft() {{ if(!rowRefs.length)return;try {{ const activeIndex=activeRow?rowRefs.indexOf(activeRow):-1;const draft={{schema_version:source.draft_schema_version,review_session_id:source.review_session_id,updated_at:new Date().toISOString(),reviewer_id:reviewerInput.value,review_method:methodInput.value,reviewed_range_sec:[rangeStartInput.value,rangeEndInput.value],active_index:activeIndex,segments:rowRefs.map(draftRowData)}};window.localStorage.setItem(draftKey,JSON.stringify(draft));discardDraftButton.disabled=false; }} catch(e) {{ return; }} }}
function scheduleDraftSave() {{ if(draftTimer!==null)window.clearTimeout(draftTimer);draftTimer=window.setTimeout(()=>{{draftTimer=null;writeLocalDraft();}},150); }}
function removeLocalDraft() {{ try {{ window.localStorage.removeItem(draftKey); }} catch(e) {{ }} discardDraftButton.disabled=true; }}
function syncTarget(tr,fallback={{}}) {{ const index=Number.parseInt(tr.querySelector('.lyric').value,10),context=lyricContext.get(index)||fallback; tr.querySelector('.target strong').textContent=context.lyric_text||('#'+index); tr.querySelector('.target .jyutping').textContent=context.jyutping||''; }}
function currentGroups() {{ const groups=[];let current=null;rowRefs.forEach(tr=>{{const key=String(Number.parseInt(tr.querySelector('.lyric').value,10));if(!current||current.key!==key){{current={{key,rows:[]}};groups.push(current);}}current.rows.push(tr);}});return groups; }}
function groupForRow(tr) {{ return currentGroups().find(group=>group.rows.includes(tr))||{{key:'-',rows:[tr]}}; }}
function groupBounds(group) {{ let start=Infinity,end=-Infinity;group.rows.forEach(tr=>{{start=Math.min(start,number(tr.querySelector('.start')));end=Math.max(end,number(tr.querySelector('.end')));}});return [start,end]; }}
function refreshGroups() {{ rowRefs.forEach(tr=>{{const target=tr.querySelector('.target');target.hidden=false;target.rowSpan=1;tr.classList.remove('group-start','group-continuation');delete tr.dataset.groupIndex;}});currentGroups().forEach((group,groupIndex)=>{{const anchor=group.rows[0],target=anchor.querySelector('.target'),reviewed=group.rows.map(tr=>tr.querySelector('.reviewed').checked);anchor.classList.add('group-start');group.rows.slice(1).forEach(tr=>{{tr.classList.add('group-continuation');tr.querySelector('.target').hidden=true;}});group.rows.forEach(tr=>{{tr.dataset.groupIndex=String(groupIndex);}});target.rowSpan=group.rows.length;target.querySelector('.group-meta').textContent='Item #'+group.key+' / '+group.rows.length+' phones';const checkbox=target.querySelector('.group-reviewed');checkbox.checked=reviewed.every(Boolean);checkbox.indeterminate=!checkbox.checked&&reviewed.some(Boolean);checkbox.setAttribute('aria-label','Reviewed all '+group.rows.length+' phone boundaries in lyric item '+group.key);const play=target.querySelector('.group-play');play.title='Loop lyric item '+group.key+' ('+group.rows.length+' phones)';play.setAttribute('aria-label',play.title);}});refreshActiveLabel(); }}
function refreshActiveLabel() {{ const phoneIndex=activeRow?rowRefs.indexOf(activeRow):-1,groups=currentGroups(),groupIndex=activeRow?groups.findIndex(group=>group.rows.includes(activeRow)):-1; document.getElementById('active-phone').textContent='Syllable '+(groupIndex>=0?groupIndex+1:'-')+' / '+groups.length+' / Phone '+(phoneIndex>=0?phoneIndex+1:'-')+' / '+rowRefs.length; }}
function updateVisibleIndices() {{ rowRefs.forEach((tr,index)=>{{ tr.querySelector('.row-index').textContent=index; const start=number(tr.querySelector('.start')),end=number(tr.querySelector('.end')),duration=tr.querySelector('.phone-duration'),play=tr.querySelector('.play'),reviewed=tr.querySelector('.reviewed'),add=tr.querySelector('.add-row'),remove=tr.querySelector('.delete-row'); duration.textContent=Number.isFinite(start)&&Number.isFinite(end)&&end>start?Math.round((end-start)*1000)+' ms':'-'; play.title='Play phone interval '+index; play.setAttribute('aria-label','Play phone '+index); reviewed.setAttribute('aria-label','Reviewed phone '+index); add.title='Add phone row after row '+index; add.setAttribute('aria-label','Add phone row after row '+index); remove.disabled=rowRefs.length<=1; remove.title=rowRefs.length<=1?'Cannot delete the only phone row':'Delete phone row '+index; remove.setAttribute('aria-label',remove.title); }}); refreshGroups(); }}
function markGroupDirty(tr,reconfigure=false) {{ const prior=groupForRow(tr).rows.slice();prior.forEach(row=>{{row.querySelector('.reviewed').checked=false;row.classList.add('dirty-row');}});if(reconfigure){{refreshGroups();groupForRow(tr).rows.forEach(row=>{{row.querySelector('.reviewed').checked=false;row.classList.add('dirty-row');}});}}tr.classList.add('dirty-row');refreshGroups();updateReadiness();scheduleDraftSave(); }}
function markDirty(tr) {{ markGroupDirty(tr); }}
function markAllDirty() {{ rowRefs.forEach(tr=>{{tr.querySelector('.reviewed').checked=false;tr.classList.add('dirty-row');}}); refreshGroups();updateReadiness();scheduleDraftSave(); }}
function setGroupReviewed(tr,checked) {{ groupForRow(tr).rows.forEach(row=>{{row.querySelector('.reviewed').checked=checked;if(checked)row.classList.remove('dirty-row');}});refreshGroups();updateReadiness();scheduleDraftSave(); }}
function bindRow(tr) {{ tr.querySelector('.play').addEventListener('click',()=>playInterval(tr)); tr.querySelector('.group-play').addEventListener('click',()=>playGroup(tr)); tr.querySelector('.group-reviewed').addEventListener('change',event=>setGroupReviewed(tr,event.target.checked)); tr.querySelector('.add-row').addEventListener('click',()=>addRowAfter(tr)); tr.querySelector('.delete-row').addEventListener('click',()=>deleteRow(tr)); tr.querySelector('.reviewed').addEventListener('change',()=>{{if(tr.querySelector('.reviewed').checked)tr.classList.remove('dirty-row');refreshGroups();updateReadiness();scheduleDraftSave();}}); tr.querySelector('.symbol').addEventListener('input',()=>markDirty(tr)); tr.querySelector('.start').addEventListener('input',()=>{{markDirty(tr);updateVisibleIndices();redrawWave();}}); tr.querySelector('.end').addEventListener('input',()=>{{markDirty(tr);updateVisibleIndices();redrawWave();}}); tr.querySelector('.lyric').addEventListener('input',()=>{{syncTarget(tr);markGroupDirty(tr,true);}}); tr.addEventListener('focusin',event=>{{if(!event.target.classList.contains('group-reviewed')&&!event.target.classList.contains('group-play'))focusPhone(tr);}}); }}
function makeRow(row, append=true) {{ const tr=document.createElement('tr'); tr.innerHTML=`<td class="row-index"></td><td class="target"><strong></strong><span class="jyutping"></span><small class="group-meta"></small><div class="group-controls"><button class="icon-button group-play" type="button" title="Loop syllable" aria-label="Loop syllable">&#8635;</button><label class="group-review-label"><input class="group-reviewed" type="checkbox"><span>Reviewed group</span></label></div></td><td><input class="symbol" value=""></td><td><input class="start" type="number" step="0.001"></td><td><input class="end" type="number" step="0.001"></td><td class="phone-duration"></td><td><input class="lyric" type="number" min="0" step="1"></td><td><button class="play" type="button" title="Play phone interval" aria-label="Play phone interval">&#9654;</button></td><td><input class="reviewed" type="checkbox" aria-label="Reviewed phone"></td><td class="row-actions"><button class="icon-button add-row" type="button" title="Add phone row" aria-label="Add phone row">&#43;</button><button class="icon-button delete-row" type="button" title="Delete phone row" aria-label="Delete phone row">&#128465;</button></td>`; tr.querySelector('.symbol').value=row.symbol||''; tr.querySelector('.start').value=formattedNumber(row.start_sec); tr.querySelector('.end').value=formattedNumber(row.end_sec); tr.querySelector('.lyric').value=row.lyric_item_index??''; syncTarget(tr,row); tr.querySelector('.reviewed').checked=row.reviewed===true; bindRow(tr); if(append){{ table.appendChild(tr); rowRefs.push(tr); }} return tr; }}
function addRowAfter(tr) {{ const index=rowRefs.indexOf(tr); if(index<0)return; const current=rowData(tr),start=current.start_sec,end=current.end_sec; let split=(start+end)/2; const validSplit=Number.isFinite(split)&&Number.isFinite(start)&&Number.isFinite(end)&&split>start+0.001&&split<end-0.001; const suggestion={{symbol:current.symbol,start_sec:validSplit?split:end,end_sec:end,lyric_item_index:current.lyric_item_index,lyric_text:current.lyric_text,jyutping:current.jyutping,reviewed:false}}; if(validSplit)tr.querySelector('.end').value=split.toFixed(3); else {{ const bounds=configuredRange(),fallbackStart=Number.isFinite(end)?end:bounds[0],next=rowRefs[index+1],nextStart=next?number(next.querySelector('.start')):bounds[1],fallbackEnd=Number.isFinite(nextStart)&&nextStart>fallbackStart?nextStart:fallbackStart+0.001; suggestion.start_sec=fallbackStart; suggestion.end_sec=fallbackEnd; }} markDirty(tr); const newRow=makeRow(suggestion,false); newRow.classList.add('dirty-row'); table.insertBefore(newRow,rowRefs[index+1]||null); rowRefs.splice(index+1,0,newRow); updateVisibleIndices(); updateReadiness(); }}
function deleteRow(tr) {{ if(rowRefs.length<=1)return; const index=rowRefs.indexOf(tr); if(index<0||!window.confirm('Delete phone row '+index+'? This cannot be undone.'))return; markGroupDirty(tr);if(activeRow===tr)activeRow=null; rowRefs.splice(index,1); tr.remove(); updateVisibleIndices(); updateReadiness();scheduleDraftSave(); }}
const localDraft=readLocalDraft();if(localDraft){{reviewerInput.value=String(localDraft.reviewer_id||'');methodInput.value=String(localDraft.review_method||methodInput.value);if(Array.isArray(localDraft.reviewed_range_sec)&&localDraft.reviewed_range_sec.length===2){{rangeStartInput.value=String(localDraft.reviewed_range_sec[0]);rangeEndInput.value=String(localDraft.reviewed_range_sec[1]);}}localDraft.segments.forEach(row=>{{const priorReviewed=row?.reviewed===true;const restored={{...row,reviewed:false}};const tr=makeRow(restored);if(row?.dirty===true||priorReviewed)tr.classList.add('dirty-row');}});restoredActiveIndex=Number.isInteger(localDraft.active_index)?localDraft.active_index:-1;draftRestored=true;discardDraftButton.disabled=false;}}else{{source.segments.forEach(row=>makeRow(row));}}updateVisibleIndices();
let viewRange=source.reviewed_range_sec.slice();
function configuredRange() {{ const start=number(document.getElementById('range-start')),end=number(document.getElementById('range-end')); return Number.isFinite(start)&&Number.isFinite(end)&&end>start?[Math.max(0,start),Math.min(source.duration_sec,end)]:source.reviewed_range_sec; }}
function focusPhone(tr,scroll=false) {{ const start=number(tr.querySelector('.start')),end=number(tr.querySelector('.end'));if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return;activeRow=tr;const bounds=configuredRange(),margin=Math.max(.45,(end-start)*2),viewStart=Math.max(bounds[0],start-margin),viewEnd=Math.min(bounds[1],end+margin);viewRange=viewEnd-viewStart<.25?[Math.max(bounds[0],start-.125),Math.min(bounds[1],end+.125)]:[viewStart,viewEnd];rowRefs.forEach(row=>{{row.classList.toggle('active-row',row===tr);row.classList.remove('active-group');}});refreshActiveLabel();if(scroll)tr.scrollIntoView({{block:'center',behavior:'smooth'}});redrawWave();scheduleDraftSave(); }}
function focusGroup(group,scroll=false) {{ const bounds=groupBounds(group);if(!Number.isFinite(bounds[0])||!Number.isFinite(bounds[1])||bounds[1]<=bounds[0])return;activeRow=group.rows[0];const configured=configuredRange(),margin=Math.max(.18,(bounds[1]-bounds[0])*.18);viewRange=[Math.max(configured[0],bounds[0]-margin),Math.min(configured[1],bounds[1]+margin)];rowRefs.forEach(row=>{{row.classList.remove('active-row');row.classList.toggle('active-group',group.rows.includes(row));}});refreshActiveLabel();if(scroll)group.rows[0].scrollIntoView({{block:'center',behavior:'smooth'}});redrawWave();scheduleDraftSave(); }}
function stopPlayback() {{ playbackToken++;if(playbackTimer!==null)window.clearTimeout(playbackTimer);playbackTimer=null;player.pause();if(loopingGroupButton)loopingGroupButton.classList.remove('active');loopingGroupButton=null; }}
function monitorPlayback(token,end,loopStart=null) {{ if(token!==playbackToken)return;if(player.currentTime>=end-0.005){{if(loopStart!==null&&loopingGroupButton){{player.currentTime=loopStart;playbackTimer=window.setTimeout(()=>monitorPlayback(token,end,loopStart),10);return;}}player.pause();playbackTimer=null;return;}}if(!player.paused)playbackTimer=window.setTimeout(()=>monitorPlayback(token,end,loopStart),10); }}
function startPlayback(start,end,loopButton=null) {{ stopPlayback();const token=++playbackToken;if(loopButton){{loopingGroupButton=loopButton;loopButton.classList.add('active');}}player.currentTime=start;const pending=player.play();if(pending)pending.then(()=>monitorPlayback(token,end,loopButton?start:null)).catch(()=>{{stopPlayback();statusNode.className='error';statusNode.textContent='Audio playback failed.';}}); }}
function playInterval(tr) {{ const start=number(tr.querySelector('.start')),end=number(tr.querySelector('.end'));if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return;focusPhone(tr);startPlayback(start,end); }}
function playGroup(tr) {{ const group=groupForRow(tr),anchor=group.rows[0],button=anchor.querySelector('.group-play');if(loopingGroupButton===button){{stopPlayback();return;}}const groupRange=groupBounds(group),configured=configuredRange(),start=Math.max(configured[0],groupRange[0]-.08),end=Math.min(configured[1],groupRange[1]+.08);if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return;focusGroup(group);startPlayback(start,end,button); }}
function findUnreviewed(direction,fromStart=false) {{ const groups=currentGroups();if(!groups.length)return null;const activeIndex=activeRow?groups.findIndex(group=>group.rows.includes(activeRow)):-1,base=fromStart?-1:activeIndex;for(let step=1;step<=groups.length;step++){{const index=(base+direction*step+groups.length)%groups.length;if(!groups[index].rows.every(tr=>tr.querySelector('.reviewed').checked))return groups[index];}}return null; }}
function goToUnreviewed(direction,fromStart=false) {{ const group=findUnreviewed(direction,fromStart);if(!group)return;focusGroup(group,true);playGroup(group.rows[0]); }}
document.getElementById('first-unreviewed').addEventListener('click',()=>goToUnreviewed(1,true));document.getElementById('previous-unreviewed').addEventListener('click',()=>goToUnreviewed(-1));document.getElementById('next-unreviewed').addEventListener('click',()=>goToUnreviewed(1));document.getElementById('play-active').addEventListener('click',()=>{{if(activeRow)playGroup(activeRow);else goToUnreviewed(1,true);}});
player.addEventListener('timeupdate',()=>document.getElementById('position').textContent=player.currentTime.toFixed(2)+' s');
    const canvas=document.getElementById('waveform'),ctx=canvas.getContext('2d'); let waveData=null;
    function activeRange() {{ return viewRange; }}
    function drawWave(data) {{ const range=activeRange(),from=Math.max(0,Math.floor(range[0]/source.duration_sec*data.length)),to=Math.min(data.length,Math.ceil(range[1]/source.duration_sec*data.length)),visible=data.subarray(from,Math.max(from+1,to)); const dpr=window.devicePixelRatio||1,w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr)); canvas.width=w;canvas.height=h;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#2d7556';ctx.lineWidth=Math.max(1,dpr);ctx.beginPath();const step=Math.max(1,Math.floor(visible.length/w));for(let x=0;x<w;x++){{let lo=1,hi=-1;for(let i=x*step;i<Math.min(visible.length,(x+1)*step);i++){{lo=Math.min(lo,visible[i]);hi=Math.max(hi,visible[i]);}}ctx.moveTo(x,h*(.5-hi*.45));ctx.lineTo(x,h*(.5-lo*.45));}}ctx.stroke(); const span=range[1]-range[0];ctx.strokeStyle='rgba(60,72,82,.35)';ctx.lineWidth=Math.max(1,dpr*.65);ctx.beginPath();rowRefs.forEach(tr=>{{const sec=number(tr.querySelector('.start'));if(sec>=range[0]&&sec<=range[1]){{const x=(sec-range[0])/span*w;ctx.moveTo(x,0);ctx.lineTo(x,h);}}}});const finalEnd=rowRefs.length?number(rowRefs[rowRefs.length-1].querySelector('.end')):NaN;if(Number.isFinite(finalEnd)&&finalEnd>=range[0]&&finalEnd<=range[1]){{const x=(finalEnd-range[0])/span*w;ctx.moveTo(x,0);ctx.lineTo(x,h);}}ctx.stroke(); }}
    function redrawWave() {{ if(waveData)drawWave(waveData); }}
    async function decodeWave() {{ try {{ const buf=await fetch('/audio').then(r=>r.arrayBuffer());const ac=new (window.AudioContext||window.webkitAudioContext)();const decoded=await ac.decodeAudioData(buf);waveData=decoded.getChannelData(0);drawWave(waveData); }} catch (e) {{ ctx.fillStyle='#66717c';ctx.font='14px system-ui';ctx.fillText('Waveform unavailable; audio playback remains available.',12,24); }} }}
    canvas.addEventListener('click',e=>{{const rect=canvas.getBoundingClientRect(),range=activeRange();const sec=range[0]+(e.clientX-rect.left)/rect.width*(range[1]-range[0]);player.currentTime=Math.max(0,Math.min(source.duration_sec,sec));}}); function resetWaveRange() {{ viewRange=configuredRange();markAllDirty();redrawWave(); }} document.getElementById('range-start').addEventListener('input',resetWaveRange);document.getElementById('range-end').addEventListener('input',resetWaveRange);window.addEventListener('resize',redrawWave); decodeWave();
    function updateReadiness() {{ const reviewed=rowRefs.filter(tr=>tr.querySelector('.reviewed').checked).length,groups=currentGroups(),reviewedGroups=groups.filter(group=>group.rows.every(tr=>tr.querySelector('.reviewed').checked)).length; const progress=reviewed+' / '+rowRefs.length+' phones / '+reviewedGroups+' / '+groups.length+' syllables'; document.getElementById('review-progress').textContent=progress; statusNode.textContent=(draftRestored?'Draft restored; re-review required. ':'')+progress; statusNode.className='muted'; saveButton.disabled=reviewed!==rowRefs.length||!reviewerInput.value.trim(); const complete=reviewed===rowRefs.length; document.getElementById('first-unreviewed').disabled=complete;document.getElementById('previous-unreviewed').disabled=complete;document.getElementById('next-unreviewed').disabled=complete; }}
    reviewerInput.addEventListener('input',()=>{{updateReadiness();scheduleDraftSave();}});methodInput.addEventListener('input',scheduleDraftSave);discardDraftButton.addEventListener('click',()=>{{if(!window.confirm('Discard the local review draft?'))return;allowUnload=true;removeLocalDraft();window.location.reload();}});window.addEventListener('beforeunload',event=>{{if(allowUnload)return;if(rowRefs.some(tr=>tr.querySelector('.reviewed').checked)){{writeLocalDraft();event.preventDefault();event.returnValue='';}}}});updateReadiness();if(restoredActiveIndex>=0&&restoredActiveIndex<rowRefs.length)window.setTimeout(()=>focusPhone(rowRefs[restoredActiveIndex],true),0);
document.getElementById('save').addEventListener('click',async()=>{{ statusNode.className='muted';statusNode.textContent='Saving...'; const segments=rowRefs.map((tr,i)=>({{index:i,...rowData(tr)}})); const payload={{reviewer_id:document.getElementById('reviewer').value,review_method:document.getElementById('method').value,reviewed_range_sec:[number(document.getElementById('range-start')),number(document.getElementById('range-end'))],segments}}; try {{ const response=await fetch('/api/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});const result=await response.json();if(!response.ok)throw new Error(result.error||'save failed');allowUnload=true;removeLocalDraft();statusNode.className='ok';statusNode.textContent='Saved: '+result.reviewed_alignment_path+' and '+result.acceptance_path;if(result.next_review_url){{statusNode.textContent='Short review saved. ';const next=document.createElement('a');next.href=result.next_review_url;next.textContent='Open full-song review';statusNode.appendChild(next);}}else if(result.next_review_error){{statusNode.className='error';statusNode.textContent='Short review saved; full-song server failed: '+result.next_review_error;}}document.getElementById('save').disabled=true; }} catch(e) {{statusNode.className='error';statusNode.textContent=e.message;}} }});
</script></body></html>"""
    return page.encode("utf-8")


class _ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, render_review_page(self.server.source), content_type="text/html; charset=utf-8")
            return
        if path == "/api/review":
            self._send_json(HTTPStatus.OK, self.server.public_source())
            return
        if path == "/audio":
            self._send_audio()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/api/save", "/save"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_POST_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body size is invalid"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ReviewError("request JSON must be an object")
            result = self.server.save(payload)
        except ReviewError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc), "error_kind": exc.error_kind})
            return
        except (UnicodeError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
            return
        self._send_json(HTTPStatus.CREATED, result)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        self._send(status, json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"), content_type="application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes | str, *, content_type: str) -> None:
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(int(status)); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(raw))); self.send_header("Cache-Control", "no-store"); self.end_headers()
        if self.command != "HEAD": self.wfile.write(raw)

    def _send_audio(self) -> None:
        path = self.server.audio_path
        try:
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range", "")
            if range_header.startswith("bytes="):
                value = range_header[6:].split(",", 1)[0].strip(); left, _, right = value.partition("-")
                if left:
                    start = int(left); end = int(right) if right else size - 1
                elif right:
                    count = int(right); start = max(0, size - count)
                if start < 0 or start >= size or end < start: raise ValueError
                end = min(end, size - 1)
            with path.open("rb") as handle:
                handle.seek(start); body = handle.read(end - start + 1)
        except (OSError, ValueError):
            self._send_json(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"error": "audio range is invalid"}); return
        partial = start or end != size - 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "audio/wav"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(len(body)))
        if partial: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command != "HEAD": self.wfile.write(body)


class ReviewHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying immutable input state and output configuration."""

    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        source: Mapping[str, Any],
        output_dir: Path,
        reviewed_alignment_name: str,
        acceptance_name: str,
        next_full_review: Mapping[str, Any] | None = None,
    ) -> None:
        self.source = dict(source)
        self.audio_path = Path(str(source["audio_path"]))
        self.output_dir = output_dir
        self.reviewed_alignment_name = reviewed_alignment_name
        self.acceptance_name = acceptance_name
        self.next_full_review = dict(next_full_review or {})
        self.next_full_server: ReviewHTTPServer | None = None
        self.next_full_thread: threading.Thread | None = None
        self._save_lock = threading.Lock()
        super().__init__(address, _ReviewHandler)

    def public_source(self) -> dict[str, Any]:
        return {key: value for key, value in self.source.items() if key != "alignment"}

    def save(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._save_lock:
            result = save_review(
                self.output_dir,
                self.source,
                request,
                reviewed_alignment_name=self.reviewed_alignment_name,
                acceptance_name=self.acceptance_name,
            )
            if self.next_full_review:
                try:
                    result.update(
                        self._start_next_full_review(Path(str(result["acceptance_path"])))
                    )
                except (OSError, ReviewError) as exc:
                    result.update(
                        {
                            "next_review_started": False,
                            "next_review_error": str(exc),
                        }
                    )
            return result

    def _start_next_full_review(self, acceptance_path: Path) -> dict[str, Any]:
        if self.next_full_server is None:
            timeline_path = self.source.get("lyrics_timeline_path")
            if not timeline_path:
                raise ReviewError(
                    "automatic full-song review requires a lyrics timeline",
                    error_kind="missing_lyrics_timing_evidence",
                )
            config = self.next_full_review
            self.next_full_server = create_server(
                self.audio_path,
                Path(str(config["alignment_path"])),
                Path(str(config["output_dir"])),
                lyrics_timeline_path=Path(str(timeline_path)),
                lyrics_timeline_audio_path=Path(
                    str(self.source["lyrics_timeline_audio_path"])
                ),
                review_scope=FULL_SONG_SCOPE,
                short_clip_acceptance_path=acceptance_path,
                host=str(config["host"]),
                port=int(config["port"]),
            )
            self.next_full_thread = threading.Thread(
                target=self.next_full_server.serve_forever,
                name="full-song-phone-review",
                daemon=True,
            )
            self.next_full_thread.start()
        host = str(self.next_full_review["host"])
        port = self.next_full_server.server_address[1]
        return {
            "next_review_started": True,
            "next_review_scope": FULL_SONG_SCOPE,
            "next_review_url": f"http://{host}:{port}/",
        }

    def server_close(self) -> None:
        if self.next_full_server is not None:
            self.next_full_server.shutdown()
            self.next_full_server.server_close()
            if self.next_full_thread is not None:
                self.next_full_thread.join(timeout=2.0)
        super().server_close()


def create_server(
    audio_path: Path | str,
    alignment_path: Path | str,
    output_dir: Path | str,
    *,
    lyrics_timeline_path: Path | str | None = None,
    lyrics_timeline_audio_path: Path | str | None = None,
    review_scope: str = SHORT_CLIP_SCOPE,
    reviewed_range_sec: Sequence[float] | None = None,
    short_clip_acceptance_path: Path | str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    reviewed_alignment_name: str = DEFAULT_REVIEWED_ALIGNMENT_NAME,
    acceptance_name: str = DEFAULT_ACCEPTANCE_NAME,
    next_full_alignment_path: Path | str | None = None,
    next_full_output_dir: Path | str | None = None,
    next_full_host: str = DEFAULT_HOST,
    next_full_port: int = DEFAULT_FULL_REVIEW_PORT,
) -> ReviewHTTPServer:
    """Load inputs once and create a local review server."""

    try:
        port_number = int(port)
    except (TypeError, ValueError) as exc:
        raise ReviewError("port must be an integer", error_kind="server_configuration") from exc
    if not 0 <= port_number <= 65535:
        raise ReviewError("port must be between 0 and 65535", error_kind="server_configuration")
    if (next_full_alignment_path is None) != (next_full_output_dir is None):
        raise ReviewError(
            "next full review requires both alignment and output directory",
            error_kind="server_configuration",
        )
    try:
        next_port_number = int(next_full_port)
    except (TypeError, ValueError) as exc:
        raise ReviewError("next full review port must be an integer", error_kind="server_configuration") from exc
    if not 0 <= next_port_number <= 65535:
        raise ReviewError(
            "next full review port must be between 0 and 65535",
            error_kind="server_configuration",
        )
    if next_full_alignment_path is not None and review_scope != SHORT_CLIP_SCOPE:
        raise ReviewError(
            "automatic full-song continuation is valid only from short_clip scope",
            error_kind="server_configuration",
        )
    source = load_review_inputs(
        audio_path,
        alignment_path,
        lyrics_timeline_path=lyrics_timeline_path,
        lyrics_timeline_audio_path=lyrics_timeline_audio_path,
        review_scope=review_scope,
        reviewed_range_sec=reviewed_range_sec,
        short_clip_acceptance_path=short_clip_acceptance_path,
    )
    output = Path(output_dir).expanduser().resolve()
    _safe_output_name(reviewed_alignment_name, label="reviewed alignment name")
    _safe_output_name(acceptance_name, label="acceptance name")
    next_full_review = None
    if next_full_alignment_path is not None and next_full_output_dir is not None:
        next_alignment = Path(next_full_alignment_path).expanduser().resolve()
        if not next_alignment.is_file():
            raise ReviewError(
                f"missing next full-song alignment: {next_alignment}",
                error_kind="missing_input",
            )
        next_full_review = {
            "alignment_path": str(next_alignment),
            "output_dir": str(Path(next_full_output_dir).expanduser().resolve()),
            "host": str(next_full_host or DEFAULT_HOST),
            "port": next_port_number,
        }
    return ReviewHTTPServer(
        (str(host or DEFAULT_HOST), port_number),
        source=source,
        output_dir=output,
        reviewed_alignment_name=reviewed_alignment_name,
        acceptance_name=acceptance_name,
        next_full_review=next_full_review,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review forced_phone_alignment_v1 phone boundaries in a local browser.")
    parser.add_argument("--audio", required=True, type=Path, help="Read-only vocal-stem PCM WAV.")
    parser.add_argument("--alignment", required=True, type=Path, help="Read-only forced_phone_alignment_v1 JSON.")
    parser.add_argument("--lyrics-timeline", type=Path, help="Optional hash-bound lyrics_timeline_v1 labels for review context.")
    parser.add_argument(
        "--lyrics-timeline-audio",
        type=Path,
        help="Original audio whose SHA-256 is declared by --lyrics-timeline.",
    )
    parser.add_argument("--scope", choices=(SHORT_CLIP_SCOPE, FULL_SONG_SCOPE), default=SHORT_CLIP_SCOPE)
    parser.add_argument("--review-start-sec", type=float)
    parser.add_argument("--review-end-sec", type=float)
    parser.add_argument("--short-clip-acceptance", type=Path, help="Required approved short-clip acceptance for full_song scope.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for new reviewed artifacts.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Interface to bind (default: {DEFAULT_HOST}).")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help=f"TCP port (default: {DEFAULT_PORT}).")
    parser.add_argument("--reviewed-alignment-name", default=DEFAULT_REVIEWED_ALIGNMENT_NAME)
    parser.add_argument("--acceptance-name", default=DEFAULT_ACCEPTANCE_NAME)
    parser.add_argument("--next-full-alignment", type=Path)
    parser.add_argument("--next-full-output-dir", type=Path)
    parser.add_argument("--next-full-host", default=DEFAULT_HOST)
    parser.add_argument("--next-full-port", default=DEFAULT_FULL_REVIEW_PORT, type=int)
    args = parser.parse_args(argv)
    if (args.review_start_sec is None) != (args.review_end_sec is None):
        parser.error("--review-start-sec and --review-end-sec must be provided together")
    reviewed_range = (
        None
        if args.review_start_sec is None
        else [args.review_start_sec, args.review_end_sec]
    )
    try:
        server = create_server(
            args.audio,
            args.alignment,
            args.output_dir,
            lyrics_timeline_path=args.lyrics_timeline,
            lyrics_timeline_audio_path=args.lyrics_timeline_audio,
            review_scope=args.scope,
            reviewed_range_sec=reviewed_range,
            short_clip_acceptance_path=args.short_clip_acceptance,
            host=args.host,
            port=args.port,
            reviewed_alignment_name=args.reviewed_alignment_name,
            acceptance_name=args.acceptance_name,
            next_full_alignment_path=args.next_full_alignment,
            next_full_output_dir=args.next_full_output_dir,
            next_full_host=args.next_full_host,
            next_full_port=args.next_full_port,
        )
    except ReviewError as exc:
        parser.error(str(exc)); return 2
    print(f"Phone review listening on http://{args.host}:{server.server_address[1]}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


__all__ = [
    "ACCEPTANCE_SCHEMA_VERSION",
    "ALIGNMENT_SCHEMA_VERSION",
    "DEFAULT_ACCEPTANCE_NAME",
    "DEFAULT_FULL_REVIEW_PORT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_REVIEWED_ALIGNMENT_NAME",
    "FULL_SONG_SCOPE",
    "HUMAN_REVIEW_ROUTE",
    "SHORT_CLIP_SCOPE",
    "ReviewError",
    "ReviewHTTPServer",
    "create_server",
    "load_review_inputs",
    "main",
    "render_review_page",
    "save_review",
    "sha256_file",
    "wav_duration",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Validated lyric timing evidence shared by ASR and forced-alignment adapters."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import wave


SCHEMA_VERSION = "lyrics_timeline_v1"
_SOURCES = frozenset({"user_lyrics", "external_asr", "qwen3_asr", "manual_alignment"})


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class LyricsTimelineError(ValueError):
    """The lyric timing evidence cannot be used for product alignment."""

    def __init__(self, error_kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


@dataclass(frozen=True, slots=True)
class LyricsTimelineItem:
    index: int
    text: str
    start_sec: float
    end_sec: float
    jyutping: str | None = None
    g2p_candidates: tuple[Mapping[str, Any], ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline item index must be non-negative")
        if not isinstance(self.text, str) or not self.text.strip():
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline item text must be non-empty")
        if (
            not _is_finite_number(self.start_sec)
            or not _is_finite_number(self.end_sec)
            or self.start_sec < 0
            or self.end_sec <= self.start_sec
        ):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline item bounds are invalid")
        if self.jyutping is not None and not isinstance(self.jyutping, str):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline item jyutping must be text")
        if not isinstance(self.g2p_candidates, Sequence) or isinstance(self.g2p_candidates, (str, bytes)):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline item candidates must be a list")
        if any(not isinstance(candidate, Mapping) for candidate in self.g2p_candidates):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline item candidates must be objects")
        if self.confidence is not None and (
            not _is_finite_number(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline item confidence is invalid")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "text": self.text,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }
        if self.jyutping:
            payload["jyutping"] = self.jyutping
        if self.g2p_candidates:
            payload["g2p_candidates"] = [dict(candidate) for candidate in self.g2p_candidates]
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


@dataclass(frozen=True, slots=True)
class LyricsTimeline:
    audio_sha256: str
    source: str
    items: tuple[LyricsTimelineItem, ...]
    provenance: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        digest = self.audio_sha256.lower() if isinstance(self.audio_sha256, str) else ""
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline audio_sha256 is invalid")
        object.__setattr__(self, "audio_sha256", digest)
        if (
            not isinstance(self.source, str)
            or self.source not in _SOURCES
            or self.schema_version != SCHEMA_VERSION
            or not self.items
            or not isinstance(self.provenance, Mapping)
        ):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline source/schema/items are invalid")
        for index, item in enumerate(self.items):
            if not isinstance(item, LyricsTimelineItem) or item.index != index or index and item.start_sec < self.items[index - 1].end_sec:
                raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline items must be indexed and non-overlapping")

    @property
    def text(self) -> str:
        return "".join(item.text for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audio_sha256": self.audio_sha256,
            "source": self.source,
            "items": [item.to_dict() for item in self.items],
            "provenance": dict(self.provenance),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_timeline_sha256(timeline: LyricsTimeline) -> str:
    try:
        encoded = json.dumps(
            timeline.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline is not canonical JSON evidence") from exc
    return hashlib.sha256(encoded).hexdigest()


def load_lyrics_timeline(path: Path | str, *, audio_path: Path | str | None = None) -> LyricsTimeline:
    timeline_path = Path(path)
    try:
        payload = json.loads(timeline_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise LyricsTimelineError("missing_lyrics_timing_evidence", f"missing lyrics timeline: {timeline_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"invalid lyrics timeline: {timeline_path}") from exc
    try:
        timeline = parse_lyrics_timeline(payload)
    except (OverflowError, TypeError, ValueError) as exc:
        if isinstance(exc, LyricsTimelineError):
            raise
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"invalid lyrics timeline: {timeline_path}") from exc
    if audio_path is not None:
        audio = Path(audio_path)
        if not audio.is_file():
            raise LyricsTimelineError("missing_audio", f"missing timeline audio: {audio}")
        try:
            actual_hash = sha256_file(audio)
        except OSError as exc:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"cannot read timeline audio: {audio}") from exc
        if actual_hash != timeline.audio_sha256:
            raise LyricsTimelineError(
                "lyrics_timing_audio_hash_mismatch",
                f"lyrics timeline audio hash {timeline.audio_sha256} does not match {actual_hash}",
            )
        duration = _wav_duration(audio)
        if duration is None:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline audio must be a readable PCM WAV: {audio}")
        if timeline.items[-1].end_sec > duration:
            raise LyricsTimelineError(
                "lyrics_timing_out_of_bounds",
                f"timeline ends at {timeline.items[-1].end_sec:.6f}s beyond audio duration {duration:.6f}s",
            )
    return timeline


def parse_lyrics_timeline(payload: object) -> LyricsTimeline:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"schema must be {SCHEMA_VERSION}")
    raw_audio_sha256 = payload.get("audio_sha256")
    audio_sha256 = raw_audio_sha256.lower() if isinstance(raw_audio_sha256, str) else ""
    if len(audio_sha256) != 64 or any(char not in "0123456789abcdef" for char in audio_sha256):
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", "audio_sha256 must be a 64-character hex digest")
    raw_source = payload.get("source")
    source = raw_source.strip() if isinstance(raw_source, str) else ""
    if source not in _SOURCES:
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"unsupported timeline source: {source}")
    raw_items = payload.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)) or not raw_items:
        raise LyricsTimelineError("missing_lyrics_timing_evidence", "lyrics timeline items must be non-empty")
    items: list[LyricsTimelineItem] = []
    previous_end = 0.0
    for expected_index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {expected_index} must be an object")
        try:
            index_value = raw_item["index"]
            start_value = raw_item["start_sec"]
            end_value = raw_item["end_sec"]
            if (
                isinstance(index_value, bool)
                or not isinstance(index_value, int)
                or isinstance(start_value, bool)
                or not isinstance(start_value, (int, float))
                or isinstance(end_value, bool)
                or not isinstance(end_value, (int, float))
            ):
                raise TypeError("index/timing fields have invalid JSON types")
            index = index_value
            start = float(start_value)
            end = float(end_value)
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {expected_index} has invalid timing") from exc
        raw_text = raw_item.get("text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if index != expected_index or not text:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {expected_index} has invalid index/text")
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {index} has invalid bounds")
        if start < previous_end:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline items must be ordered and non-overlapping")
        confidence = raw_item.get("confidence")
        try:
            if confidence is not None and (
                isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            ):
                raise TypeError("confidence has invalid JSON type")
            parsed_confidence = None if confidence is None else float(confidence)
        except (OverflowError, TypeError, ValueError) as exc:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {index} has invalid confidence") from exc
        if parsed_confidence is not None and (not math.isfinite(parsed_confidence) or not 0.0 <= parsed_confidence <= 1.0):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {index} has invalid confidence")
        raw_candidates = raw_item.get("g2p_candidates", [])
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {index} candidates must be a list")
        if any(not isinstance(candidate, Mapping) for candidate in raw_candidates):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {index} candidates must be objects")
        candidates = tuple(dict(candidate) for candidate in raw_candidates)
        raw_jyutping = raw_item.get("jyutping")
        if raw_jyutping is not None and not isinstance(raw_jyutping, str):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"timeline item {index} jyutping must be text")
        items.append(
            LyricsTimelineItem(
                index=index,
                text=text,
                start_sec=start,
                end_sec=end,
                jyutping=raw_jyutping.strip() or None if raw_jyutping is not None else None,
                g2p_candidates=candidates,
                confidence=parsed_confidence,
            )
        )
        previous_end = end
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", "timeline provenance must be an object")
    return LyricsTimeline(audio_sha256, source, tuple(items), dict(provenance))


def timeline_from_asr_transcript(
    payload: Mapping[str, Any],
    *,
    audio_path: Path | str,
    source: str = "qwen3_asr",
) -> LyricsTimeline:
    """Convert timestamped ASR segments; text-only transcripts are rejected."""
    if not isinstance(payload, Mapping):
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", "ASR transcript must be an object")
    if payload.get("schema_version") != "asr_transcript_v1":
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", "ASR schema must be asr_transcript_v1")
    if not isinstance(source, str) or source not in _SOURCES or source == "user_lyrics":
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"unsupported ASR timeline source: {source}")
    raw_segments = payload.get("segments") if isinstance(payload, Mapping) else None
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)) or not raw_segments:
        raise LyricsTimelineError(
            "missing_lyrics_timing_evidence",
            "ASR transcript has no timestamped segments; text-only evidence cannot form a product timeline",
        )
    timestamp_rows: list[Mapping[str, Any]] = []
    for segment in raw_segments:
        if not isinstance(segment, Mapping):
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "ASR transcript segment must be an object")
        nested_items = segment.get("items")
        if isinstance(nested_items, Sequence) and not isinstance(nested_items, (str, bytes)) and nested_items:
            if any(not isinstance(item, Mapping) for item in nested_items):
                raise LyricsTimelineError("invalid_lyrics_timing_evidence", "ASR timestamp items must be objects")
            timestamp_rows.extend(item for item in nested_items if isinstance(item, Mapping))
        else:
            timestamp_rows.append(segment)
    items: list[dict[str, Any]] = []
    for segment in timestamp_rows:
        start_value = segment.get("start_sec")
        if start_value is None:
            start_value = segment.get("start_time")
        end_value = segment.get("end_sec")
        if end_value is None:
            end_value = segment.get("end_time")
        if start_value is None or end_value is None:
            raise LyricsTimelineError(
                "missing_lyrics_timing_evidence",
                "ASR transcript requires start_sec/end_sec timestamps for every segment",
            )
        raw_text = segment.get("text")
        if raw_text is None:
            raw_text = segment.get("lyric")
        if raw_text is None:
            raw_text = segment.get("token")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if not text:
            raise LyricsTimelineError("invalid_lyrics_timing_evidence", "ASR timestamp segment text is empty")
        units = [unit for unit in text if not unit.isspace()]
        if len(units) != 1:
            raise LyricsTimelineError(
                "missing_lyrics_timing_evidence",
                "ASR segment timing must identify one lyric syllable; no equal-duration splitting is allowed",
            )
        item_payload = {
            "index": len(items),
            "text": units[0],
            "start_sec": start_value,
            "end_sec": end_value,
        }
        if segment.get("confidence") is not None:
            item_payload["confidence"] = segment["confidence"]
        items.append(item_payload)
    audio = Path(audio_path)
    if not audio.is_file():
        raise LyricsTimelineError("missing_audio", f"missing ASR timeline audio: {audio}")
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
    truth_layer = str(provenance.get("truth_layer") or "")
    if truth_layer != "asr_evidence_not_phone_truth":
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", "ASR transcript has an invalid truth-layer marker")
    try:
        actual_audio_hash = sha256_file(audio)
    except OSError as exc:
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"cannot read ASR timeline audio: {audio}") from exc
    duration = _wav_duration(audio)
    if duration is None:
        raise LyricsTimelineError("invalid_lyrics_timing_evidence", f"ASR timeline audio must be a readable PCM WAV: {audio}")
    raw_declared_audio_hash = provenance.get("audio_sha256")
    declared_audio_hash = raw_declared_audio_hash.lower() if isinstance(raw_declared_audio_hash, str) else ""
    if len(declared_audio_hash) != 64 or declared_audio_hash != actual_audio_hash:
        raise LyricsTimelineError(
            "lyrics_timing_audio_hash_mismatch",
            f"ASR transcript audio hash {declared_audio_hash} does not match {actual_audio_hash}",
        )
    timeline = parse_lyrics_timeline(
        {
            "schema_version": SCHEMA_VERSION,
            "audio_sha256": actual_audio_hash,
            "source": source,
            "items": items,
            "provenance": {
                **dict(provenance),
                "audio_sha256": actual_audio_hash,
                "timestamps_required": True,
                "asr_schema_version": payload.get("schema_version"),
            },
        }
    )
    raw_transcript_text = payload.get("text")
    transcript_text = (
        "".join(unit for unit in raw_transcript_text if not unit.isspace())
        if isinstance(raw_transcript_text, str)
        else ""
    )
    if not transcript_text or transcript_text != timeline.text:
        raise LyricsTimelineError(
            "invalid_lyrics_timing_evidence",
            "ASR transcript text does not match its timestamped lyric items",
        )
    if timeline.items[-1].end_sec > duration:
        raise LyricsTimelineError(
            "lyrics_timing_out_of_bounds",
            f"ASR timeline ends at {timeline.items[-1].end_sec:.6f}s beyond audio duration {duration:.6f}s",
        )
    return timeline


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (EOFError, OSError, wave.Error):
        return None


__all__ = [
    "LyricsTimeline",
    "LyricsTimelineError",
    "LyricsTimelineItem",
    "SCHEMA_VERSION",
    "canonical_timeline_sha256",
    "load_lyrics_timeline",
    "parse_lyrics_timeline",
    "timeline_from_asr_transcript",
    "sha256_file",
]

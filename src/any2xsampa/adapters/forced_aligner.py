"""Forced-phone alignment evidence adapters.

This module deliberately does not infer timing from lyric duration.  Product
mode consumes an explicit alignment artifact produced by a replaceable
external aligner (or a fixture artifact in tests).
"""
from __future__ import annotations

import json
import math
import subprocess
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from any2xsampa.cantonese.g2p import ResolvedLyricsItem
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.lyrics.timeline import (
    LyricsTimeline,
    canonical_timeline_sha256,
    sha256_file,
)

SCHEMA_VERSION = "forced_phone_alignment_v1"
MACHINE_REVIEW_DRAFT_STATUS = "machine_draft_pending_human_phone_review"
PENDING_HUMAN_PHONE_REVIEW_STATUS = "pending_human_phone_review"
HUMAN_REVIEW_PENDING_DETACHED_ACCEPTANCE_STATUS = (
    "human_reviewed_phone_alignment_pending_detached_acceptance_validation"
)


class ForcedPhoneAlignerError(ValueError):
    """Explicit forced alignment evidence is invalid or unavailable."""

    def __init__(self, error_kind: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class ForcedAlignmentResult:
    segments: tuple[PhoneSegment, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "vocal_stem_sha256": self.metadata.get("vocal_stem_sha256"),
            "lyrics_timeline_sha256": self.metadata.get("lyrics_timeline_sha256"),
            "alignment_source": self.metadata.get("alignment_source"),
            "model_output": self.metadata.get("source_mode") == "real",
            "model": dict(self.metadata.get("model", {})),
            "model_hash": self.metadata.get("model_hash"),
            "provenance": dict(self.metadata.get("provenance", {})),
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": dict(self.metadata),
        }
        provenance = self.metadata.get("provenance")
        if isinstance(provenance, Mapping):
            if "status" in provenance:
                payload["status"] = provenance["status"]
            if "product_accepted" in provenance:
                payload["product_accepted"] = provenance["product_accepted"]
        for key in ("status", "product_accepted", "machine_suggestion", "human_review_required"):
            if key in self.metadata:
                payload[key] = self.metadata[key]
        review = self.metadata.get("review")
        if isinstance(review, Mapping):
            payload["review"] = dict(review)
        return payload


class ForcedPhoneAligner(Protocol):
    def align(
        self,
        vocal_stem: Path,
        timeline: LyricsTimeline,
        resolved_items: Sequence[ResolvedLyricsItem],
    ) -> ForcedAlignmentResult: ...


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (EOFError, OSError, wave.Error):
        return None


def lyrics_timeline_sha256(timeline: LyricsTimeline) -> str:
    return canonical_timeline_sha256(timeline)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def is_human_reviewed_alignment_payload(payload: object) -> bool:
    """Return whether a payload is reviewed but still awaiting detached acceptance."""

    if not isinstance(payload, Mapping):
        return False
    review = _mapping(payload.get("review"))
    provenance = _mapping(payload.get("provenance"))
    return (
        str(review.get("status") or "").strip().lower() == "human_reviewed"
        and str(provenance.get("status") or "").strip().lower()
        == HUMAN_REVIEW_PENDING_DETACHED_ACCEPTANCE_STATUS
    )


def is_machine_review_draft_payload(payload: object) -> bool:
    """Return whether a payload carries the canonical machine-review draft markers.

    A human-reviewed artifact may retain ``product_accepted=false`` while its
    detached acceptance record is validated.  That shape is intentionally not
    classified as a machine draft here.
    """

    if not isinstance(payload, Mapping):
        return False
    review = _mapping(payload.get("review"))
    provenance = _mapping(payload.get("provenance"))
    if is_human_reviewed_alignment_payload(payload):
        return False
    statuses = {
        str(payload.get("status") or "").strip().lower(),
        str(payload.get("review_status") or "").strip().lower(),
        str(review.get("status") or "").strip().lower(),
        str(review.get("review_status") or "").strip().lower(),
        str(provenance.get("status") or "").strip().lower(),
        str(provenance.get("review_status") or "").strip().lower(),
    }
    if MACHINE_REVIEW_DRAFT_STATUS in statuses or PENDING_HUMAN_PHONE_REVIEW_STATUS in statuses:
        return True
    if payload.get("machine_suggestion") is True or provenance.get("machine_suggestion") is True:
        return True
    if "product_accepted" in payload and payload.get("product_accepted") is not True:
        return True
    return "product_accepted" in provenance and provenance.get("product_accepted") is not True


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"{label} must be numeric")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"{label} must be finite")
    return parsed


def _candidate_payload(raw: object, *, index: int) -> list[PhoneCandidate]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"segment {index} has no candidate_phones")
    candidates: list[PhoneCandidate] = []
    for candidate_index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"segment {index} candidate {candidate_index} must be an object",
            )
        try:
            symbol_value = value.get("symbol")
            if symbol_value is None:
                symbol_value = value.get("phone")
            if symbol_value is None:
                symbol_value = value.get("label")
            symbol = str(symbol_value or "").strip()
            confidence = _finite(value.get("confidence", value.get("score", 0.0)), label="candidate confidence")
            candidates.append(PhoneCandidate(symbol=symbol, confidence=confidence))
        except (KeyError, TypeError, ValueError, ForcedPhoneAlignerError) as exc:
            if isinstance(exc, ForcedPhoneAlignerError):
                raise
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"invalid candidate in segment {index}") from exc
    if any(candidates[index].confidence < candidates[index + 1].confidence for index in range(len(candidates) - 1)):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            f"segment {index} candidate_phones must be ordered by descending confidence",
        )
    return candidates


def _validate_metadata(
    raw: object,
    *,
    index: int,
    model_hash: str,
    alignment_source: str,
) -> dict[str, Any]:
    metadata = dict(raw) if isinstance(raw, Mapping) else {}
    required = ("alignment_source", "lyric_item_index", "direct_overlap_sec", "phone_overlap_ratio", "alignment_confidence")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            f"segment {index} missing alignment metadata: {', '.join(missing)}",
        )
    try:
        raw_lyric_index = metadata["lyric_item_index"]
        if isinstance(raw_lyric_index, bool) or not isinstance(raw_lyric_index, int):
            raise TypeError("lyric item index is not an integer")
        lyric_index = raw_lyric_index
    except (TypeError, ValueError) as exc:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"segment {index} has invalid lyric_item_index") from exc
    direct_overlap = _finite(metadata["direct_overlap_sec"], label="direct_overlap_sec")
    overlap_ratio = _finite(metadata["phone_overlap_ratio"], label="phone_overlap_ratio")
    alignment_confidence = _finite(metadata["alignment_confidence"], label="alignment_confidence")
    if lyric_index < 0 or direct_overlap < 0.0 or not 0.0 <= overlap_ratio <= 1.0 or not 0.0 <= alignment_confidence <= 1.0:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"segment {index} has invalid alignment metadata")
    metadata["lyric_item_index"] = lyric_index
    metadata["direct_overlap_sec"] = direct_overlap
    metadata["phone_overlap_ratio"] = overlap_ratio
    metadata["alignment_confidence"] = alignment_confidence
    raw_source = str(metadata.get("alignment_source") or "").strip()
    if not raw_source or raw_source != alignment_source:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            f"segment {index} alignment_source does not match the top-level source",
        )
    for protected_key, expected in (
        ("model_hash", model_hash),
        ("source_mode", "real"),
        ("truth_layer", "phones_ipa_articulatory_features"),
    ):
        actual = metadata.get(protected_key)
        if protected_key == "model_hash" and isinstance(actual, str):
            actual = actual.strip().lower()
        if protected_key in metadata and actual != expected:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"segment {index} has conflicting protected metadata: {protected_key}",
            )
        metadata[protected_key] = expected
    metadata.setdefault("interpolation_reason", None)
    metadata.setdefault("source_mode", "real")
    metadata.setdefault("truth_layer", "phones_ipa_articulatory_features")
    return metadata


def parse_forced_phone_alignment(
    payload: object,
    *,
    vocal_stem: Path,
    timeline: LyricsTimeline,
    resolved_items: Sequence[ResolvedLyricsItem],
) -> ForcedAlignmentResult:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"schema must be {SCHEMA_VERSION}")
    for container_name, container in (
        ("payload", payload),
        ("provenance", _mapping(payload.get("provenance"))),
    ):
        if "product_accepted" in container and not isinstance(container["product_accepted"], bool):
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"{container_name}.product_accepted must be boolean",
            )
    if is_machine_review_draft_payload(payload):
        raise ForcedPhoneAlignerError(
            "unqualified_forced_phone_evidence",
            "machine phone-review drafts cannot be parsed as accepted forced-phone evidence",
        )
    if not vocal_stem.is_file():
        raise ForcedPhoneAlignerError("missing_vocal_stem_evidence", f"missing vocal stem: {vocal_stem}")
    actual_stem_hash = sha256_file(vocal_stem)
    canonical_stem_hash = str(payload.get("vocal_stem_sha256") or "").lower()
    legacy_stem_hash = str(payload.get("audio_sha256") or "").lower()
    if canonical_stem_hash and legacy_stem_hash and canonical_stem_hash != legacy_stem_hash:
        raise ForcedPhoneAlignerError("forced_phone_audio_hash_mismatch", "conflicting vocal stem hash fields")
    declared_stem_hash = canonical_stem_hash or legacy_stem_hash
    if len(declared_stem_hash) != 64 or declared_stem_hash != actual_stem_hash:
        raise ForcedPhoneAlignerError(
            "forced_phone_audio_hash_mismatch",
            f"forced alignment stem hash does not match {actual_stem_hash}",
            details={"declared": declared_stem_hash, "actual": actual_stem_hash},
        )
    timeline_hash = lyrics_timeline_sha256(timeline)
    declared_timeline_hash = str(payload.get("lyrics_timeline_sha256") or "").lower()
    if declared_timeline_hash != timeline_hash:
        raise ForcedPhoneAlignerError(
            "forced_phone_lyrics_hash_mismatch",
            "forced alignment is not bound to the supplied lyrics timeline",
            details={"declared": declared_timeline_hash, "actual": timeline_hash},
        )
    if payload.get("model_output") is not True:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "product forced alignment must explicitly declare model_output=true",
        )
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)) or not raw_segments:
        raise ForcedPhoneAlignerError("missing_phone_evidence", "forced alignment contains no segments")
    model_payload = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    direct_model_hash = str(payload.get("model_hash") or "").strip()
    nested_model_hash = str(model_payload.get("hash") or "").strip()
    if direct_model_hash and nested_model_hash and direct_model_hash.lower() != nested_model_hash.lower():
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "conflicting forced alignment model hash fields")
    model_hash = (direct_model_hash or nested_model_hash).lower()
    if len(model_hash) != 64 or any(char not in "0123456789abcdefABCDEF" for char in model_hash):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "forced alignment model_hash must be SHA-256")
    alignment_source = str(payload.get("alignment_source") or "").strip()
    if not alignment_source:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "alignment_source must be non-empty")
    duration = _wav_duration(vocal_stem)
    if duration is None:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "vocal stem must be a readable PCM WAV")
    timeline_items = {item.index: item for item in timeline.items}
    resolved_by_index = {item.index: item for item in resolved_items}
    if set(resolved_by_index) != set(timeline_items):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "resolved G2P items must cover exactly the lyrics timeline items",
        )
    for item_index, item in resolved_by_index.items():
        timeline_item = timeline_items[item_index]
        if (
            item.text != timeline_item.text
            or abs(item.start_sec - timeline_item.start_sec) > 1e-9
            or abs(item.end_sec - timeline_item.end_sec) > 1e-9
        ):
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"resolved G2P timing/text does not match timeline item {item_index}",
            )
    expected_indices = set(timeline_items)
    segments: list[PhoneSegment] = []
    previous_end = -1.0
    covered: set[int] = set()
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"segment {index} must be an object")
        start = _finite(raw_segment.get("start_sec"), label=f"segment {index} start_sec")
        end = _finite(raw_segment.get("end_sec"), label=f"segment {index} end_sec")
        if start < 0.0 or end <= start or start < previous_end or duration is not None and end > duration + 1e-6:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"segment {index} has invalid or overlapping timing")
        metadata = _validate_metadata(
            raw_segment.get("metadata"),
            index=index,
            model_hash=model_hash,
            alignment_source=alignment_source,
        )
        lyric_index = metadata["lyric_item_index"]
        if lyric_index not in expected_indices:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"segment {index} references unknown lyric item {lyric_index}")
        lyric_item = resolved_by_index[lyric_index]
        actual_overlap = max(0.0, min(end, lyric_item.end_sec) - max(start, lyric_item.start_sec))
        actual_overlap_ratio = actual_overlap / (end - start)
        if actual_overlap <= 0.0:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence", f"segment {index} does not overlap lyric item {lyric_index}"
            )
        if (
            abs(float(metadata["direct_overlap_sec"]) - actual_overlap) > 1e-6
            or abs(float(metadata["phone_overlap_ratio"]) - actual_overlap_ratio) > 1e-6
        ):
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"segment {index} overlap metadata does not match its lyric timeline intersection",
            )
        candidates = _candidate_payload(raw_segment.get("candidate_phones"), index=index)
        confidence = _finite(raw_segment.get("confidence", candidates[0].confidence), label=f"segment {index} confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"segment {index} confidence is outside [0,1]")
        normalized_ipa = str(raw_segment.get("normalized_ipa") or "").strip()
        if not normalized_ipa:
            normalized_ipa = candidates[0].symbol.strip()
        if not normalized_ipa:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"segment {index} normalized_ipa is empty")
        segment = PhoneSegment(
            time_range=TimeRange(start, end),
            candidate_phones=candidates,
            confidence=confidence,
            normalized_ipa=normalized_ipa,
            metadata=metadata,
        )
        segments.append(segment)
        covered.add(lyric_index)
        previous_end = end
    missing = sorted(expected_indices - covered)
    if missing:
        raise ForcedPhoneAlignerError(
            "missing_phone_evidence",
            f"forced alignment does not cover lyric item(s): {missing}",
            details={"missing_lyric_item_indices": missing},
        )
    provenance = dict(payload.get("provenance")) if isinstance(payload.get("provenance"), Mapping) else {}
    metadata = {
        "alignment_source": alignment_source,
        "source_mode": "real",
        "vocal_stem_sha256": actual_stem_hash,
        "lyrics_audio_sha256": timeline.audio_sha256,
        "lyrics_timeline_sha256": timeline_hash,
        "model": {
            **dict(model_payload),
            **({"hash": model_hash} if "hash" in model_payload else {}),
        },
        "model_hash": model_hash,
        "segment_count": len(segments),
        "lyric_item_count": len(expected_indices),
        "covered_lyric_item_count": len(covered),
        "truth_layer": "phones_ipa_articulatory_features",
        "provenance": provenance,
    }
    for key in ("status", "product_accepted", "machine_suggestion", "human_review_required"):
        if key in payload:
            metadata[key] = payload[key]
        elif key in provenance:
            metadata[key] = provenance[key]
    review = payload.get("review")
    if isinstance(review, Mapping):
        metadata["review"] = dict(review)
    return ForcedAlignmentResult(tuple(segments), metadata)


class FixtureForcedPhoneAligner:
    """Load a previously produced, hash-bound forced alignment artifact."""

    def __init__(self, alignment_path: Path | str) -> None:
        self.alignment_path = Path(alignment_path)

    def align(self, vocal_stem: Path, timeline: LyricsTimeline, resolved_items: Sequence[ResolvedLyricsItem]) -> ForcedAlignmentResult:
        try:
            payload = json.loads(self.alignment_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError as exc:
            raise ForcedPhoneAlignerError("missing_phone_evidence", f"missing forced alignment: {self.alignment_path}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"invalid forced alignment: {self.alignment_path}") from exc
        return parse_forced_phone_alignment(payload, vocal_stem=vocal_stem, timeline=timeline, resolved_items=resolved_items)


class ExternalCommandForcedPhoneAligner:
    """Invoke an explicitly configured forced aligner without shell expansion."""

    def __init__(self, command: Sequence[str], *, timeout_sec: float = 300.0) -> None:
        if isinstance(command, (str, bytes)) or not command or not str(command[0]).strip():
            raise ForcedPhoneAlignerError("aligner_configuration_error", "forced aligner command is empty")
        self.command = tuple(str(part) for part in command)
        if any(not part.strip() for part in self.command):
            raise ForcedPhoneAlignerError("aligner_configuration_error", "forced aligner command contains an empty argument")
        try:
            self.timeout_sec = float(timeout_sec)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ForcedPhoneAlignerError("aligner_configuration_error", "forced aligner timeout must be numeric") from exc
        if not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0.0:
            raise ForcedPhoneAlignerError("aligner_configuration_error", "forced aligner timeout must be positive and finite")

    def align(self, vocal_stem: Path, timeline: LyricsTimeline, resolved_items: Sequence[ResolvedLyricsItem]) -> ForcedAlignmentResult:
        if not vocal_stem.is_file():
            raise ForcedPhoneAlignerError("missing_vocal_stem_evidence", f"missing vocal stem: {vocal_stem}")
        request = {
            "schema_version": SCHEMA_VERSION,
            "vocal_stem": str(vocal_stem),
            "vocal_stem_sha256": sha256_file(vocal_stem),
            "lyrics_timeline": timeline.to_dict(),
            "lyrics_timeline_sha256": lyrics_timeline_sha256(timeline),
            "g2p_items": [item.to_dict() for item in resolved_items],
        }
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=True,
                timeout=self.timeout_sec,
                shell=False,
            )
            payload = json.loads(completed.stdout)
        except (OSError, UnicodeError, TypeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise ForcedPhoneAlignerError("forced_aligner_runtime_error", f"forced aligner command failed: {exc}") from exc
        return parse_forced_phone_alignment(payload, vocal_stem=vocal_stem, timeline=timeline, resolved_items=resolved_items)


def compare_forced_and_ctc_alignment(
    forced_segments: Sequence[PhoneSegment],
    ctc_segments: Sequence[PhoneSegment],
) -> dict[str, Any]:
    """Summarize auxiliary CTC coverage without modifying forced evidence.

    The forced alignment remains the product truth layer.  The additional
    difference fields are an audit aid only: they describe duration/coverage
    disagreement and the best-overlap phone disagreement, but never select or
    rewrite a forced segment.
    """
    forced_duration = sum(segment.time_range.end_sec - segment.time_range.start_sec for segment in forced_segments)
    ctc_duration = sum(segment.time_range.end_sec - segment.time_range.start_sec for segment in ctc_segments)
    intervals: list[tuple[float, float]] = []
    overlap_pair_count = 0
    phone_disagreement_count = 0
    for forced in forced_segments:
        best_overlap = 0.0
        best_ctc: PhoneSegment | None = None
        for ctc in ctc_segments:
            start = max(forced.time_range.start_sec, ctc.time_range.start_sec)
            end = min(forced.time_range.end_sec, ctc.time_range.end_sec)
            if end > start:
                intervals.append((start, end))
                overlap_pair_count += 1
                overlap = end - start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ctc = ctc
        if best_ctc is not None:
            forced_phone = forced.candidate_phones[0].symbol if forced.candidate_phones else forced.normalized_ipa
            ctc_phone = best_ctc.candidate_phones[0].symbol if best_ctc.candidate_phones else best_ctc.normalized_ipa
            if forced_phone != ctc_phone:
                phone_disagreement_count += 1
    intervals.sort()
    overlaps = 0.0
    current_start = current_end = None
    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            overlaps += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        overlaps += current_end - current_start
    forced_coverage_ratio = overlaps / forced_duration if forced_duration > 0.0 else 0.0
    ctc_coverage_ratio = overlaps / ctc_duration if ctc_duration > 0.0 else 0.0
    duration_difference = ctc_duration - forced_duration
    denominator = max(forced_duration, ctc_duration)
    return {
        "schema_version": "ctc_forced_alignment_comparison_v1",
        "forced_segment_count": len(forced_segments),
        "ctc_segment_count": len(ctc_segments),
        "forced_duration_sec": round(forced_duration, 6),
        "ctc_duration_sec": round(ctc_duration, 6),
        "overlap_sec": round(overlaps, 6),
        "forced_covered_by_ctc_ratio": round(min(1.0, forced_coverage_ratio), 6),
        "ctc_covered_by_forced_ratio": round(min(1.0, ctc_coverage_ratio), 6),
        "duration_difference_sec": round(duration_difference, 6),
        "duration_difference_ratio": round(abs(duration_difference) / denominator, 6) if denominator > 0.0 else 0.0,
        "segment_count_difference": len(ctc_segments) - len(forced_segments),
        "overlap_pair_count": overlap_pair_count,
        "phone_disagreement_count": phone_disagreement_count,
        "ctc_is_auxiliary_only": True,
    }


__all__ = [
    "HUMAN_REVIEW_PENDING_DETACHED_ACCEPTANCE_STATUS",
    "MACHINE_REVIEW_DRAFT_STATUS",
    "PENDING_HUMAN_PHONE_REVIEW_STATUS",
    "SCHEMA_VERSION",
    "ExternalCommandForcedPhoneAligner",
    "FixtureForcedPhoneAligner",
    "ForcedAlignmentResult",
    "ForcedPhoneAligner",
    "ForcedPhoneAlignerError",
    "compare_forced_and_ctc_alignment",
    "is_human_reviewed_alignment_payload",
    "is_machine_review_draft_payload",
    "lyrics_timeline_sha256",
    "parse_forced_phone_alignment",
]

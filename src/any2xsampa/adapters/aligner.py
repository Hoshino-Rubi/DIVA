"""Minimal score-span alignment contract recovered from pipeline evidence.

The recognizer's phone candidates remain the truth layer.  This adapter only
attaches note-context ownership evidence to each segment and never rewrites a
phone, IPA, or X-SAMPA value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from any2xsampa.core.interfaces import PhoneAligner
from any2xsampa.core.models import PhoneSegment, PipelineResult


class SegmentScorePhoneAligner(PhoneAligner):
    """Attach a monotonic score-span alignment record to phone segments.

    ``NoteContextApplier`` normally places one normalized span in
    ``segment.metadata["note_context"]``.  Producers that expose competing
    spans may additionally provide ``note_context_candidates``; those are
    ranked using overlap and identity evidence while preserving segment order.
    """

    def __init__(self, future_aligner: Any | None = None) -> None:
        self.future_aligner = future_aligner

    def align_segments(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        previous_rank = -1
        for segment in segments:
            candidates = _candidate_spans(segment)
            if not candidates:
                segment.metadata.setdefault("segment_score_alignment_applied", False)
                continue

            ranked = [
                (_score_candidate(segment, span, index), index, span)
                for index, span in enumerate(candidates)
            ]
            eligible = [item for item in ranked if item[1] >= previous_rank]
            score, rank, span = max(eligible or ranked, key=lambda item: (item[0][0], -item[1]))
            previous_rank = max(previous_rank, rank)
            segment.metadata["segment_score_alignment"] = _alignment_payload(
                segment,
                span,
                score,
                candidate_count=len(candidates),
                strategy=_strategy(span),
                fallback_used=False,
            )
            segment.metadata["segment_score_alignment_applied"] = True
        return segments

    def align(self, result: PipelineResult) -> PipelineResult:
        """Satisfy the shared aligner interface for callers using PipelineResult."""

        self.align_segments(result.segments)
        return result


def _candidate_spans(segment: PhoneSegment) -> list[Mapping[str, Any]]:
    metadata = segment.metadata
    if metadata.get("note_context_applied") is False:
        return []
    values = metadata.get("note_context_candidates")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        candidates = [
            value
            for value in values
            if isinstance(value, Mapping) and _has_span_bounds(value)
        ]
        if candidates:
            return candidates
    context = metadata.get("note_context")
    if isinstance(context, Mapping) and _has_span_bounds(context):
        return [context]
    if isinstance(context, Sequence) and not isinstance(context, (str, bytes)):
        return [
            value
            for value in context
            if isinstance(value, Mapping) and _has_span_bounds(value)
        ]
    hint = metadata.get("score_hint")
    if isinstance(hint, Mapping) and _has_span_bounds(hint):
        return [hint]
    return []


def _score_candidate(
    segment: PhoneSegment,
    span: Mapping[str, Any],
    rank: int,
) -> tuple[float, list[str], float]:
    segment_start = float(segment.time_range.start_sec)
    segment_end = float(segment.time_range.end_sec)
    span_start = _number(span.get("start_sec"), segment_start)
    span_end = max(span_start, _number(span.get("end_sec"), segment_end))
    segment_duration = max(0.0, segment_end - segment_start)
    overlap = max(0.0, min(segment_end, span_end) - max(segment_start, span_start))
    overlap_ratio = overlap / segment_duration if segment_duration else 0.0
    span_duration = max(0.0, span_end - span_start)
    coverage_ratio = overlap / span_duration if span_duration else 0.0
    center_score = _center_score(segment_start, segment_end, span_start, span_end)
    identity_bonus = 0.35 if bool(span.get("note_identity_match")) else 0.0
    if str(span.get("note_binding_strategy", "")).strip().lower() == "identity":
        identity_bonus = max(identity_bonus, 0.25)
    score = (0.62 * overlap_ratio) + (0.18 * coverage_ratio) + (0.12 * center_score) + identity_bonus
    score -= min(0.08, 0.01 * rank)
    terms = [
        f"overlap_ratio={overlap_ratio:.3f}",
        f"coverage_ratio={coverage_ratio:.3f}",
        f"center_score={center_score:.3f}",
        f"identity_bonus={identity_bonus:.3f}",
    ]
    return max(0.0, min(1.0, score)), terms, overlap_ratio


def _alignment_payload(
    segment: PhoneSegment,
    span: Mapping[str, Any],
    score_data: tuple[float, list[str], float],
    *,
    candidate_count: int,
    strategy: str,
    fallback_used: bool,
) -> dict[str, Any]:
    score, terms, overlap_ratio = score_data
    aligned_note_id = span.get("normalized_note_id") or span.get("note_id")
    return {
        "adapter": "SegmentScorePhoneAligner",
        "dp_method": "monotonic_segment_score",
        "source": "note_context",
        "strategy": strategy,
        "candidate_count": candidate_count,
        "local_score": round(score, 3),
        "overlap_ratio": round(overlap_ratio, 3),
        "score_terms": terms,
        "aligned_note_id": str(aligned_note_id) if aligned_note_id is not None else None,
        "segment_start_sec": round(float(segment.time_range.start_sec), 6),
        "segment_end_sec": round(float(segment.time_range.end_sec), 6),
        "fallback_used": fallback_used,
    }


def _strategy(span: Mapping[str, Any]) -> str:
    value = span.get("note_binding_strategy") or span.get("match_strategy") or "time_overlap"
    return str(value)


def _has_span_bounds(value: Mapping[str, Any]) -> bool:
    return "start_sec" in value or "end_sec" in value


def _number(value: object, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _center_score(segment_start: float, segment_end: float, span_start: float, span_end: float) -> float:
    segment_center = (segment_start + segment_end) / 2.0
    span_center = (span_start + span_end) / 2.0
    scale = max(segment_end - segment_start, span_end - span_start, 1e-6)
    return max(0.0, 1.0 - abs(segment_center - span_center) / scale)


__all__ = ["SegmentScorePhoneAligner"]

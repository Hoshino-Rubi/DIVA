from __future__ import annotations

from copy import deepcopy

from any2xsampa.core.models import PhoneSegment


def future_aligner_sequence_hint_payload(segment: PhoneSegment) -> dict[str, object]:
    payload = segment.metadata.get("future_aligner_sequence_hint")
    return deepcopy(payload) if isinstance(payload, dict) else {}


def resolve_future_aligner_sequence_hint_identity(
    segment: PhoneSegment,
    consumer: str = "unknown",
) -> dict[str, str | None]:
    payload = future_aligner_sequence_hint_payload(segment)
    identity = {
        "track_id": _text(payload.get("aligned_track_id")) or _text(segment.metadata.get("aligned_track_id")),
        "part_id": _text(payload.get("aligned_part_id")) or _text(segment.metadata.get("aligned_part_id")),
        "note_id": _text(payload.get("aligned_note_id")) or _text(segment.metadata.get("aligned_note_id")),
    }
    consumers = segment.metadata.get("future_aligner_sequence_hint_consumers")
    if not isinstance(consumers, list):
        consumers = []
    if consumer not in consumers:
        consumers.append(consumer)
    segment.metadata["future_aligner_sequence_hint_consumers"] = consumers
    segment.metadata["future_aligner_sequence_hint_consumed"] = bool(payload)
    return identity


class FutureAlignerSequenceHintBuilder:
    """Project aligner output into the stable downstream grouping/export view."""

    def apply(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        for segment in segments:
            self._apply_segment(segment)
        return segments

    def _apply_segment(self, segment: PhoneSegment) -> None:
        future = segment.metadata.get("future_aligner")
        if not isinstance(future, dict):
            segment.metadata.setdefault("future_aligner_sequence_hint_applied", False)
            segment.metadata.setdefault("future_aligner_sequence_hint", {})
            return
        hint = {
            key: deepcopy(future.get(key))
            for key in (
                "aligned_track_id",
                "aligned_part_id",
                "aligned_note_id",
                "aligner_strategy",
                "aligner_confidence",
                "aligner_fallback_used",
                "aligner_notes",
                "future_aligner_window_size",
                "future_aligner_sequence_applied",
                "future_aligner_sequence_score",
                "future_aligner_order_penalty",
                "future_aligner_continuity_used",
                "continuity_evidence",
                "monotonic_evidence",
                "part_boundary_evidence",
                "neighbor_support_count",
                "left_neighbor_support",
                "right_neighbor_support",
                "same_note_support",
                "order_support",
                "part_continuity_support",
                "evidence_items",
                "future_aligner_provenance_applied",
                "future_aligner_provenance_summary",
                "future_aligner_provenance_items",
                "acoustic_evidence_applied",
                "acoustic_evidence_summary",
                "boundary_uncertainty",
                "part_aware_note_span_used",
                "note_span_evidence_score",
                "note_span_overlap_ratio",
                "note_span_interior_ratio",
                "note_span_anchor_count",
                "note_span_summary",
                "ownership_conflict_gap",
                "ownership_conflict_threshold",
                "ownership_stability_margin",
                "ownership_conflict_summary",
                "ownership_conflict_candidates",
                "ownership_resolution_basis",
                "boundary_conflict_likely",
                "aligner_v2_applied",
                "aligner_v2_strategy",
                "aligner_v2_confidence",
                "aligner_v2_fallback_used",
                "aligner_v2_notes",
                "aligner_v2_version",
                "alignment_competition_summary",
                "note_ownership_conflict",
                "selective_release_safety",
                "release_explanation",
                "release_risk_summary",
                "source_score_hint_note_id",
                "source_score_hint_note_identity_match",
                "source_score_hint_binding_strategy",
                "effective_score_hint_note_id",
                "effective_score_hint_note_identity_match",
                "effective_score_hint_binding_strategy",
                "ownership_binding_adjustment",
                "phrase_boundary_candidate_debug",
                "phrase_boundary_blocker_summary",
            )
            if key in future
        }
        segment.metadata["future_aligner_sequence_hint"] = hint
        segment.metadata["future_aligner_sequence_hint_applied"] = True
        segment.metadata["future_aligner_sequence_hint_summary"] = (
            "aligned_note_id=" + str(hint.get("aligned_note_id") or "")
            + ";sequence_applied=" + ("yes" if hint.get("future_aligner_sequence_applied") else "no")
        )


def _text(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


__all__ = [
    "FutureAlignerSequenceHintBuilder",
    "future_aligner_sequence_hint_payload",
    "resolve_future_aligner_sequence_hint_identity",
]


from __future__ import annotations

from any2xsampa.core.models import PhoneSegment
from any2xsampa.pipeline.future_aligner_models import (
    FutureAlignerAcousticEvidence,
    FutureAlignerHint,
)
from any2xsampa.pipeline.score_hint import future_aligner_hint_payload


def text_value(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def float_value(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_value(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def note_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    notes: list[str] = []
    for item in value:
        text = text_value(item)
        if text is not None:
            notes.append(text)
    return notes


def _has_value(value: object) -> bool:
    return value not in (None, "")


def _preferred_value(primary: object, fallback: object) -> object:
    return primary if _has_value(primary) else fallback


def _mapping_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): value[key] for key in value}


def _merged_mapping(primary: object, fallback: object) -> dict[str, object]:
    merged = _mapping_value(fallback)
    merged.update(_mapping_value(primary))
    return merged


def future_aligner_text(value: object) -> str | None:
    return text_value(value)


def future_aligner_float(value: object) -> float | None:
    return float_value(value)


def future_aligner_int(value: object) -> int | None:
    return int_value(value)


def future_aligner_note_list(value: object) -> list[str]:
    return note_list(value)


def build_future_aligner_hint(segment: PhoneSegment) -> FutureAlignerHint | None:
    payload = future_aligner_hint_payload(segment)
    if not payload:
        return None
    return FutureAlignerHint(
        normalized_track_id=text_value(payload.get("normalized_track_id")),
        normalized_part_id=text_value(payload.get("normalized_part_id")),
        normalized_note_id=text_value(payload.get("normalized_note_id")),
        part_local_note_rank=int_value(payload.get("part_local_note_rank")),
        duration_ms=float_value(payload.get("duration_ms")),
        note_start_sec=float_value(payload.get("start_sec")),
        note_end_sec=float_value(payload.get("end_sec")),
        release_context=text_value(payload.get("release_context")),
        note_binding_strategy=text_value(payload.get("note_binding_strategy")),
        note_identity_match=bool(payload.get("note_identity_match", False)),
        score_bridge_strategy=text_value(payload.get("score_bridge_strategy")),
        score_hint_confidence=float_value(payload.get("score_hint_confidence")) or 0.0,
        score_hint_notes=note_list(payload.get("score_hint_notes")),
        source_normalized_note_id=text_value(payload.get("normalized_note_id")),
        source_note_identity_match=bool(payload.get("note_identity_match", False)),
        source_note_binding_strategy=text_value(payload.get("note_binding_strategy")),
    )


def future_aligner_hint_from_segment(segment: PhoneSegment) -> FutureAlignerHint | None:
    return build_future_aligner_hint(segment)


def build_future_aligner_acoustic_evidence(
    segment: PhoneSegment,
) -> FutureAlignerAcousticEvidence | None:
    payload = segment.metadata.get("acoustic_evidence")
    acoustic = _mapping_value(payload)
    if not acoustic and not bool(segment.metadata.get("acoustic_evidence_applied", False)):
        return None
    note_alignment_map = _merged_mapping(
        acoustic.get("note_alignment"),
        segment.metadata.get("note_alignment_friendly"),
    )
    ownership_map = _merged_mapping(
        acoustic.get("ownership"),
        segment.metadata.get("ownership_signal"),
    )
    boundary_map = _merged_mapping(
        acoustic.get("boundary_uncertainty"),
        segment.metadata.get("boundary_uncertainty"),
    )
    candidate_distribution_map = _merged_mapping(
        acoustic.get("candidate_distribution"),
        segment.metadata.get("candidate_distribution"),
    )
    continuation_map = _merged_mapping(
        acoustic.get("continuation_profile"),
        segment.metadata.get("continuation_profile"),
    )
    boundary_profile_map = _merged_mapping(
        acoustic.get("boundary_profile"),
        segment.metadata.get("boundary_profile"),
    )
    adapter_map = _merged_mapping(
        acoustic.get("adapter_diagnostics"),
        segment.metadata.get("adapter_diagnostics"),
    )
    posterior_summary_map = _merged_mapping(
        acoustic.get("posterior_summary"),
        segment.metadata.get("posterior_summary"),
    )
    logit_summary_map = _merged_mapping(
        acoustic.get("logit_summary"),
        segment.metadata.get("logit_summary"),
    )
    posterior_like_available = bool(
        adapter_map.get("posterior_like_available")
        if "posterior_like_available" in adapter_map
        else posterior_summary_map
        or candidate_distribution_map.get("posterior_mean_margin") is not None
        or candidate_distribution_map.get("frame_vote_ratio") is not None
    )
    logit_like_available = bool(
        adapter_map.get("logit_like_available")
        if "logit_like_available" in adapter_map
        else logit_summary_map
        or candidate_distribution_map.get("logit_mean_margin") is not None
    )
    return FutureAlignerAcousticEvidence(
        acoustic_confidence=float_value(
            _preferred_value(acoustic.get("acoustic_confidence"), segment.metadata.get("acoustic_confidence"))
        )
        or 0.0,
        timing_reliability=float_value(
            _preferred_value(acoustic.get("timing_reliability"), segment.metadata.get("timing_reliability"))
        )
        or 0.0,
        boundary_uncertain=bool(boundary_map.get("uncertain", False)),
        boundary_uncertainty={str(key): boundary_map[key] for key in boundary_map},
        candidate_count=int_value(
            _preferred_value(acoustic.get("candidate_count"), segment.metadata.get("candidate_count"))
        )
        or 0,
        candidate_entropy=float_value(
            _preferred_value(acoustic.get("candidate_entropy"), segment.metadata.get("candidate_entropy"))
        )
        or 0.0,
        candidate_lead=(
            float_value(candidate_distribution_map.get("candidate_lead"))
            or float_value(ownership_map.get("candidate_lead"))
            or 0.0
        ),
        posterior_vote_ratio=(
            float_value(candidate_distribution_map.get("frame_vote_ratio"))
            or float_value(posterior_summary_map.get("frame_vote_ratio"))
            or 0.0
        ),
        posterior_temporal_consistency=(
            float_value(candidate_distribution_map.get("temporal_consistency"))
            or float_value(posterior_summary_map.get("temporal_consistency"))
            or 0.0
        ),
        posterior_mean_margin=(
            float_value(candidate_distribution_map.get("posterior_mean_margin"))
            or float_value(posterior_summary_map.get("mean_margin"))
            or 0.0
        ),
        logit_mean_margin=(
            float_value(candidate_distribution_map.get("logit_mean_margin"))
            or float_value(logit_summary_map.get("mean_margin"))
            or 0.0
        ),
        competition_separation=(
            float_value(candidate_distribution_map.get("competition_separation"))
            or float_value(note_alignment_map.get("competition_separation"))
            or float_value(ownership_map.get("competition_separation"))
            or 0.0
        ),
        interior_hold_strength=(
            float_value(continuation_map.get("interior_hold_strength"))
            or float_value(note_alignment_map.get("interior_hold_strength"))
            or float_value(ownership_map.get("interior_hold_strength"))
            or 0.0
        ),
        one_sided_continuation_strength=(
            float_value(continuation_map.get("one_sided_continuation_strength"))
            or float_value(note_alignment_map.get("one_sided_continuation_strength"))
            or float_value(ownership_map.get("one_sided_continuation_strength"))
            or 0.0
        ),
        boundary_pressure=(
            float_value(boundary_profile_map.get("boundary_pressure"))
            or float_value(ownership_map.get("boundary_pressure"))
            or 0.0
        ),
        dominant_candidate_stability=float_value(
            _preferred_value(
                acoustic.get("dominant_candidate_stability"),
                segment.metadata.get("dominant_candidate_stability"),
            )
        )
        or 0.0,
        boundary_softness=float_value(
            _preferred_value(acoustic.get("boundary_softness"), segment.metadata.get("boundary_softness"))
        )
        or 0.0,
        same_note_continuation_bias=float_value(note_alignment_map.get("same_note_continuation_bias")) or 0.0,
        same_note_stability=(
            float_value(ownership_map.get("same_note_stability"))
            or float_value(note_alignment_map.get("same_note_stability"))
            or 0.0
        ),
        continuation_friendliness=(
            float_value(continuation_map.get("continuation_friendliness"))
            or float_value(note_alignment_map.get("continuation_friendliness"))
            or 0.0
        ),
        score_alignment_friendly=float_value(note_alignment_map.get("score_alignment_friendly")) or 0.0,
        weak_overlap_support=float_value(note_alignment_map.get("weak_overlap_support")) or 0.0,
        weak_overlap_recovery=(
            float_value(ownership_map.get("weak_overlap_recovery"))
            or float_value(note_alignment_map.get("weak_overlap_recovery"))
            or 0.0
        ),
        weak_overlap_friendliness=(
            float_value(continuation_map.get("weak_overlap_friendliness"))
            or float_value(note_alignment_map.get("weak_overlap_friendliness"))
            or 0.0
        ),
        boundary_anchor_strength=float_value(note_alignment_map.get("boundary_anchor_strength")) or 0.0,
        supports_note_span_extension=bool(note_alignment_map.get("supports_note_span_extension", False)),
        long_note_hold_bias=float_value(note_alignment_map.get("long_note_hold_bias")) or 0.0,
        part_note_span_reliability=(
            float_value(ownership_map.get("part_note_span_reliability"))
            or float_value(note_alignment_map.get("part_note_span_reliability"))
            or 0.0
        ),
        reassignment_resistance=float_value(ownership_map.get("reassignment_resistance")) or 0.0,
        boundary_reassignment_risk=float_value(ownership_map.get("boundary_reassignment_risk")) or 0.0,
        boundary_conflict_likelihood=(
            float_value(boundary_profile_map.get("boundary_conflict_likelihood"))
            or float_value(ownership_map.get("boundary_conflict_likelihood"))
            or 0.0
        ),
        ownership_conflict_risk=float_value(ownership_map.get("ownership_conflict_risk")) or 0.0,
        frame_consistency=float_value(
            _preferred_value(acoustic.get("frame_consistency"), segment.metadata.get("frame_consistency"))
        )
        or 0.0,
        evidence_quality=float_value(
            _preferred_value(acoustic.get("evidence_quality"), segment.metadata.get("evidence_quality"))
        )
        or 0.0,
        segment_likelihood=float_value(
            _preferred_value(acoustic.get("segment_likelihood"), segment.metadata.get("segment_likelihood"))
        )
        or 0.0,
        boundary_transition_strength=float_value(
            _preferred_value(
                acoustic.get("boundary_transition_strength"),
                segment.metadata.get("boundary_transition_strength"),
            )
        )
        or 0.0,
        ownership_ambiguity=float_value(
            _preferred_value(acoustic.get("ownership_ambiguity"), segment.metadata.get("ownership_ambiguity"))
        )
        or 0.0,
        backend_kind=text_value(adapter_map.get("backend_kind")) or "",
        adapter_kind=text_value(adapter_map.get("adapter_kind")) or "",
        evidence_kind=text_value(adapter_map.get("evidence_kind")) or "",
        aggregation=text_value(adapter_map.get("aggregation")) or "",
        posterior_like_available=posterior_like_available,
        logit_like_available=logit_like_available,
        approximation_only=bool(adapter_map.get("approximation_only", False)),
        summary=text_value(segment.metadata.get("acoustic_evidence_summary")) or "",
        notes=note_list(acoustic.get("notes")),
    )


def future_aligner_acoustic_evidence_from_segment(
    segment: PhoneSegment,
) -> FutureAlignerAcousticEvidence | None:
    return build_future_aligner_acoustic_evidence(segment)


def rebind_low_confidence_phrase_boundary_hint(
    segments: list[PhoneSegment],
    index: int,
    hints: list[FutureAlignerHint | None],
    current: FutureAlignerHint,
) -> bool:
    if current.normalized_note_id is None:
        return False
    if current.note_identity_match:
        return False
    if current.note_binding_strategy not in {"time_overlap", "unmatched"}:
        return False
    if current.score_hint_confidence > 0.35:
        return False
    if index <= 0 or index >= len(hints) - 1:
        return False

    left = hints[index - 1]
    right = hints[index + 1]
    if left is None or right is None:
        return False
    if left.normalized_note_id is None or right.normalized_note_id is None:
        return False
    if left.normalized_note_id != right.normalized_note_id:
        return False
    if left.normalized_note_id == current.normalized_note_id:
        return False
    if left.normalized_track_id != current.normalized_track_id or right.normalized_track_id != current.normalized_track_id:
        return False
    if left.normalized_part_id != current.normalized_part_id or right.normalized_part_id != current.normalized_part_id:
        return False
    if not left.note_identity_match or not right.note_identity_match:
        return False
    if left.note_binding_strategy not in {"identity", "identity_partial", "segment_index"}:
        return False
    if right.note_binding_strategy not in {"identity", "identity_partial", "segment_index"}:
        return False

    effective_start_sec = min(
        float(segments[index - 1].time_range.start_sec),
        float(segments[index].time_range.start_sec),
        float(segments[index + 1].time_range.start_sec),
    )
    effective_end_sec = max(
        float(segments[index - 1].time_range.end_sec),
        float(segments[index].time_range.end_sec),
        float(segments[index + 1].time_range.end_sec),
    )

    current.normalized_note_id = left.normalized_note_id
    current.note_binding_strategy = "local_sequence_rebind"
    current.ownership_binding_adjusted = True
    current.ownership_binding_reason = "flanking_same_note_identity_anchors"
    current.note_start_sec = effective_start_sec
    current.note_end_sec = effective_end_sec
    if current.score_bridge_strategy in {None, "", "time_overlap_fallback"}:
        current.score_bridge_strategy = "local_sequence_rebind"
    if "future aligner re-bound low-confidence current note to flanking same-note anchors" not in current.score_hint_notes:
        current.score_hint_notes.append(
            "future aligner re-bound low-confidence current note to flanking same-note anchors"
        )
    if "future aligner widened low-confidence bridge-like slice to the local flanking anchor window" not in current.score_hint_notes:
        current.score_hint_notes.append(
            "future aligner widened low-confidence bridge-like slice to the local flanking anchor window"
        )
    return True


def maybe_rebind_low_confidence_phrase_boundary_hint(
    segments: list[PhoneSegment],
    index: int,
    hints: list[FutureAlignerHint | None],
    current: FutureAlignerHint,
) -> bool:
    return rebind_low_confidence_phrase_boundary_hint(segments, index, hints, current)


def normalize_hint_ownership_bindings(
    segments: list[PhoneSegment],
    hints: list[FutureAlignerHint | None],
) -> None:
    for index, hint in enumerate(hints):
        if hint is None:
            continue
        rebind_low_confidence_phrase_boundary_hint(segments, index, hints, hint)

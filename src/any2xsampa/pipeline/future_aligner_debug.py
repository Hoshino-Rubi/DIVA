from __future__ import annotations

from copy import deepcopy

from any2xsampa.core.models import PhoneSegment
from any2xsampa.pipeline.future_aligner_models import (
    FutureAlignerEvidenceItem,
    FutureAlignerHint,
    FutureAlignerResult,
)


def normalize_phrase_boundary_candidate_debug(
    payload: dict[str, object] | None,
    *,
    default_key_blocker: str,
    segment: PhoneSegment | None = None,
    current: FutureAlignerHint | None = None,
    boundary_regime: dict[str, object] | None = None,
) -> dict[str, object]:
    """Normalize the diagnostic view while retaining forward-compatible fields.

    The aligner stores this object in ``PhoneSegment.metadata``.  Normalization is
    deliberately additive: unknown keys are preserved so a newer producer can be
    consumed by an older reporter.
    """
    result = deepcopy(payload) if isinstance(payload, dict) else {}
    result.setdefault("mode", "none")
    result.setdefault("key_blocker", default_key_blocker)
    result.setdefault("current_note_id", current.normalized_note_id if current else None)
    result.setdefault(
        "winner_is_current_note",
        bool(
            current is not None
            and result.get("top_note_id") is not None
            and current.normalized_note_id == result.get("top_note_id")
        ),
    )
    result.setdefault("winner_matches_current_note", result.get("winner_is_current_note", False))

    if boundary_regime is not None:
        existing_boundary = result.get("boundary_regime")
        if isinstance(existing_boundary, dict):
            merged_boundary = deepcopy(boundary_regime)
            merged_boundary.update(existing_boundary)
            result["boundary_regime"] = merged_boundary
        else:
            result["boundary_regime"] = deepcopy(boundary_regime)
    else:
        result.setdefault("boundary_regime", {})

    winner_state = result.get("winner_state")
    if not isinstance(winner_state, dict):
        winner_state = {}
    winner_state.setdefault("winner_note_id", result.get("top_note_id"))
    winner_state.setdefault("runner_up_note_id", result.get("runner_up_note_id"))
    winner_state.setdefault("winner_matches_current_note", result.get("winner_is_current_note", False))
    winner_state.setdefault("winner_is_current_note", result.get("winner_is_current_note", False))
    winner_state.setdefault("current_note_candidate_rank", None)
    result["winner_state"] = winner_state

    candidate_shapes = result.get("candidate_shapes")
    if not isinstance(candidate_shapes, dict):
        candidate_shapes = {}
    candidate_shapes.setdefault("top_anchor_profile", "unavailable")
    candidate_shapes.setdefault("runner_up_anchor_profile", "unavailable")
    candidate_shapes.setdefault("asymmetric_chain_anchor", False)
    candidate_shapes.setdefault("opposite_edge_neighbors", False)
    candidate_shapes.setdefault("asymmetric_chain_anchor_primary_blocker", "none")
    candidate_shapes.setdefault("asymmetric_chain_anchor_blocking_reasons", [])
    result["candidate_shapes"] = candidate_shapes

    ranking = result.get("candidate_ranking")
    if not isinstance(ranking, dict):
        ranking = {}
    ranking.setdefault("current_note_candidate_rank", winner_state.get("current_note_candidate_rank"))
    ranking.setdefault("top", None)
    ranking.setdefault("runner_up", None)
    ranking.setdefault("ordered_candidates", [])
    result["candidate_ranking"] = ranking

    missing = result.get("missing_preconditions")
    if not isinstance(missing, dict):
        missing = {}
    missing.setdefault("general", [default_key_blocker])
    for category in ("asymmetric_interior", "asymmetric_dual_touch", "asymmetric_chain_anchor"):
        missing.setdefault(category, [])
    result["missing_preconditions"] = missing

    blockers = result.get("blocker_categories")
    if not isinstance(blockers, dict):
        blockers = {}
    for category in ("data", "evidence", "candidate_classification"):
        values = blockers.get(category)
        blockers[category] = list(values) if isinstance(values, list) else []
    result["blocker_categories"] = blockers

    result.setdefault("candidate_precondition_flags", {})
    result.setdefault("overlap_metrics", {})
    result.setdefault("score_hint_context", {})
    result.setdefault("segment_slicing", {})
    result.setdefault("ownership_binding_adjustment", {})
    return result


def phrase_boundary_candidate_debug_payload(
    payload: dict[str, object] | None,
    *,
    default_key_blocker: str = "candidate_preconditions_missing",
    segment: PhoneSegment | None = None,
    current: FutureAlignerHint | None = None,
    boundary_regime: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compatibility alias used by the aligner/reporting layer."""
    return normalize_phrase_boundary_candidate_debug(
        payload,
        default_key_blocker=default_key_blocker,
        segment=segment,
        current=current,
        boundary_regime=boundary_regime,
    )


def build_phrase_boundary_candidate_debug(**kwargs: object) -> dict[str, object]:
    """Compatibility re-export retained by the historical debug module."""
    from any2xsampa.pipeline.future_aligner_phrase_boundary import (
        build_phrase_boundary_candidate_debug as _build,
    )

    return _build(**kwargs)


def phrase_boundary_blocker_summary(
    debug: dict[str, object],
    *,
    segment_start_sec: float,
    segment_end_sec: float,
) -> str:
    if not debug:
        return ""
    blocker_categories = debug.get("blocker_categories")
    blocker_bits: list[str] = []
    if isinstance(blocker_categories, dict):
        for category in ("data", "evidence", "candidate_classification"):
            values = blocker_categories.get(category)
            if isinstance(values, list) and values:
                blocker_bits.append(f"{category}={'+'.join(str(value) for value in values)}")
    boundary_regime = debug.get("boundary_regime")
    boundary_bits: list[str] = []
    if isinstance(boundary_regime, dict):
        boundary_bits.append(f"ready={'yes' if boundary_regime.get('ready') else 'no'}")
        boundary_bits.append(f"uncertain={'yes' if boundary_regime.get('boundary_uncertain') else 'no'}")
        if "effective_boundary_uncertain" in boundary_regime:
            boundary_bits.append(
                f"effective_uncertain={'yes' if boundary_regime.get('effective_boundary_uncertain') else 'no'}"
            )
        boundary_bits.append(f"pressure={float(boundary_regime.get('boundary_pressure', 0.0)):.3f}")
        boundary_bits.append(
            f"transition={float(boundary_regime.get('boundary_transition_strength', 0.0)):.3f}"
        )
        boundary_bits.append(
            f"conflict={float(boundary_regime.get('boundary_conflict_likelihood', 0.0)):.3f}"
        )
        boundary_primary = boundary_regime.get("primary_blocker")
        uncertainty = boundary_regime.get("uncertainty_normalization")
        if isinstance(uncertainty, dict) and uncertainty.get("applied"):
            boundary_bits.append("uncertainty_normalized=yes")
        if boundary_primary and boundary_primary != "none":
            boundary_bits.append(f"primary={boundary_primary}")
    winner_state = debug.get("winner_state")
    current_rank = winner_state.get("current_note_candidate_rank") if isinstance(winner_state, dict) else None
    score_hint_context = debug.get("score_hint_context")
    hint_bits: list[str] = []
    if isinstance(score_hint_context, dict):
        note_id = score_hint_context.get("note_id")
        if note_id:
            hint_bits.append(f"note_id={note_id}")
        if "score_hint_confidence" in score_hint_context:
            hint_bits.append(f"hint_conf={float(score_hint_context.get('score_hint_confidence', 0.0)):.3f}")
        strategy = score_hint_context.get("note_binding_strategy")
        if strategy:
            hint_bits.append(f"binding={strategy}")
    candidate_shapes = debug.get("candidate_shapes")
    shape_bits: list[str] = []
    if isinstance(candidate_shapes, dict):
        if candidate_shapes.get("opposite_edge_neighbors"):
            shape_bits.append("opposite_edge_neighbors=yes")
        if candidate_shapes.get("asymmetric_chain_anchor"):
            shape_bits.append("asymmetric_chain_anchor=yes")
        primary_blocker = candidate_shapes.get("asymmetric_chain_anchor_primary_blocker")
        if primary_blocker and primary_blocker != "none":
            shape_bits.append(f"chain_anchor_primary={primary_blocker}")
    overlap_metrics = debug.get("overlap_metrics")
    overlap_bits: list[str] = []
    if isinstance(overlap_metrics, dict):
        if overlap_metrics.get("overlap_advantage") is not None:
            overlap_bits.append(f"overlap_advantage={float(overlap_metrics['overlap_advantage']):.3f}")
        if overlap_metrics.get("interior_advantage") is not None:
            overlap_bits.append(f"interior_advantage={float(overlap_metrics['interior_advantage']):.3f}")
    return " | ".join(
        [
            f"segment={segment_start_sec:.3f}-{segment_end_sec:.3f}",
            f"mode={debug.get('mode')}",
            f"key_blocker={debug.get('key_blocker')}",
            f"current_rank={current_rank if current_rank is not None else 'na'}",
            f"winner_is_current_note={'yes' if debug.get('winner_is_current_note') else 'no'}",
            "blockers=" + (";".join(blocker_bits) if blocker_bits else "-"),
            "boundary=" + (";".join(boundary_bits) if boundary_bits else "-"),
            "hint=" + (";".join(hint_bits) if hint_bits else "-"),
            "shape=" + (";".join(shape_bits) if shape_bits else "-"),
            "overlap=" + (";".join(overlap_bits) if overlap_bits else "-"),
        ]
    )


def annotate_future_aligner_metadata(segment: PhoneSegment, result: FutureAlignerResult) -> None:
    payload = result.to_metadata()
    segment.metadata["future_aligner"] = payload
    segment.metadata["future_aligner_applied"] = True
    segment.metadata["future_aligner_strategy"] = result.aligner_strategy
    segment.metadata["future_aligner_confidence"] = payload["aligner_confidence"]
    segment.metadata["future_aligner_fallback_used"] = result.aligner_fallback_used
    segment.metadata["future_aligner_notes"] = list(result.aligner_notes)
    segment.metadata["future_aligner_window_size"] = result.future_aligner_window_size
    segment.metadata["future_aligner_sequence_applied"] = result.future_aligner_sequence_applied
    segment.metadata["future_aligner_sequence_score"] = payload["future_aligner_sequence_score"]
    segment.metadata["future_aligner_order_penalty"] = payload["future_aligner_order_penalty"]
    segment.metadata["future_aligner_continuity_used"] = result.future_aligner_continuity_used
    segment.metadata["future_aligner_continuity_evidence"] = list(result.continuity_evidence)
    segment.metadata["future_aligner_monotonic_evidence"] = list(result.monotonic_evidence)
    segment.metadata["future_aligner_part_boundary_evidence"] = list(result.part_boundary_evidence)
    segment.metadata["future_aligner_neighbor_support_count"] = result.neighbor_support_count
    segment.metadata["future_aligner_left_neighbor_support"] = result.left_neighbor_support
    segment.metadata["future_aligner_right_neighbor_support"] = result.right_neighbor_support
    segment.metadata["future_aligner_same_note_support"] = result.same_note_support
    segment.metadata["future_aligner_order_support"] = result.order_support
    segment.metadata["future_aligner_part_continuity_support"] = result.part_continuity_support
    segment.metadata["future_aligner_evidence_items"] = [item.to_metadata() for item in result.evidence_items]
    segment.metadata["future_aligner_provenance_applied"] = result.future_aligner_provenance_applied
    segment.metadata["future_aligner_provenance_summary"] = result.future_aligner_provenance_summary
    segment.metadata["future_aligner_provenance_items"] = [item.to_metadata() for item in result.evidence_items]
    segment.metadata["acoustic_evidence_applied"] = payload["acoustic_evidence_applied"]
    segment.metadata["acoustic_evidence_summary"] = payload["acoustic_evidence_summary"]
    segment.metadata["boundary_uncertainty"] = dict(payload["boundary_uncertainty"])
    segment.metadata["part_aware_note_span_used"] = payload["part_aware_note_span_used"]
    segment.metadata["note_span_evidence_score"] = payload["note_span_evidence_score"]
    segment.metadata["note_span_overlap_ratio"] = payload["note_span_overlap_ratio"]
    segment.metadata["note_span_interior_ratio"] = payload["note_span_interior_ratio"]
    segment.metadata["note_span_anchor_count"] = payload["note_span_anchor_count"]
    segment.metadata["note_span_summary"] = payload["note_span_summary"]
    segment.metadata["ownership_conflict_gap"] = payload["ownership_conflict_gap"]
    segment.metadata["ownership_conflict_summary"] = payload["ownership_conflict_summary"]
    segment.metadata["ownership_conflict_candidates"] = [dict(item) for item in payload["ownership_conflict_candidates"]]
    segment.metadata["ownership_resolution_basis"] = payload["ownership_resolution_basis"]
    segment.metadata["boundary_conflict_likely"] = payload["boundary_conflict_likely"]
    segment.metadata["aligner_v2_applied"] = payload["aligner_v2_applied"]
    segment.metadata["aligner_v2_strategy"] = payload["aligner_v2_strategy"]
    segment.metadata["aligner_v2_confidence"] = payload["aligner_v2_confidence"]
    segment.metadata["aligner_v2_fallback_used"] = payload["aligner_v2_fallback_used"]
    segment.metadata["aligner_v2_notes"] = list(payload["aligner_v2_notes"])
    segment.metadata["aligner_v2_version"] = payload["aligner_v2_version"]
    segment.metadata["alignment_competition_summary"] = payload["alignment_competition_summary"]
    segment.metadata["note_ownership_conflict"] = payload["note_ownership_conflict"]
    from any2xsampa.pipeline.future_aligner_reporting import future_aligner_report_payload

    segment.metadata["future_aligner_report"] = future_aligner_report_payload(segment)
    segment.metadata["aligned_track_id"] = result.aligned_track_id
    segment.metadata["aligned_part_id"] = result.aligned_part_id
    segment.metadata["aligned_note_id"] = result.aligned_note_id


def provenance_summary(items: list[FutureAlignerEvidenceItem]) -> str:
    accepted = sum(1 for item in items if item.accepted)
    rejected = sum(1 for item in items if not item.accepted)
    left = sum(1 for item in items if item.accepted and item.support_direction == "left")
    right = sum(1 for item in items if item.accepted and item.support_direction == "right")
    types: list[str] = []
    for item in items:
        if item.evidence_type not in types:
            types.append(item.evidence_type)
    return (
        f"accepted={accepted},rejected={rejected},left={left},right={right},"
        f"types={'+'.join(types) if types else 'none'}"
    )

from __future__ import annotations

from any2xsampa.core.models import PhoneSegment
from any2xsampa.pipeline.future_aligner_debug import normalize_phrase_boundary_candidate_debug
from any2xsampa.pipeline.future_aligner_models import (
    FutureAlignerAcousticEvidence,
    FutureAlignerCandidateScore,
    FutureAlignerHint,
)


def _anchor_profile(*, left: bool, right: bool, same_note_neighbors: int) -> str:
    if left and right:
        return "bidirectional"
    if left:
        return "left_edge"
    if right:
        return "right_edge"
    if same_note_neighbors > 0:
        return "same_note_window_only"
    return "none"


def _segment_note_duration_metrics(
    *,
    segment: PhoneSegment,
    current: FutureAlignerHint | None,
) -> tuple[float | None, float, float]:
    note_duration_sec = None
    if current is not None and current.note_start_sec is not None and current.note_end_sec is not None:
        note_duration_sec = max(0.0, current.note_end_sec - current.note_start_sec)
    segment_duration_sec = max(0.0, float(segment.time_range.end_sec) - float(segment.time_range.start_sec))
    segment_to_note_duration_ratio = 0.0
    if note_duration_sec is not None and note_duration_sec > 0.0:
        segment_to_note_duration_ratio = segment_duration_sec / note_duration_sec
    return note_duration_sec, segment_duration_sec, segment_to_note_duration_ratio


def _chain_anchor_shape(top: FutureAlignerCandidateScore, runner_up: FutureAlignerCandidateScore) -> dict[str, object]:
    opposite_edge_neighbors = bool(
        top.same_note_neighbors == 1
        and runner_up.same_note_neighbors == 1
        and top.left_neighbor_match != runner_up.left_neighbor_match
        and top.right_neighbor_match != runner_up.right_neighbor_match
    )
    top_anchor_profile = _anchor_profile(
        left=top.left_neighbor_match,
        right=top.right_neighbor_match,
        same_note_neighbors=top.same_note_neighbors,
    )
    runner_up_anchor_profile = _anchor_profile(
        left=runner_up.left_neighbor_match,
        right=runner_up.right_neighbor_match,
        same_note_neighbors=runner_up.same_note_neighbors,
    )
    chain_anchor_shape_reasons: list[str] = []
    if top.same_note_neighbors < 2:
        chain_anchor_shape_reasons.append("top_same_note_neighbors_lt2")
    if runner_up.same_note_neighbors != 1:
        chain_anchor_shape_reasons.append("runner_up_same_note_neighbors_ne_1")
    if top.left_neighbor_match == top.right_neighbor_match:
        chain_anchor_shape_reasons.append("top_anchor_not_single_edge")
    if runner_up.left_neighbor_match == runner_up.right_neighbor_match:
        chain_anchor_shape_reasons.append("runner_up_not_single_edge_touch")
    if (
        top.left_neighbor_match == runner_up.left_neighbor_match
        and top.right_neighbor_match == runner_up.right_neighbor_match
    ):
        chain_anchor_shape_reasons.append("anchor_edge_pattern_not_separated")
    return {
        "opposite_edge_neighbors": opposite_edge_neighbors,
        "top_anchor_profile": top_anchor_profile,
        "runner_up_anchor_profile": runner_up_anchor_profile,
        "asymmetric_chain_anchor": not chain_anchor_shape_reasons,
        "asymmetric_chain_anchor_primary_blocker": (
            chain_anchor_shape_reasons[0] if chain_anchor_shape_reasons else "none"
        ),
        "asymmetric_chain_anchor_blocking_reasons": chain_anchor_shape_reasons,
    }


def _current_note_candidate_rank(
    *,
    current: FutureAlignerHint | None,
    ordered: list[FutureAlignerCandidateScore],
) -> int | None:
    if current is None or current.normalized_note_id is None:
        return None
    for index, candidate in enumerate(ordered, start=1):
        if candidate.note_id == current.normalized_note_id:
            return index
    return None


def _candidate_ranking_payload(
    ordered: list[FutureAlignerCandidateScore],
) -> list[dict[str, object]]:
    ranking: list[dict[str, object]] = []
    for index, candidate in enumerate(ordered[:3], start=1):
        ranking.append(
            {
                "position": index,
                "note_id": candidate.note_id,
                "candidate_rank": candidate.rank,
                "total_score": round(candidate.total_score, 3),
                "support_score": round(candidate.support_score, 3),
                "continuity_score": round(candidate.continuity_score, 3),
                "acoustic_score": round(candidate.acoustic_score, 3),
                "note_span_score": round(candidate.note_span_score, 3),
                "order_penalty": round(candidate.order_penalty, 3),
                "same_note_neighbors": candidate.same_note_neighbors,
                "left_neighbor_match": candidate.left_neighbor_match,
                "right_neighbor_match": candidate.right_neighbor_match,
                "note_span_overlap_ratio": round(candidate.note_span_overlap_ratio, 3),
                "note_span_interior_ratio": round(candidate.note_span_interior_ratio, 3),
                "part_aware_note_span_used": candidate.part_aware_note_span_used,
            }
        )
    return ranking


def _candidate_precondition_flags(
    *,
    boundary_regime: bool,
    guarded: bool,
    top: FutureAlignerCandidateScore,
    runner_up: FutureAlignerCandidateScore,
    opposite_edge_neighbors: bool,
    asymmetric_chain_anchor: bool,
    interior_advantage: float,
    overlap_advantage: float,
    phrase_window_pressure: float,
    has_acoustic: bool,
    current: FutureAlignerHint | None,
) -> dict[str, dict[str, bool]]:
    general_flags = {
        "has_acoustic_evidence": has_acoustic,
        "boundary_regime": boundary_regime,
        "not_phrase_window_guarded": not guarded,
        "current_note_available": current is not None and current.normalized_note_id is not None,
        "winner_is_not_current_note": current is None or current.normalized_note_id != top.note_id,
        "runner_up_available": True,
    }
    interior_flags = {
        "boundary_regime": boundary_regime,
        "not_phrase_window_guarded": not guarded,
        "opposite_edge_neighbors": opposite_edge_neighbors,
        "top_part_note_span": top.part_aware_note_span_used,
        "runner_up_part_note_span": runner_up.part_aware_note_span_used,
        "top_overlap_ge_0_88": top.note_span_overlap_ratio >= 0.88,
        "top_interior_ge_0_48": top.note_span_interior_ratio >= 0.48,
        "runner_up_overlap_le_0_35": runner_up.note_span_overlap_ratio <= 0.35,
        "runner_up_interior_le_0_12": runner_up.note_span_interior_ratio <= 0.12,
        "interior_advantage_ge_0_42": interior_advantage >= 0.42,
        "overlap_advantage_ge_0_68": overlap_advantage >= 0.68,
    }
    dual_touch_flags = {
        "boundary_regime": boundary_regime,
        "not_phrase_window_guarded": not guarded,
        "opposite_edge_neighbors": opposite_edge_neighbors,
        "top_part_note_span": top.part_aware_note_span_used,
        "runner_up_part_note_span": runner_up.part_aware_note_span_used,
        "top_overlap_ge_0_88": top.note_span_overlap_ratio >= 0.88,
        "top_interior_ge_0_46": top.note_span_interior_ratio >= 0.46,
        "runner_up_overlap_ge_0_36": runner_up.note_span_overlap_ratio >= 0.36,
        "runner_up_overlap_le_0_52": runner_up.note_span_overlap_ratio <= 0.52,
        "runner_up_interior_le_0_12": runner_up.note_span_interior_ratio <= 0.12,
        "interior_advantage_ge_0_42": interior_advantage >= 0.42,
        "overlap_advantage_ge_0_50": overlap_advantage >= 0.50,
        "phrase_window_pressure_le_0_72": phrase_window_pressure <= 0.72,
    }
    chain_anchor_flags = {
        "boundary_regime": boundary_regime,
        "not_phrase_window_guarded": not guarded,
        "asymmetric_chain_anchor_shape": asymmetric_chain_anchor,
        "top_part_note_span": top.part_aware_note_span_used,
        "runner_up_part_note_span": runner_up.part_aware_note_span_used,
        "top_overlap_ge_0_95": top.note_span_overlap_ratio >= 0.95,
        "top_interior_ge_0_34": top.note_span_interior_ratio >= 0.34,
        "runner_up_overlap_le_0_30": runner_up.note_span_overlap_ratio <= 0.30,
        "runner_up_interior_le_0_08": runner_up.note_span_interior_ratio <= 0.08,
        "interior_advantage_ge_0_36": interior_advantage >= 0.36,
        "overlap_advantage_ge_0_62": overlap_advantage >= 0.62,
        "phrase_window_pressure_le_0_73": phrase_window_pressure <= 0.73,
    }
    return {
        "general": general_flags,
        "asymmetric_interior": interior_flags,
        "asymmetric_dual_touch": dual_touch_flags,
        "asymmetric_chain_anchor": chain_anchor_flags,
    }


def _missing_preconditions(candidate_precondition_flags: dict[str, dict[str, bool]]) -> dict[str, list[str]]:
    return {
        key: [flag_name for flag_name, flag_value in flags.items() if not flag_value]
        for key, flags in candidate_precondition_flags.items()
    }


def _blocker_payload(
    *,
    mode: str,
    current: FutureAlignerHint | None,
    current_candidate: FutureAlignerCandidateScore | None,
    top: FutureAlignerCandidateScore,
    runner_up: FutureAlignerCandidateScore,
    boundary_regime: bool,
    guarded: bool,
    has_acoustic: bool,
    boundary_regime_debug: dict[str, object],
    opposite_edge_neighbors: bool,
    asymmetric_chain_anchor: bool,
    interior_flags: dict[str, bool],
) -> tuple[dict[str, list[str]], str]:
    data_blockers: list[str] = []
    evidence_blockers: list[str] = []
    classification_blockers: list[str] = []

    if not has_acoustic:
        evidence_blockers.append("missing_acoustic_evidence")
    else:
        if bool(boundary_regime_debug.get("effective_boundary_uncertain")):
            evidence_blockers.append("boundary_uncertain")
        if float(boundary_regime_debug.get("boundary_pressure", 0.0)) < 0.58:
            evidence_blockers.append("boundary_pressure_below_threshold")
        if float(boundary_regime_debug.get("boundary_transition_strength", 0.0)) < 0.72:
            evidence_blockers.append("boundary_transition_strength_below_threshold")
        if float(boundary_regime_debug.get("boundary_conflict_likelihood", 0.0)) < 0.68:
            evidence_blockers.append("boundary_conflict_likelihood_below_threshold")
    if not boundary_regime:
        classification_blockers.append("boundary_regime_inactive")
    if guarded:
        classification_blockers.append("phrase_window_guarded")
    if current is not None and current.normalized_note_id == top.note_id:
        data_blockers.append("winner_is_current_note")
    if current_candidate is not None and current_candidate.note_id == top.note_id:
        data_blockers.append("current_note_already_top_ranked")
    if not opposite_edge_neighbors:
        classification_blockers.append("opposite_edge_shape_missing")
        if top.same_note_neighbors != 1 or runner_up.same_note_neighbors != 1:
            data_blockers.append("interior_neighbor_pattern_missing")
        if (
            top.left_neighbor_match == runner_up.left_neighbor_match
            and top.right_neighbor_match == runner_up.right_neighbor_match
        ):
            data_blockers.append("edge_direction_not_asymmetric")
    if not asymmetric_chain_anchor:
        classification_blockers.append("chain_anchor_shape_missing")
        if top.same_note_neighbors < 2:
            data_blockers.append("chain_anchor_top_same_note_neighbors_lt2")
        if runner_up.same_note_neighbors != 1:
            data_blockers.append("chain_anchor_runner_up_not_single_touch")

    if mode == "none":
        if boundary_regime and opposite_edge_neighbors and interior_flags["runner_up_overlap_le_0_35"] and interior_flags["runner_up_interior_le_0_12"]:
            key_blocker = "winner_is_current_note" if current is not None and current.normalized_note_id == top.note_id else "no_candidate_shape"
        elif not boundary_regime:
            key_blocker = "boundary_regime_inactive"
        elif not opposite_edge_neighbors and not asymmetric_chain_anchor:
            key_blocker = "no_candidate_shape"
        elif current is not None and current.normalized_note_id == top.note_id:
            key_blocker = "winner_is_current_note"
        else:
            key_blocker = "candidate_preconditions_missing"
    elif mode in {
        "asymmetric_interior_candidate",
        "asymmetric_chain_anchor_candidate",
        "asymmetric_dual_touch_candidate",
    }:
        key_blocker = "winner_is_current_note" if current is not None and current.normalized_note_id == top.note_id else "candidate_ready"
    else:
        key_blocker = mode

    blocker_categories = {
        "data": list(dict.fromkeys(data_blockers)),
        "evidence": list(dict.fromkeys(evidence_blockers)),
        "candidate_classification": list(dict.fromkeys(classification_blockers)),
    }
    return blocker_categories, key_blocker


def build_phrase_boundary_candidate_debug(
    *,
    segment: PhoneSegment,
    current: FutureAlignerHint | None,
    candidate_scores: list[FutureAlignerCandidateScore] | None = None,
    ordered_candidates: list[FutureAlignerCandidateScore] | None = None,
    acoustic: FutureAlignerAcousticEvidence | None,
    mode: str,
    guarded: bool,
    boundary_regime_debug: dict[str, object],
    overlap_advantage: float,
    interior_advantage: float,
    phrase_window_pressure: float,
    current_candidate: FutureAlignerCandidateScore | None,
    has_acoustic: bool | None = None,
) -> dict[str, object]:
    effective_candidates = list(candidate_scores or ordered_candidates or [])
    if len(effective_candidates) < 2:
        return normalize_phrase_boundary_candidate_debug(
            {},
            default_key_blocker="no_candidate_competition",
            segment=segment,
            current=current,
            boundary_regime=boundary_regime_debug,
        )

    ordered = (
        list(ordered_candidates)
        if ordered_candidates is not None
        else sorted(effective_candidates, key=lambda candidate: candidate.total_score, reverse=True)
    )
    top = ordered[0]
    runner_up = ordered[1]
    acoustic_available = bool(acoustic is not None and acoustic.available) if has_acoustic is None else bool(has_acoustic)
    boundary_regime = bool(boundary_regime_debug["ready"])
    note_duration_sec, segment_duration_sec, segment_to_note_duration_ratio = _segment_note_duration_metrics(
        segment=segment,
        current=current,
    )
    shape = _chain_anchor_shape(top, runner_up)
    current_rank = _current_note_candidate_rank(current=current, ordered=ordered)
    candidate_precondition_flags = _candidate_precondition_flags(
        boundary_regime=boundary_regime,
        guarded=guarded,
        top=top,
        runner_up=runner_up,
        opposite_edge_neighbors=bool(shape["opposite_edge_neighbors"]),
        asymmetric_chain_anchor=bool(shape["asymmetric_chain_anchor"]),
        interior_advantage=interior_advantage,
        overlap_advantage=overlap_advantage,
        phrase_window_pressure=phrase_window_pressure,
        has_acoustic=acoustic_available,
        current=current,
    )
    missing_preconditions = _missing_preconditions(candidate_precondition_flags)
    blocker_categories, key_blocker = _blocker_payload(
        mode=mode,
        current=current,
        current_candidate=current_candidate,
        top=top,
        runner_up=runner_up,
        boundary_regime=boundary_regime,
        guarded=guarded,
        has_acoustic=acoustic_available,
        boundary_regime_debug=boundary_regime_debug,
        opposite_edge_neighbors=bool(shape["opposite_edge_neighbors"]),
        asymmetric_chain_anchor=bool(shape["asymmetric_chain_anchor"]),
        interior_flags=candidate_precondition_flags["asymmetric_interior"],
    )
    ranking = _candidate_ranking_payload(ordered)
    return normalize_phrase_boundary_candidate_debug(
        {
            "mode": mode,
            "current_note_id": current.normalized_note_id if current is not None else None,
            "top_note_id": top.note_id,
            "runner_up_note_id": runner_up.note_id,
            "winner_matches_current_note": bool(current is not None and current.normalized_note_id == top.note_id),
            "winner_is_current_note": bool(current is not None and current.normalized_note_id == top.note_id),
            "score_hint_context": {
                "note_id": current.normalized_note_id if current is not None else None,
                "source_note_id": current.source_normalized_note_id if current is not None else None,
                "note_identity_match": bool(current.note_identity_match) if current is not None else False,
                "source_note_identity_match": (
                    bool(current.source_note_identity_match)
                    if current is not None and current.source_note_identity_match is not None
                    else False
                ),
                "score_hint_confidence": round(current.score_hint_confidence, 3) if current is not None else 0.0,
                "note_binding_strategy": current.note_binding_strategy if current is not None else None,
                "source_note_binding_strategy": current.source_note_binding_strategy if current is not None else None,
                "note_start_sec": round(current.note_start_sec, 3) if current is not None and current.note_start_sec is not None else None,
                "note_end_sec": round(current.note_end_sec, 3) if current is not None and current.note_end_sec is not None else None,
                "note_duration_sec": round(note_duration_sec, 3) if note_duration_sec is not None else None,
                "score_hint_notes": list(current.score_hint_notes) if current is not None else [],
            },
            "ownership_binding_adjustment": {
                "applied": bool(current.ownership_binding_adjusted) if current is not None else False,
                "reason": current.ownership_binding_reason if current is not None else None,
                "source_note_id": current.source_normalized_note_id if current is not None else None,
                "effective_note_id": current.normalized_note_id if current is not None else None,
                "source_note_binding_strategy": current.source_note_binding_strategy if current is not None else None,
                "effective_note_binding_strategy": current.note_binding_strategy if current is not None else None,
            },
            "segment_slicing": {
                "segment_start_sec": round(float(segment.time_range.start_sec), 3),
                "segment_end_sec": round(float(segment.time_range.end_sec), 3),
                "segment_duration_sec": round(segment_duration_sec, 3),
                "note_start_sec": round(current.note_start_sec, 3) if current is not None and current.note_start_sec is not None else None,
                "note_end_sec": round(current.note_end_sec, 3) if current is not None and current.note_end_sec is not None else None,
                "note_duration_sec": round(note_duration_sec, 3) if note_duration_sec is not None else None,
                "segment_to_note_duration_ratio": round(segment_to_note_duration_ratio, 3),
            },
            "winner_state": {
                "winner_note_id": top.note_id,
                "runner_up_note_id": runner_up.note_id,
                "winner_matches_current_note": bool(current is not None and current.normalized_note_id == top.note_id),
                "winner_is_current_note": bool(current is not None and current.normalized_note_id == top.note_id),
                "current_note_candidate_rank": current_rank,
                "current_note_total_score": round(current_candidate.total_score, 3) if current_candidate is not None else None,
                "winner_total_score": round(top.total_score, 3),
                "runner_up_total_score": round(runner_up.total_score, 3),
                "ordered_note_ids": [candidate.note_id for candidate in ordered],
            },
            "candidate_ranking": {
                "current_note_candidate_rank": current_rank,
                "top": ranking[0] if ranking else None,
                "runner_up": ranking[1] if len(ranking) > 1 else None,
                "ordered_candidates": ranking,
            },
            "boundary_regime": {
                **boundary_regime_debug,
                "phrase_window_pressure": round(phrase_window_pressure, 3),
            },
            "candidate_shapes": {
                "opposite_edge_neighbors": shape["opposite_edge_neighbors"],
                "asymmetric_chain_anchor": shape["asymmetric_chain_anchor"],
                "asymmetric_chain_anchor_primary_blocker": shape["asymmetric_chain_anchor_primary_blocker"],
                "asymmetric_chain_anchor_blocking_reasons": shape["asymmetric_chain_anchor_blocking_reasons"],
                "top_same_note_neighbors": top.same_note_neighbors,
                "runner_up_same_note_neighbors": runner_up.same_note_neighbors,
                "top_left_neighbor_match": top.left_neighbor_match,
                "top_right_neighbor_match": top.right_neighbor_match,
                "runner_up_left_neighbor_match": runner_up.left_neighbor_match,
                "runner_up_right_neighbor_match": runner_up.right_neighbor_match,
                "top_anchor_profile": shape["top_anchor_profile"],
                "runner_up_anchor_profile": shape["runner_up_anchor_profile"],
            },
            "candidate_precondition_flags": candidate_precondition_flags,
            "missing_preconditions": missing_preconditions,
            "overlap_metrics": {
                "top_overlap": round(top.note_span_overlap_ratio, 3),
                "runner_up_overlap": round(runner_up.note_span_overlap_ratio, 3),
                "top_interior": round(top.note_span_interior_ratio, 3),
                "runner_up_interior": round(runner_up.note_span_interior_ratio, 3),
                "overlap_advantage": round(overlap_advantage, 3),
                "interior_advantage": round(interior_advantage, 3),
            },
            "blocker_categories": blocker_categories,
            "key_blocker": key_blocker,
        },
        default_key_blocker=key_blocker,
        segment=segment,
        current=current,
        boundary_regime=boundary_regime_debug,
    )

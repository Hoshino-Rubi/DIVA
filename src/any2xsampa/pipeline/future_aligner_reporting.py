from __future__ import annotations

from copy import deepcopy

from any2xsampa.core.models import PhoneSegment
from any2xsampa.pipeline.future_aligner_models import (
    FutureAlignerAcousticEvidence,
    FutureAlignerCandidateScore,
    FutureAlignerHint,
    FutureAlignerResult,
    future_aligner_output_payload,
)


FUTURE_ALIGNER_REPORT_SCHEMA_VERSION = "future_aligner_report_v1"


def build_ownership_conflict_candidates(
    candidate_scores: list[FutureAlignerCandidateScore],
    *,
    limit: int = 3,
) -> list[dict[str, object]]:
    if not candidate_scores:
        return []
    ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
    items: list[dict[str, object]] = []
    for candidate in ordered[:limit]:
        items.append(
            {
                "note_id": candidate.note_id,
                "total_score": round(candidate.total_score, 3),
                "support_score": round(candidate.support_score, 3),
                "continuity_score": round(candidate.continuity_score, 3),
                "acoustic_score": round(candidate.acoustic_score, 3),
                "note_span_score": round(candidate.note_span_score, 3),
                "order_penalty": round(candidate.order_penalty, 3),
                "same_note_neighbors": candidate.same_note_neighbors,
                "left_neighbor_match": candidate.left_neighbor_match,
                "right_neighbor_match": candidate.right_neighbor_match,
                "continuity_used": candidate.continuity_used,
                "part_aware_note_span_used": candidate.part_aware_note_span_used,
                "note_span_overlap_ratio": round(candidate.note_span_overlap_ratio, 3),
                "note_span_interior_ratio": round(candidate.note_span_interior_ratio, 3),
                "note_span_anchor_count": candidate.note_span_anchor_count,
                "candidate_summary": (
                    f"support={candidate.support_score:.3f},continuity={candidate.continuity_score:.3f},"
                    f"acoustic={candidate.acoustic_score:.3f},note_span={candidate.note_span_score:.3f},"
                    f"penalty={candidate.order_penalty:.3f}"
                ),
            }
        )
    return items


def build_ownership_conflict_summary(
    candidate_scores: list[FutureAlignerCandidateScore],
    acoustic: FutureAlignerAcousticEvidence | None,
    *,
    conflict_threshold: float,
    phrase_boundary_mode: str,
    trained_phrase_ready: bool,
) -> str:
    if not candidate_scores:
        return "no_candidates"
    ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
    top = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    if runner_up is None:
        return f"single_candidate={top.note_id}"
    gap = top.total_score - runner_up.total_score
    stability_margin = gap - conflict_threshold
    boundary_transition = acoustic.boundary_transition_strength if acoustic is not None else 0.0
    ambiguity = acoustic.ownership_ambiguity if acoustic is not None else 0.0
    conflict_risk = acoustic.ownership_conflict_risk if acoustic is not None else 0.0
    boundary_conflict = acoustic.boundary_conflict_likelihood if acoustic is not None else 0.0
    approximation_only = bool(acoustic is not None and acoustic.approximation_only)
    return (
        f"winner={top.note_id},runner_up={runner_up.note_id},gap={gap:.3f},"
        f"threshold={conflict_threshold:.3f},stability_margin={stability_margin:.3f},"
        f"boundary_transition={boundary_transition:.3f},ambiguity={ambiguity:.3f},"
        f"conflict_risk={conflict_risk:.3f},boundary_conflict={boundary_conflict:.3f},"
        f"phrase_boundary_mode={phrase_boundary_mode},trained_phrase_boundary_release_ready={'yes' if trained_phrase_ready else 'no'},"
        f"approximation_only={'yes' if approximation_only else 'no'}"
    )


def build_release_explanation(
    result: FutureAlignerResult,
    safety: dict[str, object],
) -> str:
    if not safety:
        return ""
    release_kind = str(safety.get("release_kind", "none"))
    release_mode = str(safety.get("release_mode", "none"))
    if bool(safety.get("released")):
        stability_label = "naturally stable" if bool(safety.get("natural_stability")) else "subclass-specific gate"
        return (
            f"{release_kind} accepted for {release_mode} because trained posterior/logit evidence kept the winner clear over the current note; "
            f"this is trained-only, approximation backends stay blocked, and the release used {stability_label} "
            f"(stability_margin={result.ownership_stability_margin:.3f}, threshold={result.ownership_conflict_threshold:.3f})."
        )
    blockers = safety.get("blockers")
    blocker_text = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) and blockers else "no selective gate"
    if release_kind == "phrase_boundary_selective_release_candidate":
        return (
            f"phrase-boundary release stayed blocked for {release_mode} because {blocker_text}; "
            f"the guardrail remained conservative at stability_margin={result.ownership_stability_margin:.3f}."
        )
    if release_kind == "none":
        return (
            f"no selective release path activated; ownership stayed conservative because {blocker_text} "
            f"(stability_margin={result.ownership_stability_margin:.3f})."
        )
    return (
        f"{release_kind} remained blocked for {release_mode} because {blocker_text}; "
        f"approximation backends cannot release this path and the guardrail stayed conservative "
        f"(stability_margin={result.ownership_stability_margin:.3f})."
    )


def build_release_risk_summary(
    result: FutureAlignerResult,
    safety: dict[str, object],
) -> str:
    if not safety:
        return ""
    blockers = safety.get("blockers")
    blocker_text = "+".join(str(item) for item in blockers) if isinstance(blockers, list) and blockers else "none"
    return (
        f"mode={safety.get('release_mode')};released={'yes' if safety.get('released') else 'no'};"
        f"trained_only={'yes' if safety.get('trained_only') else 'no'};"
        f"natural_stability={'yes' if safety.get('natural_stability') else 'no'};"
        f"subclass_specific_gate={'yes' if safety.get('subclass_specific_gate') else 'no'};"
        f"approximation_backend_blocked={'yes' if safety.get('approximation_backend_blocked') else 'no'};"
        f"blockers={blocker_text};stability_margin={result.ownership_stability_margin:.3f}"
    )


def build_selective_release_safety(
    result: FutureAlignerResult,
    current: FutureAlignerHint | None,
    candidate_scores: list[FutureAlignerCandidateScore],
    acoustic: FutureAlignerAcousticEvidence | None,
    *,
    phrase_boundary_mode: str,
    phrase_boundary_guarded: bool,
    trained_one_sided_ready: bool,
    trained_phrase_ready: bool,
) -> dict[str, object]:
    if not candidate_scores:
        return {}
    ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
    top = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    current_candidate = _current_candidate(
        current_note_id=current.normalized_note_id if current is not None else None,
        candidate_scores=ordered,
    )
    release_kind = "none"
    release_mode = "none"
    release_candidate = False
    if trained_one_sided_ready:
        release_kind = "one_sided_selective_release"
        release_mode = "one_sided_continuation"
        release_candidate = True
    elif trained_phrase_ready:
        release_kind = "phrase_boundary_selective_release"
        release_mode = phrase_boundary_mode
        release_candidate = phrase_boundary_mode != "none"
    elif phrase_boundary_mode in {
        "asymmetric_interior_candidate",
        "asymmetric_dual_touch_candidate",
        "asymmetric_chain_anchor_candidate",
    }:
        release_kind = "phrase_boundary_selective_release_candidate"
        release_mode = phrase_boundary_mode
        release_candidate = True
    released = bool(
        release_kind in {"one_sided_selective_release", "phrase_boundary_selective_release"}
        and not result.aligner_fallback_used
        and result.aligned_note_id == top.note_id
    )
    natural_stability = bool(released and result.ownership_stability_margin >= 0.0)
    subclass_specific_gate = bool(released and not natural_stability and release_mode != "one_sided_continuation")
    blockers: list[str] = []
    if not released:
        if phrase_boundary_mode == "mirrored_conflict_guarded":
            blockers.append("mirrored_conflict_guardrail")
        if phrase_boundary_guarded:
            blockers.append("phrase_window_guardrail")
        if acoustic is None or not acoustic.available:
            blockers.append("missing_acoustic_evidence")
        elif acoustic.approximation_only:
            blockers.append("approximation_backend")
        if result.note_ownership_conflict:
            blockers.append("ownership_conflict")
        if result.boundary_conflict_likely:
            blockers.append("boundary_conflict")
        if result.ownership_stability_margin < 0.0:
            blockers.append("negative_stability_margin")
        elif result.ownership_stability_margin < 0.08:
            blockers.append("thin_stability_margin")
        if current is not None and current.normalized_note_id == top.note_id:
            blockers.append("already_on_current_note")
        if current_candidate is None and current is not None and current.normalized_note_id is not None:
            blockers.append("missing_current_candidate")
    approximation_reason = ""
    if release_candidate:
        if acoustic is None or not acoustic.available:
            approximation_reason = "selective release needs trained posterior/logit evidence"
        elif acoustic.approximation_only:
            approximation_reason = "approximation backend stayed conservative because selective release is trained-only"
        else:
            approximation_reason = (
                "approximation backends stay blocked because this release depends on trained "
                "posterior/logit separation under boundary pressure"
            )
    return {
        "release_kind": release_kind,
        "release_mode": release_mode,
        "release_candidate": release_candidate,
        "released": released,
        "trained_only": release_candidate,
        "approximation_backend_blocked": bool(release_candidate),
        "approximation_backend_reason": approximation_reason,
        "natural_stability": natural_stability,
        "subclass_specific_gate": subclass_specific_gate,
        "stability_margin": round(result.ownership_stability_margin, 3),
        "conflict_threshold": round(result.ownership_conflict_threshold, 3),
        "ownership_gap": round(result.ownership_conflict_gap, 3),
        "winner_note_id": top.note_id,
        "runner_up_note_id": runner_up.note_id if runner_up is not None else None,
        "current_note_id": current.normalized_note_id if current is not None else None,
        "current_note_overlap": round(current_candidate.note_span_overlap_ratio, 3) if current_candidate is not None else None,
        "current_note_interior": round(current_candidate.note_span_interior_ratio, 3) if current_candidate is not None else None,
        "blockers": blockers,
    }


def build_competition_summary(
    candidate_scores: list[FutureAlignerCandidateScore],
    acoustic: FutureAlignerAcousticEvidence | None,
    *,
    conflict_threshold: float,
    evidence_mode: str,
    competition_guardrails: list[str],
    phrase_window_pressure: float,
    phrase_boundary_mode: str,
    phrase_boundary_interior_advantage: float,
    phrase_boundary_overlap_advantage: float,
) -> str:
    if not candidate_scores:
        return "no_candidates"
    ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
    top = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    guardrails = "+".join(competition_guardrails) or "none"
    if runner_up is None:
        return (
            f"winner={top.note_id}:score={top.total_score:.3f},"
            f"conflict_threshold={conflict_threshold:.3f},"
            f"evidence_mode={evidence_mode},"
            f"guardrails={guardrails},"
            f"phrase_window_pressure={phrase_window_pressure:.3f},"
            f"phrase_boundary_mode={phrase_boundary_mode},"
            f"acoustic_separation={_acoustic_metric(acoustic, 'competition_separation'):.3f},"
            f"acoustic_interior_hold={_acoustic_metric(acoustic, 'interior_hold_strength'):.3f},"
            f"acoustic_boundary_pressure={_acoustic_metric(acoustic, 'boundary_pressure'):.3f},"
            f"one_sided_strength={_acoustic_metric(acoustic, 'one_sided_continuation_strength'):.3f},"
            f"winner_same_note_neighbors={top.same_note_neighbors},"
            f"winner_overlap={top.note_span_overlap_ratio:.3f},"
            f"winner_interior={top.note_span_interior_ratio:.3f},"
            f"winner_continuity={top.continuity_score:.3f},"
            f"winner_acoustic={top.acoustic_score:.3f},"
            f"winner_note_span={top.note_span_score:.3f}"
        )
    gap = top.total_score - runner_up.total_score
    return (
        f"winner={top.note_id}:score={top.total_score:.3f},"
        f"runner_up={runner_up.note_id}:score={runner_up.total_score:.3f},"
        f"gap={gap:.3f},"
        f"conflict_threshold={conflict_threshold:.3f},"
        f"evidence_mode={evidence_mode},"
        f"guardrails={guardrails},"
        f"phrase_window_pressure={phrase_window_pressure:.3f},"
        f"phrase_boundary_mode={phrase_boundary_mode},"
        f"phrase_boundary_interior_advantage={phrase_boundary_interior_advantage:.3f},"
        f"phrase_boundary_overlap_advantage={phrase_boundary_overlap_advantage:.3f},"
        f"acoustic_separation={_acoustic_metric(acoustic, 'competition_separation'):.3f},"
        f"acoustic_interior_hold={_acoustic_metric(acoustic, 'interior_hold_strength'):.3f},"
        f"acoustic_boundary_pressure={_acoustic_metric(acoustic, 'boundary_pressure'):.3f},"
        f"one_sided_strength={_acoustic_metric(acoustic, 'one_sided_continuation_strength'):.3f},"
        f"winner_same_note_neighbors={top.same_note_neighbors},"
        f"runner_up_same_note_neighbors={runner_up.same_note_neighbors},"
        f"winner_overlap={top.note_span_overlap_ratio:.3f},"
        f"runner_up_overlap={runner_up.note_span_overlap_ratio:.3f},"
        f"winner_interior={top.note_span_interior_ratio:.3f},"
        f"runner_up_interior={runner_up.note_span_interior_ratio:.3f},"
        f"winner_continuity={top.continuity_score:.3f},"
        f"winner_acoustic={top.acoustic_score:.3f},"
        f"winner_note_span={top.note_span_score:.3f}"
    )


def build_ownership_resolution_basis(
    result: FutureAlignerResult,
    acoustic: FutureAlignerAcousticEvidence | None,
    candidate_scores: list[FutureAlignerCandidateScore],
    *,
    phrase_boundary_mode: str,
    phrase_boundary_guarded: bool,
    trained_one_sided_ready: bool,
    trained_phrase_ready: bool,
) -> str:
    if not candidate_scores:
        return "no_candidates"
    ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
    top = ordered[0]
    tokens: list[str] = []
    if result.future_aligner_sequence_applied and result.same_note_support:
        tokens.append("same_note_continuity")
    if result.part_aware_note_span_used:
        tokens.append("part_note_span")
        if result.note_span_interior_ratio >= 0.5:
            tokens.append("long_note_interior")
    if acoustic is not None and acoustic.available:
        if (
            acoustic.same_note_stability >= 0.6
            or acoustic.weak_overlap_recovery >= 0.62
            or acoustic.continuation_friendliness >= 0.68
            or acoustic.weak_overlap_friendliness >= 0.68
        ):
            tokens.append("acoustic_continuation")
        if acoustic.reassignment_resistance >= 0.66 and not result.future_aligner_sequence_applied:
            tokens.append("stable_boundary_resistance")
        if acoustic.weak_overlap_friendliness >= 0.72:
            tokens.append("weak_overlap_recovery")
        if acoustic.competition_separation >= 0.68:
            tokens.append("candidate_separation")
        if acoustic.boundary_pressure >= 0.68:
            tokens.append("boundary_pressure_guardrail")
        if acoustic.ownership_conflict_risk >= 0.62:
            tokens.append("ownership_conflict_risk")
        if acoustic.boundary_conflict_likelihood >= 0.62:
            tokens.append("boundary_conflict_guardrail")
        if top.same_note_neighbors == 1 and acoustic.one_sided_continuation_strength >= 0.72:
            tokens.append("one_sided_continuation_evidence")
        if phrase_boundary_mode == "mirrored_conflict_guarded":
            tokens.append("phrase_boundary_mirrored_conflict")
        elif phrase_boundary_mode == "asymmetric_interior_candidate":
            tokens.append("phrase_boundary_asymmetric_interior")
        elif phrase_boundary_mode == "asymmetric_dual_touch_candidate":
            tokens.append("phrase_boundary_dual_touch")
        elif phrase_boundary_mode == "asymmetric_chain_anchor_candidate":
            tokens.append("phrase_boundary_chain_anchor")
        if phrase_boundary_guarded:
            tokens.append("phrase_window_guardrail")
        if not result.aligner_fallback_used and trained_one_sided_ready:
            tokens.append("one_sided_selective_release")
        if not result.aligner_fallback_used and trained_phrase_ready:
            tokens.append("phrase_boundary_selective_release")
        if acoustic.logit_like_available and not acoustic.posterior_like_available:
            tokens.append("logit_only_guardrail")
        if not acoustic.posterior_like_available and not acoustic.logit_like_available:
            tokens.append("partial_evidence_guardrail")
    if result.aligner_fallback_used:
        tokens.append("conservative_fallback")
    if result.boundary_conflict_likely:
        tokens.append("boundary_conflict")
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return "+".join(deduped) if deduped else "score_hint_only"


def build_future_aligner_report(
    result: FutureAlignerResult,
    *,
    segment: PhoneSegment,
    current: FutureAlignerHint | None,
    phrase_boundary_mode: str,
    phrase_boundary_guarded: bool,
    phrase_window_pressure: float,
    evidence_mode: str,
    competition_guardrails: list[str],
) -> dict[str, object]:
    return {
        "report_schema_version": FUTURE_ALIGNER_REPORT_SCHEMA_VERSION,
        "alignment": {
            "aligned_track_id": result.aligned_track_id,
            "aligned_part_id": result.aligned_part_id,
            "aligned_note_id": result.aligned_note_id,
            "aligner_strategy": result.aligner_strategy,
            "aligner_confidence": round(result.aligner_confidence, 3),
            "aligner_fallback_used": result.aligner_fallback_used,
            "aligner_v2_strategy": result.aligner_v2_strategy,
            "aligner_v2_confidence": round(result.aligner_v2_confidence or result.aligner_confidence, 3),
            "aligner_v2_fallback_used": result.aligner_v2_fallback_used or result.aligner_fallback_used,
            "future_aligner_window_size": result.future_aligner_window_size,
            "future_aligner_sequence_applied": result.future_aligner_sequence_applied,
            "future_aligner_sequence_score": round(result.future_aligner_sequence_score, 3),
            "future_aligner_order_penalty": round(result.future_aligner_order_penalty, 3),
            "future_aligner_provenance_applied": result.future_aligner_provenance_applied,
            "future_aligner_provenance_summary": result.future_aligner_provenance_summary,
            "aligner_v2_notes": list(result.aligner_v2_notes or result.aligner_notes),
        },
        "segment": {
            "start_sec": round(float(segment.time_range.start_sec), 3),
            "end_sec": round(float(segment.time_range.end_sec), 3),
            "duration_sec": round(max(0.0, float(segment.time_range.end_sec) - float(segment.time_range.start_sec)), 3),
        },
        "score_hint_binding": {
            "current_note_id": current.normalized_note_id if current is not None else None,
            "source_note_id": result.source_score_hint_note_id,
            "source_note_identity_match": result.source_score_hint_note_identity_match,
            "source_note_binding_strategy": result.source_score_hint_binding_strategy,
            "effective_note_id": result.effective_score_hint_note_id,
            "effective_note_identity_match": result.effective_score_hint_note_identity_match,
            "effective_note_binding_strategy": result.effective_score_hint_binding_strategy,
            "ownership_binding_adjustment": dict(result.ownership_binding_adjustment),
        },
        "acoustic": {
            "acoustic_evidence_applied": result.acoustic_evidence_applied,
            "acoustic_evidence_summary": result.acoustic_evidence_summary,
            "boundary_uncertainty": dict(result.boundary_uncertainty),
            "evidence_mode": evidence_mode,
        },
        "note_span": {
            "part_aware_note_span_used": result.part_aware_note_span_used,
            "note_span_evidence_score": round(result.note_span_evidence_score, 3),
            "note_span_overlap_ratio": round(result.note_span_overlap_ratio, 3),
            "note_span_interior_ratio": round(result.note_span_interior_ratio, 3),
            "note_span_anchor_count": result.note_span_anchor_count,
            "note_span_summary": result.note_span_summary,
        },
        "competition": {
            "summary": result.alignment_competition_summary,
            "guardrails": list(competition_guardrails),
            "phrase_boundary_mode": phrase_boundary_mode,
            "phrase_boundary_guarded": phrase_boundary_guarded,
            "phrase_window_pressure": round(phrase_window_pressure, 3),
            "note_ownership_conflict": result.note_ownership_conflict,
            "ownership_conflict_gap": round(result.ownership_conflict_gap, 3),
            "ownership_conflict_threshold": round(result.ownership_conflict_threshold, 3),
            "ownership_stability_margin": round(result.ownership_stability_margin, 3),
        },
        "ownership": {
            "summary": result.ownership_conflict_summary,
            "candidates": [dict(item) for item in result.ownership_conflict_candidates],
            "resolution_basis": result.ownership_resolution_basis,
            "boundary_conflict_likely": result.boundary_conflict_likely,
        },
        "selective_release": {
            "safety": deepcopy(result.selective_release_safety),
            "explanation": result.release_explanation,
            "risk_summary": result.release_risk_summary,
        },
        "phrase_boundary": {
            "mode": phrase_boundary_mode,
            "guarded": phrase_boundary_guarded,
            "candidate_debug": deepcopy(result.phrase_boundary_candidate_debug),
            "blocker_summary": result.phrase_boundary_blocker_summary,
        },
    }


def future_aligner_report_payload(segment: PhoneSegment) -> dict[str, object]:
    report = segment.metadata.get("future_aligner_report")
    if isinstance(report, dict):
        return deepcopy(report)
    payload = future_aligner_output_payload(segment)
    competition_mode = None
    safety = payload.get("selective_release_safety")
    if isinstance(safety, dict):
        competition_mode = _text(safety.get("release_mode"))
    return {
        "report_schema_version": FUTURE_ALIGNER_REPORT_SCHEMA_VERSION,
        "alignment": {
            "aligned_track_id": _text(payload.get("aligned_track_id")),
            "aligned_part_id": _text(payload.get("aligned_part_id")),
            "aligned_note_id": _text(payload.get("aligned_note_id")),
            "aligner_strategy": _text(payload.get("aligner_strategy")) or "unresolved",
            "aligner_confidence": _rounded_float(payload.get("aligner_confidence")) or 0.0,
            "aligner_fallback_used": bool(payload.get("aligner_fallback_used", False)),
            "aligner_v2_strategy": _text(payload.get("aligner_v2_strategy")) or "score_aware_aligner_v2",
            "aligner_v2_confidence": _rounded_float(payload.get("aligner_v2_confidence")) or 0.0,
            "aligner_v2_fallback_used": bool(payload.get("aligner_v2_fallback_used", False)),
            "future_aligner_window_size": _int(payload.get("future_aligner_window_size")) or 1,
            "future_aligner_sequence_applied": bool(payload.get("future_aligner_sequence_applied", False)),
            "future_aligner_sequence_score": _rounded_float(payload.get("future_aligner_sequence_score")) or 0.0,
            "future_aligner_order_penalty": _rounded_float(payload.get("future_aligner_order_penalty")) or 0.0,
            "future_aligner_provenance_applied": bool(payload.get("future_aligner_provenance_applied", False)),
            "future_aligner_provenance_summary": _text(payload.get("future_aligner_provenance_summary")) or "",
            "aligner_v2_notes": _text_list(payload.get("aligner_v2_notes")),
        },
        "segment": {
            "start_sec": round(float(segment.time_range.start_sec), 3),
            "end_sec": round(float(segment.time_range.end_sec), 3),
            "duration_sec": round(max(0.0, float(segment.time_range.end_sec) - float(segment.time_range.start_sec)), 3),
        },
        "score_hint_binding": {
            "current_note_id": _text(payload.get("effective_score_hint_note_id")),
            "source_note_id": _text(payload.get("source_score_hint_note_id")),
            "source_note_identity_match": payload.get("source_score_hint_note_identity_match"),
            "source_note_binding_strategy": _text(payload.get("source_score_hint_binding_strategy")),
            "effective_note_id": _text(payload.get("effective_score_hint_note_id")),
            "effective_note_identity_match": payload.get("effective_score_hint_note_identity_match"),
            "effective_note_binding_strategy": _text(payload.get("effective_score_hint_binding_strategy")),
            "ownership_binding_adjustment": _mapping(payload.get("ownership_binding_adjustment")),
        },
        "acoustic": {
            "acoustic_evidence_applied": bool(payload.get("acoustic_evidence_applied", False)),
            "acoustic_evidence_summary": _text(payload.get("acoustic_evidence_summary")) or "",
            "boundary_uncertainty": _mapping(payload.get("boundary_uncertainty")),
            "evidence_mode": None,
        },
        "note_span": {
            "part_aware_note_span_used": bool(payload.get("part_aware_note_span_used", False)),
            "note_span_evidence_score": _rounded_float(payload.get("note_span_evidence_score")) or 0.0,
            "note_span_overlap_ratio": _rounded_float(payload.get("note_span_overlap_ratio")) or 0.0,
            "note_span_interior_ratio": _rounded_float(payload.get("note_span_interior_ratio")) or 0.0,
            "note_span_anchor_count": _int(payload.get("note_span_anchor_count")) or 0,
            "note_span_summary": _text(payload.get("note_span_summary")) or "",
        },
        "competition": {
            "summary": _text(payload.get("alignment_competition_summary")) or "",
            "guardrails": [],
            "phrase_boundary_mode": competition_mode,
            "phrase_boundary_guarded": False,
            "phrase_window_pressure": None,
            "note_ownership_conflict": bool(payload.get("note_ownership_conflict", False)),
            "ownership_conflict_gap": _rounded_float(payload.get("ownership_conflict_gap")),
            "ownership_conflict_threshold": _rounded_float(payload.get("ownership_conflict_threshold")),
            "ownership_stability_margin": _rounded_float(payload.get("ownership_stability_margin")),
        },
        "ownership": {
            "summary": _text(payload.get("ownership_conflict_summary")) or "",
            "candidates": _dict_list(payload.get("ownership_conflict_candidates")),
            "resolution_basis": _text(payload.get("ownership_resolution_basis")) or "",
            "boundary_conflict_likely": bool(payload.get("boundary_conflict_likely", False)),
        },
        "selective_release": {
            "safety": _mapping(payload.get("selective_release_safety")),
            "explanation": _text(payload.get("release_explanation")) or "",
            "risk_summary": _text(payload.get("release_risk_summary")) or "",
        },
        "phrase_boundary": {
            "mode": competition_mode,
            "guarded": False,
            "candidate_debug": _mapping(payload.get("phrase_boundary_candidate_debug")),
            "blocker_summary": _text(payload.get("phrase_boundary_blocker_summary")) or "",
        },
    }


def _current_candidate(
    *,
    current_note_id: str | None,
    candidate_scores: list[FutureAlignerCandidateScore],
) -> FutureAlignerCandidateScore | None:
    if current_note_id is None:
        return None
    for candidate in candidate_scores:
        if candidate.note_id == current_note_id:
            return candidate
    return None


def _acoustic_metric(
    acoustic: FutureAlignerAcousticEvidence | None,
    field_name: str,
) -> float:
    if acoustic is None:
        return 0.0
    return float(getattr(acoustic, field_name, 0.0) or 0.0)


def _text(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = _text(entry)
        if text is not None:
            items.append(text)
    return items


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): value[key] for key in value}


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for entry in value:
        if isinstance(entry, dict):
            items.append({str(key): entry[key] for key in entry})
    return items


def _float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rounded_float(value: object) -> float | None:
    numeric = _float(value)
    return round(numeric, 3) if numeric is not None else None


def _int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None

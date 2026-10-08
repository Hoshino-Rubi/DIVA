from __future__ import annotations

from typing import Any

from any2xsampa.adapters.recognizer_acoustic_evidence import (
    candidate_entropy as _candidate_entropy_helper,
    candidate_logit_map as _candidate_logit_map_helper,
    normalize_candidate_evidence as _normalize_candidate_evidence_helper,
)
from any2xsampa.adapters.recognizer_segment_evidence import (
    build_adapter_diagnostics as _build_adapter_diagnostics_helper,
    derive_logit_frames as _derive_logit_frames_helper,
    logit_summary as _logit_summary_helper,
    posterior_summary as _posterior_summary_helper,
)
from any2xsampa.adapters.recognizer_support import (
    _clamp_unit_interval,
    _default_articulatory_features,
    _default_continuation_bias,
    _normalize_bool,
    _normalize_unit_interval,
    _probability_to_logit,
)
from any2xsampa.core.models import AcousticEvidence, AcousticEvidenceCandidate, PhoneCandidate


def resolve_candidate_evidence(
    *,
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    candidate_evidence: list[AcousticEvidenceCandidate] | None = None,
    candidate_logits: dict[str, float] | None = None,
) -> list[AcousticEvidenceCandidate]:
    if candidate_evidence:
        return list(candidate_evidence)
    posterior = metadata.get("posterior")
    if not isinstance(posterior, list):
        return []
    evidence: list[AcousticEvidenceCandidate] = []
    for item in posterior:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip()
        if not symbol:
            continue
        evidence.append(
            AcousticEvidenceCandidate(
                symbol=symbol,
                confidence=float(item.get("confidence", 0.0)),
                articulatory_features=_default_articulatory_features(symbol),
                logit=(candidate_logits or {}).get(symbol),
            )
        )
    return evidence


def derive_primary_acoustic_metrics(
    *,
    top_confidence: float,
    margin: float,
    candidate_entropy: float,
    frame_count: int,
    duration_sec: float,
    posterior_summary: dict[str, Any],
    logit_summary: dict[str, Any],
    acoustic_overrides: dict[str, Any] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    acoustic_map = dict(acoustic_overrides or {})
    metadata_map = dict(metadata_overrides or {})
    candidate_distribution = (
        dict(acoustic_map.get("candidate_distribution"))
        if isinstance(acoustic_map.get("candidate_distribution"), dict)
        else {}
    )
    posterior_vote_ratio = float(
        candidate_distribution.get("frame_vote_ratio", posterior_summary.get("frame_vote_ratio", 1.0)) or 1.0
    )
    posterior_temporal_consistency = float(
        candidate_distribution.get(
            "temporal_consistency",
            posterior_summary.get("temporal_consistency", posterior_vote_ratio),
        )
        or posterior_vote_ratio
    )
    posterior_mean_margin = float(
        candidate_distribution.get("posterior_mean_margin", posterior_summary.get("mean_margin", margin)) or margin
    )
    posterior_interior_margin = float(
        candidate_distribution.get(
            "interior_mean_margin",
            posterior_summary.get("interior_mean_margin", posterior_mean_margin),
        )
        or posterior_mean_margin
    )
    posterior_interior_vote_ratio = float(
        candidate_distribution.get(
            "interior_vote_ratio",
            posterior_summary.get("interior_vote_ratio", posterior_vote_ratio),
        )
        or posterior_vote_ratio
    )
    posterior_entry_margin = float(
        candidate_distribution.get("entry_margin", posterior_summary.get("entry_margin", posterior_mean_margin))
        or posterior_mean_margin
    )
    posterior_exit_margin = float(
        candidate_distribution.get("exit_margin", posterior_summary.get("exit_margin", posterior_mean_margin))
        or posterior_mean_margin
    )
    edge_competition_ratio = float(
        candidate_distribution.get(
            "edge_competition_ratio",
            posterior_summary.get(
                "edge_competition_ratio",
                max(0.0, 1.0 - min(posterior_entry_margin, posterior_exit_margin)),
            ),
        )
        or 0.0
    )
    logit_mean_margin = float(
        candidate_distribution.get("logit_mean_margin", logit_summary.get("mean_margin", margin)) or margin
    )
    logit_interior_margin = float(
        candidate_distribution.get(
            "logit_interior_margin",
            logit_summary.get("interior_mean_margin", logit_mean_margin),
        )
        or logit_mean_margin
    )
    derived_candidate_lead = min(
        1.0,
        (0.38 * margin)
        + (0.22 * (1.0 - candidate_entropy))
        + (0.16 * posterior_vote_ratio)
        + (0.12 * posterior_interior_margin)
        + (0.08 * posterior_interior_vote_ratio)
        + (0.12 * max(0.0, min(logit_mean_margin, 1.0))),
    )
    explicit_candidate_lead = candidate_distribution.get("candidate_lead")
    if explicit_candidate_lead is not None:
        derived_candidate_lead = max(
            derived_candidate_lead,
            _normalize_unit_interval(explicit_candidate_lead, derived_candidate_lead),
        )
    candidate_lead = _normalize_unit_interval(
        acoustic_map.get("candidate_lead", metadata_map.get("candidate_lead")),
        derived_candidate_lead,
    )
    base_timing = 0.46 + (0.34 * margin) + (0.12 if duration_sec >= 0.08 else -0.08)
    timing_reliability = _normalize_unit_interval(
        acoustic_map.get("timing_reliability", metadata_map.get("timing_reliability")),
        min(
            1.0,
            base_timing
            + (0.12 * posterior_temporal_consistency)
            + (0.08 * posterior_vote_ratio)
            + (0.06 * posterior_interior_vote_ratio)
            - (0.08 * edge_competition_ratio),
        ),
    )
    acoustic_confidence = _normalize_unit_interval(
        acoustic_map.get("acoustic_confidence", metadata_map.get("acoustic_confidence", top_confidence)),
        min(
            1.0,
            (0.72 * top_confidence)
            + (0.12 * float(posterior_summary.get("top_phone_mean_posterior", top_confidence) or top_confidence))
            + (0.08 * posterior_vote_ratio)
            + (0.04 * posterior_interior_margin)
            + (0.08 * max(0.0, min(logit_mean_margin, 1.0))),
        ),
    )
    frame_consistency = _normalize_unit_interval(
        acoustic_map.get("frame_consistency", metadata_map.get("frame_consistency")),
        min(
            1.0,
            0.26
            + (0.18 * margin)
            + (0.08 * min(max(1, frame_count), 5))
            + (0.22 * acoustic_confidence)
            + (0.14 * posterior_vote_ratio)
            + (0.12 * posterior_temporal_consistency),
        ),
    )
    evidence_quality = _normalize_unit_interval(
        acoustic_map.get("evidence_quality", metadata_map.get("evidence_quality")),
        (0.32 * acoustic_confidence)
        + (0.2 * timing_reliability)
        + (0.18 * frame_consistency)
        + (0.12 * margin)
        + (0.1 * posterior_mean_margin)
        + (0.04 * posterior_interior_margin)
        + (0.08 * max(0.0, min(logit_mean_margin, 1.0))),
    )
    segment_likelihood = _normalize_unit_interval(
        acoustic_map.get("segment_likelihood", metadata_map.get("segment_likelihood")),
        (0.44 * acoustic_confidence)
        + (0.18 * frame_consistency)
        + (0.14 * margin)
        + (0.14 * posterior_vote_ratio)
        + (0.1 * float(posterior_summary.get("top_phone_mean_posterior", top_confidence) or top_confidence)),
    )
    return {
        "candidate_lead": candidate_lead,
        "timing_reliability": timing_reliability,
        "acoustic_confidence": acoustic_confidence,
        "frame_consistency": frame_consistency,
        "evidence_quality": evidence_quality,
        "segment_likelihood": segment_likelihood,
        "posterior_vote_ratio": posterior_vote_ratio,
        "posterior_temporal_consistency": posterior_temporal_consistency,
        "posterior_mean_margin": posterior_mean_margin,
        "posterior_interior_margin": posterior_interior_margin,
        "posterior_interior_vote_ratio": posterior_interior_vote_ratio,
        "posterior_entry_margin": posterior_entry_margin,
        "posterior_exit_margin": posterior_exit_margin,
        "edge_competition_ratio": edge_competition_ratio,
        "logit_mean_margin": logit_mean_margin,
        "logit_interior_margin": logit_interior_margin,
    }


def build_support_payload_maps(
    *,
    top_phone: str,
    runner_up_phone: str | None,
    metrics: dict[str, float],
    candidate_entropy: float,
    acoustic_overrides: dict[str, Any] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    acoustic_map = dict(acoustic_overrides or {})
    metadata_map = dict(metadata_overrides or {})
    top_features = _default_articulatory_features(top_phone)
    continuation_profile = (
        dict(acoustic_map.get("continuation_profile"))
        if isinstance(acoustic_map.get("continuation_profile"), dict)
        else {}
    )
    boundary_profile = (
        dict(acoustic_map.get("boundary_profile"))
        if isinstance(acoustic_map.get("boundary_profile"), dict)
        else {}
    )
    note_alignment = dict(acoustic_map.get("note_alignment")) if isinstance(acoustic_map.get("note_alignment"), dict) else {}
    ownership = dict(acoustic_map.get("ownership")) if isinstance(acoustic_map.get("ownership"), dict) else {}
    continuation_bias = _normalize_unit_interval(
        note_alignment.get("same_note_continuation_bias", continuation_profile.get("same_note_continuation_bias")),
        _default_continuation_bias(top_phone, top_features),
    )
    score_friendly = _normalize_unit_interval(
        note_alignment.get("score_alignment_friendly", continuation_profile.get("score_alignment_friendly")),
        min(1.0, 0.45 + (0.35 * continuation_bias) + (0.25 * (1.0 - metrics["timing_reliability"]))),
    )
    weak_overlap_support = _normalize_unit_interval(
        note_alignment.get("weak_overlap_support", continuation_profile.get("weak_overlap_support")),
        min(1.0, 0.2 + (0.55 * continuation_bias) + (0.25 * (1.0 - metrics["acoustic_confidence"]))),
    )
    boundary_anchor_strength = _normalize_unit_interval(
        note_alignment.get("boundary_anchor_strength", boundary_profile.get("boundary_anchor_strength")),
        min(1.0, 0.18 + (0.48 * metrics["timing_reliability"]) + (0.18 * metrics["frame_consistency"])),
    )
    supports_note_span_extension = _normalize_bool(
        note_alignment.get("supports_note_span_extension", continuation_profile.get("supports_note_span_extension")),
        continuation_bias >= 0.72 and (weak_overlap_support >= 0.58 or metrics["timing_reliability"] <= 0.6),
    )
    boundary_uncertainty_input = (
        dict(acoustic_map.get("boundary_uncertainty"))
        if isinstance(acoustic_map.get("boundary_uncertainty"), dict)
        else {}
    )
    start_ms = float(boundary_uncertainty_input.get("start_ms", max(12.0, (1.0 - metrics["timing_reliability"]) * 95.0)))
    end_ms = float(boundary_uncertainty_input.get("end_ms", max(12.0, (1.0 - metrics["timing_reliability"]) * 95.0)))
    boundary_uncertainty = {
        "start_ms": round(max(0.0, start_ms), 3),
        "end_ms": round(max(0.0, end_ms), 3),
        "window_ms": round(max(0.0, start_ms + end_ms), 3),
        "severity": round(max(0.0, min(1.0, 1.0 - metrics["timing_reliability"])), 3),
        "uncertain": bool(boundary_uncertainty_input.get("uncertain", metrics["timing_reliability"] < 0.62)),
    }
    long_note_hold_bias = _normalize_unit_interval(
        note_alignment.get("long_note_hold_bias", continuation_profile.get("long_note_hold_bias")),
        min(
            1.0,
            0.2
            + (0.42 * continuation_bias)
            + (0.16 * score_friendly)
            + (0.12 * weak_overlap_support)
            + (0.12 * metrics["posterior_interior_margin"])
            + (0.08 * metrics["posterior_interior_vote_ratio"]),
        ),
    )
    same_note_stability = _normalize_unit_interval(
        ownership.get("same_note_stability", continuation_profile.get("same_note_stability")),
        min(1.0, 0.16 + (0.28 * continuation_bias) + (0.16 * score_friendly) + (0.12 * metrics["frame_consistency"])),
    )
    weak_overlap_recovery = _normalize_unit_interval(
        ownership.get("weak_overlap_recovery", continuation_profile.get("weak_overlap_recovery")),
        min(1.0, 0.14 + (0.32 * weak_overlap_support) + (0.18 * score_friendly) + (0.14 * float(boundary_uncertainty["uncertain"]))),
    )
    part_note_span_reliability = _normalize_unit_interval(
        ownership.get("part_note_span_reliability", note_alignment.get("part_note_span_reliability")),
        min(1.0, 0.16 + (0.24 * boundary_anchor_strength) + (0.18 * score_friendly) + (0.16 * metrics["evidence_quality"])),
    )
    continuation_friendliness = _normalize_unit_interval(
        note_alignment.get("continuation_friendliness", continuation_profile.get("continuation_friendliness")),
        min(1.0, 0.14 + (0.24 * continuation_bias) + (0.22 * same_note_stability) + (0.14 * score_friendly)),
    )
    weak_overlap_friendliness = _normalize_unit_interval(
        note_alignment.get("weak_overlap_friendliness", continuation_profile.get("weak_overlap_friendliness")),
        min(1.0, 0.16 + (0.24 * weak_overlap_support) + (0.24 * weak_overlap_recovery) + (0.1 * score_friendly)),
    )
    competition_separation = _normalize_unit_interval(
        note_alignment.get("competition_separation", continuation_profile.get("competition_separation")),
        min(1.0, (0.38 * metrics["posterior_mean_margin"]) + (0.24 * metrics["posterior_interior_margin"]) + (0.18 * metrics["posterior_interior_vote_ratio"])),
    )
    interior_hold_strength = _normalize_unit_interval(
        note_alignment.get("interior_hold_strength", continuation_profile.get("interior_hold_strength")),
        min(1.0, (0.34 * metrics["posterior_interior_margin"]) + (0.3 * metrics["posterior_interior_vote_ratio"]) + (0.2 * metrics["posterior_temporal_consistency"])),
    )
    one_sided_continuation_strength = _normalize_unit_interval(
        note_alignment.get("one_sided_continuation_strength", continuation_profile.get("one_sided_continuation_strength")),
        min(1.0, 0.16 + (0.22 * continuation_bias) + (0.18 * weak_overlap_support) + (0.14 * metrics["posterior_interior_margin"])),
    )
    boundary_pressure = _normalize_unit_interval(
        boundary_profile.get("boundary_pressure"),
        min(1.0, (0.28 * metrics["edge_competition_ratio"]) + (0.18 * float(boundary_uncertainty["uncertain"]))),
    )
    reassignment_resistance = _normalize_unit_interval(
        ownership.get("reassignment_resistance", boundary_profile.get("reassignment_resistance")),
        min(1.0, 0.18 + (0.28 * metrics["timing_reliability"]) + (0.24 * (1.0 - metrics["edge_competition_ratio"]))),
    )
    boundary_reassignment_risk = _normalize_unit_interval(
        ownership.get("boundary_reassignment_risk", boundary_profile.get("boundary_reassignment_risk")),
        min(1.0, 0.18 + (0.26 * boundary_pressure) + (0.16 * (1.0 - reassignment_resistance))),
    )
    ownership_conflict_risk = _normalize_unit_interval(
        ownership.get("ownership_conflict_risk", boundary_profile.get("ownership_conflict_risk")),
        min(1.0, 0.12 + (0.26 * candidate_entropy) + (0.18 * boundary_reassignment_risk)),
    )
    boundary_conflict_likelihood = _normalize_unit_interval(
        ownership.get("boundary_conflict_likelihood", boundary_profile.get("boundary_conflict_likelihood")),
        min(1.0, 0.12 + (0.2 * boundary_pressure) + (0.14 * ownership_conflict_risk)),
    )
    anchor_preference = _normalize_unit_interval(
        ownership.get("anchor_preference"),
        min(1.0, 0.18 + (0.38 * boundary_anchor_strength) + (0.18 * part_note_span_reliability)),
    )
    return {
        "boundary_uncertainty": boundary_uncertainty,
        "note_alignment": {
            "same_note_continuation_bias": round(continuation_bias, 3),
            "score_alignment_friendly": round(score_friendly, 3),
            "weak_overlap_support": round(weak_overlap_support, 3),
            "boundary_anchor_strength": round(boundary_anchor_strength, 3),
            "supports_note_span_extension": supports_note_span_extension,
            "long_note_hold_bias": round(long_note_hold_bias, 3),
            "same_note_stability": round(same_note_stability, 3),
            "weak_overlap_recovery": round(weak_overlap_recovery, 3),
            "part_note_span_reliability": round(part_note_span_reliability, 3),
            "continuation_friendliness": round(continuation_friendliness, 3),
            "weak_overlap_friendliness": round(weak_overlap_friendliness, 3),
            "competition_separation": round(competition_separation, 3),
            "interior_hold_strength": round(interior_hold_strength, 3),
            "one_sided_continuation_strength": round(one_sided_continuation_strength, 3),
        },
        "ownership": {
            "same_note_stability": round(same_note_stability, 3),
            "weak_overlap_recovery": round(weak_overlap_recovery, 3),
            "long_note_support": round(long_note_hold_bias, 3),
            "part_note_span_reliability": round(part_note_span_reliability, 3),
            "reassignment_resistance": round(reassignment_resistance, 3),
            "boundary_reassignment_risk": round(boundary_reassignment_risk, 3),
            "ownership_conflict_risk": round(ownership_conflict_risk, 3),
            "anchor_preference": round(anchor_preference, 3),
            "boundary_conflict_likelihood": round(boundary_conflict_likelihood, 3),
            "candidate_lead": round(metrics["candidate_lead"], 3),
            "competition_separation": round(competition_separation, 3),
            "interior_hold_strength": round(interior_hold_strength, 3),
            "one_sided_continuation_strength": round(one_sided_continuation_strength, 3),
            "boundary_pressure": round(boundary_pressure, 3),
        },
        "candidate_distribution": {
            "top_phone": top_phone,
            "runner_up_phone": runner_up_phone,
            "candidate_lead": round(metrics["candidate_lead"], 3),
            "candidate_entropy": round(candidate_entropy, 3),
            "frame_vote_ratio": round(metrics["posterior_vote_ratio"], 3),
            "temporal_consistency": round(metrics["posterior_temporal_consistency"], 3),
            "posterior_mean_margin": round(metrics["posterior_mean_margin"], 3),
            "logit_mean_margin": round(metrics["logit_mean_margin"], 3),
            "interior_mean_margin": round(metrics["posterior_interior_margin"], 3),
            "interior_vote_ratio": round(metrics["posterior_interior_vote_ratio"], 3),
            "entry_margin": round(metrics["posterior_entry_margin"], 3),
            "exit_margin": round(metrics["posterior_exit_margin"], 3),
            "edge_competition_ratio": round(metrics["edge_competition_ratio"], 3),
            "interior_hold_strength": round(interior_hold_strength, 3),
            "competition_separation": round(competition_separation, 3),
        },
        "continuation_profile": {
            "same_note_continuation_bias": round(continuation_bias, 3),
            "same_note_stability": round(same_note_stability, 3),
            "continuation_friendliness": round(continuation_friendliness, 3),
            "score_alignment_friendly": round(score_friendly, 3),
            "weak_overlap_support": round(weak_overlap_support, 3),
            "weak_overlap_recovery": round(weak_overlap_recovery, 3),
            "weak_overlap_friendliness": round(weak_overlap_friendliness, 3),
            "long_note_hold_bias": round(long_note_hold_bias, 3),
            "supports_note_span_extension": supports_note_span_extension,
            "interior_hold_strength": round(interior_hold_strength, 3),
            "competition_separation": round(competition_separation, 3),
            "one_sided_continuation_strength": round(one_sided_continuation_strength, 3),
        },
        "boundary_profile": {
            "boundary_uncertain": bool(boundary_uncertainty["uncertain"]),
            "boundary_anchor_strength": round(boundary_anchor_strength, 3),
            "boundary_conflict_likelihood": round(boundary_conflict_likelihood, 3),
            "boundary_reassignment_risk": round(boundary_reassignment_risk, 3),
            "ownership_conflict_risk": round(ownership_conflict_risk, 3),
            "reassignment_resistance": round(reassignment_resistance, 3),
            "boundary_pressure": round(boundary_pressure, 3),
        },
    }


def calibrate_support_payload_maps(
    *,
    payload_maps: dict[str, dict[str, Any]],
    metrics: dict[str, float],
    candidate_margin: float,
    candidate_entropy: float,
    source_mode: str,
    metadata_overrides: dict[str, Any] | None = None,
    adapter_overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    metadata_map = dict(metadata_overrides or {})
    adapter_map = dict(adapter_overrides or {})
    note_alignment = dict(payload_maps.get("note_alignment", {}))
    ownership = dict(payload_maps.get("ownership", {}))
    candidate_distribution = dict(payload_maps.get("candidate_distribution", {}))
    continuation_profile = dict(payload_maps.get("continuation_profile", {}))
    boundary_profile = dict(payload_maps.get("boundary_profile", {}))
    boundary_uncertainty = dict(payload_maps.get("boundary_uncertainty", {}))
    acoustic_evidence = metadata_map.get("acoustic_evidence")
    nested_acoustic_map = (
        dict(acoustic_evidence) if isinstance(acoustic_evidence, dict) else {}
    )
    nested_boundary_profile = (
        dict(nested_acoustic_map.get("boundary_profile"))
        if isinstance(nested_acoustic_map.get("boundary_profile"), dict)
        else {}
    )

    backend_kind = str(
        adapter_map.get(
            "backend_kind",
            metadata_map.get("real_backend", metadata_map.get("recognizer_backend_family", source_mode)),
        )
    )
    approximation_only = bool(
        adapter_map.get(
            "approximation_only",
            source_mode == "real" and str(metadata_map.get("real_backend", "")).lower() == "heuristic-dsp",
        )
    )
    timing_reliability = float(metrics["timing_reliability"])
    frame_consistency = float(metrics["frame_consistency"])
    edge_competition_ratio = float(metrics["edge_competition_ratio"])
    posterior_interior_margin = float(metrics["posterior_interior_margin"])
    posterior_interior_vote_ratio = float(metrics["posterior_interior_vote_ratio"])
    posterior_temporal_consistency = float(metrics["posterior_temporal_consistency"])
    logit_interior_margin = float(metrics["logit_interior_margin"])
    candidate_lead = float(metrics["candidate_lead"])

    continuation_bias = _normalize_unit_interval(
        note_alignment.get("same_note_continuation_bias"),
        0.0,
    )
    score_friendly = _normalize_unit_interval(note_alignment.get("score_alignment_friendly"), 0.0)
    weak_overlap_support = _normalize_unit_interval(note_alignment.get("weak_overlap_support"), 0.0)
    boundary_anchor_strength = _normalize_unit_interval(note_alignment.get("boundary_anchor_strength"), 0.0)
    long_note_hold_bias = _normalize_unit_interval(note_alignment.get("long_note_hold_bias"), 0.0)
    same_note_stability = _normalize_unit_interval(note_alignment.get("same_note_stability"), 0.0)
    weak_overlap_recovery = _normalize_unit_interval(note_alignment.get("weak_overlap_recovery"), 0.0)
    part_note_span_reliability = _normalize_unit_interval(
        note_alignment.get("part_note_span_reliability"),
        0.0,
    )
    continuation_friendliness = _normalize_unit_interval(
        note_alignment.get("continuation_friendliness"),
        0.0,
    )
    weak_overlap_friendliness = _normalize_unit_interval(
        note_alignment.get("weak_overlap_friendliness"),
        0.0,
    )
    competition_separation = _normalize_unit_interval(
        note_alignment.get("competition_separation"),
        0.0,
    )
    interior_hold_strength = _normalize_unit_interval(
        note_alignment.get("interior_hold_strength"),
        0.0,
    )
    one_sided_continuation_strength = _normalize_unit_interval(
        note_alignment.get("one_sided_continuation_strength"),
        0.0,
    )
    boundary_transition_strength = _normalize_unit_interval(
        boundary_profile.get("boundary_transition_strength"),
        min(
            1.0,
            0.12
            + (0.32 * timing_reliability)
            + (0.22 * boundary_anchor_strength)
            + (0.16 * candidate_margin)
            + (0.14 * edge_competition_ratio)
            + (0.1 * max(0.0, 1.0 - min(metrics["posterior_entry_margin"], metrics["posterior_exit_margin"])))
            + (0.18 * (1.0 - continuation_bias)),
        ),
    )
    dominant_candidate_stability = _normalize_unit_interval(
        metadata_map.get("dominant_candidate_stability"),
        min(
            1.0,
            0.32
            + (0.28 * candidate_margin)
            + (0.2 * metrics["acoustic_confidence"])
            + (0.14 * frame_consistency)
            + (0.06 * (1.0 - candidate_entropy)),
        ),
    )
    boundary_softness = _normalize_unit_interval(
        boundary_profile.get(
            "boundary_softness",
            nested_boundary_profile.get(
                "boundary_softness",
                nested_acoustic_map.get("boundary_softness", metadata_map.get("boundary_softness")),
            ),
        ),
        min(
            1.0,
            0.16
            + (0.32 * float(boundary_uncertainty.get("uncertain", False)))
            + (0.24 * (1.0 - boundary_transition_strength))
            + (0.16 * (1.0 - timing_reliability))
            + (0.12 * edge_competition_ratio)
            + (0.12 * (1.0 - boundary_anchor_strength)),
        ),
    )
    ownership_ambiguity = _normalize_unit_interval(
        metadata_map.get("ownership_ambiguity"),
        min(
            1.0,
            (0.56 * (1.0 - candidate_margin))
            + (0.22 * (1.0 - timing_reliability))
            + (0.22 * (1.0 - frame_consistency)),
        ),
    )
    long_note_hold_bias = _normalize_unit_interval(
        note_alignment.get("long_note_hold_bias", continuation_profile.get("long_note_hold_bias")),
        min(
            1.0,
            max(
                0.0,
                0.2
                + (0.42 * continuation_bias)
                + (0.16 * score_friendly)
                + (0.12 * weak_overlap_support)
                + (0.12 * posterior_interior_margin)
                + (0.08 * posterior_interior_vote_ratio)
                - (0.18 * boundary_transition_strength),
            ),
        ),
    )
    same_note_stability = _normalize_unit_interval(
        ownership.get("same_note_stability", continuation_profile.get("same_note_stability")),
        min(
            1.0,
            0.16
            + (0.28 * continuation_bias)
            + (0.16 * score_friendly)
            + (0.12 * frame_consistency)
            + (0.1 * dominant_candidate_stability)
            - (0.12 * ownership_ambiguity),
        ),
    )
    weak_overlap_recovery = _normalize_unit_interval(
        ownership.get("weak_overlap_recovery", continuation_profile.get("weak_overlap_recovery")),
        min(
            1.0,
            0.14
            + (0.32 * weak_overlap_support)
            + (0.18 * score_friendly)
            + (0.14 * boundary_softness)
            + (0.08 * posterior_interior_vote_ratio)
            + (0.08 * same_note_stability),
        ),
    )
    part_note_span_reliability = _normalize_unit_interval(
        ownership.get("part_note_span_reliability", note_alignment.get("part_note_span_reliability")),
        min(
            1.0,
            0.16
            + (0.24 * boundary_anchor_strength)
            + (0.18 * score_friendly)
            + (0.16 * metrics["evidence_quality"])
            + (0.08 * long_note_hold_bias)
            - (0.12 * ownership_ambiguity),
        ),
    )
    reassignment_resistance = _normalize_unit_interval(
        ownership.get("reassignment_resistance", boundary_profile.get("reassignment_resistance")),
        min(
            1.0,
            0.18
            + (0.28 * timing_reliability)
            + (0.24 * boundary_transition_strength)
            + (0.18 * dominant_candidate_stability)
            - (0.12 * boundary_softness),
        ),
    )
    boundary_reassignment_risk = _normalize_unit_interval(
        ownership.get("boundary_reassignment_risk", boundary_profile.get("boundary_reassignment_risk")),
        min(
            1.0,
            0.18
            + (0.26 * boundary_transition_strength)
            + (0.2 * boundary_softness)
            + (0.16 * ownership_ambiguity)
            - (0.08 * reassignment_resistance),
        ),
    )
    ownership_conflict_risk = _normalize_unit_interval(
        ownership.get("ownership_conflict_risk", boundary_profile.get("ownership_conflict_risk")),
        min(
            1.0,
            0.12
            + (0.26 * ownership_ambiguity)
            + (0.18 * candidate_entropy)
            + (0.18 * boundary_reassignment_risk)
            + (0.14 * (1.0 - candidate_margin)),
        ),
    )
    continuation_friendliness = _normalize_unit_interval(
        note_alignment.get("continuation_friendliness", continuation_profile.get("continuation_friendliness")),
        min(
            1.0,
            0.14
            + (0.24 * continuation_bias)
            + (0.22 * same_note_stability)
            + (0.14 * score_friendly)
            + (0.14 * long_note_hold_bias)
            + (0.08 * posterior_temporal_consistency)
            + (0.06 * posterior_interior_vote_ratio)
            + (0.06 * candidate_lead)
            - (0.1 * boundary_transition_strength),
        ),
    )
    weak_overlap_friendliness = _normalize_unit_interval(
        note_alignment.get("weak_overlap_friendliness", continuation_profile.get("weak_overlap_friendliness")),
        min(
            1.0,
            0.16
            + (0.24 * weak_overlap_support)
            + (0.24 * weak_overlap_recovery)
            + (0.12 * boundary_softness)
            + (0.1 * score_friendly)
            + (0.08 * long_note_hold_bias)
            + (0.06 * float(boundary_uncertainty.get("uncertain", False))),
        ),
    )
    boundary_conflict_likelihood = _normalize_unit_interval(
        ownership.get(
            "boundary_conflict_likelihood",
            boundary_profile.get(
                "boundary_conflict_likelihood",
                metadata_map.get("boundary_conflict_likelihood"),
            ),
        ),
        min(
            1.0,
            0.12
            + (0.2 * boundary_transition_strength)
            + (0.16 * boundary_softness)
            + (0.14 * ownership_ambiguity)
            + (0.14 * boundary_reassignment_risk)
            + (0.14 * ownership_conflict_risk)
            + (0.1 * edge_competition_ratio)
            + (0.06 * max(0.0, 1.0 - min(logit_interior_margin, 1.0)))
            + (0.08 * float(boundary_uncertainty.get("uncertain", False)))
            - (0.08 * reassignment_resistance),
        ),
    )
    interior_hold_strength = _clamp_unit_interval(
        (0.34 * posterior_interior_margin)
        + (0.3 * posterior_interior_vote_ratio)
        + (0.2 * posterior_temporal_consistency)
        + (0.16 * max(0.0, 1.0 - edge_competition_ratio))
    )
    boundary_pressure = _clamp_unit_interval(
        (0.28 * edge_competition_ratio)
        + (0.24 * max(0.0, 1.0 - min(metrics["posterior_entry_margin"], metrics["posterior_exit_margin"])))
        + (0.18 * boundary_softness)
        + (0.14 * ownership_ambiguity)
        + (0.1 * float(boundary_uncertainty.get("uncertain", False)))
    )
    competition_separation = _clamp_unit_interval(
        (0.38 * candidate_margin)
        + (0.24 * posterior_interior_margin)
        + (0.18 * posterior_interior_vote_ratio)
        + (0.12 * max(0.0, min(logit_interior_margin, 1.0)))
        + (0.08 * (1.0 - candidate_entropy))
    )
    one_sided_continuation_strength = _normalize_unit_interval(
        note_alignment.get("one_sided_continuation_strength", continuation_profile.get("one_sided_continuation_strength")),
        min(
            1.0,
            0.16
            + (0.22 * continuation_bias)
            + (0.18 * weak_overlap_support)
            + (0.14 * posterior_interior_margin)
            + (0.1 * posterior_interior_vote_ratio)
            + (0.08 * competition_separation)
            - (0.14 * boundary_pressure),
        ),
    )
    if backend_kind == "onnx-espeak" and not approximation_only:
        competition_separation = _clamp_unit_interval(
            competition_separation
            + (0.12 * interior_hold_strength)
            + (0.08 * weak_overlap_support)
            + (0.06 * continuation_bias)
        )
        interior_hold_strength = _clamp_unit_interval(
            interior_hold_strength
            + (0.1 * competition_separation)
            + (0.08 * long_note_hold_bias)
            + (0.06 * same_note_stability)
        )
        candidate_lead = _clamp_unit_interval(
            candidate_lead + (0.06 * competition_separation) + (0.05 * interior_hold_strength)
        )
        continuation_bias = _clamp_unit_interval(
            continuation_bias + (0.06 * interior_hold_strength) + (0.02 * candidate_lead)
        )
        score_friendly = _clamp_unit_interval(score_friendly + (0.04 * interior_hold_strength))
        weak_overlap_support = _clamp_unit_interval(
            weak_overlap_support + (0.06 * interior_hold_strength) + (0.03 * competition_separation)
        )
        long_note_hold_bias = _clamp_unit_interval(
            long_note_hold_bias + (0.1 * interior_hold_strength) + (0.03 * competition_separation)
        )
        same_note_stability = _clamp_unit_interval(
            same_note_stability + (0.06 * interior_hold_strength) + (0.03 * competition_separation)
        )
        weak_overlap_recovery = _clamp_unit_interval(
            weak_overlap_recovery + (0.06 * interior_hold_strength) + (0.02 * boundary_softness)
        )
        part_note_span_reliability = _clamp_unit_interval(
            part_note_span_reliability + (0.05 * interior_hold_strength)
        )
        continuation_friendliness = _clamp_unit_interval(
            continuation_friendliness + (0.06 * interior_hold_strength) + (0.02 * candidate_lead)
        )
        weak_overlap_friendliness = _clamp_unit_interval(
            weak_overlap_friendliness + (0.05 * interior_hold_strength) + (0.02 * boundary_softness)
        )
        one_sided_continuation_strength = _clamp_unit_interval(
            one_sided_continuation_strength
            + (0.08 * competition_separation)
            + (0.06 * interior_hold_strength)
            + (0.04 * candidate_lead)
            - (0.04 * boundary_pressure)
        )
        reassignment_resistance = _clamp_unit_interval(
            reassignment_resistance + (0.03 * competition_separation)
        )
        boundary_transition_strength = _clamp_unit_interval(
            boundary_transition_strength + (0.05 * boundary_pressure)
        )
        boundary_reassignment_risk = _clamp_unit_interval(
            boundary_reassignment_risk + (0.04 * boundary_pressure)
        )
        ownership_conflict_risk = _clamp_unit_interval(
            ownership_conflict_risk + (0.04 * boundary_pressure)
        )
        boundary_conflict_likelihood = _clamp_unit_interval(
            boundary_conflict_likelihood + (0.06 * boundary_pressure)
        )
        if boundary_pressure <= 0.68:
            boundary_pressure = _clamp_unit_interval(
                boundary_pressure - (0.04 * competition_separation)
            )
    elif approximation_only:
        conservative_penalty = _clamp_unit_interval(
            (0.3 * edge_competition_ratio)
            + (0.24 * (1.0 - posterior_interior_margin))
            + (0.22 * (1.0 - posterior_interior_vote_ratio))
            + (0.12 * (1.0 - competition_separation))
        )
        competition_separation = _clamp_unit_interval(
            competition_separation - (0.08 * conservative_penalty)
        )
        interior_hold_strength = _clamp_unit_interval(
            interior_hold_strength - (0.08 * conservative_penalty)
        )
        candidate_lead = _clamp_unit_interval(candidate_lead - (0.05 * conservative_penalty))
        continuation_bias = _clamp_unit_interval(continuation_bias - (0.05 * conservative_penalty))
        score_friendly = _clamp_unit_interval(score_friendly - (0.04 * conservative_penalty))
        weak_overlap_support = _clamp_unit_interval(weak_overlap_support - (0.06 * conservative_penalty))
        long_note_hold_bias = _clamp_unit_interval(long_note_hold_bias - (0.08 * conservative_penalty))
        same_note_stability = _clamp_unit_interval(same_note_stability - (0.05 * conservative_penalty))
        weak_overlap_recovery = _clamp_unit_interval(weak_overlap_recovery - (0.05 * conservative_penalty))
        part_note_span_reliability = _clamp_unit_interval(
            part_note_span_reliability - (0.04 * conservative_penalty)
        )
        continuation_friendliness = _clamp_unit_interval(
            continuation_friendliness - (0.05 * conservative_penalty)
        )
        weak_overlap_friendliness = _clamp_unit_interval(
            weak_overlap_friendliness - (0.05 * conservative_penalty)
        )
        one_sided_continuation_strength = _clamp_unit_interval(
            one_sided_continuation_strength - (0.06 * conservative_penalty)
        )
        ownership_ambiguity = _clamp_unit_interval(ownership_ambiguity + (0.05 * conservative_penalty))
        boundary_softness = _clamp_unit_interval(boundary_softness + (0.03 * conservative_penalty))
        boundary_reassignment_risk = _clamp_unit_interval(
            boundary_reassignment_risk + (0.04 * conservative_penalty)
        )
        ownership_conflict_risk = _clamp_unit_interval(
            ownership_conflict_risk + (0.05 * conservative_penalty)
        )
        boundary_conflict_likelihood = _clamp_unit_interval(
            boundary_conflict_likelihood + (0.04 * conservative_penalty)
        )
    anchor_preference = _normalize_unit_interval(
        ownership.get("anchor_preference"),
        min(1.0, 0.18 + (0.38 * boundary_anchor_strength) + (0.18 * part_note_span_reliability)),
    )

    note_alignment.update(
        {
            "same_note_continuation_bias": round(continuation_bias, 3),
            "same_note_stability": round(same_note_stability, 3),
            "score_alignment_friendly": round(score_friendly, 3),
            "weak_overlap_support": round(weak_overlap_support, 3),
            "weak_overlap_recovery": round(weak_overlap_recovery, 3),
            "boundary_anchor_strength": round(boundary_anchor_strength, 3),
            "part_note_span_reliability": round(part_note_span_reliability, 3),
            "long_note_hold_bias": round(long_note_hold_bias, 3),
            "continuation_friendliness": round(continuation_friendliness, 3),
            "weak_overlap_friendliness": round(weak_overlap_friendliness, 3),
            "competition_separation": round(competition_separation, 3),
            "interior_hold_strength": round(interior_hold_strength, 3),
            "one_sided_continuation_strength": round(one_sided_continuation_strength, 3),
        }
    )
    ownership.update(
        {
            "same_note_stability": round(same_note_stability, 3),
            "weak_overlap_recovery": round(weak_overlap_recovery, 3),
            "long_note_support": round(long_note_hold_bias, 3),
            "part_note_span_reliability": round(part_note_span_reliability, 3),
            "reassignment_resistance": round(reassignment_resistance, 3),
            "boundary_reassignment_risk": round(boundary_reassignment_risk, 3),
            "ownership_conflict_risk": round(ownership_conflict_risk, 3),
            "anchor_preference": round(anchor_preference, 3),
            "boundary_conflict_likelihood": round(boundary_conflict_likelihood, 3),
            "candidate_lead": round(candidate_lead, 3),
            "competition_separation": round(competition_separation, 3),
            "interior_hold_strength": round(interior_hold_strength, 3),
            "one_sided_continuation_strength": round(one_sided_continuation_strength, 3),
            "boundary_pressure": round(boundary_pressure, 3),
        }
    )
    candidate_distribution.update(
        {
            "top_vs_runner_up_confidence_margin": round(candidate_margin, 3),
            "candidate_lead": round(candidate_lead, 3),
            "candidate_entropy": round(candidate_entropy, 3),
            "competition_separation": round(competition_separation, 3),
            "interior_hold_strength": round(interior_hold_strength, 3),
        }
    )
    continuation_profile.update(
        {
            "same_note_continuation_bias": round(continuation_bias, 3),
            "same_note_stability": round(same_note_stability, 3),
            "continuation_friendliness": round(continuation_friendliness, 3),
            "score_alignment_friendly": round(score_friendly, 3),
            "weak_overlap_support": round(weak_overlap_support, 3),
            "weak_overlap_recovery": round(weak_overlap_recovery, 3),
            "weak_overlap_friendliness": round(weak_overlap_friendliness, 3),
            "long_note_hold_bias": round(long_note_hold_bias, 3),
            "interior_hold_strength": round(interior_hold_strength, 3),
            "competition_separation": round(competition_separation, 3),
            "one_sided_continuation_strength": round(one_sided_continuation_strength, 3),
        }
    )
    boundary_profile.update(
        {
            "boundary_uncertain": bool(boundary_uncertainty.get("uncertain", False)),
            "boundary_transition_strength": round(boundary_transition_strength, 3),
            "boundary_softness": round(boundary_softness, 3),
            "boundary_anchor_strength": round(boundary_anchor_strength, 3),
            "boundary_conflict_likelihood": round(boundary_conflict_likelihood, 3),
            "boundary_reassignment_risk": round(boundary_reassignment_risk, 3),
            "ownership_conflict_risk": round(ownership_conflict_risk, 3),
            "reassignment_resistance": round(reassignment_resistance, 3),
            "boundary_pressure": round(boundary_pressure, 3),
        }
    )
    calibrated_metrics = {
        **metrics,
        "candidate_lead": candidate_lead,
        "boundary_transition_strength": boundary_transition_strength,
        "dominant_candidate_stability": dominant_candidate_stability,
        "boundary_softness": boundary_softness,
        "ownership_ambiguity": ownership_ambiguity,
        "competition_separation": competition_separation,
        "interior_hold_strength": interior_hold_strength,
        "one_sided_continuation_strength": one_sided_continuation_strength,
        "boundary_pressure": boundary_pressure,
        "boundary_conflict_likelihood": boundary_conflict_likelihood,
        "reassignment_resistance": reassignment_resistance,
        "boundary_reassignment_risk": boundary_reassignment_risk,
        "ownership_conflict_risk": ownership_conflict_risk,
    }
    return {
        "boundary_uncertainty": boundary_uncertainty,
        "note_alignment": note_alignment,
        "ownership": ownership,
        "candidate_distribution": candidate_distribution,
        "continuation_profile": continuation_profile,
        "boundary_profile": boundary_profile,
        "calibrated_metrics": calibrated_metrics,
        "adapter_calibration": {
            "backend_kind": backend_kind,
            "approximation_only": approximation_only,
        },
    }


def build_acoustic_evidence(
    *,
    source_mode: str,
    top_phone: str,
    duration_sec: float,
    candidate_margin: float,
    candidate_count: int,
    candidate_entropy: float,
    primary_metrics: dict[str, float],
    candidate_evidence: list[AcousticEvidenceCandidate],
    payload_maps: dict[str, dict[str, Any]],
    posterior_summary: dict[str, Any],
    logit_summary: dict[str, Any],
    adapter_diagnostics: dict[str, Any],
    advanced_metrics: dict[str, float] | None = None,
    notes: list[str] | None = None,
) -> AcousticEvidence:
    advanced_map = dict(advanced_metrics or {})
    boundary_profile = dict(payload_maps["boundary_profile"])
    boundary_profile.setdefault(
        "boundary_transition_strength",
        round(
            advanced_map.get(
                "boundary_transition_strength",
                1.0 - primary_metrics["edge_competition_ratio"],
            ),
            3,
        ),
    )
    boundary_profile.setdefault(
        "boundary_softness",
        round(
            advanced_map.get(
                "boundary_softness",
                max(0.0, min(1.0, 1.0 - primary_metrics["timing_reliability"])),
            ),
            3,
        ),
    )
    return AcousticEvidence(
        schema_version="acoustic_evidence_v1",
        source_mode=source_mode,
        top_phone=top_phone,
        acoustic_confidence=primary_metrics["acoustic_confidence"],
        timing_reliability=primary_metrics["timing_reliability"],
        duration_sec=duration_sec,
        candidate_margin=candidate_margin,
        candidate_count=candidate_count,
        candidate_entropy=candidate_entropy,
        frame_consistency=primary_metrics["frame_consistency"],
        evidence_quality=primary_metrics["evidence_quality"],
        segment_likelihood=primary_metrics["segment_likelihood"],
        dominant_candidate_stability=advanced_map.get(
            "dominant_candidate_stability",
            payload_maps["ownership"]["same_note_stability"],
        ),
        boundary_transition_strength=boundary_profile["boundary_transition_strength"],
        boundary_softness=boundary_profile["boundary_softness"],
        ownership_ambiguity=advanced_map.get(
            "ownership_ambiguity",
            payload_maps["ownership"]["ownership_conflict_risk"],
        ),
        candidate_phones=list(candidate_evidence),
        boundary_uncertainty=dict(payload_maps["boundary_uncertainty"]),
        note_alignment=dict(payload_maps["note_alignment"]),
        ownership=dict(payload_maps["ownership"]),
        candidate_distribution=dict(payload_maps["candidate_distribution"]),
        continuation_profile=dict(payload_maps["continuation_profile"]),
        boundary_profile=boundary_profile,
        adapter_diagnostics=dict(adapter_diagnostics),
        posterior_summary=dict(posterior_summary),
        logit_summary=dict(logit_summary),
        notes=list(notes or []),
    )


def build_acoustic_evidence_from_components(
    *,
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    source_mode: str,
    duration_sec: float,
    posterior_summary: dict[str, Any],
    logit_summary: dict[str, Any],
    adapter_diagnostics: dict[str, Any] | None = None,
    candidate_evidence: list[AcousticEvidenceCandidate] | None = None,
    candidate_logits: dict[str, float] | None = None,
    notes: list[str] | None = None,
) -> AcousticEvidence:
    if not candidates:
        raise ValueError("candidates must not be empty")
    resolved_evidence = resolve_candidate_evidence(
        candidates=candidates,
        metadata=metadata,
        candidate_evidence=candidate_evidence,
        candidate_logits=candidate_logits,
    )
    top_candidate = candidates[0]
    runner_up_phone = candidates[1].symbol if len(candidates) > 1 else None
    margin = max(
        0.0,
        min(
            1.0,
            top_candidate.confidence - (candidates[1].confidence if len(candidates) > 1 else 0.0),
        ),
    )
    acoustic_overrides = (
        metadata.get("acoustic_evidence")
        if isinstance(metadata.get("acoustic_evidence"), dict)
        else {}
    )
    adapter_diagnostics_override = acoustic_overrides.get("adapter_diagnostics")
    adapter_diagnostics_map = {
        **dict(adapter_diagnostics or {}),
        **(
            dict(adapter_diagnostics_override)
            if isinstance(adapter_diagnostics_override, dict)
            else {}
        ),
    }
    candidate_entropy_value = _candidate_entropy_helper(candidates)
    primary_metrics = derive_primary_acoustic_metrics(
        top_confidence=top_candidate.confidence,
        margin=margin,
        candidate_entropy=candidate_entropy_value,
        frame_count=max(1, int(metadata.get("frame_count", 1))),
        duration_sec=duration_sec,
        posterior_summary=posterior_summary,
        logit_summary=logit_summary,
        acoustic_overrides=acoustic_overrides,
        metadata_overrides=metadata,
    )
    payload_maps = build_support_payload_maps(
        top_phone=top_candidate.symbol,
        runner_up_phone=runner_up_phone,
        metrics=primary_metrics,
        candidate_entropy=candidate_entropy_value,
        acoustic_overrides=acoustic_overrides,
        metadata_overrides=metadata,
    )
    calibrated_payload_maps = calibrate_support_payload_maps(
        payload_maps=payload_maps,
        metrics=primary_metrics,
        candidate_margin=margin,
        candidate_entropy=candidate_entropy_value,
        source_mode=source_mode,
        metadata_overrides=metadata,
        adapter_overrides=adapter_diagnostics_map,
    )
    calibrated_metrics = dict(calibrated_payload_maps.get("calibrated_metrics", primary_metrics))
    adapter_calibration = (
        calibrated_payload_maps.get("adapter_calibration")
        if isinstance(calibrated_payload_maps.get("adapter_calibration"), dict)
        else {}
    )
    posterior_source = str(posterior_summary.get("source", "none"))
    logit_source = str(logit_summary.get("source", "none"))
    posterior_available = (
        bool(adapter_diagnostics_map.get("posterior_like_available"))
        if "posterior_like_available" in adapter_diagnostics_map
        else bool(
            posterior_summary.get("available", False)
            and posterior_source not in {"candidate_phones", "none"}
        )
    )
    logit_available = (
        bool(adapter_diagnostics_map.get("logit_like_available"))
        if "logit_like_available" in adapter_diagnostics_map
        else bool(
            logit_summary.get("available", False)
            and not logit_source.endswith(":derived_logits")
            and logit_source != "none"
        )
    )
    candidate_aggregation = str(
        adapter_diagnostics_map.get(
            "candidate_aggregation",
            metadata.get("candidate_aggregation", "mean"),
        )
    )
    aggregation = str(
        adapter_diagnostics_map.get(
            "aggregation",
            adapter_diagnostics_map.get("candidate_aggregation", candidate_aggregation),
        )
    )
    resolved_adapter_diagnostics = _build_adapter_diagnostics_helper(
        adapter_kind=str(
            adapter_diagnostics_map.get(
                "adapter_kind",
                metadata.get(
                    "recognizer_backend_family",
                    metadata.get("real_backend", source_mode),
                ),
            )
        ),
        backend_kind=str(
            adapter_calibration.get(
                "backend_kind",
                adapter_diagnostics_map.get(
                    "backend_kind",
                    metadata.get(
                        "real_backend",
                        metadata.get("recognizer_backend_family", source_mode),
                    ),
                ),
            )
        ),
        source=str(
            adapter_diagnostics_map.get(
                "source",
                metadata.get(
                    "source",
                    posterior_summary.get("source", logit_summary.get("source", "candidate_phones")),
                ),
            )
        ),
        aggregation=aggregation,
        candidate_source=str(
            adapter_diagnostics_map.get(
                "candidate_source",
                metadata.get("candidate_source", posterior_summary.get("source", "candidate_phones")),
            )
        ),
        candidate_aggregation=candidate_aggregation,
        posterior_like_available=posterior_available,
        logit_like_available=logit_available,
        approximation_only=bool(
            adapter_calibration.get(
                "approximation_only",
                adapter_diagnostics_map.get(
                    "approximation_only",
                    source_mode == "real" and str(metadata.get("real_backend", "")).lower() == "heuristic-dsp",
                ),
            )
        ),
        evidence_kind=str(
            adapter_diagnostics_map.get(
                "evidence_kind",
                (
                    "posterior_and_logit_adapter_like"
                    if posterior_available and logit_available
                    else "posterior_adapter_like"
                    if posterior_available
                    else "logit_adapter_like"
                    if logit_available
                    else "candidate_confidence_adapter"
                ),
            )
        ),
        posterior_source=posterior_summary.get("source"),
        logit_source=logit_summary.get("source"),
        frame_count=max(
            int(posterior_summary.get("frame_count", 0) or 0),
            int(logit_summary.get("frame_count", 0) or 0),
            max(1, int(metadata.get("frame_count", 1))),
        ),
        frame_hop_sec=adapter_diagnostics_map.get(
            "frame_hop_sec",
            posterior_summary.get("frame_hop_sec", metadata.get("frame_hop_sec")),
        ),
    )
    evidence_notes = list(notes or [])
    if not evidence_notes:
        evidence_notes = [
            text
            for text in acoustic_overrides.get("notes", [])
            if isinstance(text, str) and text.strip()
        ]
    if resolved_adapter_diagnostics["posterior_like_available"] and resolved_adapter_diagnostics["logit_like_available"]:
        evidence_notes.append("posterior/logit adapter-like evidence was available for this segment")
    elif resolved_adapter_diagnostics["posterior_like_available"]:
        evidence_notes.append("posterior-like evidence was available for this segment")
    if (
        resolved_adapter_diagnostics["backend_kind"] == "onnx-espeak"
        and not resolved_adapter_diagnostics["approximation_only"]
    ):
        evidence_notes.append("onnx-espeak calibration emphasized interior hold and boundary pressure evidence")
    elif resolved_adapter_diagnostics["approximation_only"]:
        evidence_notes.append("approximation backend calibration applied conservative ownership damping")
    if "posterior" in metadata and not evidence_notes:
        evidence_notes.append("derived from posterior-like recognizer candidates")
    elif not evidence_notes:
        evidence_notes.append("derived from recognizer top candidates and timing heuristics")
    deduped_notes: list[str] = []
    for note in evidence_notes:
        text = str(note).strip()
        if text and text not in deduped_notes:
            deduped_notes.append(text)
    return build_acoustic_evidence(
        source_mode=source_mode,
        top_phone=top_candidate.symbol,
        duration_sec=duration_sec,
        candidate_margin=margin,
        candidate_count=len(resolved_evidence) or len(candidates),
        candidate_entropy=candidate_entropy_value,
        primary_metrics=calibrated_metrics,
        candidate_evidence=resolved_evidence,
        payload_maps=calibrated_payload_maps,
        posterior_summary=posterior_summary,
        logit_summary=logit_summary,
        adapter_diagnostics=resolved_adapter_diagnostics,
        notes=deduped_notes,
    )


def build_acoustic_evidence_from_metadata(
    *,
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    source_mode: str,
    duration_sec: float,
) -> AcousticEvidence:
    if not candidates:
        raise ValueError("candidates must not be empty")
    top_phone = candidates[0].symbol
    candidate_logits = _candidate_logit_map_helper(
        candidates,
        metadata,
        derive_logit_frames=_derive_logit_frames_helper,
        probability_to_logit=_probability_to_logit,
    )
    candidate_evidence = _normalize_candidate_evidence_helper(
        candidates,
        candidate_logits=candidate_logits,
        default_articulatory_features=_default_articulatory_features,
        probability_to_logit=_probability_to_logit,
    )
    return build_acoustic_evidence_from_components(
        candidates=candidates,
        metadata=metadata,
        source_mode=source_mode,
        duration_sec=duration_sec,
        posterior_summary=_posterior_summary_helper(candidates, metadata, top_phone=top_phone),
        logit_summary=_logit_summary_helper(candidates, metadata, top_phone=top_phone),
        adapter_diagnostics=(
            dict(metadata.get("adapter_diagnostics"))
            if isinstance(metadata.get("adapter_diagnostics"), dict)
            else {}
        ),
        candidate_evidence=candidate_evidence,
        candidate_logits=candidate_logits,
    )


__all__ = [
    "build_acoustic_evidence_from_components",
    "build_acoustic_evidence_from_metadata",
    "build_acoustic_evidence",
    "calibrate_support_payload_maps",
    "build_support_payload_maps",
    "derive_primary_acoustic_metrics",
    "resolve_candidate_evidence",
]

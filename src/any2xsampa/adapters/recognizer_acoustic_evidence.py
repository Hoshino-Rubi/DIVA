from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable

from any2xsampa.core.models import AcousticEvidence, AcousticEvidenceCandidate, PhoneCandidate


def candidate_logit_map(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    *,
    derive_logit_frames: Callable[
        [list[PhoneCandidate], dict[str, Any]],
        tuple[list[dict[str, float]], str, float | None],
    ],
    probability_to_logit: Callable[[float], float],
) -> dict[str, float]:
    logit_frames, _, _ = derive_logit_frames(candidates, metadata)
    aggregated: dict[str, list[float]] = {}
    for frame in logit_frames:
        for symbol, value in frame.items():
            aggregated.setdefault(symbol, []).append(value)
    if aggregated:
        return {symbol: (sum(values) / len(values)) for symbol, values in aggregated.items()}
    return {candidate.symbol: probability_to_logit(candidate.confidence) for candidate in candidates}


def normalize_candidate_evidence(
    candidates: list[PhoneCandidate],
    *,
    candidate_logits: dict[str, float] | None,
    default_articulatory_features: Callable[[str], dict[str, str]],
    probability_to_logit: Callable[[float], float],
) -> list[AcousticEvidenceCandidate]:
    evidence_candidates: list[AcousticEvidenceCandidate] = []
    top_confidence = max((candidate.confidence for candidate in candidates), default=0.0)
    for candidate in candidates:
        evidence_candidates.append(
            AcousticEvidenceCandidate(
                symbol=candidate.symbol,
                confidence=candidate.confidence,
                articulatory_features=default_articulatory_features(candidate.symbol),
                rank=len(evidence_candidates) + 1,
                margin_from_top=max(0.0, top_confidence - candidate.confidence),
                relative_confidence=(candidate.confidence / top_confidence) if top_confidence > 0.0 else 0.0,
                logit=(candidate_logits or {}).get(candidate.symbol, probability_to_logit(candidate.confidence)),
            )
        )
    return evidence_candidates


def candidate_entropy(candidates: list[PhoneCandidate]) -> float:
    if not candidates:
        return 0.0
    total = sum(max(candidate.confidence, 0.0) for candidate in candidates)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for candidate in candidates:
        probability = max(candidate.confidence, 0.0) / total
        if probability <= 0.0:
            continue
        entropy -= probability * math.log(probability)
    normalizer = math.log(len(candidates)) if len(candidates) > 1 else 1.0
    if normalizer <= 0.0:
        return 0.0
    return max(0.0, min(entropy / normalizer, 1.0))


def apply_acoustic_evidence_metadata(
    metadata: dict[str, Any],
    *,
    evidence: AcousticEvidence,
    candidates: list[PhoneCandidate],
) -> dict[str, Any]:
    payload = evidence.to_metadata()
    metadata["acoustic_evidence"] = payload
    metadata["acoustic_evidence_applied"] = True
    metadata["acoustic_evidence_summary"] = evidence.summary()
    metadata["acoustic_confidence"] = payload["acoustic_confidence"]
    metadata["timing_reliability"] = payload["timing_reliability"]
    metadata["frame_consistency"] = payload["frame_consistency"]
    metadata["evidence_quality"] = payload["evidence_quality"]
    metadata["segment_likelihood"] = payload["segment_likelihood"]
    metadata["candidate_margin"] = payload["candidate_margin"]
    metadata["candidate_count"] = payload["candidate_count"]
    metadata["candidate_entropy"] = payload["candidate_entropy"]
    metadata["boundary_transition_strength"] = payload["boundary_transition_strength"]
    metadata["dominant_candidate_stability"] = payload["dominant_candidate_stability"]
    metadata["boundary_softness"] = payload["boundary_softness"]
    metadata["ownership_ambiguity"] = payload["ownership_ambiguity"]
    metadata["boundary_uncertainty"] = dict(payload["boundary_uncertainty"])
    metadata["note_alignment_friendly"] = dict(payload["note_alignment"])
    metadata["ownership_signal"] = dict(payload["ownership"])
    metadata["candidate_distribution"] = dict(payload["candidate_distribution"])
    metadata["continuation_profile"] = dict(payload["continuation_profile"])
    metadata["boundary_profile"] = dict(payload["boundary_profile"])
    metadata["adapter_diagnostics"] = dict(payload["adapter_diagnostics"])
    metadata["posterior_summary"] = dict(payload["posterior_summary"])
    metadata["logit_summary"] = dict(payload["logit_summary"])
    metadata.setdefault(
        "posterior",
        [{"symbol": candidate.symbol, "confidence": candidate.confidence} for candidate in candidates],
    )
    return metadata


def build_segment_summary_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "acoustic_confidence": metadata.get("acoustic_confidence"),
        "timing_reliability": metadata.get("timing_reliability"),
        "frame_consistency": metadata.get("frame_consistency"),
        "evidence_quality": metadata.get("evidence_quality"),
        "segment_likelihood": metadata.get("segment_likelihood"),
        "boundary_uncertainty": deepcopy(metadata.get("boundary_uncertainty", {})),
        "posterior_summary": deepcopy(metadata.get("posterior_summary", {})),
        "logit_summary": deepcopy(metadata.get("logit_summary", {})),
    }


def build_segment_evidence_views(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "posterior_trace": deepcopy(metadata.get("posterior_trace", {})),
        "logit_trace": deepcopy(metadata.get("logit_trace", {})),
        "posterior_summary": deepcopy(metadata.get("posterior_summary", {})),
        "logit_summary": deepcopy(metadata.get("logit_summary", {})),
        "segment_summary": build_segment_summary_payload(metadata),
        "continuation_profile": deepcopy(metadata.get("continuation_profile", {})),
        "boundary_profile": deepcopy(metadata.get("boundary_profile", {})),
        "ownership": deepcopy(metadata.get("ownership_signal", {})),
        "adapter_diagnostics": deepcopy(metadata.get("adapter_diagnostics", {})),
    }

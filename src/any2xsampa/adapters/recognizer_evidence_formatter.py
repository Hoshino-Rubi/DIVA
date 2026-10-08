from __future__ import annotations

from typing import Any

from any2xsampa.adapters.recognizer_acoustic_evidence import (
    apply_acoustic_evidence_metadata,
    build_segment_evidence_views,
    build_segment_summary_payload,
    candidate_entropy,
    candidate_logit_map,
    normalize_candidate_evidence,
)
from any2xsampa.adapters.recognizer_segment_evidence import (
    resolve_evidence_kind as _resolve_evidence_kind_helper,
)
from any2xsampa.adapters.recognizer_acoustic_evidence_builder import (
    build_acoustic_evidence as _build_acoustic_evidence_helper,
    build_acoustic_evidence_from_components as _build_acoustic_evidence_from_components_helper,
    build_acoustic_evidence_from_metadata as _build_acoustic_evidence_from_metadata_impl_helper,
    calibrate_support_payload_maps as _calibrate_support_payload_maps_helper,
    build_support_payload_maps as _build_support_payload_maps_helper,
    derive_primary_acoustic_metrics as _derive_primary_acoustic_metrics_helper,
    resolve_candidate_evidence as _resolve_candidate_evidence_helper,
)
from any2xsampa.adapters.recognizer_segment_evidence import (
    derive_logit_frames as _derive_logit_frames_helper,
    logit_summary as _logit_summary_helper,
    posterior_summary as _posterior_summary_helper,
)
from any2xsampa.adapters.recognizer_support import (
    _default_articulatory_features,
    _probability_to_logit,
)
from any2xsampa.core.models import AcousticEvidence, AcousticEvidenceCandidate, PhoneCandidate


def resolved_candidate_evidence(
    *,
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    candidate_evidence: list[AcousticEvidenceCandidate] | None = None,
    candidate_logits: dict[str, float] | None = None,
) -> list[AcousticEvidenceCandidate]:
    return _resolve_candidate_evidence_helper(
        candidates=candidates,
        metadata=metadata,
        candidate_evidence=candidate_evidence,
        candidate_logits=candidate_logits,
    )


def derive_primary_acoustic_metrics(
    *,
    top_confidence: float,
    margin: float,
    candidate_entropy_value: float,
    frame_count: int,
    duration_sec: float,
    posterior_summary: dict[str, Any],
    logit_summary: dict[str, Any],
    acoustic_overrides: dict[str, Any] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    return _derive_primary_acoustic_metrics_helper(
        top_confidence=top_confidence,
        margin=margin,
        candidate_entropy=candidate_entropy_value,
        frame_count=frame_count,
        duration_sec=duration_sec,
        posterior_summary=posterior_summary,
        logit_summary=logit_summary,
        acoustic_overrides=acoustic_overrides,
        metadata_overrides=metadata_overrides,
    )


def support_payload_maps(
    *,
    top_phone: str,
    runner_up_phone: str | None,
    metrics: dict[str, float],
    candidate_entropy_value: float,
    acoustic_overrides: dict[str, Any] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return _build_support_payload_maps_helper(
        top_phone=top_phone,
        runner_up_phone=runner_up_phone,
        metrics=metrics,
        candidate_entropy=candidate_entropy_value,
        acoustic_overrides=acoustic_overrides,
        metadata_overrides=metadata_overrides,
    )


def calibrated_support_payload_maps(
    *,
    payload_maps: dict[str, dict[str, Any]],
    metrics: dict[str, float],
    candidate_margin: float,
    candidate_entropy_value: float,
    source_mode: str,
    metadata_overrides: dict[str, Any] | None = None,
    adapter_overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return _calibrate_support_payload_maps_helper(
        payload_maps=payload_maps,
        metrics=metrics,
        candidate_margin=candidate_margin,
        candidate_entropy=candidate_entropy_value,
        source_mode=source_mode,
        metadata_overrides=metadata_overrides,
        adapter_overrides=adapter_overrides,
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
    return _build_acoustic_evidence_from_components_helper(
        candidates=candidates,
        metadata=metadata,
        source_mode=source_mode,
        duration_sec=duration_sec,
        posterior_summary=posterior_summary,
        logit_summary=logit_summary,
        adapter_diagnostics=adapter_diagnostics,
        candidate_evidence=candidate_evidence,
        candidate_logits=candidate_logits,
        notes=notes,
    )


def build_acoustic_evidence_from_metadata(
    *,
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    source_mode: str,
    duration_sec: float,
) -> AcousticEvidence:
    return _build_acoustic_evidence_from_metadata_impl_helper(
        candidates=candidates,
        metadata=metadata,
        source_mode=source_mode,
        duration_sec=duration_sec,
    )


def segment_summary_payload(metadata: dict[str, object]) -> dict[str, object]:
    return build_segment_summary_payload(metadata)


def segment_evidence_views(metadata: dict[str, object]) -> dict[str, object]:
    return build_segment_evidence_views(metadata)


__all__ = [
    "apply_acoustic_evidence_metadata",
    "build_acoustic_evidence_from_components",
    "build_acoustic_evidence_from_metadata",
    "build_segment_evidence_views",
    "build_segment_summary_payload",
    "calibrated_support_payload_maps",
    "candidate_entropy",
    "candidate_logit_map",
    "derive_primary_acoustic_metrics",
    "normalize_candidate_evidence",
    "resolved_candidate_evidence",
    "segment_evidence_views",
    "segment_summary_payload",
    "support_payload_maps",
]

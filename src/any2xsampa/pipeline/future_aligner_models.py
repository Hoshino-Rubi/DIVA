from __future__ import annotations

from dataclasses import dataclass, field

from any2xsampa.core.models import PhoneSegment


@dataclass(slots=True)
class FutureAlignerEvidenceItem:
    evidence_type: str
    source_segment_index: int | None
    source_note_id: str | None
    source_track_id: str | None
    source_part_id: str | None
    support_score: float
    support_direction: str
    accepted: bool
    rejection_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, object]:
        return {
            "evidence_type": self.evidence_type,
            "source_segment_index": self.source_segment_index,
            "source_note_id": self.source_note_id,
            "source_track_id": self.source_track_id,
            "source_part_id": self.source_part_id,
            "support_score": round(self.support_score, 3),
            "support_direction": self.support_direction,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class FutureAlignerResult:
    aligned_track_id: str | None
    aligned_part_id: str | None
    aligned_note_id: str | None
    aligner_strategy: str
    aligner_confidence: float
    aligner_fallback_used: bool
    aligner_notes: list[str] = field(default_factory=list)
    future_aligner_window_size: int = 1
    future_aligner_sequence_applied: bool = False
    future_aligner_sequence_score: float = 0.0
    future_aligner_order_penalty: float = 0.0
    future_aligner_continuity_used: bool = False
    continuity_evidence: list[str] = field(default_factory=list)
    monotonic_evidence: list[str] = field(default_factory=list)
    part_boundary_evidence: list[str] = field(default_factory=list)
    neighbor_support_count: int = 0
    left_neighbor_support: bool = False
    right_neighbor_support: bool = False
    same_note_support: bool = False
    order_support: bool = False
    part_continuity_support: bool = False
    evidence_items: list[FutureAlignerEvidenceItem] = field(default_factory=list)
    future_aligner_provenance_applied: bool = False
    future_aligner_provenance_summary: str = ""
    acoustic_evidence_applied: bool = False
    acoustic_evidence_summary: str = ""
    boundary_uncertainty: dict[str, object] = field(default_factory=dict)
    part_aware_note_span_used: bool = False
    note_span_evidence_score: float = 0.0
    note_span_overlap_ratio: float = 0.0
    note_span_interior_ratio: float = 0.0
    note_span_anchor_count: int = 0
    note_span_summary: str = ""
    ownership_conflict_gap: float = 0.0
    ownership_conflict_threshold: float = 0.0
    ownership_stability_margin: float = 0.0
    ownership_conflict_summary: str = ""
    ownership_conflict_candidates: list[dict[str, object]] = field(default_factory=list)
    ownership_resolution_basis: str = ""
    boundary_conflict_likely: bool = False
    aligner_v2_applied: bool = True
    aligner_v2_strategy: str = "score_aware_aligner_v2"
    aligner_v2_confidence: float = 0.0
    aligner_v2_fallback_used: bool = False
    aligner_v2_notes: list[str] = field(default_factory=list)
    aligner_v2_version: str = "score_aware_aligner_v2"
    alignment_competition_summary: str = ""
    note_ownership_conflict: bool = False
    selective_release_safety: dict[str, object] = field(default_factory=dict)
    release_explanation: str = ""
    release_risk_summary: str = ""
    source_score_hint_note_id: str | None = None
    source_score_hint_note_identity_match: bool | None = None
    source_score_hint_binding_strategy: str | None = None
    effective_score_hint_note_id: str | None = None
    effective_score_hint_note_identity_match: bool | None = None
    effective_score_hint_binding_strategy: str | None = None
    ownership_binding_adjustment: dict[str, object] = field(default_factory=dict)
    phrase_boundary_candidate_debug: dict[str, object] = field(default_factory=dict)
    phrase_boundary_blocker_summary: str = ""
    future_aligner_report: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "aligned_track_id": self.aligned_track_id,
            "aligned_part_id": self.aligned_part_id,
            "aligned_note_id": self.aligned_note_id,
            "aligner_strategy": self.aligner_strategy,
            "aligner_confidence": round(self.aligner_confidence, 3),
            "aligner_fallback_used": self.aligner_fallback_used,
            "aligner_notes": list(self.aligner_notes),
            "future_aligner_window_size": self.future_aligner_window_size,
            "future_aligner_sequence_applied": self.future_aligner_sequence_applied,
            "future_aligner_sequence_score": round(self.future_aligner_sequence_score, 3),
            "future_aligner_order_penalty": round(self.future_aligner_order_penalty, 3),
            "future_aligner_continuity_used": self.future_aligner_continuity_used,
            "continuity_evidence": list(self.continuity_evidence),
            "monotonic_evidence": list(self.monotonic_evidence),
            "part_boundary_evidence": list(self.part_boundary_evidence),
            "neighbor_support_count": self.neighbor_support_count,
            "left_neighbor_support": self.left_neighbor_support,
            "right_neighbor_support": self.right_neighbor_support,
            "same_note_support": self.same_note_support,
            "order_support": self.order_support,
            "part_continuity_support": self.part_continuity_support,
            "evidence_items": [item.to_metadata() for item in self.evidence_items],
            "future_aligner_provenance_applied": self.future_aligner_provenance_applied,
            "future_aligner_provenance_summary": self.future_aligner_provenance_summary,
            "future_aligner_provenance_items": [item.to_metadata() for item in self.evidence_items],
            "acoustic_evidence_applied": self.acoustic_evidence_applied,
            "acoustic_evidence_summary": self.acoustic_evidence_summary,
            "boundary_uncertainty": dict(self.boundary_uncertainty),
            "part_aware_note_span_used": self.part_aware_note_span_used,
            "note_span_evidence_score": round(self.note_span_evidence_score, 3),
            "note_span_overlap_ratio": round(self.note_span_overlap_ratio, 3),
            "note_span_interior_ratio": round(self.note_span_interior_ratio, 3),
            "note_span_anchor_count": self.note_span_anchor_count,
            "note_span_summary": self.note_span_summary,
            "ownership_conflict_gap": round(self.ownership_conflict_gap, 3),
            "ownership_conflict_threshold": round(self.ownership_conflict_threshold, 3),
            "ownership_stability_margin": round(self.ownership_stability_margin, 3),
            "ownership_conflict_summary": self.ownership_conflict_summary,
            "ownership_conflict_candidates": [dict(item) for item in self.ownership_conflict_candidates],
            "ownership_resolution_basis": self.ownership_resolution_basis,
            "boundary_conflict_likely": self.boundary_conflict_likely,
            "aligner_v2_applied": self.aligner_v2_applied,
            "aligner_v2_strategy": self.aligner_v2_strategy,
            "aligner_v2_confidence": round(self.aligner_v2_confidence or self.aligner_confidence, 3),
            "aligner_v2_fallback_used": self.aligner_v2_fallback_used or self.aligner_fallback_used,
            "aligner_v2_notes": list(self.aligner_v2_notes or self.aligner_notes),
            "aligner_v2_version": self.aligner_v2_version,
            "alignment_competition_summary": self.alignment_competition_summary,
            "note_ownership_conflict": self.note_ownership_conflict,
            "selective_release_safety": dict(self.selective_release_safety),
            "release_explanation": self.release_explanation,
            "release_risk_summary": self.release_risk_summary,
            "source_score_hint_note_id": self.source_score_hint_note_id,
            "source_score_hint_note_identity_match": self.source_score_hint_note_identity_match,
            "source_score_hint_binding_strategy": self.source_score_hint_binding_strategy,
            "effective_score_hint_note_id": self.effective_score_hint_note_id,
            "effective_score_hint_note_identity_match": self.effective_score_hint_note_identity_match,
            "effective_score_hint_binding_strategy": self.effective_score_hint_binding_strategy,
            "ownership_binding_adjustment": dict(self.ownership_binding_adjustment),
            "phrase_boundary_candidate_debug": dict(self.phrase_boundary_candidate_debug),
            "phrase_boundary_blocker_summary": self.phrase_boundary_blocker_summary,
            "future_aligner_report": dict(self.future_aligner_report),
        }


@dataclass(slots=True)
class FutureAlignerHint:
    normalized_track_id: str | None
    normalized_part_id: str | None
    normalized_note_id: str | None
    part_local_note_rank: int | None
    duration_ms: float | None
    note_start_sec: float | None
    note_end_sec: float | None
    release_context: str | None
    note_binding_strategy: str | None
    note_identity_match: bool
    score_bridge_strategy: str | None
    score_hint_confidence: float
    score_hint_notes: list[str] = field(default_factory=list)
    source_normalized_note_id: str | None = None
    source_note_identity_match: bool | None = None
    source_note_binding_strategy: str | None = None
    ownership_binding_adjusted: bool = False
    ownership_binding_reason: str | None = None

    @property
    def has_identity(self) -> bool:
        return all(
            value is not None
            for value in (
                self.normalized_track_id,
                self.normalized_part_id,
                self.normalized_note_id,
            )
        )


@dataclass(slots=True)
class FutureAlignerAcousticEvidence:
    acoustic_confidence: float
    timing_reliability: float
    boundary_uncertain: bool
    boundary_uncertainty: dict[str, object] = field(default_factory=dict)
    candidate_count: int = 0
    candidate_entropy: float = 0.0
    candidate_lead: float = 0.0
    posterior_vote_ratio: float = 0.0
    posterior_temporal_consistency: float = 0.0
    posterior_mean_margin: float = 0.0
    logit_mean_margin: float = 0.0
    competition_separation: float = 0.0
    interior_hold_strength: float = 0.0
    one_sided_continuation_strength: float = 0.0
    boundary_pressure: float = 0.0
    dominant_candidate_stability: float = 0.0
    boundary_softness: float = 0.0
    same_note_continuation_bias: float = 0.0
    same_note_stability: float = 0.0
    continuation_friendliness: float = 0.0
    score_alignment_friendly: float = 0.0
    weak_overlap_support: float = 0.0
    weak_overlap_recovery: float = 0.0
    weak_overlap_friendliness: float = 0.0
    boundary_anchor_strength: float = 0.0
    supports_note_span_extension: bool = False
    long_note_hold_bias: float = 0.0
    part_note_span_reliability: float = 0.0
    reassignment_resistance: float = 0.0
    boundary_reassignment_risk: float = 0.0
    boundary_conflict_likelihood: float = 0.0
    ownership_conflict_risk: float = 0.0
    frame_consistency: float = 0.0
    evidence_quality: float = 0.0
    segment_likelihood: float = 0.0
    boundary_transition_strength: float = 0.0
    ownership_ambiguity: float = 0.0
    backend_kind: str = ""
    adapter_kind: str = ""
    evidence_kind: str = ""
    aggregation: str = ""
    posterior_like_available: bool = False
    logit_like_available: bool = False
    approximation_only: bool = False
    summary: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.summary)


@dataclass(slots=True)
class FutureAlignerNoteSpanEstimate:
    note_id: str
    start_sec: float
    end_sec: float
    anchor_count: int
    order_value: int | None = None
    support_window_start_sec: float = 0.0
    support_window_end_sec: float = 0.0
    part_note_count: int = 0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(slots=True)
class FutureAlignerCandidateScore:
    note_id: str
    rank: int | None
    support_score: float
    continuity_score: float
    order_penalty: float
    same_note_neighbors: int
    left_neighbor_match: bool
    right_neighbor_match: bool
    acoustic_score: float = 0.0
    note_span_score: float = 0.0
    note_span_overlap_ratio: float = 0.0
    note_span_interior_ratio: float = 0.0
    note_span_anchor_count: int = 0
    part_aware_note_span_used: bool = False
    note_span_summary: str = ""
    notes: list[str] = field(default_factory=list)
    continuity_evidence: list[str] = field(default_factory=list)
    monotonic_evidence: list[str] = field(default_factory=list)
    part_boundary_evidence: list[str] = field(default_factory=list)
    order_support: bool = False
    part_continuity_support: bool = False
    evidence_items: list[FutureAlignerEvidenceItem] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return (
            self.support_score
            + self.continuity_score
            + self.acoustic_score
            + self.note_span_score
            - self.order_penalty
        )

    @property
    def continuity_used(self) -> bool:
        return self.same_note_neighbors > 0 or self.left_neighbor_match or self.right_neighbor_match


def future_aligner_output_payload(segment: PhoneSegment) -> dict[str, object]:
    payload = segment.metadata.get("future_aligner")
    return payload if isinstance(payload, dict) else {}

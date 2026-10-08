from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(slots=True)
class TimeRange:
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if isinstance(self.start_sec, bool) or isinstance(self.end_sec, bool):
            raise ValueError("time range values must be numeric")
        start = float(self.start_sec)
        end = float(self.end_sec)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("time range values must be finite")
        if start < 0.0 or end <= start:
            raise ValueError("time range must have 0 <= start_sec < end_sec")


@dataclass(slots=True)
class PhoneCandidate:
    symbol: str
    confidence: float

    def __post_init__(self) -> None:
        if not str(self.symbol).strip():
            raise ValueError("phone candidate symbol must be non-empty")
        if isinstance(self.confidence, bool):
            raise ValueError("phone candidate confidence must be numeric")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("phone candidate confidence must be finite and between zero and one")


@dataclass(slots=True)
class AcousticEvidenceCandidate:
    symbol: str
    confidence: float
    articulatory_features: dict[str, str] = field(default_factory=dict)
    rank: int | None = None
    margin_from_top: float | None = None
    relative_confidence: float | None = None
    logit: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "confidence": round(self.confidence, 3),
            "articulatory_features": dict(self.articulatory_features),
            "rank": self.rank,
            "margin_from_top": round(self.margin_from_top, 3) if self.margin_from_top is not None else None,
            "relative_confidence": (
                round(self.relative_confidence, 3) if self.relative_confidence is not None else None
            ),
            "logit": round(self.logit, 3) if self.logit is not None else None,
        }


@dataclass(slots=True)
class AcousticEvidence:
    schema_version: str
    source_mode: str
    top_phone: str | None
    acoustic_confidence: float
    timing_reliability: float
    duration_sec: float = 0.0
    candidate_margin: float = 0.0
    candidate_count: int = 0
    candidate_entropy: float = 0.0
    frame_consistency: float = 0.0
    evidence_quality: float = 0.0
    segment_likelihood: float = 0.0
    dominant_candidate_stability: float = 0.0
    boundary_transition_strength: float = 0.0
    boundary_softness: float = 0.0
    ownership_ambiguity: float = 0.0
    candidate_phones: list[AcousticEvidenceCandidate] = field(default_factory=list)
    boundary_uncertainty: dict[str, Any] = field(default_factory=dict)
    note_alignment: dict[str, Any] = field(default_factory=dict)
    ownership: dict[str, Any] = field(default_factory=dict)
    candidate_distribution: dict[str, Any] = field(default_factory=dict)
    continuation_profile: dict[str, Any] = field(default_factory=dict)
    boundary_profile: dict[str, Any] = field(default_factory=dict)
    adapter_diagnostics: dict[str, Any] = field(default_factory=dict)
    posterior_summary: dict[str, Any] = field(default_factory=dict)
    logit_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_mode": self.source_mode,
            "top_phone": self.top_phone,
            "acoustic_confidence": round(self.acoustic_confidence, 3),
            "timing_reliability": round(self.timing_reliability, 3),
            "duration_sec": round(self.duration_sec, 3),
            "candidate_margin": round(self.candidate_margin, 3),
            "candidate_count": self.candidate_count,
            "candidate_entropy": round(self.candidate_entropy, 3),
            "frame_consistency": round(self.frame_consistency, 3),
            "evidence_quality": round(self.evidence_quality, 3),
            "segment_likelihood": round(self.segment_likelihood, 3),
            "dominant_candidate_stability": round(self.dominant_candidate_stability, 3),
            "boundary_transition_strength": round(self.boundary_transition_strength, 3),
            "boundary_softness": round(self.boundary_softness, 3),
            "ownership_ambiguity": round(self.ownership_ambiguity, 3),
            "candidate_phones": [candidate.to_metadata() for candidate in self.candidate_phones],
            "boundary_uncertainty": dict(self.boundary_uncertainty),
            "note_alignment": dict(self.note_alignment),
            "ownership": dict(self.ownership),
            "candidate_distribution": dict(self.candidate_distribution),
            "continuation_profile": dict(self.continuation_profile),
            "boundary_profile": dict(self.boundary_profile),
            "adapter_diagnostics": dict(self.adapter_diagnostics),
            "posterior_summary": dict(self.posterior_summary),
            "logit_summary": dict(self.logit_summary),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        boundary_uncertain = bool(self.boundary_uncertainty.get("uncertain", False))
        continuation_bias = self.note_alignment.get("same_note_continuation_bias")
        continuation_text = (
            f"{float(continuation_bias):.2f}"
            if isinstance(continuation_bias, (int, float))
            else "na"
        )
        continuation_friendliness = self.continuation_profile.get("continuation_friendliness")
        continuation_friendliness_text = (
            f"{float(continuation_friendliness):.2f}"
            if isinstance(continuation_friendliness, (int, float))
            else "na"
        )
        weak_overlap_friendliness = self.continuation_profile.get("weak_overlap_friendliness")
        weak_overlap_friendliness_text = (
            f"{float(weak_overlap_friendliness):.2f}"
            if isinstance(weak_overlap_friendliness, (int, float))
            else "na"
        )
        return (
            f"top={self.top_phone or 'none'},"
            f"acoustic_confidence={self.acoustic_confidence:.2f},"
            f"timing_reliability={self.timing_reliability:.2f},"
            f"evidence_quality={self.evidence_quality:.2f},"
            f"boundary_uncertain={'yes' if boundary_uncertain else 'no'},"
            f"ownership_ambiguity={self.ownership_ambiguity:.2f},"
            f"dominant_candidate_stability={self.dominant_candidate_stability:.2f},"
            f"same_note_bias={continuation_text},"
            f"continuation_friendliness={continuation_friendliness_text},"
            f"weak_overlap_friendliness={weak_overlap_friendliness_text}"
        )


@dataclass(slots=True)
class DialectConstraintResult:
    dialect: str
    accepted: bool
    adjusted_ipa: str
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SyllableGroup:
    group_id: str
    member_segment_indices: list[int]
    onset: str | None = None
    nucleus: str | None = None
    coda: str | None = None
    onset_segment_indices: list[int] = field(default_factory=list)
    nucleus_segment_index: int | None = None
    coda_segment_indices: list[int] = field(default_factory=list)
    lyric_token: str | None = None
    grouping_notes: list[str] = field(default_factory=list)
    grouping_fallback_used: bool = False
    score_identity_hint_applied: bool = False
    score_identity_boundary_hit: bool = False
    grouping_identity_fallback_used: bool = False
    grouping_identity_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EnginePhoneExport:
    format: str
    original_ipa: str
    xsampa: str
    engine_phone: str
    approximation_applied: bool
    aliases: list[str] = field(default_factory=list)
    approximation_notes: list[str] = field(default_factory=list)
    unmapped: bool = False
    alias_generation_applied: bool = False
    alias_generation_notes: list[str] = field(default_factory=list)
    fallback_used: bool = False
    bank_profile: str | None = None
    alias_strategy: str | None = None
    profile_rule_applied: bool = False
    profile_rule_notes: list[str] = field(default_factory=list)
    profile_fallback_used: bool = False
    alias_inventory_checked: bool = False
    alias_inventory_hits: list[str] = field(default_factory=list)
    alias_inventory_misses: list[str] = field(default_factory=list)
    validated_aliases: list[str] = field(default_factory=list)
    repaired_aliases: list[str] = field(default_factory=list)
    unresolved_aliases: list[str] = field(default_factory=list)
    inventory_fallback_used: bool = False
    candidate_aliases: list[str] = field(default_factory=list)
    selected_alias: str | None = None
    selection_reason: str | None = None
    ranking_features: list[str] = field(default_factory=list)
    phonetic_similarity_score: float | None = None
    bank_metadata_score: float | None = None
    selected_alias_metadata: dict[str, Any] | None = None
    metadata_source: list[str] = field(default_factory=list)
    normalized_bank_metadata: dict[str, Any] | None = None
    alias_metadata_origin: list[str] = field(default_factory=list)
    metadata_fallback_used: bool = False
    repair_rank_applied: bool = False
    note_context_applied: bool = False
    note_context_source: str | None = None
    note_range_match: bool | None = None
    timing_match: bool | None = None
    release_match: bool | None = None
    note_context_fallback_used: bool = False
    selected_alias_approximation: bool = False
    selected_alias_inventory_native: bool = False


@dataclass(slots=True)
class PhoneSegment:
    time_range: TimeRange
    candidate_phones: list[PhoneCandidate]
    confidence: float
    normalized_ipa: str = ""
    xsampa: str = ""
    engine_export: EnginePhoneExport | None = None
    dialect_result: DialectConstraintResult | None = None
    syllable_group: SyllableGroup | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_phones:
            raise ValueError("phone segment requires at least one candidate phone")
        if isinstance(self.confidence, bool):
            raise ValueError("phone segment confidence must be numeric")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("phone segment confidence must be finite and between zero and one")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start_sec"] = payload.pop("time_range")["start_sec"]
        payload["end_sec"] = asdict(self.time_range)["end_sec"]
        return payload


@dataclass(slots=True)
class PipelineTrace:
    separation_adapter: str
    recognizer_adapter: str
    normalizer: str
    dialect_plugin: str
    exporter: str
    aligner: str | None = None


@dataclass(slots=True)
class PipelineResult:
    mode: str
    audio_path: str
    dialect: str
    trace: PipelineTrace
    segments: list[PhoneSegment]
    lyrics: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "audio_path": self.audio_path,
            "dialect": self.dialect,
            "lyrics": self.lyrics,
            "trace": asdict(self.trace),
            "warnings": list(self.warnings),
            "segments": [segment.to_dict() for segment in self.segments],
        }

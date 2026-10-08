from __future__ import annotations

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.pipeline.future_aligner_inputs import (
    build_future_aligner_acoustic_evidence,
    build_future_aligner_hint,
    future_aligner_acoustic_evidence_from_segment,
    future_aligner_float,
    future_aligner_hint_from_segment,
    future_aligner_int,
    future_aligner_note_list,
    future_aligner_text,
    normalize_hint_ownership_bindings,
)


def _segment(
    start_sec: float,
    end_sec: float,
    *,
    metadata: dict[str, object] | None = None,
) -> PhoneSegment:
    return PhoneSegment(
        time_range=TimeRange(start_sec, end_sec),
        candidate_phones=[PhoneCandidate("aa", 0.9)],
        confidence=0.9,
        normalized_ipa="a",
        metadata=dict(metadata or {}),
    )


def test_future_aligner_hint_from_segment_reads_future_aligner_hint_payload() -> None:
    segment = _segment(
        0.2,
        0.4,
        metadata={
            "future_aligner_hint": {
                "normalized_track_id": "track-1",
                "normalized_part_id": "part-1",
                "normalized_note_id": "note-2",
                "part_local_note_rank": 3,
                "duration_ms": 200.0,
                "start_sec": 0.15,
                "end_sec": 0.55,
                "release_context": "held",
                "note_binding_strategy": "identity",
                "note_identity_match": True,
                "score_bridge_strategy": "identity",
                "score_hint_confidence": 0.73,
                "score_hint_notes": ["identity aligned"],
            }
        },
    )

    hint = future_aligner_hint_from_segment(segment)

    assert hint is not None
    assert hint.normalized_track_id == "track-1"
    assert hint.normalized_part_id == "part-1"
    assert hint.normalized_note_id == "note-2"
    assert hint.part_local_note_rank == 3
    assert hint.note_start_sec == 0.15
    assert hint.note_end_sec == 0.55
    assert hint.note_binding_strategy == "identity"
    assert hint.note_identity_match is True
    assert hint.score_hint_confidence == 0.73
    assert hint.score_hint_notes == ["identity aligned"]


def test_future_aligner_input_helper_aliases_stay_consistent() -> None:
    segment = _segment(
        0.2,
        0.4,
        metadata={
            "future_aligner_hint": {
                "normalized_track_id": "track-1",
                "normalized_part_id": "part-1",
                "normalized_note_id": "note-2",
                "score_hint_confidence": 0.5,
                "score_hint_notes": ["hint"],
            },
            "acoustic_evidence_applied": True,
            "acoustic_evidence_summary": "ready",
            "acoustic_evidence": {
                "acoustic_confidence": 0.9,
                "timing_reliability": 0.8,
            },
        },
    )

    assert future_aligner_text(12) == "12"
    assert future_aligner_float("1.5") == 1.5
    assert future_aligner_int("2") == 2
    assert future_aligner_note_list(["a", 3, None]) == ["a", "3"]
    assert build_future_aligner_hint(segment) == future_aligner_hint_from_segment(segment)
    assert build_future_aligner_acoustic_evidence(segment) == future_aligner_acoustic_evidence_from_segment(segment)


def test_future_aligner_acoustic_evidence_from_segment_reads_flat_and_nested_views() -> None:
    segment = _segment(
        0.2,
        0.4,
        metadata={
            "acoustic_evidence_applied": True,
            "acoustic_evidence_summary": "ready",
            "acoustic_evidence": {
                "acoustic_confidence": 0.91,
                "timing_reliability": 0.82,
                "candidate_count": 2,
                "candidate_entropy": 0.18,
                "dominant_candidate_stability": 0.77,
                "boundary_softness": 0.22,
                "frame_consistency": 0.81,
                "evidence_quality": 0.84,
                "segment_likelihood": 0.88,
                "boundary_transition_strength": 0.83,
                "ownership_ambiguity": 0.14,
                "boundary_uncertainty": {"uncertain": False, "window_ms": 12.0},
                "note_alignment": {
                    "same_note_continuation_bias": 0.9,
                    "score_alignment_friendly": 0.7,
                    "weak_overlap_support": 0.62,
                    "boundary_anchor_strength": 0.86,
                    "supports_note_span_extension": True,
                    "long_note_hold_bias": 0.75,
                    "part_note_span_reliability": 0.72,
                    "continuation_friendliness": 0.88,
                    "weak_overlap_friendliness": 0.68,
                    "competition_separation": 0.66,
                    "interior_hold_strength": 0.71,
                    "one_sided_continuation_strength": 0.59,
                },
                "ownership": {
                    "same_note_stability": 0.74,
                    "weak_overlap_recovery": 0.61,
                    "part_note_span_reliability": 0.73,
                    "reassignment_resistance": 0.69,
                    "boundary_reassignment_risk": 0.28,
                    "boundary_conflict_likelihood": 0.33,
                    "ownership_conflict_risk": 0.21,
                    "candidate_lead": 0.64,
                    "competition_separation": 0.67,
                    "interior_hold_strength": 0.7,
                    "one_sided_continuation_strength": 0.58,
                    "boundary_pressure": 0.41,
                },
                "candidate_distribution": {
                    "frame_vote_ratio": 0.79,
                    "temporal_consistency": 0.83,
                    "posterior_mean_margin": 0.24,
                    "logit_mean_margin": 0.31,
                    "competition_separation": 0.68,
                    "candidate_lead": 0.66,
                },
                "continuation_profile": {
                    "continuation_friendliness": 0.89,
                    "weak_overlap_friendliness": 0.69,
                    "interior_hold_strength": 0.72,
                    "one_sided_continuation_strength": 0.6,
                },
                "boundary_profile": {
                    "boundary_pressure": 0.42,
                    "boundary_conflict_likelihood": 0.34,
                },
                "adapter_diagnostics": {
                    "backend_kind": "onnx-espeak",
                    "adapter_kind": "real",
                    "evidence_kind": "posterior",
                    "aggregation": "mean",
                    "posterior_like_available": True,
                    "logit_like_available": True,
                    "approximation_only": False,
                },
                "posterior_summary": {"frame_vote_ratio": 0.8, "mean_margin": 0.25},
                "logit_summary": {"mean_margin": 0.32},
                "notes": ["fixture acoustic"],
            },
        },
    )

    acoustic = future_aligner_acoustic_evidence_from_segment(segment)

    assert acoustic is not None
    assert acoustic.available is True
    assert acoustic.summary == "ready"
    assert acoustic.acoustic_confidence == 0.91
    assert acoustic.timing_reliability == 0.82
    assert acoustic.boundary_uncertain is False
    assert acoustic.candidate_count == 2
    assert acoustic.posterior_vote_ratio == 0.79
    assert acoustic.logit_mean_margin == 0.31
    assert acoustic.continuation_friendliness == 0.89
    assert acoustic.weak_overlap_recovery == 0.61
    assert acoustic.backend_kind == "onnx-espeak"
    assert acoustic.adapter_kind == "real"
    assert acoustic.posterior_like_available is True
    assert acoustic.logit_like_available is True
    assert acoustic.notes == ["fixture acoustic"]


def test_future_aligner_acoustic_evidence_from_segment_supports_flat_only_metadata_views() -> None:
    segment = _segment(
        0.2,
        0.4,
        metadata={
            "acoustic_evidence_applied": True,
            "acoustic_evidence_summary": "flat ready",
            "acoustic_confidence": 0.89,
            "timing_reliability": 0.77,
            "candidate_count": 3,
            "candidate_entropy": 0.22,
            "dominant_candidate_stability": 0.71,
            "boundary_softness": 0.29,
            "frame_consistency": 0.74,
            "evidence_quality": 0.76,
            "segment_likelihood": 0.81,
            "boundary_transition_strength": 0.63,
            "ownership_ambiguity": 0.18,
            "boundary_uncertainty": {"uncertain": True, "window_ms": 18.0},
            "note_alignment_friendly": {
                "same_note_continuation_bias": 0.88,
                "score_alignment_friendly": 0.69,
                "weak_overlap_support": 0.61,
                "boundary_anchor_strength": 0.83,
                "supports_note_span_extension": True,
                "long_note_hold_bias": 0.73,
                "part_note_span_reliability": 0.67,
                "continuation_friendliness": 0.81,
                "weak_overlap_friendliness": 0.59,
                "competition_separation": 0.58,
                "interior_hold_strength": 0.57,
                "one_sided_continuation_strength": 0.45,
            },
            "ownership_signal": {
                "same_note_stability": 0.66,
                "weak_overlap_recovery": 0.53,
                "part_note_span_reliability": 0.68,
                "reassignment_resistance": 0.64,
                "boundary_reassignment_risk": 0.27,
                "boundary_conflict_likelihood": 0.31,
                "ownership_conflict_risk": 0.19,
                "candidate_lead": 0.6,
                "competition_separation": 0.57,
                "interior_hold_strength": 0.56,
                "one_sided_continuation_strength": 0.44,
                "boundary_pressure": 0.39,
            },
            "candidate_distribution": {
                "frame_vote_ratio": 0.72,
                "temporal_consistency": 0.79,
                "posterior_mean_margin": 0.2,
                "logit_mean_margin": 0.28,
                "competition_separation": 0.61,
                "candidate_lead": 0.62,
            },
            "continuation_profile": {
                "continuation_friendliness": 0.84,
                "weak_overlap_friendliness": 0.6,
                "interior_hold_strength": 0.59,
                "one_sided_continuation_strength": 0.47,
            },
            "boundary_profile": {
                "boundary_pressure": 0.4,
                "boundary_conflict_likelihood": 0.32,
            },
            "adapter_diagnostics": {
                "backend_kind": "onnx-espeak",
                "adapter_kind": "real",
                "evidence_kind": "posterior",
                "aggregation": "mean",
                "posterior_like_available": True,
                "logit_like_available": False,
                "approximation_only": False,
            },
            "posterior_summary": {"frame_vote_ratio": 0.73, "mean_margin": 0.21},
            "logit_summary": {"mean_margin": 0.29},
        },
    )

    acoustic = future_aligner_acoustic_evidence_from_segment(segment)

    assert acoustic is not None
    assert acoustic.available is True
    assert acoustic.summary == "flat ready"
    assert acoustic.acoustic_confidence == 0.89
    assert acoustic.timing_reliability == 0.77
    assert acoustic.boundary_uncertain is True
    assert acoustic.candidate_count == 3
    assert acoustic.posterior_vote_ratio == 0.72
    assert acoustic.posterior_mean_margin == 0.2
    assert acoustic.logit_mean_margin == 0.28
    assert acoustic.continuation_friendliness == 0.84
    assert acoustic.weak_overlap_recovery == 0.53
    assert acoustic.boundary_pressure == 0.4
    assert acoustic.backend_kind == "onnx-espeak"
    assert acoustic.posterior_like_available is True
    assert acoustic.logit_like_available is False


def test_future_aligner_acoustic_evidence_from_segment_merges_partial_nested_and_flat_views() -> None:
    segment = _segment(
        0.2,
        0.4,
        metadata={
            "acoustic_evidence_applied": True,
            "acoustic_evidence_summary": "merged ready",
            "candidate_distribution": {
                "frame_vote_ratio": 0.74,
                "temporal_consistency": 0.8,
                "posterior_mean_margin": 0.23,
                "logit_mean_margin": 0.27,
            },
            "continuation_profile": {
                "continuation_friendliness": 0.87,
                "weak_overlap_friendliness": 0.64,
                "interior_hold_strength": 0.61,
                "one_sided_continuation_strength": 0.48,
            },
            "ownership_signal": {
                "same_note_stability": 0.69,
                "weak_overlap_recovery": 0.55,
                "reassignment_resistance": 0.66,
                "boundary_reassignment_risk": 0.25,
                "ownership_conflict_risk": 0.18,
                "candidate_lead": 0.58,
                "boundary_pressure": 0.37,
            },
            "boundary_profile": {"boundary_conflict_likelihood": 0.28},
            "posterior_summary": {"mean_margin": 0.24},
            "adapter_diagnostics": {
                "backend_kind": "fixture-json",
                "posterior_like_available": True,
                "logit_like_available": True,
            },
            "acoustic_evidence": {
                "acoustic_confidence": 0.9,
                "timing_reliability": 0.81,
                "candidate_count": 2,
                "boundary_uncertainty": {"uncertain": False},
                "adapter_diagnostics": {
                    "backend_kind": "nested-backend",
                    "adapter_kind": "real",
                    "evidence_kind": "posterior",
                    "aggregation": "nested",
                    "posterior_like_available": False,
                    "logit_like_available": True,
                },
            },
        },
    )

    acoustic = future_aligner_acoustic_evidence_from_segment(segment)

    assert acoustic is not None
    assert acoustic.summary == "merged ready"
    assert acoustic.acoustic_confidence == 0.9
    assert acoustic.timing_reliability == 0.81
    assert acoustic.candidate_count == 2
    assert acoustic.posterior_vote_ratio == 0.74
    assert acoustic.posterior_mean_margin == 0.23
    assert acoustic.continuation_friendliness == 0.87
    assert acoustic.weak_overlap_recovery == 0.55
    assert acoustic.backend_kind == "nested-backend"
    assert acoustic.adapter_kind == "real"
    assert acoustic.posterior_like_available is False
    assert acoustic.logit_like_available is True


def test_normalize_hint_ownership_bindings_rebinds_low_confidence_bridge_like_hint() -> None:
    segments = [
        _segment(0.0, 0.1),
        _segment(0.1, 0.2),
        _segment(0.2, 0.3),
    ]
    hints = [
        future_aligner_hint_from_segment(
            _segment(
                0.0,
                0.1,
                metadata={
                    "future_aligner_hint": {
                        "normalized_track_id": "track-1",
                        "normalized_part_id": "part-1",
                        "normalized_note_id": "note-1",
                        "note_binding_strategy": "identity",
                        "note_identity_match": True,
                        "score_bridge_strategy": "identity",
                        "score_hint_confidence": 0.95,
                        "score_hint_notes": ["left"],
                    }
                },
            )
        ),
        future_aligner_hint_from_segment(
            _segment(
                0.1,
                0.2,
                metadata={
                    "future_aligner_hint": {
                        "normalized_track_id": "track-1",
                        "normalized_part_id": "part-1",
                        "normalized_note_id": "note-2",
                        "note_binding_strategy": "time_overlap",
                        "note_identity_match": False,
                        "score_bridge_strategy": "time_overlap_fallback",
                        "score_hint_confidence": 0.2,
                        "score_hint_notes": ["middle"],
                    }
                },
            )
        ),
        future_aligner_hint_from_segment(
            _segment(
                0.2,
                0.3,
                metadata={
                    "future_aligner_hint": {
                        "normalized_track_id": "track-1",
                        "normalized_part_id": "part-1",
                        "normalized_note_id": "note-1",
                        "note_binding_strategy": "identity",
                        "note_identity_match": True,
                        "score_bridge_strategy": "identity",
                        "score_hint_confidence": 0.96,
                        "score_hint_notes": ["right"],
                    }
                },
            )
        ),
    ]

    normalize_hint_ownership_bindings(segments, hints)
    middle = hints[1]

    assert middle is not None
    assert middle.normalized_note_id == "note-1"
    assert middle.note_binding_strategy == "local_sequence_rebind"
    assert middle.ownership_binding_adjusted is True
    assert middle.ownership_binding_reason == "flanking_same_note_identity_anchors"
    assert middle.note_start_sec == 0.0
    assert middle.note_end_sec == 0.3
    assert middle.score_bridge_strategy == "local_sequence_rebind"
    assert (
        "future aligner re-bound low-confidence current note to flanking same-note anchors"
        in middle.score_hint_notes
    )

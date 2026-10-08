from __future__ import annotations

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.pipeline.future_aligner_models import (
    FutureAlignerAcousticEvidence,
    FutureAlignerCandidateScore,
    FutureAlignerHint,
)
from any2xsampa.pipeline.future_aligner_phrase_boundary import (
    build_phrase_boundary_candidate_debug,
)


def _segment() -> PhoneSegment:
    return PhoneSegment(
        time_range=TimeRange(0.2, 0.4),
        candidate_phones=[PhoneCandidate("aa", 0.9)],
        confidence=0.9,
        normalized_ipa="a",
    )


def _hint(*, note_id: str = "note-2") -> FutureAlignerHint:
    return FutureAlignerHint(
        normalized_track_id="track-1",
        normalized_part_id="part-1",
        normalized_note_id=note_id,
        part_local_note_rank=1,
        duration_ms=200.0,
        note_start_sec=0.15,
        note_end_sec=0.55,
        release_context="held",
        note_binding_strategy="identity",
        note_identity_match=False,
        score_bridge_strategy="identity",
        score_hint_confidence=0.73,
        score_hint_notes=["identity aligned"],
        source_normalized_note_id="note-1",
        source_note_identity_match=True,
        source_note_binding_strategy="time_overlap",
    )


def _acoustic_ready() -> FutureAlignerAcousticEvidence:
    return FutureAlignerAcousticEvidence(
        acoustic_confidence=0.91,
        timing_reliability=0.82,
        boundary_uncertain=False,
        boundary_pressure=0.82,
        boundary_transition_strength=0.84,
        boundary_conflict_likelihood=0.79,
        summary="ready",
    )


def test_build_phrase_boundary_candidate_debug_defaults_to_no_competition_shape() -> None:
    payload = build_phrase_boundary_candidate_debug(
        segment=_segment(),
        current=_hint(),
        candidate_scores=[
            FutureAlignerCandidateScore(
                note_id="note-1",
                rank=1,
                support_score=0.9,
                continuity_score=0.2,
                order_penalty=0.0,
                same_note_neighbors=1,
                left_neighbor_match=True,
                right_neighbor_match=False,
            )
        ],
        acoustic=_acoustic_ready(),
        mode="none",
        guarded=False,
        boundary_regime_debug={"ready": True},
        overlap_advantage=0.0,
        interior_advantage=0.0,
        phrase_window_pressure=0.0,
        current_candidate=None,
    )

    assert payload["key_blocker"] == "no_candidate_competition"
    assert payload["candidate_shapes"]["top_anchor_profile"] == "unavailable"
    assert payload["missing_preconditions"]["general"] == ["no_candidate_competition"]


def test_build_phrase_boundary_candidate_debug_extracts_chain_anchor_blockers_and_ranking() -> None:
    top = FutureAlignerCandidateScore(
        note_id="note-1",
        rank=1,
        support_score=0.82,
        continuity_score=0.31,
        order_penalty=0.03,
        same_note_neighbors=2,
        left_neighbor_match=True,
        right_neighbor_match=True,
        acoustic_score=0.12,
        note_span_score=0.16,
        note_span_overlap_ratio=0.97,
        note_span_interior_ratio=0.41,
        part_aware_note_span_used=True,
    )
    runner_up = FutureAlignerCandidateScore(
        note_id="note-2",
        rank=2,
        support_score=0.7,
        continuity_score=0.21,
        order_penalty=0.02,
        same_note_neighbors=1,
        left_neighbor_match=True,
        right_neighbor_match=False,
        acoustic_score=0.08,
        note_span_score=0.05,
        note_span_overlap_ratio=0.22,
        note_span_interior_ratio=0.05,
        part_aware_note_span_used=True,
    )
    boundary_regime_debug = {
        "ready": True,
        "boundary_uncertain": False,
        "effective_boundary_uncertain": False,
        "boundary_pressure": 0.82,
        "boundary_transition_strength": 0.84,
        "boundary_conflict_likelihood": 0.79,
        "primary_blocker": "none",
        "uncertainty_normalization": {"applied": False},
        "preconditions": {"activation_signal_present": True},
    }

    payload = build_phrase_boundary_candidate_debug(
        segment=_segment(),
        current=_hint(note_id="note-2"),
        candidate_scores=[top, runner_up],
        acoustic=_acoustic_ready(),
        mode="none",
        guarded=False,
        boundary_regime_debug=boundary_regime_debug,
        overlap_advantage=0.75,
        interior_advantage=0.43,
        phrase_window_pressure=0.41,
        current_candidate=runner_up,
    )

    assert payload["winner_state"]["winner_note_id"] == "note-1"
    assert payload["winner_state"]["current_note_candidate_rank"] == 2
    assert payload["candidate_ranking"]["top"]["note_id"] == "note-1"
    assert payload["candidate_ranking"]["runner_up"]["note_id"] == "note-2"
    assert payload["candidate_shapes"]["asymmetric_chain_anchor"] is False
    assert payload["candidate_shapes"]["asymmetric_chain_anchor_primary_blocker"] == "top_anchor_not_single_edge"
    assert payload["candidate_shapes"]["top_anchor_profile"] == "bidirectional"
    assert payload["candidate_shapes"]["runner_up_anchor_profile"] == "left_edge"
    assert "chain_anchor_shape_missing" in payload["blocker_categories"]["candidate_classification"]
    assert payload["missing_preconditions"]["general"] == []
    assert "asymmetric_chain_anchor_shape" in payload["missing_preconditions"]["asymmetric_chain_anchor"]
    assert payload["overlap_metrics"]["overlap_advantage"] == 0.75
    assert payload["boundary_regime"]["phrase_window_pressure"] == 0.41

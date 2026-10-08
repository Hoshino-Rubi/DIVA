from __future__ import annotations

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.pipeline.future_aligner_debug import (
    annotate_future_aligner_metadata,
    build_phrase_boundary_candidate_debug,
    phrase_boundary_blocker_summary,
    provenance_summary,
)
from any2xsampa.pipeline.future_aligner_reporting import future_aligner_report_payload
from any2xsampa.pipeline.future_aligner_models import (
    FutureAlignerCandidateScore,
    FutureAlignerEvidenceItem,
    FutureAlignerHint,
    FutureAlignerResult,
)


def test_future_aligner_debug_provenance_summary_formats_counts_and_types() -> None:
    items = [
        FutureAlignerEvidenceItem(
            evidence_type="same_note_continuity",
            source_segment_index=0,
            source_note_id="note-1",
            source_track_id="track-1",
            source_part_id="part-1",
            support_score=0.8,
            support_direction="left",
            accepted=True,
        ),
        FutureAlignerEvidenceItem(
            evidence_type="order_continuity",
            source_segment_index=2,
            source_note_id="note-1",
            source_track_id="track-1",
            source_part_id="part-1",
            support_score=0.2,
            support_direction="right",
            accepted=False,
        ),
    ]

    assert provenance_summary(items) == "accepted=1,rejected=1,left=1,right=0,types=same_note_continuity+order_continuity"


def test_annotate_future_aligner_metadata_populates_segment_contract_fields() -> None:
    segment = PhoneSegment(
        time_range=TimeRange(0.0, 0.1),
        candidate_phones=[PhoneCandidate("a", 0.9)],
        confidence=0.9,
        metadata={},
    )
    result = FutureAlignerResult(
        aligned_track_id="track-1",
        aligned_part_id="part-1",
        aligned_note_id="note-1",
        aligner_strategy="score_aware_aligner_v2",
        aligner_confidence=0.91,
        aligner_fallback_used=False,
        aligner_notes=["stable"],
        future_aligner_sequence_applied=True,
        future_aligner_sequence_score=1.2,
        future_aligner_order_penalty=0.1,
        future_aligner_provenance_applied=True,
        future_aligner_provenance_summary="accepted=1",
    )

    annotate_future_aligner_metadata(segment, result)

    assert segment.metadata["future_aligner_applied"] is True
    assert segment.metadata["future_aligner"]["aligned_note_id"] == "note-1"
    assert segment.metadata["future_aligner_strategy"] == "score_aware_aligner_v2"
    assert segment.metadata["future_aligner_provenance_summary"] == "accepted=1"
    assert segment.metadata["aligned_track_id"] == "track-1"


def test_annotate_future_aligner_metadata_writes_versioned_report_view() -> None:
    segment = PhoneSegment(
        time_range=TimeRange(0.2, 0.36),
        candidate_phones=[PhoneCandidate("a", 0.9)],
        confidence=0.9,
        metadata={},
    )
    result = FutureAlignerResult(
        aligned_track_id="track-1",
        aligned_part_id="part-1",
        aligned_note_id="note-top",
        aligner_strategy="score_aware_aligner_v2",
        aligner_confidence=0.91,
        aligner_fallback_used=False,
        aligner_notes=["stable"],
        aligner_v2_notes=["stable"],
        alignment_competition_summary="winner=note-top:score=1.420",
        ownership_conflict_summary="winner=note-top,runner_up=note-runner",
        ownership_conflict_candidates=[{"note_id": "note-top"}],
        ownership_resolution_basis="same_note_continuity",
        note_ownership_conflict=False,
        future_aligner_report={
            "report_schema_version": "future_aligner_report_v1",
            "alignment": {"aligned_note_id": "note-top"},
            "competition": {"summary": "winner=note-top:score=1.420"},
            "ownership": {"resolution_basis": "same_note_continuity"},
            "phrase_boundary": {"blocker_summary": ""},
        },
    )

    annotate_future_aligner_metadata(segment, result)
    report = future_aligner_report_payload(segment)

    assert segment.metadata["future_aligner_report"]["report_schema_version"] == "future_aligner_report_v1"
    assert report["alignment"]["aligned_note_id"] == "note-top"
    assert report["competition"]["summary"] == "winner=note-top:score=1.420"
    assert report["ownership"]["resolution_basis"] == "same_note_continuity"


def test_phrase_boundary_blocker_summary_formats_debug_surface() -> None:
    debug = {
        "mode": "none",
        "key_blocker": "boundary_regime_inactive",
        "winner_is_current_note": False,
        "winner_state": {"current_note_candidate_rank": 2},
        "score_hint_context": {"note_id": "note-2", "score_hint_confidence": 0.73, "note_binding_strategy": "identity"},
        "boundary_regime": {
            "ready": False,
            "boundary_uncertain": True,
            "effective_boundary_uncertain": False,
            "boundary_pressure": 0.41,
            "boundary_transition_strength": 0.52,
            "boundary_conflict_likelihood": 0.37,
            "primary_blocker": "boundary_uncertain",
            "uncertainty_normalization": {"applied": True},
        },
        "blocker_categories": {
            "data": ["winner_is_current_note"],
            "evidence": ["boundary_uncertain"],
            "candidate_classification": ["boundary_regime_inactive"],
        },
    }

    rendered = phrase_boundary_blocker_summary(debug, segment_start_sec=0.2, segment_end_sec=0.4)

    assert "segment=0.200-0.400" in rendered
    assert "key_blocker=boundary_regime_inactive" in rendered
    assert "hint=note_id=note-2;hint_conf=0.730;binding=identity" in rendered
    assert "blockers=data=winner_is_current_note;evidence=boundary_uncertain;candidate_classification=boundary_regime_inactive" in rendered


def test_build_phrase_boundary_candidate_debug_shapes_ranking_and_blockers() -> None:
    segment = PhoneSegment(
        time_range=TimeRange(0.2, 0.36),
        candidate_phones=[PhoneCandidate("a", 0.9)],
        confidence=0.9,
    )
    current = FutureAlignerHint(
        normalized_track_id="track-1",
        normalized_part_id="part-1",
        normalized_note_id="note-current",
        part_local_note_rank=1,
        duration_ms=160.0,
        note_start_sec=0.18,
        note_end_sec=0.42,
        release_context="held",
        note_binding_strategy="identity",
        note_identity_match=True,
        score_bridge_strategy="identity",
        score_hint_confidence=0.82,
        score_hint_notes=["identity aligned"],
    )
    top = FutureAlignerCandidateScore(
        note_id="note-top",
        rank=1,
        support_score=0.9,
        continuity_score=0.4,
        order_penalty=0.1,
        same_note_neighbors=2,
        left_neighbor_match=True,
        right_neighbor_match=False,
        acoustic_score=0.2,
        note_span_score=0.8,
        note_span_overlap_ratio=0.97,
        note_span_interior_ratio=0.4,
        part_aware_note_span_used=True,
    )
    runner_up = FutureAlignerCandidateScore(
        note_id="note-runner",
        rank=2,
        support_score=0.5,
        continuity_score=0.1,
        order_penalty=0.05,
        same_note_neighbors=1,
        left_neighbor_match=False,
        right_neighbor_match=True,
        acoustic_score=0.1,
        note_span_score=0.2,
        note_span_overlap_ratio=0.2,
        note_span_interior_ratio=0.05,
        part_aware_note_span_used=True,
    )

    payload = build_phrase_boundary_candidate_debug(
        segment=segment,
        current=current,
        ordered_candidates=[top, runner_up],
        current_candidate=None,
        acoustic=None,
        mode="none",
        guarded=False,
        has_acoustic=False,
        boundary_regime_debug={"ready": False, "primary_blocker": "boundary_regime_inactive"},
        overlap_advantage=0.77,
        interior_advantage=0.35,
        phrase_window_pressure=0.41,
    )

    assert payload["top_note_id"] == "note-top"
    assert payload["runner_up_note_id"] == "note-runner"
    assert payload["candidate_ranking"]["top"]["note_id"] == "note-top"
    assert payload["candidate_ranking"]["runner_up"]["note_id"] == "note-runner"
    assert payload["candidate_shapes"]["asymmetric_chain_anchor"] is True
    assert payload["candidate_shapes"]["top_anchor_profile"] == "left_edge"
    assert payload["candidate_shapes"]["runner_up_anchor_profile"] == "right_edge"
    assert payload["missing_preconditions"]["general"] == ["has_acoustic_evidence", "boundary_regime"]
    assert "missing_acoustic_evidence" in payload["blocker_categories"]["evidence"]
    assert payload["key_blocker"] == "boundary_regime_inactive"

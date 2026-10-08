from __future__ import annotations

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.pipeline.future_aligner import FutureAlignerResult
from any2xsampa.pipeline.future_aligner_models import FutureAlignerEvidenceItem, future_aligner_output_payload
from any2xsampa.pipeline.future_aligner_reporting import future_aligner_report_payload


def test_future_aligner_models_round_trip_metadata_and_payload_helper() -> None:
    item = FutureAlignerEvidenceItem(
        evidence_type="same_note_continuity",
        source_segment_index=0,
        source_note_id="note-1",
        source_track_id="track-1",
        source_part_id="part-1",
        support_score=0.9123,
        support_direction="left",
        accepted=True,
    )
    result = FutureAlignerResult(
        aligned_track_id="track-1",
        aligned_part_id="part-1",
        aligned_note_id="note-1",
        aligner_strategy="score_aware_aligner_v2",
        aligner_confidence=0.9123,
        aligner_fallback_used=False,
        evidence_items=[item],
        future_aligner_provenance_applied=True,
        future_aligner_provenance_summary="accepted=1",
    )
    segment = PhoneSegment(
        time_range=TimeRange(start_sec=0.0, end_sec=0.1),
        candidate_phones=[PhoneCandidate(symbol="a", confidence=0.9)],
        confidence=0.9,
        metadata={"future_aligner": result.to_metadata()},
    )

    payload = future_aligner_output_payload(segment)

    assert payload["aligned_note_id"] == "note-1"
    assert payload["evidence_items"][0]["support_score"] == 0.912
    assert payload["future_aligner_provenance_summary"] == "accepted=1"


def test_future_aligner_report_payload_falls_back_to_legacy_future_aligner_output() -> None:
    segment = PhoneSegment(
        time_range=TimeRange(start_sec=0.2, end_sec=0.4),
        candidate_phones=[PhoneCandidate(symbol="a", confidence=0.9)],
        confidence=0.9,
        metadata={
            "future_aligner": {
                "aligned_track_id": "track-1",
                "aligned_part_id": "part-1",
                "aligned_note_id": "note-2",
                "aligner_strategy": "score_aware_aligner_v2",
                "aligner_confidence": 0.87,
                "aligner_fallback_used": False,
                "aligner_v2_notes": ["summary present"],
                "alignment_competition_summary": "winner=note-2:score=1.220",
                "ownership_conflict_summary": "winner=note-2,runner_up=note-1",
                "ownership_conflict_candidates": [{"note_id": "note-2"}],
                "ownership_resolution_basis": "same_note_continuity",
                "boundary_conflict_likely": False,
                "note_ownership_conflict": False,
                "phrase_boundary_candidate_debug": {"mode": "none"},
                "phrase_boundary_blocker_summary": "segment=0.200-0.400 | key_blocker=no_candidate_competition",
                "selective_release_safety": {"release_mode": "none"},
            }
        },
    )

    report = future_aligner_report_payload(segment)

    assert report["report_schema_version"] == "future_aligner_report_v1"
    assert report["alignment"]["aligned_note_id"] == "note-2"
    assert report["competition"]["summary"] == "winner=note-2:score=1.220"
    assert report["ownership"]["resolution_basis"] == "same_note_continuity"
    assert report["phrase_boundary"]["blocker_summary"].startswith("segment=0.200-0.400")

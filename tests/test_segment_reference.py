from __future__ import annotations

import copy

import pytest

from any2xsampa.training.limited_data_artifacts import sha256_object
from any2xsampa.training.segment_reference import import_responses, validate_segment_reference


def _source() -> dict:
    notes = [
        {"note_id": "n1", "mapping_status": "resolved_proxy", "model_target_ipa": ["m", "a"]},
        {"note_id": "n2", "mapping_status": "resolved_proxy", "model_target_ipa": ["t", "s"]},
        {"note_id": "n3", "mapping_status": "resolved_proxy", "model_target_ipa": ["o"]},
    ]
    return {"songs": [{"song_id": "song", "split": "train", "original_audio": {"path": "x.wav", "sha256": "a" * 64}, "human_vsqx": {"sha256": "b" * 64}, "notes": notes}], "original_clips": [{"example_id": "ex", "song_id": "song", "split": "train", "start_sec": 1.0, "end_sec": 11.0}]}


def _packet() -> dict:
    return {"packet_sha256": "packet", "test_songs_opened": False, "items": [{"review_id": "R01", "example_id": "ex", "song_id": "song", "original_cut_sec": 5.0}]}


def test_confirmed_segment_requires_internal_phones_and_preserves_edge_subset() -> None:
    source = _source()
    responses = {"schema_version": "original_segment_review_responses_v2", "packet_sha256": "packet", "responses": [{
        "review_id": "R01", "status": "human_confirmed", "reviewer": "u", "review_statement": "听到 n1 尾部到 n3 开头", "boundary_state": "truncated_both", "note_ids": ["n1", "n2", "n3"],
        "phones": [
            {"note_id": "n1", "phone_index": 1, "phone": "a", "audible": True, "edge_status": "truncated_start"},
            {"note_id": "n2", "phone_index": 0, "phone": "t", "audible": True}, {"note_id": "n2", "phone_index": 1, "phone": "s", "audible": True},
            {"note_id": "n3", "phone_index": 0, "phone": "o", "audible": True, "edge_status": "truncated_end"},
        ],
    }]}
    result = import_responses(packet=_packet(), responses=responses, source_reference=source, source_reference_hash="c" * 64)
    assert result["human_confirmed_count"] == 1
    assert [row["phone"] for row in result["examples"][0]["phones"]] == ["a", "t", "s", "o"]
    validate_segment_reference(result)


def test_internal_note_cannot_drop_a_phone() -> None:
    source = _source()
    responses = {"schema_version": "original_segment_review_responses_v2", "packet_sha256": "packet", "responses": [{
        "review_id": "R01", "status": "human_confirmed", "reviewer": "u", "review_statement": "x", "boundary_state": "truncated_both", "note_ids": ["n1", "n2", "n3"],
        "phones": [{"note_id": "n1", "phone_index": 0, "phone": "m", "audible": True}, {"note_id": "n2", "phone_index": 0, "phone": "t", "audible": True}, {"note_id": "n3", "phone_index": 0, "phone": "o", "audible": True}],
    }]}
    with pytest.raises(ValueError, match="internal notes"):
        import_responses(packet=_packet(), responses=responses, source_reference=source, source_reference_hash="c" * 64)


def test_pending_response_has_no_phones() -> None:
    source = _source()
    responses = {"schema_version": "original_segment_review_responses_v2", "packet_sha256": "packet", "responses": [{"review_id": "R01", "status": "pending", "note_ids": [], "phones": []}]}
    result = import_responses(packet=_packet(), responses=responses, source_reference=source, source_reference_hash="c" * 64)
    assert result["examples"][0]["status"] == "pending"
    assert result["examples"][0]["phones"] == []


def test_truncated_end_keeps_first_note_complete() -> None:
    source = _source()
    responses = {"schema_version": "original_segment_review_responses_v2", "packet_sha256": "packet", "responses": [{
        "review_id": "R01", "status": "human_confirmed", "reviewer": "u", "review_statement": "x", "boundary_state": "truncated_end", "note_ids": ["n1", "n2"],
        "phones": [{"note_id": "n1", "phone_index": 0, "phone": "m", "audible": True}, {"note_id": "n1", "phone_index": 1, "phone": "a", "audible": True}, {"note_id": "n2", "phone_index": 0, "phone": "t", "audible": True}],
    }]}
    result = import_responses(packet=_packet(), responses=responses, source_reference=source, source_reference_hash="c" * 64)
    assert len(result["examples"][0]["phones"]) == 3

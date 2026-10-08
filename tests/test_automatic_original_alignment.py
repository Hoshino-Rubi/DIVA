from __future__ import annotations

import pytest

from any2xsampa.training.automatic_original_alignment import (
    align_candidate,
    build_machine_alignment,
)


def _song(split: str = "train") -> dict:
    return {
        "song_id": "song-a",
        "split": split,
        "original_audio": {"path": "original.wav", "sha256": "a" * 64},
        "human_vsqx": {"sha256": "b" * 64},
        "notes": [
            {"note_id": "song-a-n1", "mapping_status": "resolved_proxy", "model_target_ipa": ["m", "a"], "raw_lyric": "明"},
            {"note_id": "song-a-n2", "mapping_status": "resolved_proxy", "model_target_ipa": ["ŋ"], "raw_lyric": "天"},
        ],
    }


def _candidate(*, note_ids=None, disagreement: float = 0.1) -> dict:
    note_ids = note_ids or ["song-a-n1", "song-a-n2"]
    return {
        "candidate_id": "candidate-1",
        "example_id": "song-a-original-01",
        "song_id": "song-a",
        "split": "train",
        "start_sec": 10.0,
        "end_sec": 20.0,
        "candidate_references": [
            {"factor": 4, "rendered_start_sec": 30.0, "rendered_end_sec": 40.0,
             "note_ids": note_ids, "mapping_complete": True, "valid_rendered_range": True},
            {"factor": 8, "rendered_start_sec": 30.0 + disagreement, "rendered_end_sec": 40.0 + disagreement,
             "note_ids": note_ids, "mapping_complete": True, "valid_rendered_range": True},
        ],
    }


def _correspondence(similarity: float = 0.8) -> dict:
    return {
        "song_id": "song-a",
        "split": "train",
        "clips": [{"example_id": "song-a-original-01", "resolution_consistent": True, "resolution_disagreement_sec": 0.1}],
        "probes": [{"validation": [{"example_id": "song-a-original-01", "local_best": {"cosine_similarity": similarity}}]}],
    }


def test_align_candidate_transfers_phones_without_human_claim() -> None:
    row = align_candidate(_candidate(), song=_song(), correspondence=_correspondence(), reference_manifest_sha256="c" * 64)
    assert row["status"] == "machine_aligned"
    assert row["reference_kind"] == "machine_aligned_proxy"
    assert row["human_confirmed"] is False
    assert row["official_evaluation_eligible"] is False
    assert [(phone["note_id"], phone["phone_index"], phone["phone"]) for phone in row["phones"]] == [
        ("song-a-n1", 0, "m"), ("song-a-n1", 1, "a"), ("song-a-n2", 0, "ŋ")
    ]


def test_large_resolution_disagreement_enters_exception_queue() -> None:
    row = align_candidate(_candidate(disagreement=0.25), song=_song(), correspondence=_correspondence(), reference_manifest_sha256="c" * 64)
    assert row["status"] == "needs_review"
    assert row["human_review_required"] is True
    assert row["training_eligible"] is False


def test_unresolved_note_enters_exception_queue_without_training() -> None:
    result = build_machine_alignment(
        reference={"songs": [_song()]},
        candidates=[_candidate(note_ids=["song-a-n-missing"])],
        correspondences={"song-a": _correspondence()},
        source_bindings={"reference": {"sha256": "c" * 64}},
    )
    assert result["counts"] == {"total": 1, "machine_aligned": 0, "needs_review": 1, "rejected": 0}
    assert result["errors"][0]["status"] == "needs_review"
    assert result["errors"][0]["training_eligible"] is False


def test_test_song_is_never_machine_aligned() -> None:
    with pytest.raises(ValueError, match="sealed test"):
        align_candidate(_candidate(), song=_song("test"), correspondence=_correspondence(), reference_manifest_sha256="c" * 64)

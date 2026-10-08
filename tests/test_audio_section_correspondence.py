from __future__ import annotations

import pytest
np = pytest.importorskip("numpy")
pytest.importorskip("librosa")

from any2xsampa.training.audio_section_correspondence import match_song, propose_correspondence


def test_subsequence_matching_recovers_known_offset_without_claiming_truth() -> None:
    rng = np.random.default_rng(17)
    query = rng.random((12, 40), dtype=np.float32)
    reference = np.concatenate((rng.random((12, 70)), query, rng.random((12, 90))), axis=1)
    result = propose_correspondence(query, reference, query_start_sec=80, frame_seconds=0.1)
    best = result["candidates"][0]
    assert best["rendered_start_sec"] == pytest.approx(7.0)
    assert best["rendered_end_sec"] == pytest.approx(11.0)
    assert best["source_span_coverage_fraction"] == 1
    assert best["alignment_status"] == "machine_proposed"
    assert not result["original_phone_truth_established"]
    assert not result["training_or_accuracy_evaluation_allowed"]


def test_matching_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        propose_correspondence(np.full((12, 5), np.nan), np.ones((12, 50)), query_start_sec=0)


def test_sealed_test_cannot_be_opened_for_matching(tmp_path) -> None:
    with pytest.raises(ValueError, match="sealed"):
        match_song({"split": "test"}, [], tmp_path)


def test_constant_cost_does_not_reward_time_compression() -> None:
    query = np.tile([[1.0], [0.0]], (1, 40))
    reference = np.tile([[0.0], [1.0]], (1, 200))
    result = propose_correspondence(query, reference, query_start_sec=0, frame_seconds=0.1)
    assert result["candidates"][0]["normalized_cost"] == pytest.approx(1.0)
    assert result["ambiguous"]
    assert not result["original_phone_truth_established"]

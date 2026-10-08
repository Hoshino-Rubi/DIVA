from __future__ import annotations

import copy
from pathlib import Path

import pytest

from any2xsampa.training.auxiliary_slicing import audit_timestamps, gap_candidates, normalize_word, select_window, validate_context
from any2xsampa.training.limited_data_artifacts import read_json


def policy() -> dict:
    return read_json(Path("src/any2xsampa/config/quality8h_v1.json"))["slicing"]


def context() -> dict:
    return {"context_id": "freedom-flower-original-01-context", "example_id": "freedom-flower-original-01", "song_id": "freedom-flower", "split": "train", "audio_path": "source.wav", "audio_sha256": "a" * 64,
            "start_sec": 4.0, "end_sec": 34.0, "old_start_sec": 10.0, "old_end_sec": 28.0, "contains_labels_or_score_context": False}


def words() -> list[dict]:
    return [{"text": "前", "start_sec": 0.0, "end_sec": 1.0},
            {"text": "中", "start_sec": 2.0, "end_sec": 18.0},
            {"text": "後", "start_sec": 19.0, "end_sec": 20.0}]


def energy() -> list[dict]:
    return [{"start_sec": i / 100, "end_sec": i / 100 + .02, "relative_peak_db": -50.0} for i in range(2000)]


def test_context_rejects_test_leakage_and_extra_reference_context() -> None:
    validate_context(context())
    for changes in ({"song_id": "king-of-karaoke", "split": "test"}, {"lyrics": "前中後"}, {"end_sec": 34.1}, {"contains_labels_or_score_context": True}):
        row = {**context(), **changes}
        with pytest.raises(ValueError):
            validate_context(row)


def test_normalization_does_not_translate_or_silently_rewrite_words() -> None:
    assert normalize_word(" 前，後！ ") == "前後"
    assert normalize_word("後") != normalize_word("后")
    assert normalize_word("u\u0308") == "ü"


@pytest.mark.parametrize("changes,reason", [
    ({"end_sec": 0.0}, "non_positive_duration"),
    ({"start_sec": -1.0}, "timestamp_outside_audio"),
    ({"start_sec": float("nan")}, "non_finite_or_non_numeric_timestamp"),
    ({"start_sec": True}, "non_finite_or_non_numeric_timestamp"),
    ({"text": "，"}, "empty_word"),
])
def test_invalid_timestamps_are_retained_not_repaired(changes: dict, reason: str) -> None:
    rows = words()
    rows[0].update(changes)
    report = audit_timestamps(rows, 20)
    assert reason in report["rows"][0]["reasons"]
    assert len(report["rows"]) == 3 and not report["repaired_timestamps"]


def test_nested_overlap_invalidates_both_anchors() -> None:
    report = audit_timestamps([{"text": "長", "start_sec": 0, "end_sec": 8}, {"text": "短", "start_sec": 1, "end_sec": 2}, {"text": "尾", "start_sec": 4, "end_sec": 5}], 10)
    assert report["invalid_indices"] == [0, 1, 2]


def test_complete_word_window_is_only_a_machine_proposal() -> None:
    timestamps = audit_timestamps(words(), 20)
    gaps = gap_candidates(timestamps, timestamps, energy(), policy())
    assert all(r["eligible"] for r in gaps["candidates"])
    report = select_window(gaps, timestamps, timestamps, old_start=1, old_end=19, policy=policy())
    assert report["selected"]["start_sec"] == 1.5
    assert report["selected"]["end_sec"] == 18.5
    assert not report["human_approved"] and not report["official_original_accuracy_claim_allowed"]


def test_loud_gap_and_unaligned_boundary_are_rejected() -> None:
    timestamps = audit_timestamps(words(), 20)
    loud = [{**r, "relative_peak_db": -10} for r in energy()]
    assert not any(r["eligible"] for r in gap_candidates(timestamps, timestamps, loud, policy())["candidates"])
    other = copy.deepcopy(words())
    other[0]["end_sec"] += .21
    rows = gap_candidates(timestamps, audit_timestamps(other, 20), energy(), policy())["candidates"]
    assert "boundary_disagreement" in rows[0]["reasons"]


def test_missing_word_cannot_be_skipped_to_create_a_safe_cut() -> None:
    left = audit_timestamps(words(), 20)
    extra = words()
    extra.insert(1, {"text": "漏", "start_sec": 1.1, "end_sec": 1.2})
    right = audit_timestamps(extra, 20)
    gaps = gap_candidates(left, right, energy(), policy())
    assert "unmatched_words_between_gap_anchors" in gaps["candidates"][0]["reasons"]
    assert not select_window(gaps, left, right, old_start=1, old_end=19, policy=policy())["selected"]


def test_empty_timestamp_output_has_no_fabricated_pause() -> None:
    empty = audit_timestamps([], 20)
    assert empty["empty_output"]
    gaps = gap_candidates(empty, empty, [], policy())
    assert not gaps["candidates"]


def test_energy_hole_cannot_be_hidden_by_matching_first_and_last_frame() -> None:
    timestamps = audit_timestamps(words(), 20)
    interrupted = [r for r in energy() if not 1.45 <= r["start_sec"] <= 1.55]
    gaps = gap_candidates(timestamps, timestamps, interrupted, policy())
    assert "vocal_energy_does_not_support_pause" in gaps["candidates"][0]["reasons"]

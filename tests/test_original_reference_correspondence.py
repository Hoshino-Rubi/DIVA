from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("librosa")

from any2xsampa.training.original_reference_correspondence import global_path, load_policy, map_clip, match_cached_song


def test_identity_timing_and_no_automatic_review() -> None:
    values = np.random.default_rng(17).random((12, 100))
    policy = load_policy()["correspondence"]
    path = global_path(values, values, factor=1, policy=policy)
    mapped = map_clip(path, 0.4, 1.2)
    assert mapped["rendered_start_sec"] == pytest.approx(0.4)
    assert mapped["rendered_end_sec"] == pytest.approx(1.2)
    assert not path["human_reviewed"]
    assert not path["phone_model_outputs_used"]


def test_global_order_disambiguates_repeated_sections() -> None:
    motif = np.random.default_rng(29).random((12, 40))
    values = np.concatenate([motif, motif, motif], axis=1)
    path = global_path(values, values, factor=1, policy=load_policy()["correspondence"])
    assert [r["rendered_sec"] for r in path["anchors"]] == pytest.approx([r["original_sec"] for r in path["anchors"]])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        global_path(np.full((12, 20), bad), np.ones((12, 20)), factor=1, policy=load_policy()["correspondence"])


def test_resource_bound_before_cost_allocation() -> None:
    policy = {**load_policy()["correspondence"], "maximum_cost_matrix_cells": 100}
    with pytest.raises(ValueError, match="resource budget"):
        global_path(np.ones((12, 20)), np.ones((12, 20)), factor=1, policy=policy)


def test_sealed_song_rejected_before_opening_cache() -> None:
    with pytest.raises(ValueError, match="sealed"):
        match_cached_song({"split": "test"}, {}, [], load_policy()["correspondence"])


def test_no_timing_extrapolation() -> None:
    with pytest.raises(ValueError, match="outside"):
        map_clip({"anchors": [{"original_sec": 1}, {"original_sec": 2}]}, 0, 1.5)

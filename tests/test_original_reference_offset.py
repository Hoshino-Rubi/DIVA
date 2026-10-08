from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from any2xsampa.training.limited_data_artifacts import read_json
from any2xsampa.training.original_reference_offset import offset_probe


def policy() -> dict:
    return read_json(Path(__file__).resolve().parents[1] / "src/any2xsampa/config/original_reference_offset_probe_v1.json")


def test_recovers_known_offset_without_fitting_evaluation_clip() -> None:
    rng = np.random.default_rng(43)
    x = rng.random((12, 100))
    y = np.concatenate([rng.random((12, 20)), x, rng.random((12, 20))], axis=1)
    result = offset_probe(x, y, [{"example_id": "clip", "start_sec": 4, "end_sec": 6}], factor=1, frame_seconds=0.1, policy=policy())
    assert result["best"]["offset_sec"] == pytest.approx(2)
    assert result["excluded_validation_frames"] == 20
    assert not result["validation"][0]["fit_used_this_clip"]
    assert not result["human_reviewed"]


def test_constant_descriptors_remain_ambiguous() -> None:
    result = offset_probe(np.ones((12, 100)), np.ones((12, 100)), [], factor=1, frame_seconds=0.1, policy=policy())
    assert result["absolute_similarity_margin"] == pytest.approx(0)
    assert not result["official_evaluation_eligible"]


def test_requires_independent_fit_audio() -> None:
    with pytest.raises(ValueError, match="independent fit"):
        offset_probe(np.ones((12, 100)), np.ones((12, 100)), [{"start_sec": 0, "end_sec": 11}], factor=1, frame_seconds=0.1, policy=policy())


def test_search_resource_bound() -> None:
    with pytest.raises(ValueError, match="resource"):
        offset_probe(np.ones((12, 100)), np.ones((12, 100)), [], factor=1, frame_seconds=0.1, policy={**policy(), "maximum_lag_frame_products": 10})


def test_nonfinite_descriptor_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        offset_probe(np.full((12, 100), np.nan), np.ones((12, 100)), [], factor=1, frame_seconds=0.1, policy=policy())

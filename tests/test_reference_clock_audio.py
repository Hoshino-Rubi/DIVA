from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from any2xsampa.training.limited_data_artifacts import read_json
from any2xsampa.training.reference_clock_audio import common_support_offset, score_activity_offset, validate_offset_probes


def policy() -> dict:
    return read_json(Path(__file__).resolve().parents[1] / "src/any2xsampa/config/shalong_clock_audit_v1.json")


def offset_policy() -> dict:
    return {**policy()["audio_offset"], "descriptor_sample_rate": 10, "descriptor_hop_samples": 1,
            "maximum_offset_sec": 2, "minimum_common_fit_frames": 20, "fit_clip_exclusion_padding_sec": .4}


def test_known_positive_lag_with_equal_support_and_purge() -> None:
    rng = np.random.default_rng(17)
    x = rng.random((12, 150))
    y = np.concatenate((rng.random((12, 10)), x), axis=1)
    result = common_support_offset(x, y, [{"example_id": "v", "start_sec": 5, "end_sec": 7}], factor=1, policy=offset_policy())
    assert result["best"]["offset_sec"] == pytest.approx(1.0)
    assert len({r["fit_frames"] for r in result["curve"]}) == 1
    assert not any(4.6 <= i*.1 < 7.4 for i in result["support_original_frame_indices"])
    assert result["validation"][0]["deviation_from_fit_sec"] == 0


def test_bad_validation_cannot_refit_a_better_offset() -> None:
    p = policy()["audio_offset"]
    probes = [{"factor": f, "best": {"offset_sec": 1}, "frame_seconds": .1, "margin": .2,
               "validation": [{"best": {"offset_sec": 1.5}, "deviation_from_fit_sec": .5}]} for f in p["factors"]]
    result = validate_offset_probes(probes, p)
    assert result["point_offset_sec"] == 1
    assert not result["passed"]
    assert "constant_offset_not_stable_across_fixed_clips" in result["failure_reasons"]


def test_ambiguous_constant_audio_not_accepted() -> None:
    p = {**offset_policy(), "factors": [1, 2]}
    probes = [common_support_offset(np.ones((12, 150)), np.ones((12, 150)), [], factor=f, policy=p) for f in p["factors"]]
    assert "nonlocal_offset_ambiguity" in validate_offset_probes(probes, p)["failure_reasons"]


def test_resource_cap_before_curve() -> None:
    with pytest.raises(ValueError, match="resource"):
        common_support_offset(np.ones((12, 150)), np.ones((12, 150)), [], factor=1, policy={**offset_policy(), "maximum_frame_lag_products": 1})


def test_nonfinite_audio_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        common_support_offset(np.full((12, 100), np.nan), np.ones((12, 100)), [], factor=1, policy=offset_policy())


def test_activity_clock_recovers_render_delay() -> None:
    step = .01
    n = [{"start_sec": v, "end_sec": v+.4} for v in (2.1, 3.2, 4.8, 6.0, 7.1)]
    centers = (np.arange(1000) + .5) * step
    rms = np.full(1000, .0001)
    for row in n:
        rms[(centers >= row["start_sec"] + .3) & (centers < row["end_sec"] + .3)] = .5
    result = score_activity_offset(rms, n, step, {**policy()["rendered_to_score"], "maximum_offset_sec": 1})
    assert result["point_offset_sec"] == pytest.approx(.3)
    assert result["passed"]
    assert not result["phone_boundaries_validated"]


def test_all_active_score_is_not_an_identifiable_clock() -> None:
    with pytest.raises(ValueError, match="notes and rests"):
        score_activity_offset(np.ones(1000), [{"start_sec": 0, "end_sec": 10}], .01, {**policy()["rendered_to_score"], "maximum_offset_sec": 1})


def test_silent_render_has_no_clock() -> None:
    with pytest.raises(ValueError, match="invalid rendered activity"):
        score_activity_offset(np.zeros(1000), [], .01, policy()["rendered_to_score"])


@pytest.mark.parametrize("field", ["margin", "frame_seconds"])
def test_nan_cannot_pass_clock_gate(field) -> None:
    p = policy()["audio_offset"]
    probes = [{"factor": f, "best": {"offset_sec": 1}, "frame_seconds": .1, "margin": .2,
               "validation": [{"best": {"offset_sec": 1}, "deviation_from_fit_sec": 0}]} for f in p["factors"]]
    probes[0][field] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_offset_probes(probes, p)

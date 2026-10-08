from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from any2xsampa.training.reference_clock import ClockOffset, bind_score_notes, boundary_audit, exact_sample, score_clocks, safe_expanding_crop, membership_probe_offsets


def vsqx(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "score.vsqx"
    path.write_text(f'''<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/">
    <masterTrack><resolution>480</resolution><preMeasure>4</preMeasure>
    <timeSig><m>0</m><nu>4</nu><de>4</de></timeSig><tempo><t>0</t><v>14200</v></tempo>{extra}</masterTrack>
    <vsTrack><vsPart><t>7680</t><note><t>8640</t><dur>1920</dur></note></vsPart></vsTrack></vsq4>''', encoding="utf-8")
    return path


def test_exact_clock_composition_not_double_premeasure() -> None:
    a = ClockOffset("original", "rendered", Fraction(480, 71))
    b = ClockOffset("rendered", "score_absolute", Fraction(0))
    combined = a.then(b)
    assert combined.at(10) == Fraction(1190, 71)
    assert a.inverse().at(float(a.at(10))) == pytest.approx(Fraction(10))
    with pytest.raises(ValueError, match="domains"):
        b.then(a)


def test_premeasure_is_metadata_not_an_audio_origin(tmp_path) -> None:
    score = score_clocks(vsqx(tmp_path))
    assert score["premeasure_seconds"] == pytest.approx(480/71)
    note = score["notes"][0]
    assert note["absolute_start_sec"] == pytest.approx(16320/480*60/142)
    assert note["absolute_start_sec"] - note["display_start_sec"] == pytest.approx(480/71)
    assert not score["render_audio_origin_inferred_from_metadata"]


def test_changed_tempo_requires_separate_clock_version(tmp_path) -> None:
    with pytest.raises(ValueError, match="constant-tempo"):
        score_clocks(vsqx(tmp_path, '<tempo><t>1000</t><v>12000</v></tempo>'))


def test_score_reference_clock_mixup_rejected(tmp_path) -> None:
    score = score_clocks(vsqx(tmp_path))
    n = score["notes"][0]
    with pytest.raises(ValueError, match="absolute clock"):
        bind_score_notes(score, [{"start_sec": n["display_start_sec"], "end_sec": n["display_end_sec"]}])


@pytest.mark.parametrize("side,expected", [("floor", 1), ("ceil", 2)])
def test_sample_rounding_is_explicit(side, expected) -> None:
    assert exact_sample(0.000015, 96000, side=side) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_illegal_clock_values(value) -> None:
    with pytest.raises(ValueError):
        ClockOffset("a", "b", Fraction(0)).at(value)


def notes() -> list[dict]:
    return [{"note_id": str(i), "start_sec": i, "end_sec": i + .75,
             "unresolved_tokens": [], "mapping_status": "resolved_proxy"} for i in range(1, 5)]


def test_full_notes_inside_silence_boundaries() -> None:
    result = boundary_audit(notes(), start=.9, end=3.9, offset_low=-.01, offset_high=.01)
    assert result["boundary_complete"]
    assert result["fully_contained_note_ids"] == ["1", "2", "3"]


def test_cut_note_not_turned_into_full_reference() -> None:
    result = boundary_audit(notes(), start=1.4, end=3.9, offset_low=0, offset_high=0)
    assert result["partial_or_uncertain_edge_note_ids"] == ["1"]
    assert not result["boundary_complete"]
    assert not result["partial_notes_removed_to_improve_score"]


def test_offset_interval_checks_notes_between_endpoints() -> None:
    # A note can be in the crop for an intermediate offset but neither endpoint.
    result = boundary_audit(notes(), start=0, end=.1, offset_low=0, offset_high=5)
    assert result["all_potential_note_ids"] == ["1", "2", "3", "4"]
    assert not result["boundary_complete"]


def test_unknown_edge_note_cannot_be_dropped() -> None:
    rows = notes()
    rows[0].update(unresolved_tokens=["7"], mapping_status="unresolved")
    result = boundary_audit(rows, start=1.4, end=3.9, offset_low=0, offset_high=0)
    assert result["unresolved_tokens"] == ["7"]
    assert "whole_window_contains_unresolved_mapping" in result["exclusion_reasons"]


def test_half_open_boundaries() -> None:
    result = boundary_audit(notes(), start=1, end=2, offset_low=0, offset_high=0)
    assert result["all_potential_note_ids"] == ["1"]
    assert result["boundary_complete"]


def test_safe_recrop_preserves_every_old_audio_sample() -> None:
    rows = notes()
    result = safe_expanding_crop(rows, start=2.1, end=3.4, offset_low=0, offset_high=0, uncertainty=0,
                                policy={"maximum_endpoint_expansion_sec": 2, "maximum_duration_sec": 20})
    assert result["status"] == "candidate_only"
    assert result["start_sec"] <= 2.1 and result["end_sec"] >= 3.4
    assert not result["old_audio_content_discarded"]
    assert not result["official_evaluation_eligible"]


def test_no_recrop_if_rest_narrower_than_clock_uncertainty() -> None:
    result = safe_expanding_crop(notes(), start=2.1, end=3.4, offset_low=-.3, offset_high=.3, uncertainty=.15,
                                policy={"maximum_endpoint_expansion_sec": 2, "maximum_duration_sec": 20})
    assert result["status"] == "no_safe_expanding_rest_pair"


def test_safe_crop_cannot_hide_duration_budget() -> None:
    result = safe_expanding_crop(notes(), start=2.1, end=3.4, offset_low=0, offset_high=0, uncertainty=0,
                                policy={"maximum_endpoint_expansion_sec": 2, "maximum_duration_sec": 1.5})
    assert "expanded_crop_exceeds_locked_runtime_duration" in result["exclusion_reasons"]


def test_sensitivity_checks_interior_membership_change() -> None:
    offsets = membership_probe_offsets(notes(), start=0, end=.1, low=0, high=4)
    assert .9 in offsets and 1.75 in offsets
    assert any(.9 < x < 1.75 for x in offsets)

"""Singer range changes are export arrangements, not corrected model evidence."""
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from any2xsampa.local.common import read, sha, write
from any2xsampa.local.score_tasks import export_saved_score
from any2xsampa.local.vocal_range import apply_export_pitch


def tree_for(pitches=(40, 56, 61, 73), durations=(394, 42883, 25949, 1670)):
    root = ET.Element("vsq4")
    part = ET.SubElement(ET.SubElement(root, "vsTrack"), "vsPart")
    for index, (pitch, duration) in enumerate(zip(pitches, durations, strict=True)):
        note = ET.SubElement(part, "note")
        for name, value in (("t", index * 50000), ("dur", duration), ("n", pitch),
                            ("v", 64), ("y", "原始歌词"), ("p", "a")):
            ET.SubElement(note, name).text = str(value)
    return ET.ElementTree(root)


def pitches(tree):
    return [int(node.text) for node in tree.findall(".//note/n")]


def test_miku_adaptation_uses_one_octave_and_changes_no_other_note_fields():
    tree = tree_for()
    original = deepcopy(tree.getroot())
    report = apply_export_pitch(tree, mode="miku_v4c_octave")
    assert report["transpose_semitones"] == 12
    assert pitches(tree) == [52, 68, 73, 85]
    assert report["before"]["below_range_note_count"] == 2
    assert report["after"]["below_range_note_count"] == 1
    assert report["after"]["above_range_note_count"] == 1
    assert report["individual_notes_clamped_or_octave_folded"] is False
    for node in original.findall(".//note/n"):
        node.text = str(int(node.text) + 12)
    assert ET.tostring(tree.getroot()) == ET.tostring(original)


@pytest.mark.parametrize(("values", "shift"), [((61, 67, 73), 0), ((85, 88, 91), -24), ((40,), 24)])
def test_octave_fit_uses_midi_values_not_octave_labels(values, shift):
    tree = tree_for(values, (480,) * len(values))
    report = apply_export_pitch(tree, mode="miku_v4c_octave")
    assert report["transpose_semitones"] == shift
    assert pitches(tree) == [value + shift for value in values]


def test_original_mode_keeps_every_pitch():
    tree = tree_for()
    tree.find(".//note/n").text = "040"
    before = ET.tostring(tree.getroot())
    report = apply_export_pitch(tree)
    assert ET.tostring(tree.getroot()) == before
    assert report["transpose_semitones"] == 0


def test_automatic_adaptation_is_not_applied_again_to_an_adapted_register():
    tree = tree_for()
    assert apply_export_pitch(tree, mode="miku_v4c_octave")["transpose_semitones"] == 12
    adapted = ET.tostring(tree.getroot())
    assert apply_export_pitch(tree, mode="miku_v4c_octave")["transpose_semitones"] == 0
    assert ET.tostring(tree.getroot()) == adapted


def test_manual_mode_uses_the_explicit_semitone_value():
    tree = tree_for()
    report = apply_export_pitch(tree, mode="manual", transpose_semitones=7)
    assert pitches(tree) == [47, 63, 68, 80]
    assert report["transpose_semitones"] == 7


@pytest.mark.parametrize("value", [True, 12.5, 37, -37, None])
def test_invalid_manual_transpose_fails_without_partial_pitch_edits(value):
    tree = tree_for()
    before = ET.tostring(tree.getroot())
    with pytest.raises(ValueError, match="integer"):
        apply_export_pitch(tree, mode="manual", transpose_semitones=value)
    assert ET.tostring(tree.getroot()) == before


def test_manual_transpose_refuses_midi_overflow_without_clipping():
    tree = tree_for((120, 127), (100, 100))
    before = ET.tostring(tree.getroot())
    with pytest.raises(ValueError, match="MIDI range"):
        apply_export_pitch(tree, mode="manual", transpose_semitones=12)
    assert ET.tostring(tree.getroot()) == before


def test_saved_export_does_not_change_model_score_or_original_vsqx(tmp_path: Path):
    score = tmp_path / "score.json"
    write(score, {"notes": [{"pitch": 40}, {"pitch": 56}, {"pitch": 61}, {"pitch": 73}]})
    source = tmp_path / "raw.vsqx"
    tree_for().write(source, encoding="utf-8")
    delivery = tmp_path / "DELIVERY.json"
    write(delivery, {"score_sha256": sha(score), "vsqx_path": str(source),
                     "vsqx_sha256": sha(source), "note_count": 4,
                     "source": "supplied_vsqx_phone_draft"})
    preserved = {path: path.read_bytes() for path in (score, source, delivery)}
    output = tmp_path / "adapted.vsqx"
    result = export_saved_score({"score_path": str(score), "score_sha256": sha(score),
                                 "output": str(output), "pitch_mode": "miku_v4c_octave"})
    assert pitches(ET.parse(output)) == [52, 68, 73, 85]
    assert result["pitch_adaptation"]["transpose_semitones"] == 12
    assert all(path.read_bytes() == data for path, data in preserved.items())
    proof = read(output.with_name(output.name + ".provenance.json"))
    assert proof["export_sha256"] == sha(output)
    assert proof["pitch_and_duration_preserved"] is False
    assert proof["duration_preserved"] is True
    assert proof["pitch_adaptation"]["raw_model_evidence_modified"] is False
    assert proof["model_forwards"] == 0

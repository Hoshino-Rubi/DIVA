from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from any2xsampa.vocaloid.vsqx import VSQ4_NS
from tools.build_miku_v4c_draft_vsqx import run_draft_vsqx

REPO_ROOT = Path(__file__).resolve().parents[1]
NS = {"v": VSQ4_NS}


def _template(path: Path) -> Path:
    root = ET.Element(f"{{{VSQ4_NS}}}vsq4")
    ET.SubElement(root, f"{{{VSQ4_NS}}}vender").text = "Yamaha corporation"
    ET.SubElement(root, f"{{{VSQ4_NS}}}version").text = "4.0.0.3"
    table = ET.SubElement(root, f"{{{VSQ4_NS}}}vVoiceTable")
    voice = ET.SubElement(table, f"{{{VSQ4_NS}}}vVoice")
    ET.SubElement(voice, f"{{{VSQ4_NS}}}bs").text = "1"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}pc").text = "11"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}id").text = "template"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}name").text = "MIKU_V4X_Original_EVEC"
    ET.SubElement(root, f"{{{VSQ4_NS}}}mixer")
    master = ET.SubElement(root, f"{{{VSQ4_NS}}}masterTrack")
    ET.SubElement(master, f"{{{VSQ4_NS}}}seqName").text = "template"
    ET.SubElement(master, f"{{{VSQ4_NS}}}comment").text = "template"
    ET.SubElement(master, f"{{{VSQ4_NS}}}resolution").text = "480"
    ET.SubElement(master, f"{{{VSQ4_NS}}}preMeasure").text = "1"
    ET.SubElement(master, f"{{{VSQ4_NS}}}timeSig")
    ET.SubElement(master, f"{{{VSQ4_NS}}}tempo")
    track = ET.SubElement(root, f"{{{VSQ4_NS}}}vsTrack")
    ET.SubElement(track, f"{{{VSQ4_NS}}}tNo").text = "0"
    ET.SubElement(track, f"{{{VSQ4_NS}}}name").text = "Track"
    ET.SubElement(track, f"{{{VSQ4_NS}}}comment").text = "Track"
    part = ET.SubElement(track, f"{{{VSQ4_NS}}}vsPart")
    ET.SubElement(part, f"{{{VSQ4_NS}}}t").text = "3840"
    ET.SubElement(part, f"{{{VSQ4_NS}}}playTime").text = "1920"
    ET.SubElement(part, f"{{{VSQ4_NS}}}name").text = "Part"
    ET.SubElement(part, f"{{{VSQ4_NS}}}comment").text = "Part"
    ET.SubElement(part, f"{{{VSQ4_NS}}}sPlug")
    ET.SubElement(part, f"{{{VSQ4_NS}}}singer")
    ET.SubElement(part, f"{{{VSQ4_NS}}}plane").text = "0"
    ET.SubElement(root, f"{{{VSQ4_NS}}}monoTrack")
    ET.SubElement(root, f"{{{VSQ4_NS}}}stTrack")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_runner_builds_template_cloned_vsqx_from_chinese_lyrics_without_user_jyutping(tmp_path: Path) -> None:
    template = _template(tmp_path / "template.vsqx")

    rc, summary = run_draft_vsqx(
        lyrics="喜歡你",
        template=template,
        output_root=tmp_path / "run",
        pitches=[64, 67, 69],
    )

    assert rc == 0
    assert summary["ok"] is True
    assert summary["product_stage"] == "draft_candidate"
    assert summary["model_output"] is False
    assert summary["training_data"] is False
    assert summary["user_jyutping_required"] is False
    assert summary["manual_vocaloid_import_required"] is True
    assert summary["phoneme_empirical_review_required"] is True
    assert summary["xsampa_role"] == "export_only_not_internal_truth"
    assert summary["export_summary"]["export_mode"] == "template_clone_import_safe"

    draft = json.loads(Path(summary["artifacts"]["draft"]).read_text(encoding="utf-8"))
    notes = draft["tracks"][0]["notes"]
    assert [note["lyric"] for note in notes] == ["喜", "歡", "你"]
    assert [note["phonemes"] for note in notes] == [["x", "ei"], ["f", "u", "n"], ["n", "ei"]]
    assert all(note["provenance"]["user_jyutping_required"] is False for note in notes)

    root = ET.parse(summary["artifacts"]["vsqx"]).getroot()
    vsqx_notes = root.findall(".//v:note", NS)
    assert [note.findtext("v:y", namespaces=NS) for note in vsqx_notes] == ["喜", "歡", "你"]
    assert [note.findtext("v:p", namespaces=NS) for note in vsqx_notes] == ["x ei", "f u n", "n ei"]


def test_runner_accepts_optional_jyutping_override_and_pitch_analysis(tmp_path: Path) -> None:
    template = _template(tmp_path / "template.vsqx")
    pitch = {
        "schema_version": "pitch_note_analysis_v1",
        "tempo_bpm": 120.0,
        "model_output": False,
        "analysis_output": True,
        "note_candidates": [
            {"note_id": "note-a", "start_sec": 0.0, "duration_sec": 0.5, "midi": 60, "source": "rough_f0_quantized"},
            {"note_id": "note-b", "start_sec": 0.5, "duration_sec": 0.25, "midi": 62, "source": "rough_f0_quantized"},
            {"note_id": "note-c", "start_sec": 0.75, "duration_sec": 0.5, "midi": 64, "source": "rough_f0_quantized"},
        ],
    }
    pitch_path = tmp_path / "pitch.json"
    pitch_path.write_text(json.dumps(pitch), encoding="utf-8")

    rc, summary = run_draft_vsqx(
        lyrics="喜歡你",
        jyutping_override="hei2 fun1 lei5",
        template=template,
        output_root=tmp_path / "run",
        pitch_analysis=pitch_path,
    )

    assert rc == 0
    assert summary["pitch_evidence"]["provided"] is True
    assert summary["pitch_evidence"]["schema_version"] == "pitch_note_analysis_v1"
    assert summary["pitch_evidence"]["model_output"] is False
    assert summary["pitch_evidence"]["analysis_output"] is True
    assert summary["pitch_evidence"]["note_candidate_count"] == 3
    draft = json.loads(Path(summary["artifacts"]["draft"]).read_text(encoding="utf-8"))
    notes = draft["tracks"][0]["notes"]
    assert [note["provenance"]["jyutping"] for note in notes] == ["hei2", "fun1", "lei5"]
    assert [note["pitch"] for note in notes] == [64, 66, 68]
    assert [note["provenance"]["extracted_midi"] for note in notes] == [60, 62, 64]
    assert [note["provenance"]["render_midi"] for note in notes] == [64, 66, 68]
    assert [note["provenance"]["pitch_transpose_semitones"] for note in notes] == [4, 4, 4]
    assert all(note["provenance"]["pitch_safety_applied"] is True for note in notes)
    assert [note["start_tick"] for note in notes] == [0, 480, 720]
    assert [note["duration_ticks"] for note in notes] == [480, 240, 480]
    assert draft["provenance"]["pitch_render_policy"]["floor_midi"] == 64
    assert summary["pitch_evidence"]["render_pitch_policy"]["floor_note_name"] == "E4"


def test_runner_can_preserve_raw_extracted_pitch_when_render_safety_disabled(tmp_path: Path) -> None:
    template = _template(tmp_path / "template.vsqx")
    pitch = {
        "schema_version": "pitch_note_analysis_v1",
        "tempo_bpm": 120.0,
        "model_output": False,
        "analysis_output": True,
        "note_candidates": [
            {"note_id": "note-a", "start_sec": 0.0, "duration_sec": 0.5, "midi": 48, "source": "rough_f0_quantized"},
        ],
    }
    pitch_path = tmp_path / "pitch.json"
    pitch_path.write_text(json.dumps(pitch), encoding="utf-8")

    rc, summary = run_draft_vsqx(
        lyrics="喜",
        jyutping_override="hei2",
        template=template,
        output_root=tmp_path / "run",
        pitch_analysis=pitch_path,
        render_pitch_safety=False,
    )

    assert rc == 0
    draft = json.loads(Path(summary["artifacts"]["draft"]).read_text(encoding="utf-8"))
    note = draft["tracks"][0]["notes"][0]
    assert note["pitch"] == 48
    assert note["provenance"]["extracted_midi"] == 48
    assert note["provenance"]["render_midi"] == 48
    assert note["provenance"]["pitch_safety_applied"] is False
    assert draft["provenance"]["pitch_render_policy"]["enabled"] is False


def test_runner_missing_template_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.vsqx"

    rc, summary = run_draft_vsqx(
        lyrics="喜歡你",
        template=missing,
        output_root=tmp_path / "run",
    )

    assert rc == 2
    assert summary["ok"] is False
    assert summary["error_kind"] == "missing_template"
    assert any("missing VSQX template" in error for error in summary["errors"])
    assert not Path(summary["artifacts"]["vsqx"]).exists()


def test_runner_invalid_render_pitch_floor_is_argument_error(tmp_path: Path) -> None:
    template = _template(tmp_path / "template.vsqx")

    rc, summary = run_draft_vsqx(
        lyrics="喜",
        jyutping_override="hei2",
        template=template,
        output_root=tmp_path / "run",
        render_pitch_floor=200,
    )

    assert rc == 2
    assert summary["ok"] is False
    assert summary["error_kind"] == "invalid_arguments"
    assert any("--render-pitch-floor" in error for error in summary["errors"])


def test_runner_rejects_invalid_pitch_evidence_and_template_xml(tmp_path: Path) -> None:
    template = _template(tmp_path / "template.vsqx")
    bad_pitch = tmp_path / "bad-pitch.json"
    bad_pitch.write_text(
        json.dumps(
            {
                "schema_version": "pitch_note_analysis_v1",
                "tempo_bpm": 120,
                "note_candidates": [
                    {"midi": 164, "start_sec": 0, "duration_sec": 0.5}
                ],
            }
        ),
        encoding="utf-8",
    )

    rc, result = run_draft_vsqx(
        lyrics="喜",
        template=template,
        output_root=tmp_path / "bad-pitch-run",
        pitch_analysis=bad_pitch,
    )
    assert rc == 2
    assert result["error_kind"] == "invalid_pitch_evidence"

    invalid_template = tmp_path / "invalid.vsqx"
    invalid_template.write_text("<vsq4>", encoding="utf-8")
    rc, result = run_draft_vsqx(
        lyrics="喜",
        template=invalid_template,
        output_root=tmp_path / "bad-template-run",
        pitches=[64],
    )
    assert rc == 2
    assert result["error_kind"] == "invalid_template_xml"


def test_runner_rejects_pitch_safety_overflow(tmp_path: Path) -> None:
    pitch_path = tmp_path / "pitch-overflow.json"
    pitch_path.write_text(
        json.dumps(
            {
                "schema_version": "pitch_note_analysis_v1",
                "tempo_bpm": 120,
                "note_candidates": [
                    {"midi": 0, "start_sec": 0, "duration_sec": 0.5},
                    {"midi": 127, "start_sec": 0.5, "duration_sec": 0.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    rc, result = run_draft_vsqx(
        lyrics="喜歡",
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "overflow-run",
        pitch_analysis=pitch_path,
    )
    assert rc == 2
    assert result["error_kind"] == "invalid_pitch_evidence"


def test_runner_rejects_non_miku_template(tmp_path: Path) -> None:
    template = _template(tmp_path / "other-singer.vsqx")
    tree = ET.parse(template)
    voice_name = tree.getroot().find(".//v:vVoice/v:name", NS)
    assert voice_name is not None
    voice_name.text = "CYBER_SONGMAN"
    tree.write(template, encoding="utf-8", xml_declaration=True)

    rc, result = run_draft_vsqx(
        lyrics="喜",
        template=template,
        output_root=tmp_path / "run",
        pitches=[64],
    )
    assert rc == 2
    assert result["error_kind"] == "incompatible_template_singer"


def test_runner_recomputes_part_play_time_for_long_draft(tmp_path: Path) -> None:
    rc, result = run_draft_vsqx(
        lyrics="喜喜喜喜喜喜",
        jyutping_override="hei2 hei2 hei2 hei2 hei2 hei2",
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        pitches=[64],
    )
    assert rc == 0
    root = ET.parse(result["artifacts"]["vsqx"]).getroot()
    notes = root.findall(".//v:note", NS)
    expected_end = max(
        int(note.findtext("v:t", namespaces=NS) or 0)
        + int(note.findtext("v:dur", namespaces=NS) or 0)
        for note in notes
    )
    assert int(root.findtext(".//v:vsPart/v:playTime", namespaces=NS) or 0) == expected_end


def test_runner_policy_rejects_known_silent_phoneme_candidate(tmp_path: Path, monkeypatch) -> None:
    from tools import build_miku_v4c_draft_vsqx as runner

    class BadResolver:
        def resolve_text(self, lyrics: str, *, jyutping_override: str | None = None):
            return {
                "schema_version": "cantonese_phoneme_lattice_v1",
                "target": "miku_v4c",
                "text": lyrics,
                "syllables": [
                    {
                        "index": 0,
                        "text": "歡",
                        "candidates": [
                            {
                                "jyutping": "fun1",
                                "internal_phones": ["f", "un"],
                                "ipa": [],
                                "v4c_phonemes": ["f", "UN"],
                                "score": 1.0,
                                "source": "fixture_bad_candidate",
                                "notes": [],
                            }
                        ],
                    }
                ],
            }

    monkeypatch.setattr(runner, "CantonesePhonemeResolver", BadResolver)
    template = _template(tmp_path / "template.vsqx")

    rc, summary = runner.run_draft_vsqx(
        lyrics="歡",
        template=template,
        output_root=tmp_path / "run",
    )

    assert rc == 1
    assert summary["ok"] is False
    assert summary["error_kind"] == "phoneme_policy_failure"
    assert summary["phoneme_policy_issues"][0]["sequence"] == "f UN"
    assert not Path(summary["artifacts"]["vsqx"]).exists()


def test_runner_cli_smoke_writes_summary_stdout(tmp_path: Path) -> None:
    template = _template(tmp_path / "template.vsqx")
    output_root = tmp_path / "cli-run"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_miku_v4c_draft_vsqx.py",
            "--lyrics",
            "喜歡你",
            "--template",
            str(template),
            "--output-root",
            str(output_root),
            "--pitches",
            "64,67,69",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["ok"] is True
    assert summary["schema_version"] == "miku_v4c_draft_vsqx_run_v1"
    assert Path(summary["artifacts"]["product_summary"]).exists()
    assert Path(summary["artifacts"]["vsqx"]).exists()

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from any2xsampa.vocaloid.vsqx_alignment import (
    SCHEMA_VERSION,
    build_vsqx_phone_timing_report,
    compare_phone_segments_to_vsqx,
    parse_vsqx_timeline,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_parse_vsqx_timeline_applies_part_offset_and_tempo_changes(tmp_path: Path) -> None:
    vsqx_path = _write_vsqx(tmp_path / "reference.vsqx")

    timeline = parse_vsqx_timeline(vsqx_path)

    assert timeline["resolution"] == 480
    assert timeline["note_count"] == 2
    notes = timeline["notes"]
    assert notes[0]["start_tick"] == 480
    assert notes[0]["start_sec"] == 0.5
    assert notes[0]["end_sec"] == 1.0
    assert notes[1]["start_tick"] == 1440
    assert notes[1]["start_sec"] == 2.0
    assert notes[1]["end_sec"] == 3.0


def test_compare_phone_segments_to_vsqx_reports_posthoc_overlap(tmp_path: Path) -> None:
    timeline = parse_vsqx_timeline(_write_vsqx(tmp_path / "reference.vsqx"))
    payload = _phone_payload()

    comparison = compare_phone_segments_to_vsqx(payload, timeline)

    assert comparison["phone_segment_count"] == 5
    assert comparison["phone_non_silence_segment_count"] == 3
    assert comparison["vsqx_note_count"] == 2
    assert comparison["phone_non_silence_duration_sec"] == 1.3
    assert comparison["vsqx_note_active_duration_sec"] == 1.5
    assert comparison["overlap_duration_sec"] == 1.0
    assert comparison["phone_duration_covered_by_notes_ratio"] == pytest.approx(0.769231)
    assert comparison["note_duration_covered_by_phones_ratio"] == pytest.approx(0.666667)
    assert comparison["first_phone_minus_first_note_sec"] == -0.1
    assert comparison["last_phone_minus_last_note_sec"] == -0.5
    assert comparison["per_note"][0]["phones_by_overlap_duration"][0]["phone"] == "a"


def test_report_and_cli_preserve_recognizer_vs_reference_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vsqx_path = _write_vsqx(tmp_path / "reference.vsqx")
    phone_path = tmp_path / "phones.json"
    phone_path.write_text(json.dumps(_phone_payload()), encoding="utf-8")
    output_path = tmp_path / "report.json"

    report = build_vsqx_phone_timing_report(phone_path, vsqx_path)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["recognizer_redecoded"] is False
    assert report["vsqx_used_as_recognizer_input"] is False
    assert report["lyrics_used_as_recognizer_input"] is False
    assert report["phone_segments"]["producer_backend_kind"] == "onnx-espeak"
    assert report["phone_segments"]["raw_model_output_kind"] == "trained_onnx_ctc_model"

    tool = _load_tool()
    exit_code = tool.main(
        [
            "--phone-segments",
            str(phone_path),
            "--vsqx",
            str(vsqx_path),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report


def _phone_payload() -> dict[str, object]:
    metadata = {
        "source_mode": "real",
        "real_backend": "posterior-json",
        "lyrics_guided": False,
        "note_context_applied": False,
        "producer_backend_kind": "onnx-espeak",
        "producer_raw_model_output": {"raw_model_output_kind": "trained_onnx_ctc_model"},
    }
    return {
        "mode": "infer-open",
        "audio_path": "reference.wav",
        "segments": [
            _segment(0.0, 0.4, "sil", metadata),
            _segment(0.4, 0.9, "a", metadata),
            _segment(0.9, 1.1, "l", metadata),
            _segment(1.1, 1.9, "sil", metadata),
            _segment(1.9, 2.5, "i", metadata),
        ],
    }


def _segment(start: float, end: float, phone: str, metadata: dict[str, object]) -> dict[str, object]:
    return {
        "start_sec": start,
        "end_sec": end,
        "normalized_ipa": phone,
        "metadata": metadata,
    }


def _write_vsqx(path: Path) -> Path:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/">
  <masterTrack>
    <resolution>480</resolution>
    <preMeasure>1</preMeasure>
    <tempo><t>0</t><v>12000</v></tempo>
    <tempo><t>960</t><v>6000</v></tempo>
  </masterTrack>
  <vsTrack>
    <tNo>2</tNo>
    <vsPart>
      <t>480</t>
      <note><t>0</t><dur>480</dur><n>60</n><y>a</y><p>m a</p></note>
      <note><t>960</t><dur>480</dur><n>62</n><y>i</y><p>l i</p></note>
    </vsPart>
  </vsTrack>
</vsq4>
""",
        encoding="utf-8",
    )
    return path


def _load_tool() -> ModuleType:
    tool_path = REPO_ROOT / "tools" / "compare_phone_segments_to_vsqx.py"
    spec = importlib.util.spec_from_file_location("compare_phone_segments_to_vsqx_tool_under_test", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

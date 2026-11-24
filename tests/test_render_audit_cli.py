"""Explicit diagnostic CLI integration using retained, small fault fixtures."""
from array import array
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import wave

from any2xsampa.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / ".state/runs/render-audit-entry-20261116/cli-fixtures"


def folder() -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="cli-", dir=FIXTURES))


def wav(path: Path, samples: array) -> None:
    data = array("h", samples)
    if sys.byteorder != "little":
        data.byteswap()
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(data.tobytes())


def request(case: str, kind: str, start: int, end: int, **extra) -> dict:
    return dict(id=case, kind=kind, start_sample=start, end_sample=end,
                raw_ipa=["y"] if kind == "note" else [],
                export_tokens=["y"] if kind == "note" else [],
                placeholder=False, approximate=False,
                role="voiced_nucleus" if kind == "note" else "unknown", **extra)


def invoke(arguments: list[str]) -> tuple[int, str, str]:
    out, error = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(error):
        code = main(arguments)
    return code, out.getvalue(), error.getvalue()


class RenderAuditCliTests(unittest.TestCase):
    def test_route_is_explicit_and_existing_inference_default_stays(self):
        parser = build_parser()
        audit = parser.parse_args(["audit-render", "--wav", "a.wav", "--schedule", "s.json"])
        self.assertFalse(audit.check_period)
        original = parser.parse_args(["infer-open", "--audio", "a.wav", "--dialect", "cantonese"])
        self.assertIsNone(original.recognizer_phone_mapping)

    def test_complete_fault_report_and_existing_output_preserved(self):
        base = folder()
        tone = array("h", (round(6000 * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(1600)))
        samples = tone + array("h", [0]) * 320 + array("h", [1])
        samples += array("h", [0]) * 1920 + array("h", [32767]) * 1600 + tone
        audio, schedule, output = base / "faults.wav", base / "schedule.json", base / "report.json"
        wav(audio, samples)
        windows = [request("good", "note", 0, 1600),
                   request("one_count_tail", "rest", 1600, 1921),
                   request("silent", "note", 1921, 3521),
                   request("zero_rest", "rest", 3521, 3841),
                   request("clipped", "note", 3841, 5441),
                   {**request("fallback", "note", 5441, 7041), "placeholder": True}]
        payload = dict(schema_version="render_audit_schedule_v1", clock="rendered_wave_samples",
                       sample_rate=16000, origin="synthetic_sine_and_faults_not_speech", windows=windows)
        schedule.write_text(json.dumps(payload), encoding="utf-8")
        before = audio.read_bytes(), schedule.read_bytes()
        args = ["audit-render", "--wav", str(audio), "--schedule", str(schedule), "--output", str(output)]
        code, text, error = invoke(args)
        self.assertEqual((code, error), (0, ""))
        report = json.loads(text)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
        self.assertEqual((audio.read_bytes(), schedule.read_bytes()), before)
        self.assertEqual((report["summary"]["notes"], report["summary"]["rests"]), (4, 2))
        rows = {r["id"]: r for r in report["rows"]}
        self.assertEqual(rows["good"]["physical_emission_status"], "level_pass")
        self.assertEqual(rows["silent"]["physical_emission_status"], "level_fail")
        self.assertEqual(rows["clipped"]["physical_emission_status"], "level_fail")
        self.assertEqual(rows["fallback"]["physical_emission_status"], "unknown")
        self.assertFalse(rows["one_count_tail"]["strict_zero"])
        self.assertTrue(rows["zero_rest"]["strict_zero"])
        self.assertTrue(all(r["pronunciation_correct"] is None for r in report["rows"]))
        self.assertEqual([r["request"] for r in report["rows"]], windows)
        saved = output.read_bytes()
        code, _, error = invoke(args)
        self.assertEqual(code, 1)
        self.assertTrue(json.loads(error))
        self.assertEqual(output.read_bytes(), saved)

    def test_period_flag_reaches_audio_only_adapter_then_compares_requested_midi(self):
        base = folder()
        audio, schedule = base / "period.wav", base / "schedule.json"
        tone = array("h", (round(6000 * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(3201)))
        wav(audio, tone)
        schedule.write_text(json.dumps(dict(schema_version="render_audit_schedule_v1",
            clock="rendered_wave_samples", sample_rate=16000,
            windows=[request("requested", "note", 0, 3201, pitch_midi=57)])), encoding="utf-8")
        code, text, error = invoke(["audit-render", "--wav", str(audio), "--schedule", str(schedule), "--check-period"])
        self.assertEqual((code, error), (0, ""))
        row = json.loads(text)["rows"][0]
        self.assertEqual(row["period_observation"]["f0_status"], "period_estimate_unqualified")
        comparison = row["requested_pitch_comparison"]
        self.assertTrue(comparison["period_estimate_within_50c"])
        self.assertFalse(comparison["reference_used_by_estimator"])
        self.assertIsNone(comparison["pitch_truth"])
        self.assertIsNone(row["pronunciation_correct"])


if __name__ == "__main__":
    unittest.main()

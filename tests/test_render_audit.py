from __future__ import annotations

import array
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

from any2xsampa.local import render_audit


ROOT = Path(__file__).resolve().parents[3]
FIXTURES_ROOT = ROOT / ".state/runs/render-audit-entry-20261116/fixtures"


def _wave_bytes(samples: list[int], rate: int, *, channels: int = 1, width: int = 2) -> bytes:
    if width == 2:
        pcm = array.array("h", samples)
        if sys.byteorder != "little":
            pcm.byteswap()
        payload = pcm.tobytes()
    else:
        payload = bytes(samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        output.writeframes(payload)
    return buffer.getvalue()


def _window(identifier: str = "note", start: int = 0, end: int = 1600, **updates: object) -> dict:
    window = {
        "id": identifier, "kind": "note", "start_sample": start, "end_sample": end,
        "raw_ipa": ["a", "n"], "export_tokens": ["a", "N"],
        "placeholder": False, "approximate": False, "role": "voiced_nucleus",
        "pitch_midi": 69,
    }
    window.update(updates)
    return window


def _schedule(*windows: dict, rate: int = 16000) -> dict:
    return {
        "schema_version": "render_audit_schedule_v1", "clock": "rendered_wave_samples",
        "sample_rate": rate, "windows": list(windows),
    }


class RenderAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Retain only this tiny, newly synthesized fixture set; never delete it.
        FIXTURES_ROOT.mkdir(parents=True, exist_ok=True)
        cls.fixture_dir = Path(tempfile.mkdtemp(prefix="render-audit-", dir=FIXTURES_ROOT))
        cls.tone_samples = [round(8192 * math.sin(2 * math.pi * 440 * i / 16000))
                            for i in range(1600)]
        cls.contents = {
            "tone": _wave_bytes(cls.tone_samples, 16000),
            "zero": _wave_bytes([0] * 1600, 16000),
            "clip": _wave_bytes([32767, -32768] * 800, 16000),
            # Two adjacent windows: the rest ends with a single PCM count;
            # the note has two zero 20ms frames and an active 1ms tail frame.
            "tail": _wave_bytes([0] * 60 + [1] + [0] * 40 + [1000], 1000),
            "stereo": _wave_bytes([0] * 200, 16000, channels=2),
            "pcm8": _wave_bytes([128] * 100, 16000, width=1),
        }
        cls.paths = {}
        for name, contents in cls.contents.items():
            path = cls.fixture_dir / f"{name}.wav"
            path.write_bytes(contents)
            cls.paths[name] = path
        if len(cls.paths) > 12 or sum(map(len, cls.contents.values())) > 300 * 1024:
            raise AssertionError("render audit fixtures exceeded their resource budget")
        print(f"Retained render-audit fixtures: {cls.fixture_dir}")

    def test_tone_levels_hash_contract_and_no_mutation(self) -> None:
        request = _window(approximate=True, extra={"nested": ["preserved"]})
        schedule = _schedule(request)
        original = deepcopy(schedule)
        result = render_audit.audit_wave(self.paths["tone"], schedule)
        row = result["rows"][0]
        levels = row["levels"]
        self.assertEqual(schedule, original)
        self.assertEqual(row["request"], original["windows"][0])
        self.assertIsNot(row["request"], request)
        self.assertEqual([row[key] for key in ("id", "kind", "start_sample", "end_sample")],
                         [request[key] for key in ("id", "kind", "start_sample", "end_sample")])
        row["request"]["extra"]["nested"].append("only the copy")
        row["request"]["raw_ipa"].append("only the copy")
        self.assertEqual(schedule, original)
        self.assertEqual(row["physical_emission_status"], "level_pass")
        self.assertAlmostEqual(levels["rms"], math.sqrt(sum(s * s for s in self.tone_samples)
                                                      / 1600) / 32768)
        self.assertEqual(levels["peak"], 8192 / 32768)
        self.assertEqual(levels["clipped_samples"], 0)
        self.assertFalse(levels["digital_zero"])
        self.assertEqual(levels["activity"]["frame_count"], 5)
        self.assertEqual(levels["activity"]["active_fraction"], 1)
        self.assertEqual(levels["edges"]["first_20ms"]["sample_count"], 320)
        self.assertEqual(levels["edges"]["last_20ms"]["sample_count"], 320)
        self.assertEqual(result["sample_rate"], 16000)
        self.assertEqual(result["samples"], 1600)
        self.assertEqual(result["wav_sha256"], hashlib.sha256(self.contents["tone"]).hexdigest())
        self.assertEqual(result["wav_bytes"], len(self.contents["tone"]))
        self.assertEqual(row["phone_emission"], "unknown")
        self.assertIsNone(row["pronunciation_correct"])
        self.assertIsNone(row["strict_zero"])
        self.assertEqual(row["period_observation"], {"f0_status": "not_requested"})
        self.assertEqual(result["summary"]["note_status_counts"]["level_pass"], 1)
        json.dumps(result, allow_nan=False)

    def test_zero_clipping_and_strict_40ms_gate(self) -> None:
        for name, status in (("zero", "level_fail"), ("clip", "level_fail")):
            with self.subTest(name=name):
                result = render_audit.audit_wave(self.paths[name], _schedule(_window()))
                row = result["rows"][0]
                self.assertEqual(row["physical_emission_status"], status)
                self.assertEqual(row["phone_emission"], "unknown")
                self.assertIsNone(row["pronunciation_correct"])
                if name == "zero":
                    self.assertEqual(row["levels"]["rms"], 0)
                    self.assertTrue(row["levels"]["digital_zero"])
                    self.assertEqual(row["levels"]["digital_zero_fraction"], 1)
                    self.assertEqual(row["levels"]["activity"]["active_frames"], 0)
                else:
                    self.assertEqual(row["levels"]["peak"], 1)
                    self.assertEqual(row["levels"]["clipped_samples"], 1600)
                    self.assertEqual(row["levels"]["clip_fraction"], 1)
                    self.assertIn("clipped_samples", row["reasons"])
        short = render_audit.audit_wave(self.paths["tone"], _schedule(_window(end=640)))
        self.assertEqual(short["rows"][0]["physical_emission_status"], "unknown")
        self.assertIn("duration_not_over_40ms", short["rows"][0]["reasons"])
        longer = render_audit.audit_wave(self.paths["tone"], _schedule(_window(end=641)))
        self.assertEqual(longer["rows"][0]["physical_emission_status"], "level_pass")

    def test_roles_and_placeholder_retain_fallback_levels(self) -> None:
        cases = (
            ({"placeholder": True}, "unknown"),
            ({"role": "unknown"}, "unknown"),
            ({"role": "unvoiced_or_transient"}, "energy_available"),
            ({"role": "glide_or_nasal"}, "level_pass"),
        )
        for updates, status in cases:
            with self.subTest(updates=updates):
                result = render_audit.audit_wave(self.paths["tone"], _schedule(_window(**updates)))
                row = result["rows"][0]
                self.assertEqual(row["physical_emission_status"], status)
                self.assertGreater(row["levels"]["rms"], 1e-4)
                self.assertEqual(row["phone_emission"], "unknown")
                self.assertIsNone(row["pronunciation_correct"])
        silent = render_audit.audit_wave(self.paths["zero"], _schedule(
            _window(role="unvoiced_or_transient")))
        self.assertEqual(silent["rows"][0]["physical_emission_status"], "unknown")
        self.assertIn("no_transient_energy_observed", silent["rows"][0]["reasons"])

    def test_rest_last_count_and_incomplete_activity_tail_are_not_dropped(self) -> None:
        schedule = _schedule(
            _window("rest", 0, 61, kind="rest", role="unknown"),
            _window("tail", 61, 102), rate=1000,
        )
        result = render_audit.audit_wave(self.paths["tail"], schedule)
        rest, note = result["rows"]
        self.assertFalse(rest["rest_digital_zero"])
        self.assertFalse(rest["strict_zero"])
        self.assertIsNone(note["strict_zero"])
        self.assertEqual(rest["physical_emission_status"], "unknown")
        self.assertEqual(rest["levels"]["digital_zero_samples"], 60)
        self.assertEqual(rest["levels"]["peak"], 1 / 32768)
        self.assertAlmostEqual(rest["levels"]["rms"], 1 / math.sqrt(61) / 32768)
        self.assertEqual(rest["levels"]["activity"]["frame_count"], 4)
        self.assertEqual(rest["levels"]["activity"]["last_frame_samples"], 1)
        self.assertEqual(rest["levels"]["activity"]["active_frames"], 0)
        self.assertEqual(rest["levels"]["edges"]["last_20ms"]["peak"], 1 / 32768)
        self.assertEqual(note["levels"]["sample_count"], 41)
        self.assertEqual(note["levels"]["activity"]["frame_count"], 3)
        self.assertEqual(note["levels"]["activity"]["active_frames"], 1)
        self.assertEqual(note["levels"]["activity"]["active_fraction"], 1 / 3)
        self.assertEqual(note["levels"]["edges"]["last_activity_frame"]["sample_count"], 1)
        self.assertEqual(note["levels"]["edges"]["last_activity_frame"]["rms"], 1000 / 32768)
        self.assertEqual(note["physical_emission_status"], "level_fail")
        self.assertEqual(result["summary"]["nonzero_rests"], 1)
        zero = render_audit.audit_wave(self.paths["zero"], _schedule(_window(kind="rest")))
        self.assertTrue(zero["rows"][0]["rest_digital_zero"])
        self.assertTrue(zero["rows"][0]["strict_zero"])
        self.assertEqual(zero["summary"]["digital_zero_rests"], 1)

    def test_out_of_bounds_keep_full_denominator_order_and_original_bounds(self) -> None:
        windows = [
            _window("absent", 1800, 2000, kind="rest"),
            _window("partial", 1200, 1800),
            _window("complete", 0, 1200),
        ]
        result = render_audit.audit_wave(self.paths["tone"], _schedule(*windows))
        self.assertEqual([row["id"] for row in result["rows"]], [w["id"] for w in windows])
        self.assertEqual(result["summary"]["windows"], 3)
        self.assertEqual(result["summary"]["notes"], 2)
        self.assertEqual(result["summary"]["rests"], 1)
        self.assertEqual(result["summary"]["missing_windows"], 2)
        self.assertEqual(result["summary"]["missing_notes"], 1)
        self.assertEqual(result["summary"]["missing_rests"], 1)
        for row, window in zip(result["rows"][:2], windows[:2]):
            self.assertEqual(row["request"], window)
            self.assertTrue(row["missing"])
            self.assertEqual(row["missing_samples"], 200)
            self.assertEqual(row["physical_emission_status"], "unknown")
            self.assertIsNone(row["levels"])
            self.assertIsNone(row["rest_digital_zero"])
            self.assertIsNone(row["strict_zero"])
            self.assertIsNone(row["pronunciation_correct"])
            self.assertEqual(row["start_sample"], window["start_sample"])
            self.assertEqual(row["end_sample"], window["end_sample"])
        self.assertEqual(result["rows"][1]["available_samples"], 400)

    def test_invalid_schema_clock_rate_types_and_window_metadata_raise(self) -> None:
        schedule_cases = (
            ("schema_version", "other"), ("clock", "source_audio_samples"),
            ("sample_rate", True), ("sample_rate", 16000.0),
            ("sample_rate", "16000"), ("sample_rate", 0), ("sample_rate", 8000),
            ("windows", ()), ("windows", [None]),
        )
        for field, value in schedule_cases:
            with self.subTest(field=field, value=value):
                schedule = _schedule(_window())
                schedule[field] = value
                with self.assertRaises(ValueError):
                    render_audit.audit_wave(self.paths["tone"], schedule)
        window_cases = (
            ("start_sample", True), ("start_sample", 0.0), ("start_sample", -1),
            ("end_sample", False), ("end_sample", 1600.0), ("end_sample", 0),
            ("id", 1), ("id", ""), ("kind", []), ("role", ["unknown"]),
            ("placeholder", 0), ("approximate", 1), ("pitch_midi", True),
            ("pitch_midi", 69.0), ("pitch_midi", -1), ("pitch_midi", 128),
            ("raw_ipa", "a"), ("raw_ipa", [1]),
            ("export_tokens", [False]),
        )
        for field, value in window_cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    render_audit.audit_wave(self.paths["tone"], _schedule(_window(**{field: value})))
        for missing_field in ("id", "start_sample", "raw_ipa", "placeholder", "role"):
            with self.subTest(missing=missing_field):
                window = _window()
                del window[missing_field]
                with self.assertRaises(ValueError):
                    render_audit.audit_wave(self.paths["tone"], _schedule(window))
        with self.assertRaises(ValueError):
            render_audit.audit_wave(self.paths["tone"], None)

    def test_duplicate_overlap_and_unsorted_adjacent_windows(self) -> None:
        invalid = (
            [_window("same", 0, 800), _window("same", 800, 1600)],
            [_window("later", 700, 1600), _window("earlier", 0, 800)],
            [_window("outer", 0, 1600), _window("inner", 100, 200)],
        )
        for windows in invalid:
            with self.subTest(windows=windows):
                with self.assertRaises(ValueError):
                    render_audit.audit_wave(self.paths["tone"], _schedule(*windows))
        result = render_audit.audit_wave(self.paths["tone"], _schedule(
            _window("second", 800, 1600), _window("first", 0, 800)))
        self.assertEqual([row["id"] for row in result["rows"]], ["second", "first"])

    def test_wrong_pcm_float_compression_and_truncation_raise(self) -> None:
        for name in ("stereo", "pcm8"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    render_audit.audit_wave(self.paths[name], _schedule(_window()))
        float_pcm = bytearray(_wave_bytes([0] * 100, 16000))
        struct.pack_into("<H", float_pcm, 20, 3)  # WAVE_FORMAT_IEEE_FLOAT
        compressed = bytearray(float_pcm)
        struct.pack_into("<H", compressed, 20, 6)  # WAVE_FORMAT_ALAW
        odd_pcm = io.BytesIO()
        with wave.open(odd_pcm, "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(b"\0" * 201)
        for contents in (bytes(float_pcm), bytes(compressed), self.contents["tone"][:-2],
                         odd_pcm.getvalue(), b"not a WAV"):
            with self.subTest(header=contents[:24]):
                with patch.object(Path, "open", return_value=io.BytesIO(contents)):
                    with self.assertRaises(ValueError):
                        render_audit.audit_wave(self.paths["tone"], _schedule(_window()))

    def test_one_bounded_read_and_size_limit(self) -> None:
        class ReadOnce(io.BytesIO):
            def __init__(self, contents: bytes) -> None:
                super().__init__(contents)
                self.read_sizes: list[int] = []

            def read(self, size: int = -1) -> bytes:
                self.read_sizes.append(size)
                if len(self.read_sizes) != 1:
                    raise AssertionError("WAV source was read more than once")
                return super().read(size)

        source = ReadOnce(self.contents["tone"])
        with patch.object(Path, "open", return_value=source) as opened:
            result = render_audit.audit_wave(self.paths["tone"], _schedule(_window()))
        opened.assert_called_once_with("rb")
        self.assertEqual(source.read_sizes, [128 * 1024 * 1024 + 1])
        self.assertEqual(result["wav_sha256"], hashlib.sha256(self.contents["tone"]).hexdigest())
        source = ReadOnce(self.contents["tone"])
        limit = len(self.contents["tone"]) - 1
        with patch.object(render_audit, "MAX_WAV_BYTES", limit):
            with patch.object(Path, "open", return_value=source):
                with self.assertRaisesRegex(ValueError, "128 MiB"):
                    render_audit.audit_wave(self.paths["tone"], _schedule(_window()))
        self.assertEqual(source.read_sizes, [limit + 1])

    def test_period_receives_only_pcm_copy_and_rate_for_complete_notes(self) -> None:
        calls = []
        observation = {"f0_status": "ok", "f0_quantiles_hz": {"p50": 440},
                       "f0_error": None, "parameters": {"method": "test"}}

        def analyzer(pcm: array.array, rate: int) -> dict:
            self.assertIsInstance(pcm, array.array)
            self.assertEqual(pcm.typecode, "h")
            calls.append((list(pcm), rate))
            pcm[0] = 32767
            return observation

        windows = [
            _window("note", 0, 400), _window("placeholder", 400, 800, placeholder=True),
            _window("unknown", 800, 1000, role="unknown"),
            _window("transient", 1000, 1200, role="unvoiced_or_transient"),
            _window("rest", 1200, 1400, kind="rest"), _window("missing", 1400, 1800),
        ]
        schedule = _schedule(*windows)
        before = deepcopy(schedule)
        result = render_audit.audit_wave(self.paths["tone"], schedule, period_analyzer=analyzer)
        self.assertEqual(schedule, before)
        self.assertEqual(len(calls), 4)
        for call, window, row in zip(calls, windows, result["rows"]):
            self.assertEqual(call, (self.tone_samples[window["start_sample"]:window["end_sample"]], 16000))
            self.assertEqual(row["period_observation"], observation)
            self.assertIsNot(row["period_observation"], observation)
            self.assertEqual(row["levels"]["clipped_samples"], 0)
        for row in result["rows"][4:]:
            self.assertEqual(row["period_observation"], {"f0_status": "not_requested"})
        for row in result["rows"]:
            self.assertEqual(row["phone_emission"], "unknown")
            self.assertIsNone(row["pronunciation_correct"])
        observation["parameters"]["method"] = "changed afterward"
        self.assertEqual(result["rows"][0]["period_observation"]["parameters"]["method"], "test")
        self.assertEqual(result["rows"][1]["physical_emission_status"], "unknown")

    def test_period_errors_and_wrong_return_keep_physical_results_and_denominator(self) -> None:
        def broken(pcm: array.array, rate: int) -> dict:
            raise RuntimeError("test analyzer failure")

        for analyzer in (broken, lambda pcm, rate: None):
            with self.subTest(analyzer=analyzer):
                result = render_audit.audit_wave(self.paths["tone"], _schedule(_window()),
                                                period_analyzer=analyzer)
                row = result["rows"][0]
                self.assertEqual(row["physical_emission_status"], "level_pass")
                self.assertGreater(row["levels"]["rms"], 1e-4)
                self.assertEqual(row["period_observation"]["f0_status"], "unknown")
                self.assertTrue(row["period_observation"]["f0_error"])
                self.assertEqual(result["summary"]["notes"], 1)
                self.assertEqual(result["summary"]["windows"], 1)
                self.assertEqual(result["summary"]["period_error_notes"], 1)
                self.assertIsNone(row["pronunciation_correct"])
        with self.assertRaises(ValueError):
            render_audit.audit_wave(self.paths["tone"], _schedule(_window()), period_analyzer=1)

    def test_reported_parameters_are_fixed_and_empty_schedule_keeps_zero_denominator(self) -> None:
        result = render_audit.audit_wave(self.paths["tone"], _schedule())
        params = result["parameters"]
        self.assertEqual(params["max_wav_bytes"], 128 * 1024 * 1024)
        self.assertEqual(params["pcm_scale"], 32768)
        self.assertEqual(params["clip_threshold"], 0.999)
        self.assertEqual(params["activity_frame_ms"], 20)
        self.assertEqual(params["activity_frame_samples"], 320)
        self.assertEqual(params["activity_threshold_rms"], 1e-4)
        self.assertEqual(params["activity_fraction_basis"], "frames_including_partial_tail")
        self.assertEqual(params["minimum_note_duration_ms_exclusive"], 40)
        self.assertEqual(params["level_rms_threshold"], 1e-4)
        self.assertEqual(params["level_active_fraction_threshold"], 0.5)
        self.assertTrue(params["level_requires_no_clipping"])
        self.assertEqual(result["rows"], [])
        for name in ("windows", "notes", "rests", "missing_windows"):
            self.assertEqual(result["summary"][name], 0)


if __name__ == "__main__":
    unittest.main()

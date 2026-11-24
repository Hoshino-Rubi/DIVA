"""Zero-model local mapping transport and whole-chunk contract checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from any2xsampa.cli import build_parser
from any2xsampa.local.common import phone_mapping_identity, read, request, sha, write
from any2xsampa.local.note_score import score_from_notes
from any2xsampa.local import whole_task

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / ".state/runs/render-audit-entry-20261116/request-fixtures"


def folder() -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="mapping-", dir=FIXTURES))


def mapping(path: Path, version: str = "c3") -> None:
    path.write_text(json.dumps({"profile": version, "schema_version": "test_mapping_v1",
                                "mappings": {"y": ["y"]}}), encoding="utf-8")


class RequestMappingTests(unittest.TestCase):
    def test_optional_request_and_no_reference_conditions(self):
        base = folder()
        selected = base / "c3.json"
        mapping(selected)
        original = {"audio": "input.wav", "model": "fixture-model", "device": "cpu"}
        self.assertNotIn("recognizer_phone_mapping", request(original))
        value = {**original, "recognizer_phone_mapping": str(selected)}
        before = deepcopy(value)
        self.assertEqual(request(value)["recognizer_phone_mapping"], str(selected.resolve()))
        self.assertEqual(value, before)
        for extra in ({"lyrics": "x"}, {"known_boundaries": []},
                      {"recognizer_phone_mapping": ""}, {"recognizer_phone_mapping": 1}):
            with self.assertRaises(ValueError):
                request({**original, **extra})
        with self.assertRaises(FileNotFoundError):
            request({**original, "recognizer_phone_mapping": str(base / "absent")})
        self.assertIsNone(phone_mapping_identity(None, "guarded-phoneticxeus"))
        with self.assertRaises(ValueError):
            phone_mapping_identity(str(selected), "guarded-phoneticxeus")
        self.assertEqual(phone_mapping_identity(str(selected), "onnx-espeak")["sha256"], sha(selected))

    def test_cli_clip_and_whole_keep_same_optional_selection(self):
        parser = build_parser()
        for extra in ([], ["--whole-file"]):
            args = parser.parse_args(["recognize", "input.wav", "--model", "m",
                                      "--recognizer-phone-mapping", "c3.json", *extra])
            self.assertEqual(args.recognizer_phone_mapping, "c3.json")
        default = parser.parse_args(["recognize", "input.wav", "--model", "m"])
        self.assertIsNone(default.recognizer_phone_mapping)
        with self.assertRaises(ValueError):
            whole_task.whole_request({"audio": "a.wav", "model": "m", "start": 1})

    def _whole(self, change_after_first: bool = False):
        base = folder()
        audio, selected = base / "binding.bin", base / "c3.json"
        audio.write_bytes(b"explicit fake-worker input binding; not speech")
        mapping(selected)
        identity = phone_mapping_identity(str(selected), "onnx-espeak")
        info = {"path": str(audio), "sha256": sha(audio), "duration": 0.2}
        chunks = [dict(index=i, start_sec=i / 10, end_sec=(i + 1) / 10,
                       keep_start_sec=i / 10, keep_end_sec=(i + 1) / 10) for i in range(2)]
        spec = {"backend": "onnx-espeak", "devices": ["cpu"], "package_sha256": "fake-package"}

        class FakeEngine:
            def __init__(self):
                self.calls = []
                self.closed = False

            def execute(self, action, params, cancel, progress):
                self.calls.append(deepcopy(params))
                target = Path(params["output_dir"])
                target.mkdir(parents=True)
                write(target / "result.json", {"fixture_no_model": True, "phone_mapping": identity})
                if change_after_first:
                    mapping(selected, "changed")
                return {"result_path": str(target / "result.json")}

            def save_result(self, output, mode, source, evidence):
                write(output / "result.json", {**evidence, "fixture_no_model": True})
                return evidence

            def close(self):
                self.closed = True

        engine = FakeEngine()
        params = dict(audio=str(audio), model="fixture-model", device="cpu",
                      recognizer_phone_mapping=str(selected), output_dir=str(base / "whole"))
        captured = []

        def merge(manifest, cancel):
            captured.append(deepcopy(manifest))
            return {"segments": [], "load_count": 1, "phone_mapping": identity}

        with patch.object(whole_task, "package", return_value=spec), \
             patch.object(whole_task, "whole_input", return_value=info), \
             patch.object(whole_task, "plan_chunks", return_value=chunks), \
             patch.object(whole_task, "merge_results", side_effect=merge):
            if change_after_first:
                with self.assertRaisesRegex(ValueError, "mapping changed"):
                    whole_task.run_whole(engine, params, threading.Event(), lambda _: None)
            else:
                whole_task.run_whole(engine, params, threading.Event(), lambda _: None)
        return base, engine, identity, captured

    def test_every_whole_chunk_receives_selection_and_manifest_binds_it(self):
        base, engine, identity, captured = self._whole()
        self.assertEqual(len(engine.calls), 2)
        self.assertTrue(all(c["recognizer_phone_mapping"] == identity["path"] for c in engine.calls))
        self.assertEqual(captured[0]["phone_mapping"], identity)
        self.assertEqual(read(base / "whole/PLAN.json")["phone_mapping"], identity)

    def test_mapping_change_stops_before_second_chunk_and_preserves_failure(self):
        base, engine, _, captured = self._whole(True)
        self.assertEqual(len(engine.calls), 1)
        self.assertTrue(engine.closed)
        self.assertEqual(captured, [])
        failure = read(base / "whole/EXIT.json")
        self.assertEqual((failure["status"], failure["chunks_completed"], failure["retried"]),
                         ("failed", 1, False))
        self.assertTrue((base / "whole/chunks/0000/result.json").is_file())

    def test_score_consumes_selected_phone_without_changing_note_geometry(self):
        prediction = dict(schema_version="note_transcription_v1", time_basis="original_audio",
            source_start_sec=0, source_end_sec=0.2, mode="fixture",
            input=dict(audio_path="fixture.wav", audio_sha256="synthetic-binding"),
            notes=[dict(start_sec=0, end_sec=0.2, duration_sec=0.2, pitch_midi=59, presence=True)])
        phones = dict(schema_version="local_task_result_v1", time_basis="original_audio",
            input=dict(path="fixture.wav", sha256="synthetic-binding", start_sec=0, end_sec=0.2),
            segments=[dict(start_sec=0.05, end_sec=0.06, normalized_ipa="u")])
        before = deepcopy((prediction, phones))
        a = score_from_notes(prediction, phones)
        c3 = deepcopy(phones)
        c3["segments"][0]["normalized_ipa"] = "y"
        b = score_from_notes(prediction, c3)
        self.assertEqual(a["notes"][0]["v4c_phonemes"], ["u"])
        self.assertEqual(b["notes"][0]["v4c_phonemes"], ["y"])
        geometry = ("pitch", "start_sec", "end_sec", "duration_sec")
        self.assertEqual([a["notes"][0][k] for k in geometry], [b["notes"][0][k] for k in geometry])
        self.assertEqual((prediction, phones), before)
        self.assertEqual(a["phone_count"], b["phone_count"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from any2xsampa.adapters.onnx_espeak import (
    OnnxEspeakCTCUniversalPhoneRecognizer,
    OnnxEspeakInferenceError,
    OnnxEspeakResourceError,
)
from any2xsampa.core.models import PhoneCandidate
from any2xsampa.local import common, onnx_frames, runtime


FIXTURE_PARENT = (Path(__file__).resolve().parents[3] / ".state/runs/"
                  "render-audit-entry-20261116/runtime-fixtures")
SCHEMA = "onnx_espeak_internal_phone_mapping_v1"


class RuntimePhoneMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FIXTURE_PARENT.mkdir(parents=True, exist_ok=True)
        cls.fixture_root = Path(tempfile.mkdtemp(prefix="runtime-mapping-", dir=FIXTURE_PARENT))

    @classmethod
    def tearDownClass(cls) -> None:
        # These small fixtures remain available for inspection; never delete them.
        size = sum(p.stat().st_size for p in cls.fixture_root.rglob("*") if p.is_file())
        if size > 100 * 1024:
            raise AssertionError(f"Runtime mapping fixtures exceed 100 KiB: {size}")

    def setUp(self) -> None:
        self.directory = self.fixture_root / self._testMethodName
        self.directory.mkdir()
        self.mapping = self.mapping_file("C3 映射 with spaces.json")
        self.artifacts = {}
        self.instances = []
        self.progress = []
        self.lock_counts = {"entered": 0, "exited": 0}
        self.load_hook = None
        self.forward_hook = None
        self.constructor_hook = None
        self.runtime_load_count = 0
        self.blank_error = False
        self.worker = runtime.RuntimeEngine()
        self.addCleanup(self.worker.close)
        self.spec = dict(
            python=sys.executable, devices=["cpu"], package_sha256="fake-package-hash",
            backend="onnx-espeak", id="fake-model", quality="fixture-only",
            paths={"model": "never-open.onnx", "vocabulary": "never-open-vocab.json"},
            model_sha256="fake-model-hash", vocabulary_sha256="fake-vocab-hash",
        )
        owner = self

        class FakeRecordedOnnx(OnnxEspeakCTCUniversalPhoneRecognizer):
            def __init__(self, **kwargs):
                self.requested_mapping_path = kwargs["phone_mapping_path"]
                super().__init__(**kwargs)  # Reuse only the real JSON mapping loader.
                self.last_frames = []
                self.forward_count = 0
                owner.instances.append(self)
                if owner.constructor_hook is not None:
                    owner.constructor_hook(self)

            def _load_runtime(self):
                owner.runtime_load_count += 1
                if owner.load_hook is not None:
                    owner.load_hook(self)
                return None, None

            def recognize(self, audio, lyrics=None):
                self.forward_count += 1
                if owner.forward_hook is not None:
                    owner.forward_hook(self)
                symbol = None if owner.blank_error else self._project_raw_label("f")
                self.last_frames = [{"symbol": symbol, "candidates": (
                    [] if symbol is None else [PhoneCandidate(symbol, 0.9)]
                )}]
                if owner.blank_error:
                    raise OnnxEspeakInferenceError("fake all-blank frames")
                return []

        @contextmanager
        def serial_job():
            self.lock_counts["entered"] += 1
            try:
                yield
            finally:
                self.lock_counts["exited"] += 1

        modules = {}
        for name, attributes in {
            "phoneticxeus_runtime": {"enforce_offline": Mock()},
            "phoneticxeus_monitor": {
                "query_total_gpu_bytes": Mock(side_effect=AssertionError("No GPU query")),
                "ProcessResourceMonitor": Mock(side_effect=AssertionError("No GPU monitor")),
            },
            "candidate_inference": {"serial_candidate_job": serial_job},
        }.items():
            module = ModuleType(f"any2xsampa.adapters.{name}")
            module.__dict__.update(attributes)
            modules[module.__name__] = module
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(onnx_frames, "RecordedOnnxRecognizer", FakeRecordedOnnx))
        self.enterContext(patch.object(runtime, "package", lambda _: deepcopy(self.spec)))
        self.enterContext(patch.object(runtime, "write", self.record_artifact))
        self.hash_artifact = self.enterContext(patch.object(runtime, "sha", return_value="fake-artifact-hash"))

    def mapping_file(self, name: str, payload: dict | None = None) -> Path:
        path = self.directory / name
        if payload is None:
            payload = {
                "schema_version": SCHEMA, "profile": "C3", "blank_labels": ["<pad>"],
                "normalization": {"strip_trailing_tone_digits": True, "aliases": {"eff": "f"}},
                "mappings": {"f": ["f"], "ŋ": ["ŋ"]},
            }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def record_artifact(self, path: Path, payload: dict) -> None:
        self.assertNotIn(path, self.artifacts)
        self.artifacts[path] = deepcopy(payload)

    def execute(self, selection=None, *, omit=False, progress=None):
        output = self.directory / f"output-{len(self.progress)}-{len(self.artifacts)}"
        params = dict(model="fake-model", audio="never-read.wav", device="cpu", _output=str(output))
        if not omit:
            params["recognizer_phone_mapping"] = str(selection) if isinstance(selection, Path) else selection

        def report(event):
            self.progress.append(event)
            if progress is not None:
                progress(event)

        response = self.worker.execute("recognize", params, threading.Event(), report)
        self.assertEqual(set(response), {"evidence_path", "evidence_sha256"})
        evidence = self.artifacts[Path(response["evidence_path"])]
        return evidence, self.artifacts[Path(evidence["frame_evidence_path"])]

    def assert_no_success(self) -> None:
        self.assertFalse(any(p.name == "runtime-result.json" for p in self.artifacts))
        self.assertIsNone(self.worker.model)
        self.assertIsNone(self.worker.key)
        self.assertIsNone(self.worker.phone_mapping)

    def test_default_to_c3_to_c3_to_default_reloads_exactly(self) -> None:
        evidence = []
        symbols = []
        models = []
        for selection in (None, self.mapping, self.mapping, None):
            result, frames = self.execute(selection)
            evidence.append(result)
            symbols.append(frames["frames"][0]["symbol"])
            models.append(self.worker.model)
        self.assertEqual([r["load_count"] for r in evidence], [1, 2, 2, 3])
        self.assertEqual(symbols, ["f", "f", "f", "f"])
        self.assertIs(models[1], models[2])
        self.assertIsNot(models[0], models[1])
        self.assertIsNot(models[2], models[3])
        self.assertEqual([m.requested_mapping_path for m in self.instances], [None, self.mapping, None])
        self.assertEqual(self.lock_counts, {"entered": 3, "exited": 2})
        self.worker.close()
        self.assertEqual(self.lock_counts["exited"], 3)

    def test_default_omission_and_none_keep_evidence_shape(self) -> None:
        for omit in (True, False):
            result, frames = self.execute(omit=omit)
            self.assertNotIn("phone_mapping", result)
            self.assertEqual(set(frames), {"schema_version", "frames"})
            self.assertEqual(set(result), {
                "segments", "time_basis", "model_metadata", "frame_evidence_path",
                "frame_evidence_sha256", "model_id", "package_sha256", "device", "quality", "load_count",
            })
        self.assertEqual(len(self.instances), 1)

    def test_metadata_matches_actual_mapping_and_raw_file_hash(self) -> None:
        expected = dict(path=str(self.mapping.resolve()), sha256=hashlib.sha256(self.mapping.read_bytes()).hexdigest(),
                        profile="C3", schema_version=SCHEMA)
        result, frames = self.execute(self.mapping)
        self.assertEqual(result["phone_mapping"], expected)
        self.assertEqual(frames["phone_mapping"], expected)
        self.assertEqual(result["model_metadata"]["phone_mapping_profile"], "C3")
        self.assertEqual(self.worker.model._project_raw_label("eff5"), "f")
        self.assertIsNone(self.worker.model._project_raw_label("<pad>"))
        self.assertEqual(self.worker.key[-1], (str(self.mapping), expected["sha256"]))

    def test_same_path_changed_content_invalidates_even_with_same_profile(self) -> None:
        first, _ = self.execute(self.mapping)
        payload = json.loads(self.mapping.read_text(encoding="utf-8"))
        payload["mappings"] = {"h": ["f"]}
        self.mapping_file(self.mapping.name, payload)
        second, frames = self.execute(self.mapping)
        self.assertEqual(second["load_count"], 2)
        self.assertNotEqual(first["phone_mapping"]["sha256"], second["phone_mapping"]["sha256"])
        self.assertEqual(second["phone_mapping"]["profile"], "C3")
        self.assertEqual(frames["frames"][0]["symbol"], "h")

    def test_same_bytes_at_different_paths_are_distinct_selections(self) -> None:
        other = self.directory / "other.json"
        other.write_bytes(self.mapping.read_bytes())
        first, _ = self.execute(self.mapping)
        second, _ = self.execute(other)
        self.assertEqual(first["phone_mapping"]["sha256"], second["phone_mapping"]["sha256"])
        self.assertNotEqual(first["phone_mapping"]["path"], second["phone_mapping"]["path"])
        self.assertEqual(len(self.instances), 2)

    def test_identity_is_read_and_hashed_once_per_explicit_request(self) -> None:
        original = Path.open
        reads = []
        data = self.mapping.read_bytes()

        def open_file(path, *args, **kwargs):
            if path == self.mapping and args and args[0] == "rb":
                reads.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", open_file), patch.object(
            common.hashlib, "sha256", wraps=hashlib.sha256
        ) as digest, patch.object(runtime, "phone_mapping_identity", wraps=runtime.phone_mapping_identity) as identity:
            self.execute(omit=True)
            self.execute(None)
            self.assertEqual(reads, [])
            self.assertFalse(any(call.args == (data,) for call in digest.call_args_list))
            self.execute(self.mapping)
            self.execute(self.mapping)
            self.assertEqual(reads, [self.mapping, self.mapping])
            self.assertEqual(sum(call.args == (data,) for call in digest.call_args_list), 2)
            self.assertEqual(identity.call_count, 4)
        self.assertTrue(all(call.args[0].name in {"onnx-frames.json", "runtime-result.json"}
                            for call in self.hash_artifact.call_args_list))

    def test_non_onnx_backends_reject_selection_before_read_or_load(self) -> None:
        for backend in ("guarded-phoneticxeus", "unknown"):
            with self.subTest(backend=backend), patch.object(
                runtime, "phone_mapping_identity", wraps=runtime.phone_mapping_identity
            ) as identity:
                self.spec["backend"] = backend
                with self.assertRaisesRegex(ValueError, "requires an onnx-espeak model"):
                    self.execute(self.mapping)
                identity.assert_called_once_with(str(self.mapping), backend)
        self.assertEqual(self.instances, [])

    def test_load_also_rejects_explicit_mapping_for_other_backend(self) -> None:
        mapping = common.phone_mapping_identity(str(self.mapping), "onnx-espeak")
        self.spec["backend"] = "guarded-phoneticxeus"
        with self.assertRaisesRegex(ValueError, "requires an onnx-espeak model"):
            self.worker._load(self.spec, "cpu", self.directory, mapping)
        self.assertEqual(self.instances, [])

    def test_invalid_selection_and_missing_files_never_load(self) -> None:
        for value in ("", "  ", 1, False):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonempty path string"):
                self.execute(value)
        for path in (self.directory / "missing.json", self.directory):
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                self.execute(path)
        self.assertEqual(self.instances, [])

    def test_invalid_json_and_mappings_are_rejected(self) -> None:
        invalid = (
            (b"{broken", "Expecting"),
            (b"[]", "mappings"),
            (b'{}', "mappings"),
            (json.dumps(dict(schema_version=SCHEMA, profile="C3", mappings=[])).encode(), "mappings"),
        )
        for content, message in invalid:
            with self.subTest(content=content):
                self.mapping.write_bytes(content)
                with self.assertRaisesRegex(ValueError, message):
                    self.execute(self.mapping)
        self.assertEqual(self.instances, [])

    def test_existing_loader_rejects_invalid_mapping_rules_without_fallback(self) -> None:
        for field, value, message in (
            ("mappings", {"a": ["same"], "i": ["same"]}, "mapped to both"),
            ("mappings", {"a": "same"}, "must be a list"),
            ("blank_labels", {}, "blank_labels must be a list"),
            ("normalization", [], "normalization must be an object"),
            ("normalization", {"aliases": []}, "aliases must be an object"),
        ):
            with self.subTest(field=field, value=value):
                payload = dict(schema_version=SCHEMA, profile="C3", mappings={})
                payload[field] = value
                self.mapping_file(self.mapping.name, payload)
                with self.assertRaisesRegex(OnnxEspeakResourceError, message):
                    self.execute(self.mapping)
                self.assert_no_success()

    def test_same_profile_stale_cached_instance_cannot_be_relabelled(self) -> None:
        self.execute(self.mapping)
        model = self.worker.model
        identity = dict(self.worker.phone_mapping)
        state = self.worker._phone_mapping_model_state
        payload = json.loads(self.mapping.read_text(encoding="utf-8"))
        payload["mappings"] = {"h": ["f"]}
        self.mapping_file(self.mapping.name, payload)
        self.artifacts.clear()

        def stale_load(*args):
            self.worker.model = model
            self.worker.phone_mapping = identity
            self.worker._phone_mapping_model_state = state

        with patch.object(self.worker, "_load", stale_load), self.assertRaisesRegex(ValueError, "does not match request identity"):
            self.execute(self.mapping)
        self.assert_no_success()
        self.assertEqual(model.forward_count, 1)

    def test_loaded_schema_and_profile_must_match_identity(self) -> None:
        for field in ("_mapping_schema", "_mapping_profile"):
            with self.subTest(field=field):
                self.constructor_hook = lambda model: setattr(model, field, "stale")
                with self.assertRaisesRegex(ValueError, "does not match request identity"):
                    self.execute(self.mapping)
                self.assert_no_success()

    def test_loaded_instance_using_a_different_path_is_rejected(self) -> None:
        self.constructor_hook = lambda model: setattr(model, "phone_mapping_path", self.directory / "wrong.json")
        with self.assertRaisesRegex(ValueError, "does not match request identity"):
            self.execute(self.mapping)
        self.assert_no_success()

    def test_reused_instance_identity_is_rechecked(self) -> None:
        self.execute(self.mapping)
        self.artifacts.clear()
        self.worker.model._raw_label_aliases = {}
        with self.assertRaisesRegex(ValueError, "does not match request identity"):
            self.execute(self.mapping)
        self.assert_no_success()
        self.assertEqual(len(self.instances), 1)
        self.assertEqual(self.instances[0].forward_count, 1)

    def change_mapping(self) -> None:
        # A size change makes the request-lifetime stat guard deterministic on all filesystems.
        with self.mapping.open("ab") as stream:
            stream.write(b" ")

    def test_change_during_identity_read_is_rejected(self) -> None:
        original = runtime.phone_mapping_identity

        def identify(selection, backend):
            identity = original(selection, backend)
            self.change_mapping()
            return identity

        with patch.object(runtime, "phone_mapping_identity", identify), self.assertRaisesRegex(ValueError, "changed during request"):
            self.execute(self.mapping)
        self.assertEqual(self.instances, [])

    def test_change_between_identity_and_constructor_is_rejected(self) -> None:
        def progress(event):
            if event["stage"] == "loading":
                self.change_mapping()

        with self.assertRaisesRegex(ValueError, "changed during request"):
            self.execute(self.mapping, progress=progress)
        self.assert_no_success()
        self.assertEqual(self.instances, [])

    def test_change_during_mapping_loader_is_rejected(self) -> None:
        self.constructor_hook = lambda model: self.change_mapping()
        with self.assertRaisesRegex(ValueError, "changed during request"):
            self.execute(self.mapping)
        self.assert_no_success()
        self.assertEqual(self.runtime_load_count, 0)

    def test_change_during_session_load_is_rejected_before_forward(self) -> None:
        self.load_hook = lambda model: self.change_mapping()
        with self.assertRaisesRegex(ValueError, "changed during request"):
            self.execute(self.mapping)
        self.assert_no_success()
        self.assertEqual(self.instances[0].forward_count, 0)

    def test_change_during_forward_is_rejected_before_frame_evidence(self) -> None:
        self.forward_hook = lambda model: self.change_mapping()
        with self.assertRaisesRegex(ValueError, "changed during request"):
            self.execute(self.mapping)
        self.assert_no_success()
        self.assertFalse(any(p.name == "onnx-frames.json" for p in self.artifacts))

    def test_file_disappearing_during_forward_is_rejected_without_deletion(self) -> None:
        original = runtime._mapping_file_state
        unavailable = False

        def state(path):
            if unavailable:
                raise FileNotFoundError(path)
            return original(path)

        def forward(model):
            nonlocal unavailable
            unavailable = True

        self.forward_hook = forward
        with patch.object(runtime, "_mapping_file_state", state), self.assertRaisesRegex(ValueError, "changed during request"):
            self.execute(self.mapping)
        self.assert_no_success()
        self.assertTrue(self.mapping.is_file())

    def test_touch_without_content_change_between_requests_reuses_model(self) -> None:
        first, _ = self.execute(self.mapping)
        stat = self.mapping.stat()
        os.utime(self.mapping, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        second, _ = self.execute(self.mapping)
        self.assertEqual(first["phone_mapping"], second["phone_mapping"])
        self.assertEqual(len(self.instances), 1)

    def test_all_blank_frames_keep_opt_in_mapping_identity(self) -> None:
        self.blank_error = True
        result, frames = self.execute(self.mapping)
        self.assertEqual(result["segments"], [])
        self.assertIsNone(frames["frames"][0]["symbol"])
        self.assertEqual(result["phone_mapping"], frames["phone_mapping"])


if __name__ == "__main__":
    unittest.main()

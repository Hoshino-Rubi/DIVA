from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from any2xsampa.adapters.onnx_espeak import (
    OnnxEspeakCTCUniversalPhoneRecognizer,
    OnnxEspeakResourceError,
)


class _RuntimeSession:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers
        self.profile_ended = False

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_values")]

    def run(self, outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        del outputs, feed
        return [np.asarray([[[4.0, 0.0]]], dtype=np.float32)]

    def end_profiling(self) -> str:
        self.profile_ended = True
        return ".state/onnx-profile.json"


def _mapping(path: Path) -> Path:
    path.write_text(
        '{"schema_version":"test_v1","profile":"test","blank_labels":[],"mappings":{"a":["a"]}}',
        encoding="utf-8",
    )
    return path


def test_default_bundle_and_manifest_hashes_are_verified(tmp_path: Path) -> None:
    session = _RuntimeSession(["CPUExecutionProvider"])
    recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
        device="cpu",
        phone_mapping_path=_mapping(tmp_path / "mapping.json"),
        session_factory=lambda path, providers: session,
    )

    recognizer._load_runtime()

    metadata = recognizer.model_metadata()
    assert metadata["manifest_verified"] is True
    assert metadata["manifest_path"].endswith("models\\MANIFEST.json")
    assert metadata["model_sha256"]
    assert metadata["vocab_sha256"]
    assert metadata["session_providers"] == ["CPUExecutionProvider"]


def test_bundled_manifest_hash_cannot_be_overridden(tmp_path: Path) -> None:
    recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
        device="cpu",
        phone_mapping_path=_mapping(tmp_path / "mapping.json"),
        expected_model_sha256="0" * 64,
    )
    with pytest.raises(OnnxEspeakResourceError, match="model SHA-256 mismatch"):
        recognizer._load_runtime()


def test_explicit_cuda_rejects_cpu_only_runtime(tmp_path: Path) -> None:
    mapping = _mapping(tmp_path / "mapping.json")
    recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
        model_path=tmp_path / "missing.onnx",
        vocab_path=tmp_path / "missing-vocab.json",
        phone_mapping_path=mapping,
        device="cuda",
    )
    with pytest.raises(OnnxEspeakResourceError, match="model resource not found"):
        recognizer._load_runtime()


def test_auto_cpu_fallback_records_reason_and_preload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"a":0}', encoding="utf-8")
    mapping = _mapping(tmp_path / "mapping.json")
    session = _RuntimeSession(["CPUExecutionProvider"])
    recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
        model_path=model,
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="auto",
        session_factory=lambda path, providers: session,
    )

    class _Ort:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

        @staticmethod
        def preload_dlls() -> None:
            raise RuntimeError("CUDA DLLs unavailable")

    def _import(name: str) -> object:
        if name == "numpy":
            return np
        if name == "onnxruntime":
            return _Ort
        raise ImportError(name)

    monkeypatch.setattr("any2xsampa.adapters.onnx_espeak.importlib.import_module", _import)
    recognizer._load_runtime()
    metadata = recognizer.model_metadata()
    assert metadata["requested_providers"] == ["CPUExecutionProvider"]
    assert metadata["provider_fallback_reason"]
    assert metadata["cuda_dll_preload"]["attempted"] is True
    assert metadata["cuda_dll_preload"]["success"] is False


def test_cuda_session_requires_active_cuda_provider(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"a":0}', encoding="utf-8")
    mapping = _mapping(tmp_path / "mapping.json")
    session = _RuntimeSession(["CPUExecutionProvider"])
    recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
        model_path=model,
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="cuda",
        session_factory=lambda path, providers: session,
    )
    monkeypatch = pytest.MonkeyPatch()
    try:
        class _Ort:
            @staticmethod
            def get_available_providers() -> list[str]:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            @staticmethod
            def preload_dlls() -> None:
                return None

        monkeypatch.setattr("any2xsampa.adapters.onnx_espeak.importlib.import_module", lambda name: (
            np if name == "numpy" else _Ort
        ))
        with pytest.raises(OnnxEspeakResourceError, match="did not activate CUDAExecutionProvider"):
            recognizer._load_runtime()
    finally:
        monkeypatch.undo()


def test_auto_retries_cpu_when_cuda_session_creation_fails(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"a":0}', encoding="utf-8")
    mapping = _mapping(tmp_path / "mapping.json")
    calls: list[list[str]] = []

    def session_factory(path: str, providers: list[str]) -> _RuntimeSession:
        del path
        calls.append(providers)
        if "CUDAExecutionProvider" in providers:
            raise RuntimeError("CUDA initialization failed")
        return _RuntimeSession(["CPUExecutionProvider"])

    recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
        model_path=model,
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="auto",
        session_factory=session_factory,
    )
    recognizer._available_providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    recognizer._requested_providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    recognizer._session_factory = session_factory
    monkeypatch = pytest.MonkeyPatch()
    try:
        class _Ort:
            @staticmethod
            def get_available_providers() -> list[str]:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            @staticmethod
            def preload_dlls() -> None:
                return None

        monkeypatch.setattr("any2xsampa.adapters.onnx_espeak.importlib.import_module", lambda name: (
            np if name == "numpy" else _Ort
        ))
        recognizer._load_runtime()
    finally:
        monkeypatch.undo()

    assert calls == [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    assert recognizer.model_metadata()["provider_fallback_reason"]


def test_profile_is_finalized_and_reported(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"a":0}', encoding="utf-8")
    mapping = _mapping(tmp_path / "mapping.json")
    session = _RuntimeSession(["CPUExecutionProvider"])
    recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
        model_path=model,
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="cpu",
        enable_profiling=True,
        session_factory=lambda path, providers: session,
    )
    recognizer._numpy = np
    recognizer._session = session
    assert recognizer.finalize_profile() == ".state/onnx-profile.json"
    assert session.profile_ended is True
    assert recognizer.model_metadata()["profile_path"] == ".state/onnx-profile.json"


def test_profile_session_options_are_scoped_to_state(tmp_path: Path) -> None:
    profile_root = tmp_path / ".state" / "profiles"
    observed: dict[str, object] = {}

    class _Options:
        enable_profiling = False
        profile_file_prefix = ""

    class _Ort:
        SessionOptions = _Options

        @staticmethod
        def InferenceSession(
            model_path: str,
            *,
            sess_options: _Options,
            providers: list[str],
        ) -> _RuntimeSession:
            observed.update(
                model_path=model_path,
                enable_profiling=sess_options.enable_profiling,
                profile_file_prefix=sess_options.profile_file_prefix,
                providers=providers,
            )
            return _RuntimeSession(providers)

    recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
        device="cpu",
        phone_mapping_path=_mapping(tmp_path / "mapping.json"),
        enable_profiling=True,
        profile_output_dir=profile_root,
    )
    recognizer._create_session(_Ort, "model.onnx", ("CPUExecutionProvider",))

    assert observed["enable_profiling"] is True
    assert Path(str(observed["profile_file_prefix"])).parent == profile_root.resolve()
    assert str(recognizer.model_metadata()["profile_output_dir"]) == str(profile_root)

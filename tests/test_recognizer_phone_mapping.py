from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from any2xsampa import cli
from any2xsampa.adapters.onnx_espeak import (
    OnnxEspeakCTCUniversalPhoneRecognizer as OnnxRecognizer,
    OnnxEspeakResourceError,
)
from any2xsampa.adapters.recognizer import ConfigurableUniversalPhoneRecognizer, RecognizerConfig
from any2xsampa.adapters.recognizer_dispatch import build_recognizer_dispatch_plan
from any2xsampa.adapters.recognizer_real import RealBackendUniversalPhoneRecognizer
from any2xsampa.services import inference
from any2xsampa.services.inference import InferenceRequest, InferenceService


@pytest.fixture(autouse=True)
def forbid_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Mapping contract tests must not load a model or read PCM")

    monkeypatch.setattr(OnnxRecognizer, "_load_runtime", forbidden)
    monkeypatch.setattr(OnnxRecognizer, "_read_pcm16_mono", forbidden)


@pytest.fixture
def mapping_file(tmp_path: Path) -> Path:
    path = tmp_path / "phone mapping with spaces.json"
    path.write_text(json.dumps({
        "schema_version": "onnx_espeak_internal_phone_mapping_v1",
        "profile": "test_optin",
        "blank_labels": ["<pad>"],
        "normalization": {"strip_trailing_tone_digits": True, "aliases": {"ng": "ŋ"}},
        "mappings": {"ŋ": ["ŋ"]},
    }), encoding="utf-8")
    return path


def request_for_mapping(path: str | None, **overrides) -> InferenceRequest:
    values = dict(mode="infer-open", audio_path="unused", dialect="cantonese",
                  recognizer_mode="real", recognizer_real_backend="onnx-espeak",
                  recognizer_phone_mapping=path)
    values.update(overrides)
    return InferenceRequest(**values)


@pytest.mark.parametrize("command", ["infer-open", "infer-with-lyrics"])
@pytest.mark.parametrize("backend", ["onnx-espeak", "onnx_espeak", "trained-onnx"])
def test_cli_mapping_reaches_real_factory_without_inference(
    command: str, backend: str, mapping_file: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    audio = tmp_path / "unread input"
    audio.touch()
    observed = []
    expected = {"segments": [], "marker": "unchanged output"}

    class FakePipeline:
        def __init__(self, recognizer_config):
            self.recognizer = ConfigurableUniversalPhoneRecognizer(recognizer_config)

        def run(self, **kwargs):
            wrapper = self.recognizer._build_primary()
            adapter = wrapper._build_backend()
            observed.append(adapter)
            assert adapter.phone_mapping_path == mapping_file
            assert adapter._map_raw_label("ng5") == "ŋ"
            assert adapter._project_raw_label("unmapped") == "unmapped"
            assert adapter._project_raw_label("<pad>") is None
            assert adapter.model_metadata()["phone_mapping_profile"] == "test_optin"
            assert adapter._session is None
            return SimpleNamespace(to_dict=lambda: expected)

    monkeypatch.setattr(inference, "SingingToPhonesPipeline", FakePipeline)
    monkeypatch.setattr(cli, "augment_pipeline_payload", lambda payload: payload)
    args = [command, "--audio", str(audio), "--dialect", "cantonese",
            "--recognizer-mode", "real", "--recognizer-real-backend", backend,
            "--recognizer-phone-mapping", str(mapping_file)]
    if command == "infer-with-lyrics":
        args += ["--lyrics", "test lyrics"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert len(observed) == 1


def test_runtime_round_trip_keeps_mapping_out_of_public_contract(mapping_file: Path) -> None:
    request = request_for_mapping(str(mapping_file))
    original = request_for_mapping(None)
    assert request.to_mapping() == original.to_mapping()
    runtime = request.to_runtime_mapping()
    assert runtime["recognizer_phone_mapping"] == str(mapping_file)
    restored = InferenceRequest.from_mapping(runtime, mode=request.mode)
    assert restored.recognizer_phone_mapping == str(mapping_file)
    assert original.to_runtime_mapping()["recognizer_phone_mapping"] is None


def test_default_mapping_and_dispatch_remain_unchanged() -> None:
    config = RecognizerConfig(mode="real", real_backend="onnx-espeak")
    plan = build_recognizer_dispatch_plan(config)
    assert plan.phone_mapping_path is None
    adapter = ConfigurableUniversalPhoneRecognizer(config)._build_primary()._build_backend()
    assert adapter.phone_mapping_path.name == "recognizer_onnx_espeak_ipa_map.json"
    assert adapter._map_raw_label("f") == "f"
    assert adapter._session is None
    args = cli.build_parser().parse_args(["infer-open", "--audio", "unused", "--dialect", "cantonese"])
    assert args.recognizer_phone_mapping is None
    assert InferenceRequest.from_args_namespace(args, mode="infer-open").recognizer_phone_mapping is None


@pytest.mark.parametrize("mode,backend,command", [
    ("fixture", "onnx-espeak", None),
    ("real", None, "producer {audio} {output}"),
    ("real", "posterior-json", None),
    ("real", "cantonese-singing-ctc", None),
    ("real", "unknown", None),
    ("real", "onnx-espeak", "producer {audio} {output}"),
])
def test_inapplicable_selection_is_rejected_at_request_and_dispatch(
    mode: str, backend: str | None, command: str | None, mapping_file: Path,
) -> None:
    with pytest.raises(ValueError):
        request_for_mapping(str(mapping_file), recognizer_mode=mode,
                            recognizer_real_backend=backend, recognizer_real_command=command)
    with pytest.raises(ValueError, match="phone_mapping_path requires real mode"):
        build_recognizer_dispatch_plan(RecognizerConfig(
            mode=mode, real_backend=backend, real_command=command,
            fixture_path=Path("unused fixture"), phone_mapping_path=mapping_file,
        ))


def test_direct_real_wrapper_rejects_mapping_for_other_backend(mapping_file: Path) -> None:
    with pytest.raises(ValueError, match="phone_mapping_path requires real mode"):
        RealBackendUniversalPhoneRecognizer("posterior-json", phone_mapping_path=mapping_file)


@pytest.mark.parametrize("mapping_kind", ["missing", "directory"])
def test_mapping_path_must_be_a_file_before_pipeline(
    mapping_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "unread input"
    audio.touch()
    mapping = tmp_path if mapping_kind == "directory" else tmp_path / "missing.json"

    def forbidden_pipeline(**kwargs):
        pytest.fail("Invalid mapping paths must fail before pipeline construction")

    monkeypatch.setattr(inference, "SingingToPhonesPipeline", forbidden_pipeline)
    with pytest.raises(FileNotFoundError, match="Recognizer phone mapping file not found"):
        InferenceService().run(request_for_mapping(str(mapping), audio_path=str(audio)))


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_explicit_mapping_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="non-empty path"):
        request_for_mapping(value)
    with pytest.raises(ValueError, match="non-empty path"):
        build_recognizer_dispatch_plan(RecognizerConfig(
            mode="real", real_backend="onnx-espeak", phone_mapping_path=value,
        ))


@pytest.mark.parametrize("payload,message", [
    ("{broken", "invalid JSON"),
    ('{"mappings":{"a":["same"],"i":["same"]}}', "mapped to both"),
    ('{"mappings":[]}', "mappings must be an object"),
])
def test_invalid_mapping_never_becomes_debug_stub(
    payload: str, message: str, tmp_path: Path,
) -> None:
    path = tmp_path / "invalid mapping.json"
    path.write_text(payload, encoding="utf-8")
    recognizer = ConfigurableUniversalPhoneRecognizer(RecognizerConfig(
        mode="real", real_backend="onnx-espeak", phone_mapping_path=path,
        allow_stub_fallback=True,
    ))
    with pytest.raises(OnnxEspeakResourceError, match=message):
        recognizer.recognize(Path("never read PCM"))


def test_cli_rejects_fixture_mapping_before_inference(mapping_file: Path, capsys) -> None:
    assert cli.main(["infer-open", "--audio", "unused", "--dialect", "cantonese",
                     "--recognizer-phone-mapping", str(mapping_file)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    error = json.loads(captured.err)
    assert "phone_mapping_path requires real mode" in error["error"]
    assert error["error_code"] == "cli.validation_error"

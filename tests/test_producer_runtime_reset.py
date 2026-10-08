from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from any2xsampa.adapters import onnx_espeak_runtime
from any2xsampa.adapters.recognizer_real import ExternalPosteriorJSONUniversalPhoneRecognizer
from any2xsampa.adapters.recognizer import ConfigurableUniversalPhoneRecognizer, RecognizerConfig
from any2xsampa.core.models import PhoneSegment


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_TOOL = REPO_ROOT / "tools" / "posterior_json_producer.py"
BUNDLE_TOOL = REPO_ROOT / "tools" / "onnx_espeak_bundle_packager.py"
DEMO_AUDIO = REPO_ROOT / "samples" / "demo.wav"


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(REPO_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    return environment


def _run_tool(tool: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(tool), *arguments],
        cwd=REPO_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_success(result: subprocess.CompletedProcess[str], context: str) -> None:
    assert result.returncode == 0, (
        f"{context} failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_original_route_tools_are_importable() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import tools.posterior_json_producer as producer; "
                "import tools.onnx_espeak_bundle_packager as packager; "
                "assert callable(producer.main); assert callable(packager.main)"
            ),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_success(probe, "producer/runtime tool import")


@pytest.mark.parametrize("tool", [PRODUCER_TOOL, BUNDLE_TOOL], ids=["producer", "bundle-packager"])
def test_original_route_tools_expose_help(tool: Path) -> None:
    result = _run_tool(tool, "--help")
    _assert_success(result, f"{tool.name} --help")
    assert "usage:" in result.stdout.lower()


def test_producer_demo_output_round_trips_to_phone_segments(tmp_path: Path) -> None:
    assert DEMO_AUDIO.is_file(), f"required demo audio is missing: {DEMO_AUDIO}"
    output_path = tmp_path / "demo.posterior.json"
    output_path.write_text(
        json.dumps({"stale_runtime_marker": True, "segments": []}),
        encoding="utf-8",
    )

    result = _run_tool(
        PRODUCER_TOOL,
        "--audio",
        str(DEMO_AUDIO),
        "--output",
        str(output_path),
        "--diagnostic-backend",
        "prototype-logit",
    )
    _assert_success(result, "posterior producer demo run")
    assert output_path.is_file()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "external_posterior_json_v1"
    assert "stale_runtime_marker" not in payload
    assert isinstance(payload.get("segments"), list)
    assert payload["segments"]
    assert isinstance(payload.get("producer"), dict)

    recognizer = ExternalPosteriorJSONUniversalPhoneRecognizer(
        output_path,
        backend_name="posterior-json",
    )
    segments = recognizer.recognize(DEMO_AUDIO)
    assert segments
    assert all(isinstance(segment, PhoneSegment) for segment in segments)

    serialized = [segment.to_dict() for segment in segments]
    json.dumps(serialized, ensure_ascii=False, sort_keys=True)
    for segment in serialized:
        assert segment["end_sec"] > segment["start_sec"]
        assert segment["candidate_phones"]
        assert all(candidate["symbol"].strip() for candidate in segment["candidate_phones"])
        assert all(0.0 <= candidate["confidence"] <= 1.0 for candidate in segment["candidate_phones"])
        assert segment["metadata"]["source_mode"] == "real"
        assert segment["metadata"]["backend_kind"] == "posterior-json"


def test_original_producer_output_enters_configured_posterior_json_recognizer(
    tmp_path: Path,
) -> None:
    """Keep the external-producer -> real backend -> PhoneSegment route wired."""

    output_path = tmp_path / "configured.posterior.json"
    result = _run_tool(
        PRODUCER_TOOL,
        "--audio",
        str(DEMO_AUDIO),
        "--output",
        str(output_path),
        "--diagnostic-backend",
        "prototype-logit",
    )
    _assert_success(result, "configured posterior producer run")

    recognizer = ConfigurableUniversalPhoneRecognizer(
        RecognizerConfig(
            mode="real",
            real_backend="posterior-json",
            real_output_path=output_path,
        )
    )
    segments = recognizer.recognize(DEMO_AUDIO, lyrics="ignored")

    assert segments
    assert all(isinstance(segment, PhoneSegment) for segment in segments)
    assert all(segment.metadata["source_mode"] == "real" for segment in segments)
    assert all(segment.metadata["backend_kind"] == "posterior-json" for segment in segments)
    assert all(segment.metadata["lyrics_guided"] is True for segment in segments)


def test_runtime_diagnostics_command_emits_archivable_json() -> None:
    result = _run_tool(PRODUCER_TOOL, "--diagnose-runtime")
    _assert_success(result, "onnx-espeak runtime diagnostics")
    diagnostics = json.loads(result.stdout)
    assert isinstance(diagnostics, dict)
    assert diagnostics


def test_onnx_runtime_bundle_helper_persists_manifest_and_state(tmp_path: Path) -> None:
    runtime_root = tmp_path / "onnx-espeak-runtime"
    layout = onnx_espeak_runtime.prepare_onnx_espeak_runtime_bundle(
        runtime_root,
        bundle_name="test-runtime",
    )

    manifest_path = Path(layout["manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == onnx_espeak_runtime.BUNDLE_SCHEMA_VERSION
    assert manifest["bundle_name"] == "test-runtime"
    assert manifest["worker_entrypoint"] == "worker/posterior_json_producer.py"

    state = onnx_espeak_runtime.write_onnx_espeak_bundle_state(
        runtime_root,
        runtime_diagnostics={
            "runtime_status": "unavailable",
            "ready": False,
            "model_resources": {"status": "missing"},
        },
    )
    status_path = Path(layout["status_file"])
    assert status_path.is_file()
    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == onnx_espeak_runtime.BUNDLE_SCHEMA_VERSION
    assert persisted["runtime_root"] == str(runtime_root.resolve())
    assert persisted["runtime_status"] == "unavailable"
    assert state["missing_components"] == persisted["missing_components"]


def test_prepared_bundle_worker_is_a_source_resolving_launcher(tmp_path: Path) -> None:
    runtime_root = tmp_path / "onnx-espeak-runtime"
    layout = onnx_espeak_runtime.prepare_onnx_espeak_runtime_bundle(runtime_root)

    worker = Path(layout["worker_entrypoint"])
    source = worker.read_text(encoding="utf-8")

    assert "ANY2XSAMPA_REPO_ROOT" in source
    assert "runpy.run_path" in source
    assert "tools/posterior_json_producer.py" in source
    assert "from any2xsampa.adapters" not in source


@pytest.mark.parametrize(
    ("previous_runtime", "previous_python"),
    [(None, None), ("previous-runtime", "previous-python")],
    ids=["unset", "restores-existing-values"],
)
def test_bundle_packager_restores_temporary_runtime_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    previous_runtime: str | None,
    previous_python: str | None,
) -> None:
    import tools.onnx_espeak_bundle_packager as packager

    runtime_root = tmp_path / "runtime"
    if previous_runtime is None:
        monkeypatch.delenv("ANY2XSAMPA_ONNX_ESPEAK_RUNTIME", raising=False)
    else:
        monkeypatch.setenv("ANY2XSAMPA_ONNX_ESPEAK_RUNTIME", previous_runtime)
    if previous_python is None:
        monkeypatch.delenv("ANY2XSAMPA_ONNX_ESPEAK_PYTHON", raising=False)
    else:
        monkeypatch.setenv("ANY2XSAMPA_ONNX_ESPEAK_PYTHON", previous_python)

    monkeypatch.setattr(
        packager,
        "describe_onnx_espeak_runtime",
        lambda: {
            "runtime_status": "unavailable",
            "ready": False,
            "model_resources": {"status": "missing"},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUNDLE_TOOL),
            "--runtime-root",
            str(runtime_root),
            "--external-python",
            "temporary-python",
        ],
    )

    packager.main()

    if previous_runtime is None:
        assert "ANY2XSAMPA_ONNX_ESPEAK_RUNTIME" not in os.environ
    else:
        assert os.environ["ANY2XSAMPA_ONNX_ESPEAK_RUNTIME"] == previous_runtime
    if previous_python is None:
        assert "ANY2XSAMPA_ONNX_ESPEAK_PYTHON" not in os.environ
    else:
        assert os.environ["ANY2XSAMPA_ONNX_ESPEAK_PYTHON"] == previous_python

    output = json.loads(capsys.readouterr().out)
    assert output["runtime_status"] == "unavailable"
    assert Path(output["status_file"]).is_file()

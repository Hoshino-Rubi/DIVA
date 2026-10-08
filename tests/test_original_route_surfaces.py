from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from any2xsampa.adapters.recognizer_real import ExternalPosteriorJSONUniversalPhoneRecognizer
from any2xsampa.core.models import PhoneSegment


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_TOOL = REPO_ROOT / "tools" / "posterior_json_producer.py"
BUNDLE_TOOL = REPO_ROOT / "tools" / "onnx_espeak_bundle_packager.py"


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


def test_original_producer_and_runtime_tools_are_importable() -> None:
    producer = importlib.import_module("tools.posterior_json_producer")
    packager = importlib.import_module("tools.onnx_espeak_bundle_packager")

    assert callable(producer.main)
    assert callable(producer.available_producer_backend_names)
    assert {"onnx-espeak", "prototype-logit", "heuristic-dsp"}.issubset(
        set(producer.available_producer_backend_names())
    )
    assert callable(packager.main)


def test_external_posterior_json_round_trips_to_phone_segments(tmp_path: Path) -> None:
    audio_path = tmp_path / "vocal.wav"
    audio_path.write_bytes(b"RIFF fixture audio")
    output_path = tmp_path / "posterior.json"
    payload = {
        "schema_version": "external_posterior_json_v1",
        "producer": {
            "backend_kind": "prototype-logit",
            "raw_model_output_kind": "prototype_frame_logit_model",
        },
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 0.25,
                "confidence": 0.72,
                "candidate_phones": [
                    {"symbol": "a", "confidence": 0.72},
                    {"symbol": "i", "confidence": 0.28},
                ],
            },
            {
                "start_sec": 0.25,
                "end_sec": 0.55,
                "posterior_trace": {
                    "frames": [
                        {
                            "candidates": [
                                {"symbol": "s", "confidence": 0.8},
                                {"symbol": "z", "confidence": 0.2},
                            ]
                        },
                        {
                            "candidates": [
                                {"symbol": "s", "confidence": 0.6},
                                {"symbol": "z", "confidence": 0.4},
                            ]
                        },
                    ]
                },
            },
        ],
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    segments = ExternalPosteriorJSONUniversalPhoneRecognizer(
        output_path,
        backend_name="posterior-json",
    ).recognize(audio_path)

    assert len(segments) == 2
    assert all(isinstance(segment, PhoneSegment) for segment in segments)
    serialized = [segment.to_dict() for segment in segments]
    json.dumps(serialized, ensure_ascii=False, sort_keys=True)
    assert serialized[0]["start_sec"] == 0.0
    assert serialized[0]["end_sec"] == 0.25
    assert serialized[0]["candidate_phones"][0]["symbol"] == "a"
    assert serialized[1]["candidate_phones"][0]["symbol"] == "s"
    for segment in serialized:
        assert segment["end_sec"] > segment["start_sec"]
        assert 0.0 <= segment["confidence"] <= 1.0
        assert segment["candidate_phones"]
        assert segment["metadata"]["source_mode"] == "real"
        assert segment["metadata"]["backend_kind"] == "posterior-json"
        assert segment["metadata"]["audio_binding"]["status"] == "legacy_unbound"


def test_external_posterior_rejects_audio_hash_mismatch(tmp_path: Path) -> None:
    audio_path = REPO_ROOT / "samples" / "demo.wav"
    output_path = tmp_path / "posterior.json"
    payload = {
        "schema_version": "external_posterior_json_v1",
        "producer": {"input_audio": {"sha256": "0" * 64}},
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 0.1,
                "candidate_phones": [{"symbol": "a", "confidence": 1.0}],
            }
        ],
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="audio provenance error.*hash mismatch"):
        ExternalPosteriorJSONUniversalPhoneRecognizer(output_path).recognize(audio_path)


def test_external_posterior_orders_direct_candidates_by_confidence(tmp_path: Path) -> None:
    audio_path = REPO_ROOT / "samples" / "demo.wav"
    output_path = tmp_path / "posterior.json"
    payload = {
        "schema_version": "external_posterior_json_v1",
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 0.1,
                "candidate_phones": [
                    {"symbol": "n", "confidence": 0.1},
                    {"symbol": "a", "confidence": 0.9},
                ],
            }
        ],
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    segments = ExternalPosteriorJSONUniversalPhoneRecognizer(output_path).recognize(audio_path)

    assert [candidate.symbol for candidate in segments[0].candidate_phones] == ["a", "n"]


def test_onnx_runtime_bundle_and_diagnostics_contract(tmp_path: Path) -> None:
    runtime_root = tmp_path / "onnx-espeak-runtime"
    result = _run_tool(
        BUNDLE_TOOL,
        "--runtime-root",
        str(runtime_root),
        "--bundle-name",
        "contract-test",
    )
    _assert_success(result, "onnx-espeak bundle packager")
    summary = json.loads(result.stdout)

    expected_summary_keys = {
        "runtime_root",
        "manifest_path",
        "worker_entrypoint",
        "status_file",
        "bundle_status",
        "runtime_status",
        "ready_components",
        "missing_components",
        "external_runtime",
        "model_resources",
    }
    assert expected_summary_keys <= summary.keys()
    assert Path(summary["runtime_root"]).resolve() == runtime_root.resolve()

    def _bundle_path(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else runtime_root / path

    manifest_path = _bundle_path(summary["manifest_path"])
    status_path = _bundle_path(summary["status_file"])
    worker_entrypoint = _bundle_path(summary["worker_entrypoint"])
    assert manifest_path.is_file()
    assert status_path.is_file()
    assert worker_entrypoint.parent.is_dir()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "onnx_espeak_runtime_bundle_v1"
    assert manifest["bundle_name"] == "contract-test"
    assert manifest["backend_kind"] == "onnx-espeak"
    assert manifest["worker_entrypoint"] == "worker/posterior_json_producer.py"
    assert manifest["status_file"] == "bundle_state.json"
    assert isinstance(manifest["required_dependencies"], list)

    state = json.loads(status_path.read_text(encoding="utf-8"))
    assert state["schema_version"] == "onnx_espeak_runtime_bundle_v1"
    assert state["runtime_root"] == str(runtime_root.resolve())
    assert state["bundle_status"] == summary["bundle_status"]
    assert state["runtime_status"] == summary["runtime_status"]
    assert isinstance(state["ready_components"], list)
    assert isinstance(state["missing_components"], list)
    assert isinstance(state["model_resources"], dict)
    assert isinstance(state["runtime_diagnostics"], dict)

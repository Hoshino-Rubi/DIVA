from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from any2xsampa.cli import _request_mapping_from_args, build_parser
from any2xsampa.services.inference import InferenceRequest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, "-m", "any2xsampa.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_help_smoke() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "infer-open" in result.stdout
    assert "infer-with-lyrics" in result.stdout


def test_cli_infer_open_success_writes_same_payload_to_stdout_and_output(tmp_path: Path) -> None:
    output_path = tmp_path / "cli_result.json"

    result = _run_cli(
        "infer-open",
        "--audio",
        "samples/demo.wav",
        "--dialect",
        "cantonese",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0
    stdout_payload = json.loads(result.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["mode"] == "infer-open"
    assert stdout_payload["segments"][0]["metadata"]["future_aligner_provenance_view"] is not None


def test_cli_stdout_is_safe_under_cp950_with_non_ascii_ipa(tmp_path: Path) -> None:
    fixture = tmp_path / "ipa_fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 0.1,
                        "confidence": 1.0,
                        "candidate_phones": [{"symbol": "ɡ", "confidence": 1.0}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "infer-open",
        "--audio",
        "samples/demo.wav",
        "--dialect",
        "cantonese",
        "--recognizer-fixture",
        str(fixture),
        env_overrides={"PYTHONIOENCODING": "cp950:strict"},
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["segments"][0]["candidate_phones"][0]["symbol"] == "ɡ"


def test_cli_failure_emits_structured_error_to_stderr() -> None:
    result = _run_cli(
        "infer-open",
        "--audio",
        "samples/does_not_exist.wav",
        "--dialect",
        "cantonese",
    )

    assert result.returncode == 1
    error_payload = json.loads(result.stderr)
    assert error_payload["error"] == "Audio file not found: samples/does_not_exist.wav"
    assert error_payload["error_code"] == "cli.file_not_found"
    assert error_payload["error_kind"] == "file_not_found"
    assert error_payload["exit_code"] == 1
    assert error_payload["details"] == {"transport": "cli"}


def test_cli_request_mapping_round_trips_through_inference_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "infer-with-lyrics",
            "--audio",
            "samples/demo.wav",
            "--dialect",
            "cantonese",
            "--lyrics",
            "laa ng",
            "--export-format",
            "openutau_like",
            "--engine-profile",
            "generic_openutau",
            "--engine-inventory",
            "tests/fixtures/openutau_bank_inventory_behavior.json",
            "--note-context",
            "tests/fixtures/note_context_openutau_project.ustx",
            "--note-context-format",
            "ustx",
            "--engine-tag",
            "lead",
        ]
    )

    payload = _request_mapping_from_args(args)
    request = InferenceRequest.from_mapping(payload, mode="infer-with-lyrics")

    assert request.to_mapping() == {
        "audio_path": "samples/demo.wav",
        "dialect": "cantonese",
        "lyrics": "laa ng",
        "export_format": "openutau_like",
        "engine_profile": "generic_openutau",
        "engine_inventory": "tests/fixtures/openutau_bank_inventory_behavior.json",
        "engine_subbank": None,
        "engine_color": None,
        "engine_style": None,
        "engine_tags": ["lead"],
        "note_context": "tests/fixtures/note_context_openutau_project.ustx",
        "note_context_format": "ustx",
        "recognizer_mode": "fixture",
        "recognizer_fixture": None,
        "recognizer_real_backend": None,
        "recognizer_real_command": None,
        "recognizer_real_output": "real_recognizer_output.json",
        "disable_recognizer_fallback": False,
    }


def test_runtime_mapping_preserves_real_model_fields_without_widening_public_mapping() -> None:
    request = InferenceRequest(
        mode="infer-open",
        audio_path="audio.wav",
        dialect="cantonese",
        recognizer_mode="real",
        recognizer_real_backend="cantonese-singing-ctc",
        recognizer_model="bundle",
        recognizer_device="cuda",
        debug_allow_recognizer_stub_fallback=True,
    )

    assert "recognizer_model" not in request.to_mapping()
    runtime = request.to_runtime_mapping()
    assert runtime["recognizer_model"] == "bundle"
    assert runtime["recognizer_device"] == "cuda"
    assert runtime["debug_allow_recognizer_stub_fallback"] is True


def test_real_backend_and_command_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        InferenceRequest(
            mode="infer-open",
            audio_path="audio.wav",
            dialect="cantonese",
            recognizer_mode="real",
            recognizer_real_backend="posterior-json",
            recognizer_real_command="producer {audio} {output}",
        )


@pytest.mark.parametrize(
    "field_name",
    ["disable_recognizer_fallback", "debug_allow_recognizer_stub_fallback"],
)
def test_inference_mapping_rejects_string_booleans(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be a JSON boolean"):
        InferenceRequest.from_mapping(
            {
                "audio_path": "audio.wav",
                "dialect": "cantonese",
                field_name: "false",
            },
            mode="infer-open",
        )

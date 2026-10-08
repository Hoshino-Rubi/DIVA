from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from any2xsampa.asr import qwen3 as qwen3_compat

from any2xsampa.asr.qwen3_adapter import (
    ASRTranscriptError,
    Qwen3ASRAdapter,
    Qwen3ASRLoadError,
    Qwen3ASRRequest,
    dumps_transcript_json,
    parse_forced_align_items,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FakeForcedAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class FakeForcedAlignResult:
    items: list[FakeForcedAlignItem]


@dataclass(frozen=True)
class FakeTranscription:
    language: str
    text: str
    time_stamps: FakeForcedAlignResult | None = None


class FakeQwen3Backend:
    backend = "transformers"
    device = "cpu"
    dtype = "float32"
    processor = object()
    model = None
    forced_aligner = object()

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio: Any,
        context: str = "",
        language: str | None = None,
        return_time_stamps: bool = False,
    ) -> list[FakeTranscription]:
        self.calls.append(
            {
                "audio": audio,
                "context": context,
                "language": language,
                "return_time_stamps": return_time_stamps,
            }
        )
        timestamps = None
        if return_time_stamps:
            timestamps = FakeForcedAlignResult(
                items=[
                    FakeForcedAlignItem("你", 0.0, 0.31),
                    FakeForcedAlignItem("好", 0.31, 0.62),
                ]
            )
        return [FakeTranscription(language=language or "Chinese", text="你好", time_stamps=timestamps)]


def test_fake_transcribe_outputs_asr_transcript_v1_contract() -> None:
    backend = FakeQwen3Backend()
    adapter = Qwen3ASRAdapter(model_name_or_path="fake/model", backend=backend)

    payload = adapter.transcribe(
        Qwen3ASRRequest(
            audio_path=REPO_ROOT / "samples" / "demo.wav",
            language="Chinese",
            context="粤语提示",
            timestamps=True,
        )
    )

    assert backend.calls == [
        {
            "audio": str(REPO_ROOT / "samples" / "demo.wav"),
            "context": "粤语提示",
            "language": "Chinese",
            "return_time_stamps": True,
        }
    ]
    assert payload["schema_version"] == "asr_transcript_v1"
    assert payload["model_output"] is True
    assert payload["provenance"]["truth_layer"] == "asr_evidence_not_phone_truth"
    assert payload["model"]["name_or_path"] == "fake/model"
    assert payload["model"]["device"] == "cpu"
    assert payload["model"]["model_output"] is True
    assert payload["text"] == "你好"
    segment = payload["segments"][0]
    assert segment["text"] == "你好"
    assert segment["timestamp_source"] == "forced_aligner"
    assert segment["items"] == [
        {"index": 0, "text": "你", "start_sec": 0.0, "end_sec": 0.31},
        {"index": 1, "text": "好", "start_sec": 0.31, "end_sec": 0.62},
    ]
    assert "FakeForcedAlignResult" not in dumps_transcript_json(payload)


def test_normalized_qwen_transcript_binds_audio_hash() -> None:
    backend = FakeQwen3Backend()
    adapter = Qwen3ASRAdapter(model_name_or_path="fake/model", backend=backend)
    payload = adapter.transcribe(
        Qwen3ASRRequest(audio_path=REPO_ROOT / "samples" / "demo.wav", timestamps=True)
    )
    assert payload["provenance"]["audio_sha256"] == hashlib.sha256(
        (REPO_ROOT / "samples" / "demo.wav").read_bytes()
    ).hexdigest()


def test_forced_align_result_dict_items_are_parsed_not_repr() -> None:
    items = parse_forced_align_items(
        {
            "items": [
                {"text": "a", "start_time": "0.1254", "end_time": 0.2255},
                {"text": "b", "start_sec": 0.2255, "end_sec": "0.3255", "confidence": 0.8},
            ]
        }
    )

    assert items == [
        {"index": 0, "text": "a", "start_sec": 0.125, "end_sec": 0.226},
        {"index": 1, "text": "b", "start_sec": 0.226, "end_sec": 0.326, "confidence": 0.8},
    ]


def test_model_missing_without_injection_has_clear_error() -> None:
    adapter = Qwen3ASRAdapter()

    with pytest.raises(Qwen3ASRLoadError, match="model is not configured"):
        adapter.transcribe(Qwen3ASRRequest(audio_path=REPO_ROOT / "samples" / "demo.wav"))


def test_runtime_loader_forwards_isolated_model_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    loaded_backend = object()

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, model_ref: str, **kwargs: object) -> object:
            calls.append((model_ref, kwargs))
            return loaded_backend

    fake_module = types.ModuleType("qwen_asr")
    fake_module.Qwen3ASRModel = FakeQwen3ASRModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_module)

    adapter = Qwen3ASRAdapter(
        model_name_or_path=str(model_dir),
        forced_aligner="forced-aligner",
        device="cuda:0",
        dtype="bfloat16",
        model_kwargs={"local_files_only": True, "max_inference_batch_size": 1},
    )

    assert adapter._load_backend() is loaded_backend
    assert adapter._load_backend() is loaded_backend
    assert calls == [
        (
            str(model_dir),
            {
                "local_files_only": True,
                "max_inference_batch_size": 1,
                "forced_aligner": "forced-aligner",
                "device_map": "cuda:0",
                "dtype": "bfloat16",
            },
        )
    ]


def test_invalid_forced_align_repr_string_is_rejected() -> None:
    with pytest.raises(ASRTranscriptError, match="not text repr"):
        parse_forced_align_items("ForcedAlignResult(items=[...])")


def test_forced_alignment_rejects_reversed_intervals() -> None:
    with pytest.raises(ASRTranscriptError, match="positive duration"):
        parse_forced_align_items(
            [{"text": "a", "start_sec": 1.0, "end_sec": 0.5}]
        )


def test_forced_alignment_rejects_zero_duration_and_rounding_to_negative_zero() -> None:
    with pytest.raises(ASRTranscriptError, match="positive duration"):
        parse_forced_align_items(
            [{"text": "a", "start_sec": 0.0, "end_sec": 0.0}]
        )
    with pytest.raises(ASRTranscriptError, match="non-negative"):
        parse_forced_align_items(
            [{"text": "a", "start_sec": -0.0004, "end_sec": 0.1}]
        )


def test_forced_alignment_rejects_unsorted_or_overlapping_items() -> None:
    with pytest.raises(ASRTranscriptError, match="out of order or overlaps"):
        parse_forced_align_items(
            [
                {"text": "a", "start_sec": 0.0, "end_sec": 0.5},
                {"text": "b", "start_sec": 0.4, "end_sec": 0.8},
            ]
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_forced_alignment_rejects_non_finite_timestamps(value: float) -> None:
    with pytest.raises(ASRTranscriptError, match="finite"):
        parse_forced_align_items(
            [{"text": "a", "start_sec": value, "end_sec": 1.0}]
        )


def test_compat_provenance_cannot_override_asr_only_truth_boundary() -> None:
    payload = {
        "schema_version": "asr_transcript_v1",
        "text": "a",
        "segments": [],
        "provenance": {
            "truth_layer": "asr_evidence_not_phone_truth",
            "adapter": "Qwen3ASRAdapter",
            "model_output": True,
            "audio_sha256": "a" * 64,
            "request_sha256": "b" * 64,
        },
    }
    with pytest.raises(ValueError, match="protected fields"):
        qwen3_compat._legacy_payload(
            payload,
            audio_path="audio.wav",
            context="",
            model_name="fixture",
            forced_aligner=None,
            device="cpu",
            dtype="float32",
            extra_provenance={"truth_layer": "phones"},
        )


def test_cli_no_model_errors_cleanly() -> None:
    completed = _run_tool("--audio", "samples/demo.wav")

    assert completed.returncode == 1
    payload = json.loads(completed.stderr)
    assert payload["error_code"] == "qwen3_asr.model_required"
    assert "Pass --model" in payload["error"]
    assert completed.stdout == ""


def test_cli_fake_path_serializes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "asr.json"

    completed = _run_tool(
        "--audio",
        "samples/demo.wav",
        "--fake",
        "--language",
        "Chinese",
        "--context",
        "abc",
        "--timestamps",
        "--output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "asr_transcript_v1"
    assert payload["text"] == "abc"
    assert payload["segments"][0]["items"][1] == {
        "index": 1,
        "text": "b",
        "start_sec": 0.25,
        "end_sec": 0.5,
    }


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "tools/qwen3_asr_transcribe.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

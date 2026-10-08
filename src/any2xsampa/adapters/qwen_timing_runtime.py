"""Qwen auxiliary evidence with native timestamp repair explicitly bypassed."""
from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path
import time
from typing import Any

from any2xsampa.adapters.melband_runtime import verify_model_assets
from any2xsampa.asr.qwen3_adapter import normalize_qwen3_transcript
from any2xsampa.training.auxiliary_slicing import audit_timestamps


class RawTimestampCapture:
    """Instance-local adapter hook. Upstream code/model bytes are never edited."""
    def __init__(self, processor: Any) -> None:
        self.native = processor.fix_timestamp
        self.calls: list[dict] = []

    def __call__(self, values: Any) -> list:
        raw = values.tolist()
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in raw):
            raise ValueError("non-finite raw timestamp token evidence")
        try:
            fixed = self.native(values) if len(raw) else []
            native_error = None
        except Exception as exc:
            fixed, native_error = None, f"{type(exc).__name__}: {exc}"
        self.calls.append({"raw_ms": raw, "native_repaired_ms_diagnostic_only": fixed, "native_repair_error": native_error,
                           "native_changed_values": fixed != raw if fixed is not None else None})
        return raw


class QwenAuxiliaryRuntime:
    def __init__(self, run: Path, cache: Path, role: str) -> None:
        from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget, configure_cuda, enforce_offline

        if role not in {"asr", "timestamps"}:
            raise ValueError("unsupported Qwen auxiliary role")
        self.identity, self.policy = verify_model_assets(run, role)
        enforce_offline(cache)
        self.torch = torch = configure_cuda()
        if importlib.metadata.version("qwen-asr") != "0.0.6":
            raise RuntimeError("Qwen runtime must equal the predeclared 0.0.6 version")
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

        before = time.monotonic()
        kwargs = {"device_map": "cuda:0", "dtype": torch.bfloat16, "local_files_only": True}
        if role == "asr":
            self.backend = Qwen3ASRModel.from_pretrained(self.identity["root"], max_inference_batch_size=1, max_new_tokens=512, **kwargs)
            self.capture = None
        else:
            self.backend = Qwen3ForcedAligner.from_pretrained(self.identity["root"], **kwargs)
            self.capture = RawTimestampCapture(self.backend.aligner_processor)
            self.backend.aligner_processor.fix_timestamp = self.capture
        self.backend.model.eval().requires_grad_(False)
        if any(p.device.type != "cuda" for p in self.backend.model.parameters()):
            raise RuntimeError("Qwen parameters are not all on CUDA")
        from any2xsampa.adapters.loaded_weight_verification import verify_loaded_tensors
        self.loading_verification = verify_loaded_tensors(torch, self.backend.model, Path(self.identity["root"]) / "model.safetensors")
        self.role = role
        self.load_seconds = time.monotonic() - before
        self.initial_memory = check_gpu_budget(torch)

    def _begin(self) -> float:
        from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
        check_gpu_budget(self.torch)
        self.torch.cuda.reset_peak_memory_stats()
        return time.monotonic()

    def _finish(self, before: float) -> dict:
        from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
        torch = self.torch
        torch.cuda.synchronize()
        return {"elapsed_seconds": time.monotonic() - before, "load_seconds": self.load_seconds,
                "device": "cuda:0", "dtype": "bfloat16", "cpu_threads": 4,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "memory_after": check_gpu_budget(torch), "offline": True,
                "loading_verification": self.loading_verification,
                "qwen_frontend": "native_mono_librosa16k_peak_normalize_only_if_above_1_then_clip"}

    def transcribe(self, path: Path) -> tuple[dict, dict]:
        if self.role != "asr":
            raise ValueError("the timing model cannot transcribe")
        before = self._begin()
        with self.torch.inference_mode():
            raw = self.backend.transcribe(audio=str(path), context="", language="Cantonese", return_time_stamps=False)
        value = raw[0]
        raw_record = {"text": value.text, "language": value.language, "context": "", "timestamps_requested": False}
        runtime = self._finish(before)
        if not str(value.text).strip():
            return {"status": "empty_asr_output", "raw": raw_record, "runtime": runtime}, {}
        payload = normalize_qwen3_transcript(raw, audio_path=path, requested_model=self.identity["model"]["id"], requested_language="Cantonese", backend=self.backend)
        return {"status": "transcribed", "raw": raw_record, "runtime": runtime}, payload

    def align(self, path: Path, text: str, duration: float) -> dict:
        if self.role != "timestamps" or not text.strip():
            raise ValueError("timestamps require an existing nonempty audio-only ASR transcript")
        assert self.capture is not None
        self.capture.calls.clear()
        before = self._begin()
        results = self.backend.align(audio=str(path), text=text, language="Cantonese")
        rows = [{"text": item.text, "start_sec": item.start_time, "end_sec": item.end_time} for item in results[0].items]
        audit = audit_timestamps(rows, duration)
        return {"schema_version": "auxiliary_word_timing_v1", "raw_items": rows, "audit": audit,
                "raw_timestamp_capture": list(self.capture.calls), "runtime": self._finish(before),
                "upstream_repair_bypassed_for_proposals": True, "native_repair_is_diagnostic_only": True,
                "text_source": "same_audio_Qwen_ASR_without_context", "human_approved": False,
                "phone_reference_or_durations_produced": False}

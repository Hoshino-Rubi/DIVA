"""External producer adapter for the stable posterior JSON contract.

The producer is deliberately separate from the recognizer adapter. A trained
ONNX/eSpeak producer delegates to the existing CTC runtime; diagnostic
backends are explicit and are never presented as model output.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import wave
from typing import Any, Callable, Mapping

from any2xsampa.adapters.onnx_espeak import OnnxEspeakCTCUniversalPhoneRecognizer
from any2xsampa.adapters.audio_provenance import describe_audio_file
from any2xsampa.adapters.onnx_espeak_runtime import (
    build_onnx_espeak_runtime_diagnostics_snapshot,
    describe_onnx_espeak_runtime,
    write_onnx_espeak_runtime_diagnostics,
)
from any2xsampa.core.models import PhoneSegment


SCHEMA_VERSION = "external_posterior_json_v1"


class PosteriorJSONProducerError(RuntimeError):
    """Raised when a producer cannot emit a valid posterior JSON payload."""


@dataclass(frozen=True, slots=True)
class ProducerBackendSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    producer_name: str
    adapter_kind: str
    raw_model_output_kind: str
    approximation_only: bool
    diagnostic_baseline: bool
    real_backend: bool
    real_model_output: bool
    aggregation: str

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "backend_kind": self.canonical_name,
            "posterior_like_available": True,
            "logit_like_available": True,
            "approximation_only": self.approximation_only,
            "source": f"producer:{self.canonical_name}",
            "aggregation": self.aggregation,
            "backend_role": (
                "diagnostic_baseline" if self.diagnostic_baseline else "trained_model_backend"
            ),
            "diagnostic_baseline": self.diagnostic_baseline,
            "real_backend": self.real_backend,
            "real_model_output": self.real_model_output,
        }


_BACKEND_SPECS: tuple[ProducerBackendSpec, ...] = (
    ProducerBackendSpec(
        canonical_name="onnx-espeak",
        aliases=("onnx-espeak", "onnx_espeak", "trained-onnx"),
        producer_name="score-aware-onnx-espeak-posterior-json",
        adapter_kind="score-aware-onnx-espeak-producer",
        raw_model_output_kind="trained_onnx_ctc_model",
        approximation_only=False,
        diagnostic_baseline=False,
        real_backend=True,
        real_model_output=True,
        aggregation="sum_mapped_posterior_mass",
    ),
    ProducerBackendSpec(
        canonical_name="prototype-logit",
        aliases=("prototype-logit", "prototype_logit", "prototype"),
        producer_name="score-aware-prototype-logit-posterior-json",
        adapter_kind="score-aware-prototype-logit-producer",
        raw_model_output_kind="prototype_frame_logit_model",
        approximation_only=True,
        diagnostic_baseline=True,
        real_backend=False,
        real_model_output=False,
        aggregation="mean_frame_posterior",
    ),
    ProducerBackendSpec(
        canonical_name="heuristic-dsp",
        aliases=("heuristic-dsp", "heuristic", "score-aware-heuristic"),
        producer_name="score-aware-heuristic-posterior-json",
        adapter_kind="score-aware-heuristic-producer",
        raw_model_output_kind="heuristic_frame_classifier",
        approximation_only=True,
        diagnostic_baseline=True,
        real_backend=False,
        real_model_output=False,
        aggregation="mean_frame_confidence",
    ),
)

_BACKEND_ALIASES = {alias: spec for spec in _BACKEND_SPECS for alias in spec.aliases}


def available_producer_backend_names(*, include_auto: bool = True) -> list[str]:
    """Return canonical producer names, optionally including ``auto``."""

    names = [spec.canonical_name for spec in _BACKEND_SPECS]
    return (["auto", *names] if include_auto else names)


def _resolve_backend(name: str) -> tuple[ProducerBackendSpec, str]:
    requested = str(name or "auto").strip().lower() or "auto"
    if requested == "auto":
        return _BACKEND_ALIASES["onnx-espeak"], "auto"
    spec = _BACKEND_ALIASES.get(requested)
    if spec is None:
        names = ", ".join(available_producer_backend_names())
        raise ValueError(f"Unknown producer backend '{name}'. Available backends: {names}")
    return spec, requested


def _audio_profile(path: Path) -> dict[str, Any]:
    """Read only WAV facts used by explicit diagnostic baselines."""

    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            raw = stream.readframes(min(frame_count, 4096))
    except (OSError, wave.Error) as exc:
        raise PosteriorJSONProducerError(f"Producer requires a readable PCM WAV file: {path}") from exc
    if channels < 1 or sample_rate < 1 or frame_count < 1:
        raise PosteriorJSONProducerError(f"Producer WAV metadata is invalid: {path}")
    # This is a diagnostic activity hint, not an acoustic model prediction.
    activity = (len(set(raw)) / 256.0) if raw else 0.0
    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "duration_sec": frame_count / sample_rate,
        "activity_hint": max(0.0, min(1.0, activity)),
    }


def _resolve_note_context_format(path: Path | None, requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value != "auto" or path is None:
        return value or "auto"
    suffix = path.suffix.lower()
    return {".json": "json", ".ustx": "ustx", ".ust": "ust"}.get(suffix, "auto")


def _diagnostic_segment(profile: Mapping[str, Any], spec: ProducerBackendSpec) -> dict[str, Any]:
    duration = float(profile["duration_sec"])
    activity = float(profile["activity_hint"])
    top_confidence = max(0.55, min(0.82, 0.55 + 0.27 * activity))
    runner_confidence = 1.0 - top_confidence
    symbols = ("a", "i") if spec.canonical_name == "prototype-logit" else ("s", "z")
    frame_count = 4
    frame_hop = duration / frame_count
    posterior_frames: list[dict[str, Any]] = []
    logit_frames: list[dict[str, Any]] = []
    for index in range(frame_count):
        wobble = 0.01 * ((index % 2) - 0.5)
        confidence = max(0.01, min(0.99, top_confidence + wobble))
        runner = 1.0 - confidence
        posterior_frames.append(
            {
                "start_sec": round(index * frame_hop, 9),
                "end_sec": round((index + 1) * frame_hop, 9),
                "candidates": [
                    {"symbol": symbols[0], "confidence": confidence},
                    {"symbol": symbols[1], "confidence": runner},
                ],
            }
        )
        logit_frames.append(
            {
                "start_sec": round(index * frame_hop, 9),
                "end_sec": round((index + 1) * frame_hop, 9),
                "candidates": [
                    {"symbol": symbols[0], "logit": math.log(max(confidence, 1e-9))},
                    {"symbol": symbols[1], "logit": math.log(max(runner, 1e-9))},
                ],
            }
        )
    posterior_trace = {
        "source": f"producer:{spec.canonical_name}:diagnostic_frame_posteriors",
        "frame_hop_sec": frame_hop,
        "frames": posterior_frames,
    }
    logit_trace = {
        "source": f"producer:{spec.canonical_name}:diagnostic_frame_logits",
        "frame_hop_sec": frame_hop,
        "frames": logit_frames,
    }
    metadata = {
        "source_mode": "diagnostic",
        "backend_kind": spec.canonical_name,
        "real_backend": False,
        "real_model_output": False,
        "model_output": False,
        "diagnostic_baseline": True,
        "baseline_backend": spec.canonical_name,
        "approximation_only": True,
        "adapter_kind": spec.adapter_kind,
        "candidate_aggregation": spec.aggregation,
        "posterior_like_available": True,
        "logit_like_available": True,
        "raw_model_output_kind": spec.raw_model_output_kind,
        "diagnostic_reason": "explicit diagnostic baseline; no trained recognizer model was used",
        "posterior_trace": posterior_trace,
        "logit_trace": logit_trace,
        "producer_raw_model_output": {
            "kind": "audio_activity_hint",
            "model_output": False,
            "activity_hint": activity,
        },
        "producer_contract_layers": {
            "payload": "raw",
            "segments": "phone_segment",
            "posterior": "posterior_trace",
            "logit": "logit_trace",
            "raw_model_output": "metadata.producer_raw_model_output",
        },
        "producer_generated_fields": ["posterior_trace", "logit_trace"],
        "adapter_diagnostics": {
            **spec.capability_metadata(),
            "adapter_kind": spec.adapter_kind,
            "candidate_source": posterior_trace["source"],
        },
    }
    return {
        "start_sec": 0.0,
        "end_sec": duration,
        "confidence": top_confidence,
        "candidate_phones": [
            {"symbol": symbols[0], "confidence": top_confidence},
            {"symbol": symbols[1], "confidence": runner_confidence},
        ],
        "posterior_trace": posterior_trace,
        "logit_trace": logit_trace,
        "segment_summary": {
            "acoustic_confidence": top_confidence,
            "timing_reliability": 0.5,
            "evidence_quality": 0.25,
            "notes": ["diagnostic baseline derived from WAV container/activity only"],
        },
        "adapter_diagnostics": metadata["adapter_diagnostics"],
        "metadata": metadata,
    }


def _segment_payload(segment: PhoneSegment, spec: ProducerBackendSpec) -> dict[str, Any]:
    payload = segment.to_dict()
    metadata = dict(payload.get("metadata") or {})
    raw_output = dict(metadata.get("producer_raw_model_output") or {})
    for trace_name in ("raw_label_trace", "mapped_frame_trace"):
        if trace_name in metadata and trace_name not in raw_output:
            raw_output[trace_name] = metadata[trace_name]
        if trace_name in metadata:
            payload[trace_name] = metadata[trace_name]
    for trace_name in ("posterior_trace", "logit_trace"):
        if trace_name in metadata:
            payload[trace_name] = metadata[trace_name]
    raw_output.setdefault("kind", spec.raw_model_output_kind)
    raw_output.setdefault("model_output", spec.real_model_output)
    metadata.update(
        {
            "producer_backend_kind": spec.canonical_name,
            "producer_name": spec.producer_name,
            "producer_version": "v1",
            "producer_raw_model_output": raw_output,
            "diagnostic_baseline": spec.diagnostic_baseline,
            "approximation_only": spec.approximation_only,
            "producer_contract_layers": {
                "payload": "raw",
                "segments": "phone_segment",
                "posterior": "posterior_trace",
                "logit": "logit_trace",
                "raw_model_output": "metadata.producer_raw_model_output",
            },
            "producer_generated_fields": [
                "posterior_trace",
                "logit_trace",
                "raw_label_trace",
                "mapped_frame_trace",
            ],
        }
    )
    payload["metadata"] = metadata
    return payload


class ScoreAwarePosteriorJSONProducer:
    """Produce ``external_posterior_json_v1`` from audio."""

    def __init__(
        self,
        backend_name: str = "auto",
        *,
        model_path: Path | str | None = None,
        device: str = "auto",
        cpu_thread_limit: int | None = None,
        chunk_duration_sec: float | None = None,
        chunk_overlap_sec: float = 0.5,
        runtime_describer: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.requested_backend_name = str(backend_name or "auto").strip().lower() or "auto"
        self.backend, self.requested_backend_name = _resolve_backend(self.requested_backend_name)
        self.backend_name = self.backend.canonical_name
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = str(device or "auto").strip().lower() or "auto"
        if cpu_thread_limit is not None and int(cpu_thread_limit) < 1:
            raise ValueError("Producer cpu_thread_limit must be at least one.")
        self.cpu_thread_limit = int(cpu_thread_limit) if cpu_thread_limit is not None else None
        self.chunk_duration_sec = float(chunk_duration_sec) if chunk_duration_sec is not None else None
        self.chunk_overlap_sec = float(chunk_overlap_sec)
        self._runtime_describer = runtime_describer or describe_onnx_espeak_runtime

    def produce(
        self,
        audio_path: Path | str,
        output_path: Path | str | None = None,
        *,
        note_context_path: Path | str | None = None,
        note_context_format: str = "auto",
        lyrics: str | None = None,
    ) -> dict[str, Any]:
        audio = Path(audio_path)
        if output_path is None:
            raise ValueError("Producer output_path is required.")
        output = Path(output_path)
        if not audio.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio}")
        input_audio = describe_audio_file(audio)
        note_path = Path(note_context_path) if note_context_path is not None else None
        if note_path is not None and not note_path.is_file():
            raise FileNotFoundError(f"Note context file not found: {note_path}")
        resolved_note_context_format = _resolve_note_context_format(note_path, note_context_format)

        if self.backend.diagnostic_baseline:
            profile = _audio_profile(audio)
            segments = [_diagnostic_segment(profile, self.backend)]
        else:
            recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
                model_path=self.model_path,
                device=self.device,
                cpu_thread_limit=self.cpu_thread_limit,
                chunk_duration_sec=self.chunk_duration_sec,
                chunk_overlap_sec=self.chunk_overlap_sec,
            )
            try:
                recognized = recognizer.recognize(audio, lyrics=lyrics)
            except Exception as exc:
                raise PosteriorJSONProducerError(
                    f"ONNX/eSpeak producer failed for {audio}: {type(exc).__name__}: {exc}"
                ) from exc
            segments = [_segment_payload(segment, self.backend) for segment in recognized]

        producer_context = {
            "audio_path": str(audio),
            "input_audio": input_audio,
            "note_context_path": str(note_path) if note_path is not None else None,
            "note_context_format": resolved_note_context_format,
            "note_context_role": "metadata_only_not_acoustic_input",
            "note_context_applied_to_acoustic_decoder": False,
            "note_context_pending_consumer": "pipeline_note_context_applier",
            "lyrics_guided": False,
            "lyrics_input_ignored": bool(lyrics),
        }
        for segment in segments:
            metadata = dict(segment.get("metadata") or {})
            metadata["producer_context"] = producer_context
            segment["metadata"] = metadata

        producer: dict[str, Any] = {
            "name": self.backend.producer_name,
            "version": "v1",
            "backend_kind": self.backend.canonical_name,
            "requested_backend_kind": self.requested_backend_name,
            "approximation_only": self.backend.approximation_only,
            "diagnostic_baseline": self.backend.diagnostic_baseline,
            "real_backend": self.backend.real_backend,
            "real_model_output": self.backend.real_model_output,
            "model_output": self.backend.real_model_output,
            "raw_model_output": self.backend.raw_model_output_kind,
            "raw_model_output_kind": self.backend.raw_model_output_kind,
            "audio_path": str(audio),
            "input_audio": input_audio,
            "lyrics_guided": False,
            "lyrics_input_ignored": bool(lyrics),
            "note_context_path": str(note_path) if note_path is not None else None,
            "note_context_format": resolved_note_context_format,
            "note_context_role": "metadata_only_not_acoustic_input",
            "note_context_applied_to_acoustic_decoder": False,
            "note_context_pending_consumer": "pipeline_note_context_applier",
            "dispatch": {
                "requested_backend_kind": self.requested_backend_name,
                "resolved_backend_kind": self.backend.canonical_name,
                "auto_fallback_used": False,
                "strict_real_backend": self.backend.real_backend,
            },
            "backend_capability": self.backend.capability_metadata(),
            "backend_registry_entry": {
                "canonical_name": self.backend.canonical_name,
                "aliases": list(self.backend.aliases),
                "adapter_kind": self.backend.adapter_kind,
            },
        }
        if self.backend.canonical_name == "onnx-espeak" and segments:
            runtime_metadata = segments[0].get("metadata", {}).get("adapter_diagnostics", {})
            producer["model_metadata"] = {
                key: value
                for key, value in runtime_metadata.items()
                if key in {"model_path", "model_sha256", "vocab_path", "vocab_sha256", "providers"}
            }
            try:
                runtime = self._runtime_describer(prefetch=False)
            except TypeError:
                runtime = self._runtime_describer()
            if isinstance(runtime, Mapping):
                producer["onnx_espeak_runtime"] = dict(runtime)
                producer["dispatch"].update(
                    {
                        "onnx_espeak_ready": runtime.get("ready"),
                        "onnx_espeak_bundle_status": runtime.get("bundle_status"),
                        "onnx_espeak_runtime_status": runtime.get("runtime_status"),
                        "onnx_espeak_failure_category": runtime.get("failure_category"),
                        "onnx_espeak_failure_reason": runtime.get("failure_reason"),
                    }
                )
        payload = {"schema_version": SCHEMA_VERSION, "producer": producer, "segments": segments}
        try:
            encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise PosteriorJSONProducerError(f"Producer payload is not JSON serializable: {exc}") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{id(payload)}")
        try:
            temporary.write_text(encoded + "\n", encoding="utf-8")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return payload


__all__ = [
    "PosteriorJSONProducerError",
    "ProducerBackendSpec",
    "ScoreAwarePosteriorJSONProducer",
    "available_producer_backend_names",
    "build_onnx_espeak_runtime_diagnostics_snapshot",
    "describe_onnx_espeak_runtime",
    "write_onnx_espeak_runtime_diagnostics",
]

"""Isolated reconstruction of the real recognizer adapter boundary.

This module is intentionally self-contained at the adapter boundary.  It consumes
the project's stable ``PhoneSegment`` model and never falls back to a synthetic
recognizer when a real backend is unavailable.

The implementation is a functional reconstruction from surviving source,
bytecode, and fixture evidence.  It is not a byte-for-byte recovery of the
corrupted ``recognizer.py`` module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Protocol, TypeVar

from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.adapters.audio_provenance import AudioProvenanceError, validate_audio_binding
from any2xsampa.adapters.onnx_espeak import OnnxEspeakCTCUniversalPhoneRecognizer
from any2xsampa.adapters.recognizer_cantonese_singing_ctc import CantoneseSingingCTCRecognizer
from any2xsampa.adapters.recognizer_dispatch import render_real_command_args


class RealRecognizerFactory(Protocol):
    def __call__(self, output_path: Path | None) -> UniversalPhoneRecognizer:
        ...


@dataclass(slots=True)
class ProducerBackendSpec:
    """Producer-side catalog entry recovered from valid backend bytecode."""

    canonical_name: str
    aliases: tuple[str, ...]
    factory: Callable[[], UniversalPhoneRecognizer]
    producer_name: str
    adapter_kind: str
    candidate_source: str
    logit_source: str
    raw_model_output_kind: str
    capability: BackendCapabilityProfile


@dataclass(slots=True, frozen=True)
class RealRecognizerRuntimeOptions:
    """Runtime options recovered from ``recognizer_backend_catalog`` bytecode."""

    output_path: Path | None = None
    model_path: Path | None = None
    device: str = "auto"


@dataclass(slots=True)
class BackendCapabilityProfile:
    """Stable capability metadata for a real or diagnostic backend."""

    backend_kind: str
    posterior_like_available: bool
    logit_like_available: bool
    approximation_only: bool
    source: str
    aggregation: str
    backend_role: str = "trained_model_backend"
    intended_use: str = "primary trained recognizer backend"
    diagnostic_baseline: bool = False
    real_backend: bool = True
    real_model_output: bool = True

    def to_metadata(
        self,
        *,
        source: str | None = None,
        aggregation: str | None = None,
        posterior_like_available: bool | None = None,
        logit_like_available: bool | None = None,
        approximation_only: bool | None = None,
    ) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "posterior_like_available": (
                self.posterior_like_available
                if posterior_like_available is None
                else posterior_like_available
            ),
            "logit_like_available": (
                self.logit_like_available
                if logit_like_available is None
                else logit_like_available
            ),
            "approximation_only": (
                self.approximation_only
                if approximation_only is None
                else approximation_only
            ),
            "source": source or self.source,
            "aggregation": aggregation or self.aggregation,
            "backend_role": self.backend_role,
            "intended_use": self.intended_use,
            "diagnostic_baseline": self.diagnostic_baseline,
            "real_backend": self.real_backend,
            "real_model_output": self.real_model_output,
        }


@dataclass(slots=True)
class RealRecognizerBackendSpec:
    """Backend catalog entry matching the recovered ``build`` contract."""

    canonical_name: str
    aliases: tuple[str, ...]
    factory: RealRecognizerFactory | None
    capability: BackendCapabilityProfile
    requires_output_path: bool = False
    runtime_factory: Callable[[RealRecognizerRuntimeOptions], UniversalPhoneRecognizer] | None = None
    requires_model_path: bool = False

    def build(self, options: RealRecognizerRuntimeOptions) -> UniversalPhoneRecognizer:
        if self.capability.diagnostic_baseline or not self.capability.real_backend:
            raise ValueError(
                f"{self.canonical_name} is diagnostic-only and cannot be selected as a real backend."
            )
        if self.requires_output_path and options.output_path is None:
            raise ValueError(
                f"{self.canonical_name} backend requires real_output_path "
                "pointing to the external JSON file."
            )
        if self.requires_model_path and options.model_path is None:
            raise ValueError(
                f"{self.canonical_name} backend requires recognizer_model "
                "pointing to a recognizer bundle."
            )
        if self.requires_model_path and options.model_path is not None:
            model_path = Path(options.model_path).expanduser()
            if not model_path.is_file() and not model_path.is_dir():
                raise FileNotFoundError(
                    f"{self.canonical_name} recognizer_model does not exist: {model_path}"
                )
        if self.runtime_factory is not None:
            return self.runtime_factory(options)
        if self.factory is None:
            raise ValueError(f"{self.canonical_name} backend has no runtime factory.")
        return self.factory(options.output_path)


def build_producer_backend_specs(
    *,
    onnx_factory: Callable[[], UniversalPhoneRecognizer],
    prototype_factory: Callable[[], UniversalPhoneRecognizer],
    heuristic_factory: Callable[[], UniversalPhoneRecognizer],
) -> tuple[ProducerBackendSpec, ...]:
    """Build the producer catalog, including explicit baseline metadata."""

    return (
        ProducerBackendSpec(
            canonical_name="onnx-espeak",
            aliases=("onnx-espeak", "onnx_espeak", "trained-onnx"),
            factory=onnx_factory,
            producer_name="score-aware-onnx-espeak-posterior-json",
            adapter_kind="score-aware-onnx-espeak-producer",
            candidate_source="producer:onnx_mapped_frame_posteriors",
            logit_source="producer:onnx_mapped_frame_logits",
            raw_model_output_kind="trained_onnx_ctc_model",
            capability=BackendCapabilityProfile(
                backend_kind="onnx-espeak",
                posterior_like_available=True,
                logit_like_available=True,
                approximation_only=False,
                source="producer:onnx-espeak",
                aggregation="sum_mapped_posterior_mass",
            ),
        ),
        ProducerBackendSpec(
            canonical_name="prototype-logit",
            aliases=("prototype-logit", "prototype_logit", "prototype"),
            factory=prototype_factory,
            producer_name="score-aware-prototype-logit-posterior-json",
            adapter_kind="score-aware-prototype-logit-producer",
            candidate_source="producer:prototype_frame_posteriors",
            logit_source="producer:prototype_frame_logits",
            raw_model_output_kind="prototype_frame_logit_model",
            capability=BackendCapabilityProfile(
                backend_kind="prototype-logit",
                posterior_like_available=True,
                logit_like_available=True,
                approximation_only=True,
                source="producer:prototype-logit",
                aggregation="mean_frame_posterior",
                backend_role="diagnostic_baseline",
                intended_use="diagnostic/baseline posterior-logit fixture; not a real recognizer backend",
                diagnostic_baseline=True,
                real_backend=False,
                real_model_output=False,
            ),
        ),
        ProducerBackendSpec(
            canonical_name="heuristic-dsp",
            aliases=("heuristic-dsp", "heuristic", "score-aware-heuristic"),
            factory=heuristic_factory,
            producer_name="score-aware-heuristic-posterior-json",
            adapter_kind="score-aware-heuristic-producer",
            candidate_source="producer:frame_posteriors",
            logit_source="producer:frame_logits",
            raw_model_output_kind="heuristic_frame_classifier",
            capability=BackendCapabilityProfile(
                backend_kind="heuristic-dsp",
                posterior_like_available=True,
                logit_like_available=True,
                approximation_only=True,
                source="producer:heuristic-dsp",
                aggregation="mean_frame_confidence",
                backend_role="diagnostic_baseline",
                intended_use="diagnostic/baseline DSP heuristic; not a real recognizer backend",
                diagnostic_baseline=True,
                real_backend=False,
                real_model_output=False,
            ),
        ),
    )


def build_real_backend_specs(
    *,
    producer_specs: Mapping[str, ProducerBackendSpec],
    onnx_factory: RealRecognizerFactory,
    heuristic_factory: RealRecognizerFactory,
    prototype_factory: RealRecognizerFactory,
    posterior_json_factory: RealRecognizerFactory,
    allosaurus_factory: RealRecognizerFactory | None = None,
    cantonese_singing_ctc_factory: Callable[[RealRecognizerRuntimeOptions], UniversalPhoneRecognizer]
    | None = None,
) -> tuple[RealRecognizerBackendSpec, ...]:
    """Build real backend specs while retaining diagnostic baselines in the catalog."""

    specs: list[RealRecognizerBackendSpec] = [
        RealRecognizerBackendSpec(
            canonical_name="onnx-espeak",
            aliases=("onnx-espeak", "onnx_espeak", "trained-onnx"),
            factory=onnx_factory,
            capability=producer_specs["onnx-espeak"].capability,
        ),
        RealRecognizerBackendSpec(
            canonical_name="heuristic-dsp",
            aliases=("heuristic-dsp",),
            factory=heuristic_factory,
            capability=producer_specs["heuristic-dsp"].capability,
        ),
        RealRecognizerBackendSpec(
            canonical_name="prototype-logit",
            aliases=("prototype-logit", "prototype_logit"),
            factory=prototype_factory,
            capability=producer_specs["prototype-logit"].capability,
        ),
        RealRecognizerBackendSpec(
            canonical_name="posterior-json",
            aliases=("posterior-json", "posterior_json"),
            factory=posterior_json_factory,
            requires_output_path=True,
            capability=BackendCapabilityProfile(
                backend_kind="posterior-json",
                posterior_like_available=False,
                logit_like_available=False,
                approximation_only=False,
                source="external-posterior-json",
                aggregation="external_segment_candidates",
                backend_role="external_evidence_adapter",
                intended_use="external posterior/logits JSON adapter",
                diagnostic_baseline=False,
                real_backend=True,
                real_model_output=False,
            ),
        ),
    ]
    if allosaurus_factory is not None:
        specs.insert(
            1,
            RealRecognizerBackendSpec(
                canonical_name="allosaurus",
                aliases=("allosaurus", "allo"),
                factory=allosaurus_factory,
                capability=BackendCapabilityProfile(
                    backend_kind="allosaurus",
                    posterior_like_available=True,
                    logit_like_available=False,
                    approximation_only=False,
                    source="allosaurus:decoded_topk",
                    aggregation="decoder_emission_window",
                    backend_role="trained_model_backend",
                    intended_use="real offline Allosaurus phone recognizer",
                    real_backend=True,
                    real_model_output=True,
                ),
            ),
        )
    if cantonese_singing_ctc_factory is not None:
        specs.insert(
            1,
            RealRecognizerBackendSpec(
                canonical_name="cantonese-singing-ctc",
                aliases=("cantonese-singing-ctc", "cantonese_singing_ctc", "yue-singing-ctc"),
                factory=None,
                runtime_factory=cantonese_singing_ctc_factory,
                requires_model_path=True,
                capability=BackendCapabilityProfile(
                    backend_kind="cantonese-singing-ctc",
                    posterior_like_available=True,
                    logit_like_available=True,
                    approximation_only=False,
                    source="local:cantonese-singing-ctc",
                    aggregation="calibrated_ctc_phone_runs",
                    backend_role="trained_model_backend",
                    intended_use="primary Cantonese singing acoustic phone recognizer",
                    real_backend=True,
                    real_model_output=True,
                ),
            ),
        )
    return tuple(specs)


TSpec = TypeVar("TSpec")


def build_backend_alias_map(specs: tuple[ProducerBackendSpec, ...] | tuple[RealRecognizerBackendSpec, ...]) -> dict[str, Any]:
    return {alias: spec for spec in specs for alias in spec.aliases}


def available_backend_names(
    specs: tuple[ProducerBackendSpec, ...] | tuple[RealRecognizerBackendSpec, ...],
    *,
    include_auto: bool = False,
) -> list[str]:
    names: list[str] = []
    for spec in specs:
        if isinstance(spec, RealRecognizerBackendSpec) and (
            not spec.capability.real_backend and not spec.requires_output_path
        ):
            continue
        names.append(spec.canonical_name)
    return (["auto"] + names) if include_auto else names


def resolve_backend_spec(
    alias_map: Mapping[str, TSpec],
    backend_name: str,
    *,
    kind_label: str,
    available_names: list[str],
    allow_auto: bool = False,
) -> TSpec:
    normalized = backend_name.strip().lower()
    if not allow_auto and normalized in {"", "auto"}:
        raise ValueError("auto must be resolved through ScoreAwarePosteriorJSONProducer dispatch.")
    spec = alias_map.get(normalized)
    if spec is None:
        raise ValueError(
            f"Unknown {kind_label} '{backend_name}'. Available backends: {', '.join(available_names)}"
        )
    return spec


def _number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def _confidence(value: Any, *, label: str) -> float:
    result = _number(value, label=label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _candidate_symbol(item: Mapping[str, Any], *, index: int) -> str:
    symbol = item.get("symbol")
    if symbol is None or not str(symbol).strip():
        raise ValueError(f"candidate {index} is missing symbol.")
    return str(symbol)


def _direct_candidates(value: Any, *, label: str) -> list[PhoneCandidate]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list.")
    candidates: list[PhoneCandidate] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}[{index}] must be an object.")
        symbol = _candidate_symbol(item, index=index)
        confidence = _confidence(item.get("confidence"), label=f"{label}[{index}].confidence")
        candidates.append(PhoneCandidate(symbol=symbol, confidence=confidence))
    candidates.sort(key=lambda candidate: -candidate.confidence)
    return candidates


def _trace_candidates(trace: Any, *, label: str) -> tuple[list[PhoneCandidate], str]:
    if not isinstance(trace, Mapping):
        raise ValueError(f"{label} must be an object.")
    frames = trace.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{label}.frames must be a non-empty list.")
    sums: dict[str, list[float]] = {}
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"{label}.frames[{frame_index}] must be an object.")
        values = frame.get("candidates")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{label}.frames[{frame_index}].candidates must be a non-empty list.")
        for item_index, item in enumerate(values):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"{label}.frames[{frame_index}].candidates[{item_index}] must be an object."
                )
            symbol = _candidate_symbol(item, index=item_index)
            key = "confidence" if "confidence" in item else "posterior"
            if key in item:
                value = _confidence(
                    item[key],
                    label=f"{label}.frames[{frame_index}].candidates[{item_index}].{key}",
                )
            elif "logit" in item:
                value = _number(
                    item["logit"],
                    label=f"{label}.frames[{frame_index}].candidates[{item_index}].logit",
                )
            else:
                raise ValueError(
                    f"{label}.frames[{frame_index}].candidates[{item_index}] "
                    "requires confidence, posterior, or logit."
                )
            sums.setdefault(symbol, []).append(value)

    means = {symbol: sum(values) / len(values) for symbol, values in sums.items()}
    if "logit" in str(label):
        candidates = _softmax_candidates(means)
        return candidates, "softmax_from_frame_logits"
    total = sum(means.values())
    if total <= 0.0:
        raise ValueError(f"{label} has no positive posterior mass.")
    candidates = [
        PhoneCandidate(symbol=symbol, confidence=max(0.0, min(1.0, value / total)))
        for symbol, value in means.items()
    ]
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates, "mean_frame_confidence"


def _softmax_candidates(logits: Mapping[str, float]) -> list[PhoneCandidate]:
    if not logits:
        raise ValueError("logits must be a non-empty object.")
    peak = max(logits.values())
    weights = {symbol: math.exp(value - peak) for symbol, value in logits.items()}
    total = sum(weights.values())
    candidates = [
        PhoneCandidate(symbol=symbol, confidence=weight / total)
        for symbol, weight in weights.items()
    ]
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates


def _logit_candidates(value: Any, *, label: str) -> list[PhoneCandidate]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object.")
    logits: dict[str, float] = {}
    for index, (symbol, raw_value) in enumerate(value.items()):
        if not str(symbol).strip():
            raise ValueError(f"{label}[{index}] has an empty symbol.")
        logits[str(symbol)] = _number(raw_value, label=f"{label}.{symbol}")
    return _softmax_candidates(logits)


def _segment_candidates(segment: Mapping[str, Any]) -> tuple[list[PhoneCandidate], str, bool, bool]:
    posterior_like = "posterior" in segment or "posterior_trace" in segment
    logit_like = "logits" in segment or "logit_trace" in segment
    if "candidate_phones" in segment:
        return (
            _direct_candidates(segment["candidate_phones"], label="candidate_phones"),
            "direct_candidates",
            posterior_like,
            logit_like,
        )
    if "posterior" in segment:
        return (
            _direct_candidates(segment["posterior"], label="posterior"),
            "external_segment_candidates",
            posterior_like,
            logit_like,
        )
    if "logits" in segment:
        return (
            _logit_candidates(segment["logits"], label="logits"),
            "softmax_from_segment_logits",
            posterior_like,
            logit_like,
        )
    if "posterior_trace" in segment:
        candidates, aggregation = _trace_candidates(segment["posterior_trace"], label="posterior_trace")
        return candidates, aggregation, posterior_like, logit_like
    if "logit_trace" in segment:
        candidates, aggregation = _trace_candidates(segment["logit_trace"], label="logit_trace")
        return candidates, aggregation, posterior_like, logit_like
    raise ValueError("Recognizer segment is missing posterior, logits, or candidate_phones.")


def _segment_metadata(
    segment: Mapping[str, Any],
    *,
    producer_metadata: Mapping[str, Any] | None,
    source_mode: str,
    backend_name: str,
    schema_version: str,
    aggregation: str,
    posterior_like: bool,
    logit_like: bool,
    source_label: str,
    lyrics: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_mode": source_mode,
        "source": source_label,
        "schema_version": schema_version,
        "backend_kind": backend_name,
        "real_backend": True,
        "real_model_output": False,
        "adapter_kind": "external_evidence_adapter",
        "candidate_aggregation": aggregation,
        "posterior_like_available": posterior_like,
        "logit_like_available": logit_like,
        "approximation_only": False,
        "lyrics_guided": bool(lyrics),
        "raw_candidates": [],
    }
    if producer_metadata:
        # Preserve producer/runtime lineage at the PhoneSegment boundary so
        # downstream projection does not depend on reopening the raw JSON.
        producer = dict(producer_metadata)
        metadata["producer"] = producer
        metadata["producer_name"] = producer.get("name")
        metadata["producer_backend_kind"] = producer.get("backend_kind")
        producer_is_approximate = bool(producer.get("approximation_only", False))
        metadata["producer_approximation_only"] = producer_is_approximate
        metadata["producer_diagnostic_baseline"] = bool(
            producer.get("diagnostic_baseline", producer_is_approximate)
        )
        metadata["producer_real_backend"] = bool(
            producer.get("real_backend", not producer_is_approximate)
        )
        metadata["producer_real_model_output"] = bool(
            producer.get("real_model_output", not producer_is_approximate)
        )
        metadata["producer_raw_model_output"] = {
            "kind": producer.get("raw_model_output"),
            "model_metadata": producer.get("model_metadata", {}),
            "onnx_espeak_runtime": producer.get("onnx_espeak_runtime", {}),
        }
        # The adapter remains the selected backend, while evidence quality
        # follows the producer capability (diagnostic baselines stay marked
        # approximate instead of being treated as trained model output).
        metadata["approximation_only"] = metadata["producer_approximation_only"]
        metadata["real_backend"] = metadata["producer_real_backend"]
        metadata["real_model_output"] = metadata["producer_real_model_output"]
    for key in (
        "posterior_trace",
        "logit_trace",
        "raw_label_trace",
        "mapped_frame_trace",
        "posterior_summary",
        "logit_summary",
        "segment_summary",
        "continuation_profile",
        "ownership",
        "candidate_distribution",
        "boundary_uncertainty",
        "notes",
    ):
        if key in segment:
            metadata[key] = segment[key]
    diagnostics = segment.get("adapter_diagnostics")
    if diagnostics is not None:
        if not isinstance(diagnostics, Mapping):
            raise ValueError("adapter_diagnostics must be an object.")
        metadata["adapter_diagnostics"] = dict(diagnostics)
    else:
        metadata["adapter_diagnostics"] = {}
    metadata["adapter_diagnostics"].update(
        {
            "backend_kind": backend_name,
            "adapter_kind": "external_evidence_adapter",
            "candidate_aggregation": aggregation,
            "posterior_like_available": posterior_like,
            "logit_like_available": logit_like,
            "approximation_only": metadata["approximation_only"],
            "real_backend": metadata["real_backend"],
            "real_model_output": metadata["real_model_output"],
        }
    )
    return metadata


def _build_segments(
    payload: Mapping[str, Any],
    *,
    source_mode: str,
    backend_name: str,
    source_label: str,
    lyrics: str | None,
) -> list[PhoneSegment]:
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "external_posterior_json_v1":
        raise ValueError(
            "Recognizer payload schema_version must be 'external_posterior_json_v1'."
        )
    segment_payloads = payload.get("segments")
    if not isinstance(segment_payloads, list) or not segment_payloads:
        raise ValueError("Recognizer payload must contain a non-empty 'segments' list.")
    built: list[PhoneSegment] = []
    for index, raw_segment in enumerate(segment_payloads):
        if not isinstance(raw_segment, Mapping):
            raise ValueError(f"Recognizer segment {index} must be an object.")
        start = _number(raw_segment.get("start_sec"), label=f"segments[{index}].start_sec", minimum=0.0)
        end = _number(raw_segment.get("end_sec"), label=f"segments[{index}].end_sec", minimum=0.0)
        if end < start:
            raise ValueError(f"Recognizer segment {index} end_sec must not precede start_sec.")
        candidates, aggregation, posterior_like, logit_like = _segment_candidates(raw_segment)
        metadata = _segment_metadata(
            raw_segment,
            producer_metadata=(
                payload.get("producer")
                if isinstance(payload.get("producer"), Mapping)
                else None
            ),
            source_mode=source_mode,
            backend_name=backend_name,
            schema_version=schema_version,
            aggregation=aggregation,
            posterior_like=posterior_like,
            logit_like=logit_like,
            source_label=source_label,
            lyrics=lyrics,
        )
        # Keep the isolated real adapter on the same evidence contract as
        # fixture replay: posterior/logit traces must reach aligner v2 as
        # acoustic evidence without changing the phone truth layer.
        from any2xsampa.adapters.recognizer import _annotate_acoustic_evidence

        metadata = _annotate_acoustic_evidence(
            metadata,
            candidates,
            source_mode,
            start,
            end,
        )
        metadata["raw_candidates"] = [candidate.symbol for candidate in candidates]
        confidence = candidates[0].confidence
        if raw_segment.get("confidence") is not None:
            confidence = _confidence(raw_segment["confidence"], label=f"segments[{index}].confidence")
        built.append(
            PhoneSegment(
                time_range=TimeRange(start_sec=start, end_sec=end),
                candidate_phones=candidates,
                confidence=confidence,
                metadata=metadata,
            )
        )
    return built


class ExternalPosteriorJSONUniversalPhoneRecognizer(UniversalPhoneRecognizer):
    """Replay an external posterior/logit JSON artifact as real evidence."""

    def __init__(
        self,
        output_path: Path,
        *,
        backend_name: str = "posterior-json",
        source_label: str = "external posterior JSON",
    ) -> None:
        self.output_path = Path(output_path)
        self.backend_name = backend_name
        self.source_label = source_label

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        audio_path = Path(vocal_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Vocal audio file not found: {audio_path}")
        if not self.output_path.exists():
            raise FileNotFoundError(f"Real recognizer output not found: {self.output_path}")
        try:
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.source_label} replay schema error in {self.output_path}: invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.source_label} must contain a JSON object payload.")
        try:
            binding_status = validate_audio_binding(
                (payload.get("producer") or {}).get("input_audio")
                if isinstance(payload.get("producer"), Mapping)
                else None,
                audio_path,
            )
        except AudioProvenanceError as exc:
            raise ValueError(f"{self.source_label} audio provenance error: {exc}") from exc
        try:
            segments = _build_segments(
                payload,
                source_mode="real",
                backend_name=self.backend_name,
                source_label=self.source_label,
                lyrics=lyrics,
            )
        except ValueError as exc:
            raise ValueError(
                f"{self.source_label} replay schema error in {self.output_path}: {exc}"
            ) from exc
        for segment in segments:
            segment.metadata["audio_binding"] = binding_status
        return segments


def _validate_command_template(command_template: str) -> None:
    formatter = __import__("string").Formatter()
    allowed = {"audio", "output"}
    unknown: list[str] = []
    for _, field_name, _, _ in formatter.parse(command_template):
        if field_name and field_name.strip() not in allowed and field_name.strip() not in unknown:
            unknown.append(field_name.strip())
    if unknown:
        names = ", ".join(f"{{{name}}}" for name in sorted(allowed))
        fields = ", ".join(f"{{{name}}}" for name in unknown)
        raise ValueError(
            "Real recognizer command template contains unknown placeholder(s): "
            f"{fields}. Allowed placeholders: {names}. "
            "If you need literal braces in the command, escape them as '{{' and '}}'."
        )


class RealCommandUniversalPhoneRecognizer(UniversalPhoneRecognizer):
    """Run an explicitly configured producer and consume its JSON artifact."""

    def __init__(self, command_template: str, output_path: Path) -> None:
        self.command_template = command_template
        self.output_path = Path(output_path)

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        audio_path = Path(vocal_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Vocal audio file not found: {audio_path}")
        command = render_real_command_args(
            self.command_template,
            audio_path=audio_path,
            output_path=self.output_path,
        )
        if self.output_path.exists():
            try:
                self.output_path.unlink()
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot remove stale real recognizer output: {self.output_path}"
                ) from exc
        try:
            completed = subprocess.run(
                command,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=120.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Real recognizer command timed out after 120 seconds") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"Real recognizer command failed with exit code {completed.returncode}{suffix}"
            )
        if not self.output_path.exists():
            raise RuntimeError(f"Real recognizer command did not produce output: {self.output_path}")
        return ExternalPosteriorJSONUniversalPhoneRecognizer(
            self.output_path,
            backend_name="command",
            source_label="real recognizer command output",
        ).recognize(audio_path, lyrics=lyrics)


class RealBackendUniversalPhoneRecognizer(UniversalPhoneRecognizer):
    """Select a catalogued real backend without a synthetic fallback."""

    def __init__(
        self,
        backend_name: str,
        output_path: Path | None = None,
        *,
        model_path: Path | None = None,
        device: str = "auto",
        backend_specs: Mapping[str, RealRecognizerBackendSpec] | None = None,
    ) -> None:
        self.backend_name = backend_name.strip().lower()
        self.output_path = Path(output_path) if output_path is not None else None
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device.strip().lower() or "auto"
        self.backend_specs = dict(backend_specs or {})

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        backend = self._build_backend()
        return backend.recognize(vocal_audio_path, lyrics=lyrics)

    def _build_backend(self) -> UniversalPhoneRecognizer:
        spec = self.backend_specs.get(self.backend_name)
        if spec is None:
            if self.backend_name in {"onnx-espeak", "onnx_espeak", "trained-onnx"}:
                spec = RealRecognizerBackendSpec(
                    canonical_name="onnx-espeak",
                    aliases=("onnx-espeak", "onnx_espeak", "trained-onnx"),
                    factory=None,
                    runtime_factory=lambda options: OnnxEspeakCTCUniversalPhoneRecognizer(
                        model_path=options.model_path,
                        device=options.device,
                    ),
                    capability=BackendCapabilityProfile(
                        backend_kind="onnx-espeak",
                        posterior_like_available=True,
                        logit_like_available=True,
                        approximation_only=False,
                        source="onnxruntime:wav2vec2-espeak-ctc",
                        aggregation="mean_mapped_frame_posterior",
                        backend_role="trained_model_backend",
                        intended_use="real local ONNX/eSpeak acoustic phone recognizer",
                        real_backend=True,
                        real_model_output=True,
                    ),
                )
            elif self.backend_name in {"posterior-json", "posterior_json"}:
                spec = RealRecognizerBackendSpec(
                    canonical_name="posterior-json",
                    aliases=("posterior-json", "posterior_json"),
                    factory=lambda output: ExternalPosteriorJSONUniversalPhoneRecognizer(
                        output or Path(""), backend_name="posterior-json"
                    ),
                    requires_output_path=True,
                    capability=BackendCapabilityProfile(
                        backend_kind="posterior-json",
                        posterior_like_available=False,
                        logit_like_available=False,
                        approximation_only=False,
                        source="external-posterior-json",
                        aggregation="external_segment_candidates",
                        backend_role="external_evidence_adapter",
                        intended_use="external posterior/logits JSON adapter",
                        real_model_output=False,
                    ),
                )
            elif self.backend_name in {
                "cantonese-singing-ctc",
                "cantonese_singing_ctc",
                "yue-singing-ctc",
            }:
                spec = RealRecognizerBackendSpec(
                    canonical_name="cantonese-singing-ctc",
                    aliases=(
                        "cantonese-singing-ctc",
                        "cantonese_singing_ctc",
                        "yue-singing-ctc",
                    ),
                    factory=None,
                    runtime_factory=lambda options: CantoneseSingingCTCRecognizer(
                        options.model_path or Path(""),
                        device=options.device,
                    ),
                    requires_model_path=True,
                    capability=BackendCapabilityProfile(
                        backend_kind="cantonese-singing-ctc",
                        posterior_like_available=True,
                        logit_like_available=True,
                        approximation_only=False,
                        source="local:cantonese-singing-ctc",
                        aggregation="calibrated_ctc_phone_runs",
                        backend_role="trained_model_backend",
                        intended_use="primary Cantonese singing acoustic phone recognizer",
                        real_backend=True,
                        real_model_output=True,
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown real recognizer backend '{self.backend_name}'. "
                    "Provide a catalog spec for a trained backend."
                )
        return spec.build(
            RealRecognizerRuntimeOptions(
                output_path=self.output_path,
                model_path=self.model_path,
                device=self.device,
            )
        )


__all__ = [
    "BackendCapabilityProfile",
    "ExternalPosteriorJSONUniversalPhoneRecognizer",
    "RealBackendUniversalPhoneRecognizer",
    "RealCommandUniversalPhoneRecognizer",
    "RealRecognizerBackendSpec",
    "RealRecognizerRuntimeOptions",
]

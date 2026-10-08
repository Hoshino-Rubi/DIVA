from __future__ import annotations

"""Replaceable boundary for the original producer/runtime route.

The recognizer's stable truth layer is still ``PhoneSegment``.  This module
only gathers the external producer and ONNX/eSpeak runtime surfaces used by
the command-line workers.  Keeping the imports here makes the worker boundary
replaceable without making recognizer parsing depend on a particular runtime
implementation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from any2xsampa.adapters.onnx_espeak_runtime import (
    BUNDLE_SCHEMA_VERSION,
    OnnxEspeakRuntimeCandidate,
    build_onnx_espeak_runtime_diagnostics_snapshot,
    classify_onnx_espeak_runtime_failure,
    describe_onnx_espeak_runtime,
    inspect_onnx_espeak_runtime_bundle,
    normalize_dependency_probe,
    normalize_python_probe,
    onnx_espeak_bundle_status,
    prepare_onnx_espeak_runtime_bundle,
    probe_external_python_runtime,
    probe_external_python_runtime_with_env,
    probe_onnx_espeak_runtime_candidate,
    resolve_bundle_relative_path,
    resolve_onnx_espeak_runtime_status,
    safe_relative_path,
    utc_timestamp,
    write_onnx_espeak_bundle_state,
    write_onnx_espeak_runtime_diagnostics,
)
from any2xsampa.adapters.posterior_json_producer import (
    SCHEMA_VERSION,
    PosteriorJSONProducerError,
    ProducerBackendSpec,
    ScoreAwarePosteriorJSONProducer,
    available_producer_backend_names,
)


class PosteriorProducer(Protocol):
    """Minimal producer contract consumed by the runtime boundary."""

    def produce(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        ...


class ProducerFactory(Protocol):
    """Callable shape used to inject a producer in a worker or test."""

    def __call__(self, *args: Any, **kwargs: Any) -> PosteriorProducer:
        ...


@dataclass(frozen=True, slots=True)
class RecognizerProducerRuntimeAdapter:
    """Small dependency-injection seam for producer and runtime diagnostics.

    The defaults point at the current repo-local implementations.  A bundled
    worker or an integration test can provide another producer factory or
    diagnostics callable without changing recognizer JSON parsing.
    """

    producer_factory: ProducerFactory = ScoreAwarePosteriorJSONProducer
    runtime_describer: Callable[..., Mapping[str, Any]] | None = None

    def create_producer(self, *args: Any, **kwargs: Any) -> PosteriorProducer:
        return self.producer_factory(*args, **kwargs)

    def describe_runtime(self, *, prefetch: bool = False) -> dict[str, Any]:
        describer = self.runtime_describer or describe_onnx_espeak_runtime
        try:
            diagnostics = describer(prefetch=prefetch)
        except TypeError:
            # Keep compatibility with simple injected callables that predate
            # the optional prefetch keyword.
            diagnostics = describer()
        if not isinstance(diagnostics, Mapping):
            raise TypeError("runtime_describer must return a mapping")
        return dict(diagnostics)


DEFAULT_RECOGNIZER_PRODUCER_RUNTIME = RecognizerProducerRuntimeAdapter()


def create_posterior_json_producer(
    backend_name: str = "auto",
    *,
    model_path: Path | str | None = None,
    device: str = "auto",
    cpu_thread_limit: int | None = None,
    chunk_duration_sec: float | None = None,
    chunk_overlap_sec: float = 0.5,
    runtime_describer: Callable[..., Mapping[str, Any]] | None = None,
) -> ScoreAwarePosteriorJSONProducer:
    """Construct the default producer through the replaceable runtime seam."""

    adapter = DEFAULT_RECOGNIZER_PRODUCER_RUNTIME
    if runtime_describer is not None:
        adapter = RecognizerProducerRuntimeAdapter(
            producer_factory=adapter.producer_factory,
            runtime_describer=runtime_describer,
        )
    return adapter.create_producer(
        backend_name=backend_name,
        model_path=model_path,
        device=device,
        cpu_thread_limit=cpu_thread_limit,
        chunk_duration_sec=chunk_duration_sec,
        chunk_overlap_sec=chunk_overlap_sec,
        runtime_describer=runtime_describer,
    )


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_RECOGNIZER_PRODUCER_RUNTIME",
    "OnnxEspeakRuntimeCandidate",
    "PosteriorProducer",
    "PosteriorJSONProducerError",
    "ProducerBackendSpec",
    "ProducerFactory",
    "RecognizerProducerRuntimeAdapter",
    "SCHEMA_VERSION",
    "ScoreAwarePosteriorJSONProducer",
    "available_producer_backend_names",
    "build_onnx_espeak_runtime_diagnostics_snapshot",
    "classify_onnx_espeak_runtime_failure",
    "create_posterior_json_producer",
    "describe_onnx_espeak_runtime",
    "inspect_onnx_espeak_runtime_bundle",
    "normalize_dependency_probe",
    "normalize_python_probe",
    "onnx_espeak_bundle_status",
    "prepare_onnx_espeak_runtime_bundle",
    "probe_external_python_runtime",
    "probe_external_python_runtime_with_env",
    "probe_onnx_espeak_runtime_candidate",
    "resolve_bundle_relative_path",
    "resolve_onnx_espeak_runtime_status",
    "safe_relative_path",
    "utc_timestamp",
    "write_onnx_espeak_bundle_state",
    "write_onnx_espeak_runtime_diagnostics",
]

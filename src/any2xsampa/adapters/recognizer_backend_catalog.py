from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from any2xsampa.core.interfaces import UniversalPhoneRecognizer


@dataclass(frozen=True, slots=True)
class BackendCapabilityProfile:
    backend_kind: str
    posterior_like_available: bool
    logit_like_available: bool
    approximation_only: bool
    source: str
    aggregation: str

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
                self.approximation_only if approximation_only is None else approximation_only
            ),
            "source": source or self.source,
            "aggregation": aggregation or self.aggregation,
        }


@dataclass(frozen=True, slots=True)
class ProducerBackendSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    factory: Callable[[], UniversalPhoneRecognizer]
    producer_name: str
    adapter_kind: str
    candidate_source: str
    logit_source: str
    raw_model_output_kind: str
    capability: BackendCapabilityProfile


@dataclass(frozen=True, slots=True)
class RealRecognizerBackendSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    factory: Callable[[Path | None], UniversalPhoneRecognizer]
    capability: BackendCapabilityProfile
    requires_output_path: bool = False


def build_producer_backend_specs(
    *,
    onnx_factory: Callable[[], UniversalPhoneRecognizer],
    prototype_factory: Callable[[], UniversalPhoneRecognizer],
    heuristic_factory: Callable[[], UniversalPhoneRecognizer],
) -> tuple[ProducerBackendSpec, ...]:
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
            ),
        ),
    )


def build_real_backend_specs(
    *,
    producer_specs: dict[str, ProducerBackendSpec],
    heuristic_factory: Callable[[Path | None], UniversalPhoneRecognizer],
    prototype_factory: Callable[[Path | None], UniversalPhoneRecognizer],
    posterior_json_factory: Callable[[Path | None], UniversalPhoneRecognizer],
) -> tuple[RealRecognizerBackendSpec, ...]:
    return (
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
            capability=BackendCapabilityProfile(
                backend_kind="posterior-json",
                posterior_like_available=False,
                logit_like_available=False,
                approximation_only=False,
                source="external-posterior-json",
                aggregation="external_segment_candidates",
            ),
            requires_output_path=True,
        ),
    )


TSpec = TypeVar("TSpec", ProducerBackendSpec, RealRecognizerBackendSpec)


def build_backend_alias_map(specs: tuple[TSpec, ...]) -> dict[str, TSpec]:
    return {
        alias: spec
        for spec in specs
        for alias in spec.aliases
    }


def available_backend_names(
    specs: tuple[ProducerBackendSpec, ...] | tuple[RealRecognizerBackendSpec, ...],
    *,
    include_auto: bool = False,
) -> list[str]:
    names = [spec.canonical_name for spec in specs]
    return (["auto"] + names) if include_auto else names


def resolve_backend_spec(
    alias_map: dict[str, TSpec],
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
            f"Unknown {kind_label} '{backend_name}'. "
            f"Available backends: {', '.join(available_names)}"
        )
    return spec

from __future__ import annotations

"""Fixture-first universal phone recognizer adapter.

The original recognizer module is truncated in the source tree.  This module
keeps the public recognizer/configuration contract while making fixture replay
the deterministic, runnable path.  External posterior/logit payloads are
normalized before they reach the stable ``PhoneSegment`` model.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable
import warnings

from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange

from any2xsampa.adapters.recognizer_acoustic_evidence import (
    apply_acoustic_evidence_metadata,
    build_segment_evidence_views,
    build_segment_summary_payload,
    candidate_entropy as _candidate_entropy_helper,
    candidate_logit_map as _candidate_logit_map_helper,
    normalize_candidate_evidence as _normalize_candidate_evidence_helper,
)
from any2xsampa.adapters.recognizer_acoustic_evidence_builder import (
    build_acoustic_evidence_from_metadata as _build_acoustic_evidence_from_metadata_helper,
)
from any2xsampa.adapters.recognizer_dispatch import (
    build_recognizer_dispatch_plan,
    render_real_command_args,
)
from any2xsampa.adapters.recognizer_external_payload import _normalize_json_payload
from any2xsampa.adapters.onnx_espeak import OnnxEspeakCTCUniversalPhoneRecognizer
from any2xsampa.adapters.recognizer_segment_builder import build_segments_from_payload
from any2xsampa.adapters.recognizer_segment_evidence import (
    build_adapter_diagnostics as _build_adapter_diagnostics_helper,
    derive_logit_frames as _derive_logit_frames_helper,
    derive_posterior_frames as _derive_posterior_frames_helper,
    frame_trace as _frame_trace_helper,
    infer_candidates_from_metadata as _infer_candidates_from_metadata_helper,
    logit_summary as _logit_summary_helper,
    posterior_summary as _posterior_summary_helper,
)
from any2xsampa.adapters.recognizer_support import (
    _candidate_payload_from_scores,
    _default_articulatory_features,
    _normalize_symbol_scores_frame,
    _probability_to_logit,
    _softmax_symbol_scores,
    _sorted_symbol_scores,
)

# Runtime diagnostics do not depend on recognizer construction and are safe to
# expose while this fixture-first module is still initializing.
from any2xsampa.adapters.onnx_espeak_runtime import (
    build_onnx_espeak_runtime_diagnostics_snapshot,
    describe_onnx_espeak_runtime,
    write_onnx_espeak_runtime_diagnostics,
)


@dataclass(slots=True)
class RecognizerConfig:
    """Configuration shared by fixture and real recognizer adapters."""

    mode: str = "fixture"
    fixture_path: Path | None = None
    real_backend: str | None = None
    real_command: str | None = None
    real_output_path: Path | None = None
    model_path: Path | None = None
    device: str = "auto"
    allow_stub_fallback: bool = False


def _frame_trace(
    metadata: dict[str, Any],
    trace_key: str,
    score_key: str,
) -> tuple[list[dict[str, float]], str | None, float | None]:
    return _frame_trace_helper(metadata, trace_key, score_key)


def _derive_posterior_frames(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, float]], str, float | None]:
    return _derive_posterior_frames_helper(candidates, metadata)


def _derive_logit_frames(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, float]], str, float | None]:
    return _derive_logit_frames_helper(candidates, metadata)


def _posterior_summary(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    top_phone: str,
) -> dict[str, Any]:
    return _posterior_summary_helper(candidates, metadata, top_phone)


def _logit_summary(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    top_phone: str,
) -> dict[str, Any]:
    return _logit_summary_helper(candidates, metadata, top_phone)


def _infer_candidates_from_metadata(metadata: dict[str, Any]) -> list[PhoneCandidate]:
    return _infer_candidates_from_metadata_helper(metadata)


def _candidate_logit_map(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
) -> dict[str, float]:
    return _candidate_logit_map_helper(
        candidates,
        metadata,
        derive_logit_frames=_derive_logit_frames,
        probability_to_logit=_probability_to_logit,
    )


def _normalize_candidate_evidence(
    candidates: list[PhoneCandidate],
    candidate_logits: dict[str, float] | None = None,
) -> list[Any]:
    return _normalize_candidate_evidence_helper(
        candidates,
        candidate_logits=candidate_logits,
        default_articulatory_features=_default_articulatory_features,
        probability_to_logit=_probability_to_logit,
    )


def _candidate_entropy(candidates: list[PhoneCandidate]) -> float:
    return _candidate_entropy_helper(candidates)


def _build_acoustic_evidence_entry(
    *,
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    source_mode: str,
    duration_sec: float,
) -> Any:
    return _build_acoustic_evidence_from_metadata_helper(
        candidates=candidates,
        metadata=metadata,
        source_mode=source_mode,
        duration_sec=duration_sec,
    )


def _build_acoustic_evidence(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    source_mode: str,
    duration_sec: float,
) -> Any:
    return _build_acoustic_evidence_entry(
        candidates=candidates,
        metadata=metadata,
        source_mode=source_mode,
        duration_sec=duration_sec,
    )


# Keep the historical monkeypatch seam callable without routing normal fixture
# recognition through it.  A few downstream integrations still inspect this
# state while migrating to ``_build_acoustic_evidence_entry``.
_LEGACY_ACOUSTIC_EVIDENCE_COMPAT_NAME = "_build_acoustic_evidence_impl"
_LEGACY_ACOUSTIC_EVIDENCE_COMPAT_STATUS = "deprecated_monkeypatch_seam"
_LEGACY_ACOUSTIC_EVIDENCE_COMPAT_ALIAS: Callable[..., Any] | None = None


def _build_acoustic_evidence_impl(
    *,
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    source_mode: str,
    duration_sec: float,
) -> Any:
    warnings.warn(
        "_build_acoustic_evidence_impl() is deprecated; use _build_acoustic_evidence_entry().",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_acoustic_evidence_entry(
        candidates=candidates,
        metadata=metadata,
        source_mode=source_mode,
        duration_sec=duration_sec,
    )


_LEGACY_ACOUSTIC_EVIDENCE_COMPAT_ALIAS = _build_acoustic_evidence_impl


def _legacy_acoustic_evidence_builder_seam_state() -> dict[str, object]:
    legacy_builder = globals().get(_LEGACY_ACOUSTIC_EVIDENCE_COMPAT_NAME)
    override_active = callable(legacy_builder) and legacy_builder is not _LEGACY_ACOUSTIC_EVIDENCE_COMPAT_ALIAS
    return {
        "compat_symbol": _LEGACY_ACOUSTIC_EVIDENCE_COMPAT_NAME,
        "status": _LEGACY_ACOUSTIC_EVIDENCE_COMPAT_STATUS,
        "preferred_entrypoint": "_build_acoustic_evidence_entry",
        "override_active": override_active,
        "active_builder_source": "legacy_override" if override_active else "entry",
    }


def _active_acoustic_evidence_builder() -> Callable[..., Any]:
    legacy_builder = globals().get(_LEGACY_ACOUSTIC_EVIDENCE_COMPAT_NAME)
    if callable(legacy_builder) and legacy_builder is not _LEGACY_ACOUSTIC_EVIDENCE_COMPAT_ALIAS:
        return legacy_builder
    return _build_acoustic_evidence_entry


def segment_summary_payload(metadata: dict[str, object]) -> dict[str, object]:
    return build_segment_summary_payload(metadata)


def segment_evidence_views(metadata: dict[str, object]) -> dict[str, object]:
    return build_segment_evidence_views(metadata)


def _annotate_acoustic_evidence(
    metadata: dict[str, Any],
    candidates: list[PhoneCandidate],
    source_mode: str,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    evidence = _build_acoustic_evidence_entry(
        candidates=candidates,
        metadata=metadata,
        source_mode=source_mode,
        duration_sec=max(0.0, end_sec - start_sec),
    )
    return apply_acoustic_evidence_metadata(
        metadata,
        evidence=evidence,
        candidates=candidates,
    )


def _build_segments(payload: dict[str, Any], source_mode: str) -> list[PhoneSegment]:
    return build_segments_from_payload(
        payload,
        source_mode=source_mode,
        infer_candidates_from_metadata=_infer_candidates_from_metadata,
        default_articulatory_features=_default_articulatory_features,
        annotate_acoustic_evidence=_annotate_acoustic_evidence,
    )


def _coerce_candidate_list(value: object) -> list[dict[str, Any]]:
    """Convert phones/candidate shorthand into the canonical candidate list."""
    if isinstance(value, dict):
        if value.get("symbol", value.get("phone", value.get("label"))) is not None:
            symbol = str(value.get("symbol", value.get("phone", value.get("label", "")))).strip()
            if not symbol:
                return []
            score = value.get("confidence", value.get("score", value.get("probability", 1.0)))
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                confidence = 1.0
            return [{"symbol": symbol, "confidence": confidence}]
        normalized: list[dict[str, Any]] = []
        for symbol, score in value.items():
            if isinstance(score, bool):
                continue
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                continue
            normalized.append({"symbol": str(symbol), "confidence": confidence})
        return normalized
    if not isinstance(value, list):
        if value is None:
            return []
        return [{"symbol": str(value), "confidence": 1.0}]

    normalized = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            candidate = _coerce_candidate_list(item)
            normalized.extend(candidate)
            continue
        if isinstance(item, (str, int, float)) and not isinstance(item, bool):
            normalized.append({"symbol": str(item), "confidence": 1.0 if index == 0 else 0.0})
    return normalized


def _prepare_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Recognizer fixture JSON must contain an object payload.")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Recognizer payload must contain a non-empty 'segments' list.")

    prepared_segments: list[dict[str, Any]] = []
    trace_keys = (
        "posterior",
        "posteriors",
        "posterior_trace",
        "posterior_frames",
        "logits",
        "logit_scores",
        "logit_trace",
        "logit_frames",
    )
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"Recognizer segment {index} must be an object.")
        segment = dict(raw_segment)
        metadata_payload = segment.get("metadata")
        metadata = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}

        if not segment.get("candidate_phones"):
            shorthand = segment.get("phones", segment.get("phone", segment.get("symbol")))
            candidates = _coerce_candidate_list(shorthand)
            if candidates:
                segment["candidate_phones"] = candidates
        elif isinstance(segment.get("candidate_phones"), (dict, list)):
            segment["candidate_phones"] = _coerce_candidate_list(segment["candidate_phones"])

        for key in trace_keys:
            if key in segment and key not in metadata:
                metadata[key] = segment[key]
        segment["metadata"] = metadata
        prepared_segments.append(segment)

    prepared = dict(payload)
    prepared["segments"] = prepared_segments
    return prepared


def _adapter_diagnostics(**kwargs: Any) -> dict[str, Any]:
    return _build_adapter_diagnostics_helper(**kwargs)


def _build_adapter_diagnostics(**kwargs: Any) -> dict[str, Any]:
    """Preserve the historical helper seam used by downstream integrations."""
    return _adapter_diagnostics(**kwargs)


def _normalize_payload(payload: dict[str, Any], *, source_mode: str, backend_name: str, source_label: str) -> dict[str, Any]:
    return _normalize_json_payload(
        payload,
        source_mode=source_mode,
        backend_name=backend_name,
        source_label=source_label,
        build_adapter_diagnostics=_adapter_diagnostics,
    )


def _replay_info(payload: dict[str, Any]) -> dict[str, Any] | None:
    if str(payload.get("schema_version", "")).strip() != "open_corpus_replay_v1":
        return None
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Open-corpus replay requires a provenance object.")
    replay_kind = str(provenance.get("replay_kind", "")).strip()
    if not replay_kind:
        raise ValueError("Open-corpus replay provenance requires replay_kind.")
    replay_type = str(payload.get("replay_type", "")).strip()
    if replay_type not in {"phones_timing", "posterior_logits"}:
        raise ValueError("Open-corpus replay replay_type must be phones_timing or posterior_logits.")
    return {
        "schema_version": "open_corpus_replay_v1",
        "replay_kind": replay_kind,
        "replay_type": replay_type,
        "model_output": bool(payload.get("model_output", provenance.get("model_output", False))),
    }


class FixtureUniversalPhoneRecognizer(UniversalPhoneRecognizer):
    """Read fixture/replay JSON and return stable ``PhoneSegment`` objects."""

    def __init__(self, fixture_path: Path | str) -> None:
        self.fixture_path = Path(fixture_path)

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        audio_path = Path(vocal_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Vocal audio file not found: {audio_path}")
        if not self.fixture_path.exists():
            raise FileNotFoundError(f"Recognizer fixture file not found: {self.fixture_path}")
        try:
            payload = json.loads(self.fixture_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Recognizer fixture JSON is invalid: {self.fixture_path}") from exc

        payload = _prepare_payload(payload)
        replay = _replay_info(payload)
        source_mode = "fixture"
        backend_name = "fixture-json"
        normalized = _normalize_payload(
            payload,
            source_mode=source_mode,
            backend_name=backend_name,
            source_label=str(self.fixture_path),
        )
        segments = _build_segments(normalized, source_mode)
        for segment in segments:
            segment.metadata["source"] = str(self.fixture_path)
            segment.metadata["lyrics_guided"] = bool(lyrics)
            if replay is not None:
                segment.metadata["fixture_schema_status"] = "provenance_backed_replay"
                segment.metadata["replay_schema_version"] = replay["schema_version"]
                segment.metadata["replay_kind"] = replay["replay_kind"]
                segment.metadata["replay_type"] = replay["replay_type"]
                segment.metadata.setdefault("model_output", replay["model_output"])
        return segments


class ExternalPosteriorJSONUniversalPhoneRecognizer(FixtureUniversalPhoneRecognizer):
    """Consume an external posterior/logit JSON output using fixture parsing."""

    def __init__(self, output_path: Path | str, backend_name: str = "external-posterior-json") -> None:
        super().__init__(output_path)
        self.backend_name = backend_name

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        audio_path = Path(vocal_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Vocal audio file not found: {audio_path}")
        if not self.fixture_path.exists():
            raise FileNotFoundError(f"Recognizer output file not found: {self.fixture_path}")
        payload = _prepare_payload(json.loads(self.fixture_path.read_text(encoding="utf-8-sig")))
        replay = _replay_info(payload)
        normalized = _normalize_payload(
            payload,
            source_mode="real",
            backend_name=self.backend_name,
            source_label=str(self.fixture_path),
        )
        segments = _build_segments(normalized, "real")
        for segment in segments:
            segment.metadata["source"] = str(self.fixture_path)
            segment.metadata["lyrics_guided"] = bool(lyrics)
            if replay is not None:
                segment.metadata["fixture_schema_status"] = "provenance_backed_replay"
                segment.metadata["replay_kind"] = replay["replay_kind"]
                segment.metadata["replay_type"] = replay["replay_type"]
                segment.metadata.setdefault("model_output", replay["model_output"])
        return segments


class RealCommandUniversalPhoneRecognizer(UniversalPhoneRecognizer):
    """Run a configured real command, then parse its JSON output."""

    def __init__(self, command_template: str, output_path: Path | str) -> None:
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
                capture_output=True,
                text=True,
                check=False,
                timeout=120.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Real recognizer command timed out after 120 seconds") from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            suffix = f": {details}" if details else ""
            raise RuntimeError(f"Real recognizer command failed with exit code {completed.returncode}{suffix}")
        if not self.output_path.exists():
            raise RuntimeError(f"Real recognizer command did not produce output: {self.output_path}")
        return ExternalPosteriorJSONUniversalPhoneRecognizer(self.output_path, "real-command-json").recognize(
            audio_path,
            lyrics=lyrics,
        )


class RealBackendUniversalPhoneRecognizer(UniversalPhoneRecognizer):
    """Isolated real backend adapter; backend output is always explicit JSON."""

    def __init__(
        self,
        backend_name: str,
        output_path: Path | str | None = None,
        *,
        model_path: Path | str | None = None,
        device: str = "auto",
    ) -> None:
        self.backend_name = str(backend_name)
        self.output_path = Path(output_path) if output_path is not None else None
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        if self.output_path is None:
            raise RuntimeError(
                f"Real recognizer backend '{self.backend_name}' is unavailable without real_output_path."
            )
        if not self.output_path.exists():
            raise RuntimeError(
                f"Real recognizer backend '{self.backend_name}' did not produce output: {self.output_path}"
            )
        return ExternalPosteriorJSONUniversalPhoneRecognizer(
            self.output_path,
            backend_name=self.backend_name,
        ).recognize(vocal_audio_path, lyrics=lyrics)


class ProducerPosteriorJSONUniversalPhoneRecognizer(RealBackendUniversalPhoneRecognizer):
    """Compatibility name for a producer-backed posterior JSON adapter."""

    def __init__(self, *, backend_name: str, output_path: Path | str | None = None) -> None:
        super().__init__(backend_name, output_path)


def _stub_segments(
    *,
    requested_mode: str,
    primary_kind: str,
    error: Exception,
) -> list[PhoneSegment]:
    segment = PhoneSegment(
        time_range=TimeRange(start_sec=0.0, end_sec=0.001),
        candidate_phones=[PhoneCandidate(symbol="sil", confidence=0.0)],
        confidence=0.0,
        metadata={
            "source_mode": "stub",
            "raw_candidates": ["sil"],
            "fallback_from_mode": requested_mode,
            "recognizer_fallback": {
                "requested_mode": requested_mode,
                "primary_kind": primary_kind,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "fallback_status": "explicit_debug_stub",
            },
            "normalization_notes": ["recognizer fallback to stub mode"],
        },
    )
    return [segment]


class ConfigurableUniversalPhoneRecognizer(UniversalPhoneRecognizer):
    """Route fixture, real backend, or real command mode from ``RecognizerConfig``."""

    def __init__(self, config: RecognizerConfig) -> None:
        self.config = config

    def _build_primary(self) -> UniversalPhoneRecognizer:
        plan = build_recognizer_dispatch_plan(self.config)
        return self._build_from_plan(plan)

    @staticmethod
    def _build_from_plan(plan: Any) -> UniversalPhoneRecognizer:
        if plan.primary_kind == "fixture":
            if plan.fixture_path is None:
                raise ValueError("Fixture mode requires a fixture_path.")
            return FixtureUniversalPhoneRecognizer(plan.fixture_path)
        if plan.primary_kind == "real_backend":
            if plan.real_backend is None:
                raise ValueError("Real mode requires a real_backend or real_command.")
            from any2xsampa.adapters.recognizer_real import (
                RealBackendUniversalPhoneRecognizer as IsolatedRealBackendRecognizer,
            )

            return IsolatedRealBackendRecognizer(
                plan.real_backend,
                output_path=getattr(plan, "real_output_path", None),
                model_path=getattr(plan, "model_path", None),
                device=getattr(plan, "device", "auto"),
            )
        if plan.primary_kind == "real_command":
            if plan.real_command is None or plan.real_output_path is None:
                raise ValueError("Real mode requires real_command with real_output_path.")
            return RealCommandUniversalPhoneRecognizer(plan.real_command, plan.real_output_path)
        raise ValueError(f"Unknown recognizer primary kind: {plan.primary_kind}")

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        plan = build_recognizer_dispatch_plan(self.config)
        try:
            return self._build_from_plan(plan).recognize(vocal_audio_path, lyrics=lyrics)
        except Exception as exc:
            mode = str(getattr(plan, "mode", self.config.mode)).strip().lower()
            if mode == "real" and bool(self.config.allow_stub_fallback):
                return _stub_segments(
                    requested_mode=mode,
                    primary_kind=str(plan.primary_kind),
                    error=exc,
                )
            raise


def frame_evidence_summary_from_posterior_payload(
    payload: dict[str, Any],
    *,
    anchor_start_sec: float | None = None,
    anchor_end_sec: float | None = None,
) -> dict[str, Any]:
    """Expose producer projection when present, with a small deterministic fallback."""
    try:
        from any2xsampa.adapters.recognizer_producer_projection import (
            frame_evidence_summary_from_posterior_payload as helper,
        )

        return helper(payload, anchor_start_sec=anchor_start_sec, anchor_end_sec=anchor_end_sec)
    except (ImportError, AttributeError):
        records: list[dict[str, Any]] = []
        for segment in payload.get("segments", []) if isinstance(payload.get("segments"), list) else []:
            if not isinstance(segment, dict):
                continue
            trace = segment.get("posterior_trace")
            frames = trace.get("frames", []) if isinstance(trace, dict) else []
            for frame in frames if isinstance(frames, list) else []:
                scores = _normalize_symbol_scores_frame(frame, score_key="confidence")
                if scores:
                    records.append({"top_symbol": _sorted_symbol_scores(scores)[0][0]})
        sequence = [record["top_symbol"] for record in records]
        return {
            "full_top_symbol_sequence": sequence,
            "full_run_symbols": sequence,
            "anchor_window_frame_count": len(sequence),
            "target_neighborhood_frame_count": len(sequence),
        }


def prediction_indices_for_reference_window(
    predictions: list[dict[str, Any]],
    *,
    start_sec: float,
    end_sec: float,
    selection_mode: str = "center_inside_window",
) -> list[int]:
    try:
        from any2xsampa.adapters.recognizer_producer_projection import (
            prediction_indices_for_reference_window as helper,
        )

        return helper(
            predictions,
            start_sec=start_sec,
            end_sec=end_sec,
            selection_mode=selection_mode,
        )
    except (ImportError, AttributeError):
        selected = []
        for index, prediction in enumerate(predictions):
            try:
                pred_start = float(prediction["start_sec"])
                pred_end = float(prediction["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if pred_end >= start_sec and pred_start <= end_sec:
                selected.append(index)
        if selected:
            return selected
        return [max(0, len(predictions) - 1)] if predictions else []


def producer_runtime_contract_from_posterior_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from any2xsampa.adapters.recognizer_producer_projection import (
            producer_runtime_contract_from_posterior_payload as helper,
        )

        return helper(payload)
    except (ImportError, AttributeError):
        producer = payload.get("producer") if isinstance(payload.get("producer"), dict) else {}
        return {
            "backend_kind": producer.get("backend_kind"),
            "dispatch": producer.get("dispatch", {}),
            "onnx_runtime": producer.get("onnx_espeak_runtime", {}),
            "payload_signature": {},
        }


# Keep the historical producer names importable from this module without
# pulling the producer back into recognizer truth or creating a package import
# cycle.  The producer imports the ONNX adapter, so this compatibility import
# must happen only after all recognizer classes are defined.
from any2xsampa.adapters.producer_compat import (
    ScoreAwarePosteriorJSONProducer,
    available_producer_backend_names,
)


__all__ = [
    "ConfigurableUniversalPhoneRecognizer",
    "ExternalPosteriorJSONUniversalPhoneRecognizer",
    "FixtureUniversalPhoneRecognizer",
    "OnnxEspeakCTCUniversalPhoneRecognizer",
    "ProducerPosteriorJSONUniversalPhoneRecognizer",
    "RealBackendUniversalPhoneRecognizer",
    "RealCommandUniversalPhoneRecognizer",
    "RecognizerConfig",
    "_build_acoustic_evidence",
    "_build_acoustic_evidence_entry",
    "_build_acoustic_evidence_impl",
    "_build_adapter_diagnostics",
    "_build_segments",
    "_candidate_entropy",
    "_candidate_logit_map",
    "_derive_logit_frames",
    "_derive_posterior_frames",
    "_infer_candidates_from_metadata",
    "_normalize_candidate_evidence",
    "_posterior_summary",
    "_logit_summary",
    "_legacy_acoustic_evidence_builder_seam_state",
    "_active_acoustic_evidence_builder",
    "segment_evidence_views",
    "segment_summary_payload",
    "frame_evidence_summary_from_posterior_payload",
    "prediction_indices_for_reference_window",
    "producer_runtime_contract_from_posterior_payload",
    "ScoreAwarePosteriorJSONProducer",
    "available_producer_backend_names",
    "build_onnx_espeak_runtime_diagnostics_snapshot",
    "describe_onnx_espeak_runtime",
    "write_onnx_espeak_runtime_diagnostics",
]

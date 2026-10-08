from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class ASRTranscriptError(ValueError):
    pass


class Qwen3ASRLoadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Qwen3ASRRequest:
    audio_path: Path
    model: str | None = None
    language: str | None = None
    context: str = ""
    timestamps: bool = False


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ASRTranscriptError(f"{label} must be numeric")
    try:
        raw = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ASRTranscriptError(f"{label} must be numeric") from exc
    if not math.isfinite(raw):
        raise ASRTranscriptError(f"{label} must be finite")
    if raw < 0:
        raise ASRTranscriptError(f"{label} must be non-negative")
    return round(raw, 3)


def parse_forced_align_items(value: object) -> list[dict[str, object]]:
    if isinstance(value, str):
        raise ASRTranscriptError("forced alignment value is not text repr; pass structured items")
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, Mapping):
        items = value.get("items", value.get("forced_align_items", []))
    else:
        items = getattr(value, "items", None)
    if callable(items):
        items = items()
    if not isinstance(items, (list, tuple)):
        raise ASRTranscriptError("forced alignment items must be a list")
    parsed: list[dict[str, object]] = []
    previous_end = -math.inf
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            text = item.get("text")
            if text is None:
                text = item.get("token", "")
            start = item.get("start_sec")
            if start is None:
                start = item.get("start_time")
            end = item.get("end_sec")
            if end is None:
                end = item.get("end_time")
            confidence = item.get("confidence")
        else:
            text = getattr(item, "text", "")
            start = getattr(item, "start_sec", None)
            if start is None:
                start = getattr(item, "start_time", None)
            end = getattr(item, "end_sec", None)
            if end is None:
                end = getattr(item, "end_time", None)
            confidence = getattr(item, "confidence", None)
        if start is None or end is None:
            raise ASRTranscriptError(f"forced alignment item {index} is missing timestamps")
        row: dict[str, object] = {
            "index": index,
            "text": str(text),
            "start_sec": _number(start, label=f"items[{index}].start_time"),
            "end_sec": _number(end, label=f"items[{index}].end_time"),
        }
        if not str(row["text"]).strip():
            raise ASRTranscriptError(f"forced alignment item {index} has empty text")
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        if end_sec <= start_sec:
            raise ASRTranscriptError(
                f"forced alignment item {index} must have positive duration"
            )
        if start_sec < previous_end:
            raise ASRTranscriptError(
                f"forced alignment item {index} is out of order or overlaps the previous item"
            )
        previous_end = end_sec
        if confidence is not None:
            if isinstance(confidence, bool):
                raise ASRTranscriptError(
                    f"forced alignment item {index} confidence must be numeric"
                )
            try:
                parsed_confidence = float(confidence)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ASRTranscriptError(
                    f"forced alignment item {index} confidence must be numeric"
                ) from exc
            if not math.isfinite(parsed_confidence) or not 0.0 <= parsed_confidence <= 1.0:
                raise ASRTranscriptError(
                    f"forced alignment item {index} confidence must be between zero and one"
                )
            row["confidence"] = parsed_confidence
        parsed.append(row)
    return parsed


def _first_result(results: object) -> object:
    if isinstance(results, (list, tuple)):
        if not results:
            raise ASRTranscriptError("Qwen3 backend returned no transcription")
        return results[0]
    return results


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class Qwen3ASRAdapter:
    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        backend: Any | None = None,
        forced_aligner: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.backend = backend
        self.forced_aligner = forced_aligner
        self.model_kwargs = dict(model_kwargs or {})
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        forced_aligner: Any | None = None,
        device_map: str | None = None,
        dtype: str | None = None,
        **model_kwargs: Any,
    ) -> "Qwen3ASRAdapter":
        return cls(
            model_name_or_path=model_name_or_path,
            forced_aligner=forced_aligner,
            device=device_map,
            dtype=dtype,
            model_kwargs=model_kwargs,
        )

    def _load_backend(self) -> Any:
        if self.backend is not None:
            return self.backend
        if not self.model_name_or_path:
            raise Qwen3ASRLoadError(
                "Qwen3-ASR model is not configured; pass model_name_or_path or inject a backend"
            )
        model_ref = str(self.model_name_or_path).strip()
        if not model_ref:
            raise Qwen3ASRLoadError(
                "Qwen3-ASR model is not configured; pass model_name_or_path or inject a backend"
            )
        local_ref = Path(model_ref)
        if local_ref.exists() and not local_ref.is_dir():
            raise Qwen3ASRLoadError(f"Qwen3-ASR model path must be a directory: {local_ref}")
        try:
            from qwen_asr import Qwen3ASRModel
        except Exception as exc:
            raise Qwen3ASRLoadError(
                "Qwen3-ASR model runtime is unavailable; install the optional "
                "qwen-asr runtime or inject a backend"
            ) from exc

        load_kwargs = dict(self.model_kwargs)
        if self.forced_aligner is not None:
            load_kwargs.setdefault("forced_aligner", self.forced_aligner)
        if self.device and self.device != "auto":
            load_kwargs.setdefault("device_map", self.device)
        if self.dtype is not None:
            load_kwargs.setdefault("dtype", self.dtype)
        try:
            self.backend = Qwen3ASRModel.from_pretrained(model_ref, **load_kwargs)
        except Exception as exc:
            raise Qwen3ASRLoadError(
                f"Qwen3-ASR model failed to load from {model_ref}: {exc}"
            ) from exc
        return self.backend

    def transcribe(self, request: Qwen3ASRRequest) -> dict[str, Any]:
        audio_path = Path(request.audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        backend = self._load_backend()
        try:
            results = backend.transcribe(
                str(audio_path),
                context=request.context or "",
                language=request.language,
                return_time_stamps=bool(request.timestamps),
            )
        except Exception as exc:
            raise ASRTranscriptError(f"Qwen3-ASR transcription failed: {exc}") from exc
        result = _first_result(results)
        text = str(_field(result, "text", "")).strip()
        if not text:
            raise ASRTranscriptError("Qwen3 backend returned an empty transcript")
        language = _field(result, "language", request.language)
        raw_timestamps = _field(result, "time_stamps", None)
        if raw_timestamps is None:
            raw_timestamps = _field(result, "timestamps", None)
        items = parse_forced_align_items(raw_timestamps) if request.timestamps else []
        model_name = self.model_name_or_path or str(getattr(backend, "model_name_or_path", "injected-backend"))
        model_device = self.device or str(getattr(backend, "device", "auto"))
        model_dtype = self.dtype or str(getattr(backend, "dtype", "auto"))
        audio_sha256 = _sha256_file(audio_path)
        request_sha256 = _sha256_json(
            {
                "audio_sha256": audio_sha256,
                "model": self.model_name_or_path,
                "language": request.language,
                "context": request.context,
                "timestamps": bool(request.timestamps),
                "device": self.device,
                "dtype": self.dtype,
                "model_kwargs": self.model_kwargs,
            }
        )
        return {
            "schema_version": "asr_transcript_v1",
            "model_output": True,
            "audio_path": str(audio_path),
            "language": str(language) if language is not None else None,
            "text": text,
            "segments": [
                {
                    "index": 0,
                    "text": text,
                    "timestamp_source": "forced_aligner" if items else "none",
                    "items": items,
                }
            ],
            "model": {
                "name_or_path": model_name,
                "backend": str(getattr(backend, "backend", "injected")),
                "device": model_device,
                "dtype": model_dtype,
                "model_output": True,
            },
            "provenance": {
                "truth_layer": "asr_evidence_not_phone_truth",
                "adapter": self.__class__.__name__,
                "timestamps_requested": bool(request.timestamps),
                "context_supplied": bool(request.context),
                "audio_sha256": audio_sha256,
                "request_sha256": request_sha256,
                "model_reference_sha256": hashlib.sha256(
                    str(model_name).encode("utf-8")
                ).hexdigest(),
            },
        }


def validate_transcript_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "asr_transcript_v1":
        raise ASRTranscriptError("ASR transcript schema must be asr_transcript_v1")
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ASRTranscriptError("ASR transcript text must be non-empty")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("truth_layer") != "asr_evidence_not_phone_truth":
        raise ASRTranscriptError("ASR transcript must declare evidence-not-phone-truth provenance")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ASRTranscriptError("ASR transcript segments must be a list")
    normalized = dict(payload)
    normalized["text"] = text
    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ASRTranscriptError(f"ASR transcript segment {index} must be an object")
        row = dict(segment)
        if not str(row.get("text") or "").strip():
            raise ASRTranscriptError(f"ASR transcript segment {index} has empty text")
        row["items"] = parse_forced_align_items(row.get("items"))
        normalized_segments.append(row)
    normalized["segments"] = normalized_segments
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dumps_transcript_json(payload: Mapping[str, Any]) -> str:
    validated = validate_transcript_payload(payload)
    return json.dumps(
        validated,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def normalize_qwen3_transcript(
    raw_results: object,
    *,
    audio_path: Path,
    requested_model: str,
    requested_language: str | None,
    requested_context: str = "",
    timestamps_requested: bool = False,
    backend: Any | None = None,
    forced_aligner: Any | None = None,
    model_output: bool = True,
) -> dict[str, Any]:
    result = _first_result(raw_results)
    text = str(_field(result, "text", "")).strip()
    if not text:
        raise ASRTranscriptError("Qwen3 backend returned an empty transcript")
    language = _field(result, "language", requested_language)
    raw_timestamps = _field(result, "time_stamps", None)
    if raw_timestamps is None:
        raw_timestamps = _field(result, "timestamps", None)
    items = parse_forced_align_items(raw_timestamps) if timestamps_requested else []
    audio_sha256 = _sha256_file(audio_path)
    device = str(getattr(backend, "device", "auto")) if backend is not None else "auto"
    dtype = str(getattr(backend, "dtype", "auto")) if backend is not None else "auto"
    payload = {
        "schema_version": "asr_transcript_v1",
        "audio_path": str(audio_path),
        "language": str(language) if language is not None else None,
        "text": text,
        "segments": [
            {
                "index": 0,
                "text": text,
                "timestamp_source": "forced_aligner" if items else "none",
                "items": items,
            }
        ],
        "model": {
            "name_or_path": requested_model,
            "backend": str(getattr(backend, "backend", "injected")),
            "device": device,
            "dtype": dtype,
            "model_output": bool(model_output),
        },
        "provenance": {
            "truth_layer": "asr_evidence_not_phone_truth",
            "adapter": "Qwen3ASRAdapter",
            "timestamps_requested": bool(timestamps_requested),
            "forced_aligner": str(forced_aligner) if forced_aligner is not None else None,
            "context_supplied": bool(requested_context),
            "audio_sha256": audio_sha256,
        },
    }
    return validate_transcript_payload(payload)


def build_qwen3_asr_model(
    model_name_or_path: str,
    *,
    forced_aligner: Any | None = None,
    device_map: str | None = None,
    dtype: str | None = None,
    **model_kwargs: Any,
) -> Qwen3ASRAdapter:
    return Qwen3ASRAdapter.from_pretrained(
        model_name_or_path,
        forced_aligner=forced_aligner,
        device_map=device_map,
        dtype=dtype,
        **model_kwargs,
    )


__all__ = [
    "ASRTranscriptError",
    "Qwen3ASRAdapter",
    "Qwen3ASRLoadError",
    "Qwen3ASRRequest",
    "build_qwen3_asr_model",
    "dumps_transcript_json",
    "normalize_qwen3_transcript",
    "parse_forced_align_items",
    "validate_transcript_payload",
]

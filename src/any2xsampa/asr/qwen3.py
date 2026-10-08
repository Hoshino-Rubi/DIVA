from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from any2xsampa.asr.qwen3_adapter import (
    Qwen3ASRAdapter as _EvidenceAdapter,
    normalize_qwen3_transcript,
)

ASR_TRANSCRIPT_SCHEMA_VERSION = "asr_transcript_v1"
ASR_EVIDENCE_ROLE = "lyric_time_evidence_not_phone_truth"


class ASRTranscriptDict(dict):
    def to_dict(self) -> dict[str, Any]:
        return dict(self)


def transcript_from_qwen_result(
    raw_result: Any,
    *,
    audio_path: str | Path,
    model_name: str,
    language: str | None,
    context: str = "",
    forced_aligner: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    model_output: bool = True,
    extra_provenance: dict[str, Any] | None = None,
) -> ASRTranscriptDict:
    raw_results = raw_result if isinstance(raw_result, (list, tuple)) else [raw_result]
    backend = _MetadataBackend(
        model_name_or_path=model_name,
        device=device,
        dtype=dtype,
        forced_aligner=object() if forced_aligner else None,
    )
    payload = normalize_qwen3_transcript(
        raw_results,
        audio_path=Path(audio_path),
        requested_model=model_name,
        requested_language=language,
        requested_context=context,
        timestamps_requested=True,
        backend=backend,
        forced_aligner=forced_aligner,
        model_output=model_output,
    )
    return _legacy_payload(
        payload,
        audio_path=audio_path,
        context=context,
        model_name=model_name,
        forced_aligner=forced_aligner,
        device=device,
        dtype=dtype,
        extra_provenance=extra_provenance,
    )


@dataclass(frozen=True, slots=True)
class _MetadataBackend:
    model_name_or_path: str
    device: str | None = None
    dtype: str | None = None
    forced_aligner: object | None = None
    backend: str = "injected"
    processor: object | None = None
    model: object | None = None


class Qwen3ASRAdapter:
    """Compatibility wrapper that keeps fake-model injection available."""

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        forced_aligner: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        extra_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.forced_aligner = forced_aligner
        self.device = device
        self.dtype = dtype
        self.extra_provenance = dict(extra_provenance or {})

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        *,
        forced_aligner: str | None = None,
        device_map: str | None = None,
        dtype: str | None = None,
        **kwargs: Any,
    ) -> _EvidenceAdapter:
        model_kwargs = dict(kwargs)
        if device_map:
            model_kwargs["device_map"] = device_map
        if dtype:
            model_kwargs["dtype"] = dtype
        return _EvidenceAdapter.from_pretrained(
            model_name,
            forced_aligner=forced_aligner,
            **model_kwargs,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = "Cantonese",
        context: str = "",
        return_timestamps: bool = False,
    ) -> ASRTranscriptDict:
        raw_results = self.model.transcribe(
            audio=str(audio_path),
            context=context,
            language=language,
            return_time_stamps=return_timestamps,
        )
        if not isinstance(raw_results, (list, tuple)):
            raw_results = [raw_results]
        backend = _MetadataBackend(
            model_name_or_path=self.model_name,
            device=self.device,
            dtype=self.dtype,
            forced_aligner=object() if self.forced_aligner else None,
        )
        payload = normalize_qwen3_transcript(
            raw_results,
            audio_path=Path(audio_path),
            requested_model=self.model_name,
            requested_language=language,
            requested_context=context,
            timestamps_requested=return_timestamps,
            backend=backend,
            forced_aligner=self.forced_aligner,
            model_output=True,
        )
        return _legacy_payload(
            payload,
            audio_path=audio_path,
            context=context,
            model_name=self.model_name,
            forced_aligner=self.forced_aligner,
            device=self.device,
            dtype=self.dtype,
            extra_provenance=self.extra_provenance,
        )


def _legacy_payload(
    payload: dict[str, Any],
    *,
    audio_path: str | Path,
    context: str,
    model_name: str,
    forced_aligner: str | None,
    device: str | None,
    dtype: str | None,
    extra_provenance: dict[str, Any] | None,
) -> ASRTranscriptDict:
    payload["evidence_role"] = ASR_EVIDENCE_ROLE
    payload["audio_path"] = str(audio_path)
    payload["context"] = context
    payload["provenance"]["model"] = model_name
    payload["provenance"]["forced_aligner"] = forced_aligner
    payload["provenance"]["device"] = device
    payload["provenance"]["dtype"] = dtype
    payload["provenance"]["contract_note"] = (
        "ASR transcript is lyric/time evidence only; phones remain downstream truth."
    )
    if extra_provenance:
        reserved = {"truth_layer", "adapter", "model_output", "audio_sha256", "request_sha256"}
        conflicts = sorted(reserved.intersection(extra_provenance))
        if conflicts:
            raise ValueError("extra Qwen provenance cannot override protected fields: " + ", ".join(conflicts))
        payload["provenance"].update(extra_provenance)
    for segment in payload["segments"]:
        segment.setdefault("confidence", None)
        segment["segment_id"] = segment.get("id", "seg-0").replace("-", "_")
        for item in segment.get("items", []):
            item.setdefault("confidence", None)
            item.setdefault("item_type", "char")
            item.setdefault("raw", {})
    return ASRTranscriptDict(payload)

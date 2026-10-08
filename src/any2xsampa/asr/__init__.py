"""ASR evidence adapters.

The project truth layer remains phones / IPA / articulatory evidence. ASR
transcripts are treated as time-aligned lyric evidence, not recognizer truth.
"""

from __future__ import annotations

from any2xsampa.asr.qwen3_adapter import (
    ASRTranscriptError,
    Qwen3ASRAdapter,
    Qwen3ASRLoadError,
    Qwen3ASRRequest,
    build_qwen3_asr_model,
    dumps_transcript_json,
    normalize_qwen3_transcript,
    validate_transcript_payload,
)

__all__ = [
    "ASRTranscriptError",
    "Qwen3ASRAdapter",
    "Qwen3ASRLoadError",
    "Qwen3ASRRequest",
    "build_qwen3_asr_model",
    "dumps_transcript_json",
    "normalize_qwen3_transcript",
    "validate_transcript_payload",
]

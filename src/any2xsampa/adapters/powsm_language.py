"""Explicit known-language ablation, preserving the historical unknown-language adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any

from any2xsampa.adapters.powsm import PowsmPosteriorRecognizer
from any2xsampa.adapters.powsm_runtime import PowsmRuntime
from any2xsampa.training.limited_data_artifacts import read_json


def validate_condition(language: str) -> str:
    if language not in {"<unk>", "<yue>"}:
        raise ValueError("only the predeclared unknown/Cantonese conditions are allowed")
    return language


class PowsmLanguageRuntime(PowsmRuntime):
    def __init__(self, run: Path, cache: Path, *, language: str) -> None:
        self.language = validate_condition(language)
        super().__init__(run, cache)
        original_encode = self.model.encode
        language_id = self.labels.index(language)
        expected = [self.labels.index("<unk>"), self.labels.index("<pr>")]

        def conditioned_encode(speech: Any, speech_lengths: Any, text_prev: Any,
                               text_prev_lengths: Any, prefix: Any, prefix_lengths: Any) -> Any:
            if prefix.tolist() != [expected] or text_prev.tolist() != [[self.labels.index("<na>")]]:
                raise ValueError("upstream prompt changed or external text was supplied")
            explicit_prefix = prefix.clone()
            explicit_prefix[:, 0] = language_id
            return original_encode(speech, speech_lengths, text_prev, text_prev_lengths,
                                   explicit_prefix, prefix_lengths)

        self.model.encode = conditioned_encode
        self.identity = {**self.identity, "language_condition": language,
                         "language_condition_kind": "known_cantonese" if language == "<yue>" else "unknown"}
        self.runtime = {**self.runtime, "language_condition": language,
                        "task_condition": "<pr>", "text_context": "<na>"}


class PowsmLanguagePosteriorRecognizer(PowsmPosteriorRecognizer):
    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list:
        payload = read_json(self.path)
        producer = payload.get("producer", {})
        language = validate_condition(producer.get("model_metadata", {}).get("language_condition"))
        if producer.get("language_hint_used") is not (language == "<yue>"):
            raise ValueError("language condition provenance differs from its declared hint")
        segments = super().recognize(vocal_audio_path, lyrics)
        for segment in segments:
            segment.metadata["language_condition"] = language
            segment.metadata["language_hint_used"] = language == "<yue>"
        return segments

"""Typed bridge from phone truth segments to a Miku V4C draft export."""

from __future__ import annotations

import math
from typing import Any, Sequence

from any2xsampa.core.models import PhoneSegment
from any2xsampa.vocaloid.draft import (
    DEFAULT_SINGER_TARGET,
    DEFAULT_TICKS_PER_QUARTER,
    VocalNoteDraft,
    VocalTrackDraft,
    VocalTrackDraftTrack,
)
from any2xsampa.vocaloid.phoneme_policy import assert_miku_v4c_note_phonemes_playable


_RESERVED_PHONES = {"<blank>", "<pad>", "<unk>", "UNMAPPED"}
_PROVENANCE_FIELDS = (
    "backend_kind",
    "checkpoint_sha256",
    "bundle_schema_version",
    "backbone_id",
    "backbone_revision",
    "dataset_sha256",
    "model_sha256",
    "split_identity",
    "split_sha256",
    "vocabulary_sha256",
    "configuration_sha256",
    "torch_version",
    "cuda_version",
)


def _sequence_value(values: Sequence[Any] | None, index: int, default: Any) -> Any:
    if values is None:
        return default
    if index >= len(values):
        raise ValueError("per-segment export values must cover every phone segment")
    return values[index]


def draft_from_phone_segments(
    segments: Sequence[PhoneSegment],
    *,
    tempo: float = 120.0,
    ticks_per_quarter: int = DEFAULT_TICKS_PER_QUARTER,
    pitches: Sequence[int] | None = None,
    lyrics: Sequence[str] | None = None,
    track_name: str = "Any2Xsampa phone export",
    singer_target: str = DEFAULT_SINGER_TARGET,
) -> VocalTrackDraft:
    """Build a draft without treating engine phonemes as recognizer truth."""

    if not segments:
        raise ValueError("phone-segment export requires at least one segment")
    if not math.isfinite(float(tempo)) or float(tempo) <= 0.0:
        raise ValueError("tempo must be finite and positive")
    if isinstance(ticks_per_quarter, bool) or ticks_per_quarter <= 0:
        raise ValueError("ticks_per_quarter must be a positive integer")
    if pitches is not None and len(pitches) != len(segments):
        raise ValueError("pitches must cover every phone segment")
    if lyrics is not None and len(lyrics) != len(segments):
        raise ValueError("lyrics must cover every phone segment")

    ticks_per_second = float(tempo) * ticks_per_quarter / 60.0
    notes: list[VocalNoteDraft] = []
    for index, segment in enumerate(segments):
        if not segment.candidate_phones:
            raise ValueError(f"phone segment {index} has no candidate phones")
        source_phone = segment.candidate_phones[0].symbol.strip()
        normalized_ipa = segment.normalized_ipa.strip()
        if not source_phone or source_phone in _RESERVED_PHONES or not normalized_ipa:
            raise ValueError(f"phone segment {index} has invalid phone/IPA truth")
        export = segment.engine_export
        if (
            export is None
            or export.format != "vocaloid_like"
            or segment.metadata.get("engine_export_format") != export.format
            or export.unmapped
            or export.fallback_used
            or not export.engine_phone.strip()
            or export.engine_phone.strip() in _RESERVED_PHONES
        ):
            raise ValueError(f"phone segment {index} has no approved Vocaloid mapping")
        phonemes = [export.engine_phone.strip()]
        assert_miku_v4c_note_phonemes_playable(phonemes)
        start_tick = round(segment.time_range.start_sec * ticks_per_second)
        end_tick = round(segment.time_range.end_sec * ticks_per_second)
        if start_tick < 0 or end_tick <= start_tick:
            raise ValueError(f"phone segment {index} has an invalid time range")
        pitch = int(_sequence_value(pitches, index, 60))
        if not 0 <= pitch <= 127:
            raise ValueError(f"phone segment {index} pitch must be a MIDI note")
        lyric = str(_sequence_value(lyrics, index, "la")).strip()
        if not lyric:
            raise ValueError(f"phone segment {index} lyric must be non-empty")
        recognizer_provenance = {
            field: segment.metadata[field]
            for field in _PROVENANCE_FIELDS
            if field in segment.metadata
        }
        notes.append(
            VocalNoteDraft(
                id=f"phone-segment-{index + 1}",
                start_tick=start_tick,
                duration_ticks=end_tick - start_tick,
                pitch=pitch,
                lyric=lyric,
                phonemes=phonemes,
                phoneme_source=f"engine-export:{export.format}",
                provenance={
                    "truth_layer": "phones_ipa_articulatory_features",
                    "source_phone": source_phone,
                    "normalized_ipa": normalized_ipa,
                    "recognizer": recognizer_provenance,
                    "engine_mapping": {
                        "format": export.format,
                        "engine_phone": export.engine_phone,
                        "approximation_applied": export.approximation_applied,
                        "approximation_notes": list(export.approximation_notes),
                    },
                },
            )
        )
    return VocalTrackDraft(
        tempo=float(tempo),
        ticks_per_quarter=ticks_per_quarter,
        singer_target=singer_target,
        tracks=[
            VocalTrackDraftTrack(
                id="phone-segment-track",
                name=track_name,
                singer_target=singer_target,
                notes=notes,
                provenance={
                    "truth_layer": "phones_ipa_articulatory_features",
                    "engine_phonemes_are_export_only": True,
                },
            )
        ],
        provenance={
            "adapter": "draft_from_phone_segments",
            "truth_layer": "phones_ipa_articulatory_features",
            "engine_phonemes_are_export_only": True,
        },
    )


__all__ = ["draft_from_phone_segments"]

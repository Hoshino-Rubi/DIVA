"""Miku V4C syllable-note bridge built from phone/IPA truth segments."""
from __future__ import annotations

import math
from typing import Sequence

from any2xsampa.core.models import PhoneSegment
from any2xsampa.vocaloid.draft import (
    DEFAULT_SINGER_TARGET,
    DEFAULT_TICKS_PER_QUARTER,
    VocalNoteDraft,
    VocalTrackDraft,
    VocalTrackDraftTrack,
)
from any2xsampa.vocaloid.phoneme_mapper import MikuV4CPhonemeMapper
from any2xsampa.vocaloid.phoneme_policy import assert_miku_v4c_note_phonemes_playable


_VOWELS = frozenset({"a", "@", "e", "i", "o", "u"})
_CODAS = frozenset({"m", "n", "N", "p", "t", "k"})
_ONSET_TO_INTERNAL = {
    "p": "b",
    "t": "d",
    "k": "g",
    "ts": "z",
    "s": "s",
    "l": "l",
    "m": "m",
    "n": "n",
    "N": "ng",
}
_NUCLEUS_TO_INTERNAL = {"a": "aa", "@": "a", "e": "e", "i": "i", "o": "o", "u": "u"}
_CODA_TO_INTERNAL = {"N": "ng", "m": "m", "n": "n", "p": "p", "t": "t", "k": "k"}


def draft_miku_v4c_from_phone_segments(
    segments: Sequence[PhoneSegment],
    *,
    tempo: float = 120.0,
    ticks_per_quarter: int = DEFAULT_TICKS_PER_QUARTER,
    pitches: Sequence[int] | None = None,
    lyrics: Sequence[str] | None = None,
    minimum_note_ticks: int = 120,
    maximum_onset_gap_sec: float = 0.18,
    maximum_coda_gap_sec: float = 0.12,
    singer_target: str = DEFAULT_SINGER_TARGET,
) -> VocalTrackDraft:
    """Build editable Miku notes around vowel nuclei; consonants remain evidence."""

    if not segments:
        raise ValueError("Miku V4C export requires at least one phone segment")
    if not math.isfinite(float(tempo)) or tempo <= 0:
        raise ValueError("tempo must be finite and positive")
    if ticks_per_quarter <= 0 or minimum_note_ticks <= 0:
        raise ValueError("tick values must be positive")
    mapper = MikuV4CPhonemeMapper.for_cantonese()
    ticks_per_second = float(tempo) * ticks_per_quarter / 60.0
    notes: list[VocalNoteDraft] = []
    used: set[int] = set()

    for nucleus_index, segment in enumerate(segments):
        nucleus = _phone(segment)
        if nucleus not in _VOWELS:
            continue
        members = [nucleus_index]
        internal = [_NUCLEUS_TO_INTERNAL[nucleus]]
        if nucleus_index > 0 and nucleus_index - 1 not in used:
            onset_segment = segments[nucleus_index - 1]
            onset = _phone(onset_segment)
            gap = segment.time_range.start_sec - onset_segment.time_range.end_sec
            if onset in _ONSET_TO_INTERNAL and 0.0 <= gap <= maximum_onset_gap_sec:
                members.insert(0, nucleus_index - 1)
                internal.insert(0, _ONSET_TO_INTERNAL[onset])
        if nucleus_index + 1 < len(segments):
            coda_segment = segments[nucleus_index + 1]
            coda = _phone(coda_segment)
            gap = coda_segment.time_range.start_sec - segment.time_range.end_sec
            if coda in _CODAS and 0.0 <= gap <= maximum_coda_gap_sec:
                members.append(nucleus_index + 1)
                internal.append(_CODA_TO_INTERNAL[coda])

        phonemes, mapping_notes = mapper.map_internal(internal)
        assert_miku_v4c_note_phonemes_playable(phonemes)
        start_sec = segments[members[0]].time_range.start_sec
        observed_end_sec = segments[members[-1]].time_range.end_sec
        next_nucleus_start = _next_nucleus_start(segments, nucleus_index)
        end_sec = max(observed_end_sec, segment.time_range.start_sec + minimum_note_ticks / ticks_per_second)
        if next_nucleus_start is not None:
            end_sec = min(end_sec, next_nucleus_start)
        if end_sec <= start_sec:
            continue
        start_tick = round(start_sec * ticks_per_second)
        duration_ticks = max(minimum_note_ticks, round((end_sec - start_sec) * ticks_per_second))
        pitch = int(pitches[len(notes)]) if pitches is not None and len(notes) < len(pitches) else 60
        lyric = str(lyrics[len(notes)]) if lyrics is not None and len(notes) < len(lyrics) else "la"
        used.update(members)
        notes.append(
            VocalNoteDraft(
                id=f"miku-syllable-{len(notes) + 1}",
                start_tick=start_tick,
                duration_ticks=duration_ticks,
                pitch=pitch,
                lyric=lyric,
                phonemes=phonemes,
                phoneme_source="miku-v4c-cantonese-syllable-mapper",
                provenance={
                    "truth_layer": "phones_ipa_articulatory_features",
                    "member_segment_indices": members,
                    "internal_phones": internal,
                    "mapping_notes": mapping_notes,
                    "timing_policy": "vowel_nucleus_with_nearby_onset_coda_minimum_note_ticks",
                    "minimum_note_ticks": minimum_note_ticks,
                    "functional_reconstruction": True,
                },
            )
        )
    if not notes:
        raise ValueError("Miku V4C export found no vowel nuclei; refusing consonant-only notes")
    return VocalTrackDraft(
        tempo=float(tempo),
        ticks_per_quarter=ticks_per_quarter,
        singer_target=singer_target,
        tracks=[
            VocalTrackDraftTrack(
                id="miku-syllable-track",
                name="Any2Xsampa Miku V4C syllable export",
                singer_target=singer_target,
                notes=notes,
                provenance={
                    "truth_layer": "phones_ipa_articulatory_features",
                    "engine_phonemes_are_export_only": True,
                    "functional_reconstruction": True,
                },
            )
        ],
        provenance={
            "adapter": "draft_miku_v4c_from_phone_segments",
            "truth_layer": "phones_ipa_articulatory_features",
            "engine_phonemes_are_export_only": True,
            "functional_reconstruction": True,
        },
    )


def _phone(segment: PhoneSegment) -> str:
    return (segment.normalized_ipa or "").strip()


def _next_nucleus_start(segments: Sequence[PhoneSegment], index: int) -> float | None:
    for candidate in segments[index + 1 :]:
        if _phone(candidate) in _VOWELS:
            return candidate.time_range.start_sec
    return None


__all__ = ["draft_miku_v4c_from_phone_segments"]

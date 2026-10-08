from __future__ import annotations

import pytest

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.vocaloid import draft_miku_v4c_from_phone_segments


def segment(phone: str, start: float, end: float) -> PhoneSegment:
    return PhoneSegment(
        time_range=TimeRange(start, end),
        candidate_phones=[PhoneCandidate(phone, 0.9)],
        confidence=0.9,
        normalized_ipa=phone,
    )


def test_builds_one_visible_note_from_onset_nucleus_coda() -> None:
    draft = draft_miku_v4c_from_phone_segments(
        [segment("t", 0.0, 0.04), segment("a", 0.05, 0.09), segment("n", 0.10, 0.13)],
        tempo=120,
    )

    note = draft.tracks[0].notes[0]
    assert note.phonemes == ["t", "a", "n"]
    assert note.start_tick == 0
    assert note.duration_ticks >= 120
    assert note.provenance["member_segment_indices"] == [0, 1, 2]


def test_drops_isolated_consonants_instead_of_creating_notes() -> None:
    draft = draft_miku_v4c_from_phone_segments(
        [segment("t", 0.0, 0.02), segment("a", 0.5, 0.52), segment("s", 1.0, 1.02)]
    )

    assert len(draft.tracks[0].notes) == 1
    assert draft.tracks[0].notes[0].phonemes == ["a"]


def test_rejects_consonant_only_input() -> None:
    with pytest.raises(ValueError, match="no vowel nuclei"):
        draft_miku_v4c_from_phone_segments([segment("t", 0.0, 0.02)])

from __future__ import annotations

import pytest

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.vocaloid.evidence_fusion import (
    EvidenceFusionError,
    build_pitch_skeleton_payload,
)
from tools.build_vocal_track_draft import build_draft_payload


def _segment(
    phone: str, start: float, end: float, *, lyric_item_index: int | None = None
) -> PhoneSegment:
    metadata = {"source_mode": "real", "real_model_output": True}
    if lyric_item_index is not None:
        metadata["lyric_item_index"] = lyric_item_index
    return PhoneSegment(
        time_range=TimeRange(start, end),
        candidate_phones=[PhoneCandidate(phone, 0.9)],
        confidence=0.9,
        normalized_ipa=phone,
        metadata=metadata,
    )


def test_pitch_skeleton_keeps_every_pitch_and_uses_dash_continuations() -> None:
    payload = build_draft_payload("喜歡", jyutping_override="hei2 fun1", pitches=[64, 65])
    phones = [
        _segment("s", 0.0, 0.04),
        _segment("e", 0.04, 0.45),
        _segment("s", 0.5, 0.54),
        _segment("u", 0.54, 1.0),
    ]
    pitches = [
        {"note_id": "p0", "start_sec": 0.0, "duration_sec": 0.2, "midi": 60, "source": "pyin"},
        {"note_id": "p1", "start_sec": 0.2, "duration_sec": 0.2, "midi": 62, "source": "pyin"},
        {"note_id": "p2", "start_sec": 0.55, "duration_sec": 0.4, "midi": 64, "source": "pyin"},
    ]

    fused, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="user_lyrics",
        tempo_bpm=120.0,
    )

    notes = fused["tracks"][0]["notes"]
    assert [note["lyric"] for note in notes] == ["喜", "-", "歡"]
    assert [note["pitch"] for note in notes] == [64, 66, 68]
    assert len(notes) == len(pitches)
    assert summary["direct_overlap_ratio"] == 1.0
    assert summary["interpolation_count"] == 0
    assert all(note["phoneme_source"] for note in notes)
    assert all(note["provenance"]["pitch_candidate_id"] for note in notes)
    assert notes[1]["provenance"]["continuation_note"] is True
    assert len(notes[1]["phonemes"]) == 1


def test_single_interpolation_is_recorded_within_five_percent() -> None:
    payload = build_draft_payload("喜", jyutping_override="hei2", pitches=[64])
    phones = [_segment("e", 0.0, 4.0)]
    pitches = [
        {
            "note_id": f"p{index}",
            "start_sec": index * 0.2,
            "duration_sec": 0.1,
            "midi": 64,
            "source": "pyin",
        }
        for index in range(20)
    ]
    pitches[-1]["start_sec"] = 4.2

    _, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="external_asr_transcript",
        tempo_bpm=120.0,
    )

    assert summary["direct_overlap_ratio"] == 0.95
    assert summary["interpolation_ratio"] == 0.05
    assert summary["interpolations"][0]["pitch_candidate_id"] == "p19"
    assert summary["interpolations"][0]["reason"]


def test_phone_group_members_are_not_reused_across_syllables() -> None:
    payload = build_draft_payload("\u559c\u6b61", jyutping_override="hei2 fun1", pitches=[64, 65])
    phones = [
        _segment("e", 0.0, 0.4),
        _segment("t", 0.4, 0.5),
        _segment("u", 0.5, 0.9),
    ]
    pitches = [
        {"start_sec": 0.0, "duration_sec": 0.3, "midi": 64},
        {"start_sec": 0.55, "duration_sec": 0.3, "midi": 65},
    ]

    fused, _ = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="user_lyrics",
        tempo_bpm=120.0,
    )

    memberships = [note["provenance"]["phone_segment_indices"] for note in fused["tracks"][0]["notes"]]
    assert memberships == [[0, 1], [2]]


def test_phone_groups_do_not_cross_explicit_lyric_item_boundaries() -> None:
    payload = build_draft_payload("久到", jyutping_override="gau2 dou3", pitches=[64, 65])
    phones = [
        _segment("ɐu", 0.0, 0.4, lyric_item_index=0),
        _segment("t", 0.4, 0.45, lyric_item_index=1),
        _segment("ou", 0.45, 0.8, lyric_item_index=1),
    ]
    pitches = [
        {"start_sec": 0.0, "duration_sec": 0.3, "midi": 64},
        {"start_sec": 0.45, "duration_sec": 0.3, "midi": 65},
    ]

    fused, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="mfa_cv19_v3",
        tempo_bpm=120.0,
    )

    assert summary["phone_group_count"] == 2
    memberships = [note["provenance"]["phone_segment_indices"] for note in fused["tracks"][0]["notes"]]
    assert memberships == [[0], [1, 2]]


def test_pitch_candidate_crossing_explicit_lyrics_is_audited_boundary_split() -> None:
    payload = build_draft_payload("久到", jyutping_override="gau2 dou3", pitches=[64])
    phones = [
        _segment("ɐu", 0.0, 0.4, lyric_item_index=0),
        _segment("ou", 0.4, 0.8, lyric_item_index=1),
    ]
    pitches = [
        {"note_id": "held-note", "start_sec": 0.0, "duration_sec": 0.8, "midi": 64}
    ]

    fused, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="mfa_cv19_v3",
        tempo_bpm=120.0,
    )

    notes = fused["tracks"][0]["notes"]
    assert [note["lyric"] for note in notes] == ["久", "到"]
    assert [note["provenance"]["pitch_source_candidate_id"] for note in notes] == [
        "held-note",
        "held-note",
    ]
    assert summary["note_count"] == 2
    assert summary["coverage"]["pitch_candidate_count"] == 1
    assert summary["pitch_boundary_split_count"] == 1
    assert summary["pitch_boundary_splits"][0]["split_boundaries_sec"] == [0.4]
    assert summary["direct_overlap_ratio"] == 1.0
    assert summary["interpolation_ratio"] == 0.0


def test_rejects_more_lyrics_than_phone_groups() -> None:
    payload = build_draft_payload(
        "\u559c\u6b61\u4f60", jyutping_override="hei2 fun1 nei5", pitches=[64]
    )
    pitches = [{"start_sec": 0.0, "duration_sec": 0.4, "midi": 64}]

    with pytest.raises(EvidenceFusionError, match="cannot be assigned") as raised:
        build_pitch_skeleton_payload(
            base_payload=payload,
            pitch_rows=pitches,
            phone_segments=[_segment("e", 0.0, 0.4)],
            lyric_source="user_lyrics",
            tempo_bpm=120.0,
        )

    assert raised.value.error_kind == "insufficient_phone_evidence"


@pytest.mark.parametrize(
    ("pitches", "phones", "error_kind"),
    [
        (
            [
                {"start_sec": 1.0, "duration_sec": 0.2, "midi": 64},
                {"start_sec": 0.0, "duration_sec": 0.2, "midi": 64},
            ],
            [_segment("e", 0.0, 2.0)],
            "invalid_pitch_evidence",
        ),
        (
            [{"start_sec": 0.0, "duration_sec": 0.2, "midi": 64}],
            [_segment("e", 1.0, 1.2), _segment("u", 0.0, 0.2)],
            "invalid_phone_evidence",
        ),
    ],
)
def test_rejects_out_of_order_evidence(
    pitches: list[dict[str, float]], phones: list[PhoneSegment], error_kind: str
) -> None:
    payload = build_draft_payload("\u559c", jyutping_override="hei2", pitches=[64])

    with pytest.raises(EvidenceFusionError) as raised:
        build_pitch_skeleton_payload(
            base_payload=payload,
            pitch_rows=pitches,
            phone_segments=phones,
            lyric_source="user_lyrics",
            tempo_bpm=120.0,
        )

    assert raised.value.error_kind == error_kind


def test_rejects_overlapping_phone_segments_and_fractional_midi() -> None:
    payload = build_draft_payload("\u559c", jyutping_override="hei2", pitches=[64])
    valid_pitch = [{"start_sec": 0.0, "duration_sec": 0.2, "midi": 64}]

    with pytest.raises(EvidenceFusionError) as overlapping:
        build_pitch_skeleton_payload(
            base_payload=payload,
            pitch_rows=valid_pitch,
            phone_segments=[_segment("e", 0.0, 1.0), _segment("u", 0.5, 1.5)],
            lyric_source="user_lyrics",
            tempo_bpm=120.0,
        )
    assert overlapping.value.error_kind == "invalid_phone_evidence"

    with pytest.raises(EvidenceFusionError) as fractional:
        build_pitch_skeleton_payload(
            base_payload=payload,
            pitch_rows=[{"start_sec": 0.0, "duration_sec": 0.2, "midi": 64.5}],
            phone_segments=[_segment("e", 0.0, 1.0)],
            lyric_source="user_lyrics",
            tempo_bpm=120.0,
        )
    assert fractional.value.error_kind == "invalid_pitch_evidence"


def test_gate_thresholds_cannot_be_lowered() -> None:
    payload = build_draft_payload("喜", jyutping_override="hei2", pitches=[64])
    pitch = [{"start_sec": 0.0, "duration_sec": 0.2, "midi": 64}]
    phones = [_segment("e", 0.0, 1.0)]

    with pytest.raises(EvidenceFusionError) as direct:
        build_pitch_skeleton_payload(
            base_payload=payload,
            pitch_rows=pitch,
            phone_segments=phones,
            lyric_source="user_lyrics",
            tempo_bpm=120.0,
            direct_overlap_threshold=0.90,
        )
    assert direct.value.error_kind == "invalid_evidence_threshold"

    with pytest.raises(EvidenceFusionError) as interpolation:
        build_pitch_skeleton_payload(
            base_payload=payload,
            pitch_rows=pitch,
            phone_segments=phones,
            lyric_source="user_lyrics",
            tempo_bpm=120.0,
            maximum_interpolation_ratio=0.10,
        )
    assert interpolation.value.error_kind == "invalid_evidence_threshold"


def test_summary_declares_full_coverage_and_monotonic_alignment() -> None:
    payload = build_draft_payload("喜歡", jyutping_override="hei2 fun1", pitches=[64, 65, 66])
    phones = [_segment("e", 0.0, 0.4), _segment("u", 0.4, 0.8)]
    pitches = [
        {"start_sec": 0.0, "duration_sec": 0.2, "midi": 64},
        {"start_sec": 0.2, "duration_sec": 0.2, "midi": 65},
        {"start_sec": 0.4, "duration_sec": 0.2, "midi": 66},
    ]

    _, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="user_lyrics",
        tempo_bpm=120.0,
    )

    assert summary["pitch_coverage_ratio"] == 1.0
    assert summary["phone_coverage_ratio"] == 1.0
    assert summary["lyric_coverage_ratio"] == 1.0
    assert summary["monotonic_alignment"] is True
    assert summary["coverage"]["phone_segment_covered_count"] == 2


def test_unassigned_content_phone_is_a_structured_hard_failure() -> None:
    payload = build_draft_payload("喜", jyutping_override="hei2", pitches=[64])
    pitch = [{"start_sec": 0.0, "duration_sec": 0.2, "midi": 64}]

    with pytest.raises(EvidenceFusionError) as raised:
        build_pitch_skeleton_payload(
            base_payload=payload,
            pitch_rows=pitch,
            phone_segments=[_segment("e", 0.0, 0.4), _segment("x", 0.4, 0.5)],
            lyric_source="user_lyrics",
            tempo_bpm=120.0,
        )

    assert raised.value.error_kind == "incomplete_phone_alignment"


def test_continuation_uses_acoustic_nucleus_not_lexical_export() -> None:
    payload = build_draft_payload("喜", jyutping_override="hei2", pitches=[64])
    phones = [_segment("e", 0.0, 0.4)]
    pitches = [
        {"start_sec": 0.0, "duration_sec": 0.2, "midi": 64},
        {"start_sec": 0.2, "duration_sec": 0.2, "midi": 65},
    ]

    fused, _ = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="user_lyrics",
        tempo_bpm=120.0,
    )

    notes = fused["tracks"][0]["notes"]
    assert notes[0]["lyric"] != "-"
    assert notes[1]["lyric"] == "-"
    assert notes[1]["phonemes"] == ["@`"]
    assert notes[1]["provenance"]["truth_layer"] == "phones_ipa_articulatory_features"


def test_m_and_ng_are_onsets_before_vowels_not_extra_nuclei() -> None:
    payload = build_draft_payload("喜", jyutping_override="hei2", pitches=[64])
    phones = [_segment("m", 0.0, 0.04), _segment("e", 0.04, 0.4)]
    pitches = [{"start_sec": 0.0, "duration_sec": 0.3, "midi": 64}]

    fused, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="user_lyrics",
        tempo_bpm=120.0,
    )

    assert summary["phone_group_count"] == 1
    assert fused["tracks"][0]["notes"][0]["provenance"]["phone_segment_indices"] == [0, 1]


@pytest.mark.parametrize(
    ("ipa", "internal"),
    [
        ("\u0250", "a"),
        ("\u0275", "eo"),
        ("\u0153\u02d0", "oe"),
        ("\u025b\u02d0", "e"),
        ("\u0254\u02d0", "o"),
        ("\u026a", "i"),
        ("\u028a", "u"),
        ("\u0250i", "ai"),
        ("\u0250u", "au"),
        ("\u0275y", "eoi"),
        ("\u0254\u02d0i", "oi"),
    ],
)
def test_mfa_ipa_nuclei_remain_single_groups_and_keep_ipa_truth(
    ipa: str, internal: str
) -> None:
    payload = build_draft_payload("喜", jyutping_override="hei2", pitches=[64])
    pitches = [
        {"start_sec": 0.0, "duration_sec": 0.2, "midi": 64},
        {"start_sec": 0.2, "duration_sec": 0.2, "midi": 65},
    ]

    fused, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=[_segment(ipa, 0.0, 0.4)],
        lyric_source="mfa_cv19_v3",
        tempo_bpm=120.0,
    )

    assert summary["phone_group_count"] == 1
    assert summary["direct_overlap_ratio"] == 1.0
    continuation = fused["tracks"][0]["notes"][1]
    assert continuation["phonemes"]
    assert continuation["provenance"]["phone_ipa"] == [ipa]
    assert continuation["provenance"]["phone_internal"] == [internal]


@pytest.mark.parametrize("ipa", ["m\u0329", "\u014b\u0329"])
def test_explicit_mfa_syllabic_nasals_are_nuclei(ipa: str) -> None:
    payload = build_draft_payload("喜", jyutping_override="hei2", pitches=[64])
    pitches = [
        {"start_sec": 0.0, "duration_sec": 0.2, "midi": 64},
        {"start_sec": 0.2, "duration_sec": 0.2, "midi": 65},
    ]

    fused, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=[_segment(ipa, 0.0, 0.4)],
        lyric_source="mfa_cv19_v3",
        tempo_bpm=120.0,
    )

    assert summary["phone_group_count"] == 1
    continuation = fused["tracks"][0]["notes"][1]
    assert continuation["phonemes"] == (["m"] if ipa.startswith("m") else ["N"])
    assert continuation["provenance"]["phone_ipa"] == [ipa]


def test_mfa_syllabic_nasal_is_not_attached_as_a_vowel_coda() -> None:
    payload = build_draft_payload("喜歡", jyutping_override="hei2 fun1", pitches=[64, 65])
    phones = [_segment("\u0250", 0.0, 0.2), _segment("m\u0329", 0.2, 0.4)]
    pitches = [
        {"start_sec": 0.0, "duration_sec": 0.2, "midi": 64},
        {"start_sec": 0.2, "duration_sec": 0.2, "midi": 65},
    ]

    fused, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=pitches,
        phone_segments=phones,
        lyric_source="mfa_cv19_v3",
        tempo_bpm=120.0,
    )

    assert summary["phone_group_count"] == 2
    notes = fused["tracks"][0]["notes"]
    assert notes[0]["provenance"]["phone_ipa"] == ["\u0250"]
    assert notes[1]["provenance"]["phone_ipa"] == ["m\u0329"]


def test_mfa_velar_nasal_onset_attaches_to_unicode_vowel() -> None:
    payload = build_draft_payload("喜", jyutping_override="hei2", pitches=[64])
    fused, summary = build_pitch_skeleton_payload(
        base_payload=payload,
        pitch_rows=[{"start_sec": 0.0, "duration_sec": 0.3, "midi": 64}],
        phone_segments=[_segment("\u014b", 0.0, 0.04), _segment("\u0250", 0.04, 0.4)],
        lyric_source="mfa_cv19_v3",
        tempo_bpm=120.0,
    )

    assert summary["phone_group_count"] == 1
    assert fused["tracks"][0]["notes"][0]["provenance"]["phone_ipa"] == ["\u014b", "\u0250"]

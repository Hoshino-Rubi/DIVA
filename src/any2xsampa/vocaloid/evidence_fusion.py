from __future__ import annotations

"""Fuse pitch, lyric, and phone evidence into a pitch-skeleton Vocaloid draft."""

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from any2xsampa.core.models import PhoneSegment
from any2xsampa.vocaloid.phoneme_mapper import MikuV4CPhonemeMapper
from any2xsampa.vocaloid.phoneme_policy import assert_miku_v4c_note_phonemes_playable

SCHEMA_VERSION = "audio_evidence_fusion_v1"
_MIN_DIRECT_OVERLAP_THRESHOLD = 0.95
_MAX_INTERPOLATION_RATIO = 0.05
# The product fusion receives canonical IPA from the MFA adapter.  Keep the
# historical ASCII aliases as well: fixture and non-MFA recognizers still use
# those labels.  A diphthong is one phone in the MFA phone set and must remain
# one nucleus rather than being split into its component glyphs.
_VOWELS = frozenset(
    {
        "a", "@", "e", "i", "o", "u",
        "aː", "aːaː", "aːi", "aːu",
        "eɪ", "ei", "iː", "iːa", "iːu", "ou", "uː", "uːi", "yː",
        "ă", "ɐ", "ɐi", "ɐu", "œː", "ɔː", "ɔːi",
        "ɛː", "ɛːu", "ɪ", "ɵ", "ɵy", "ʊ",
        # Stress-bearing MFA entries are still vowel nuclei; retain the mark
        # in the IPA provenance while mapping their export approximation.
        "ˈi", "ˈɛː", "ˌe", "ˌi",
    }
)
_SYLLABIC_NUCLEI = frozenset({"m", "N", "ng", "ŋ", "m̩", "m̍", "ŋ̩", "ŋ̍"})
_EXPLICIT_SYLLABIC_NUCLEI = frozenset({"m̩", "m̍", "ŋ̩", "ŋ̍"})
_NON_CONTENT_PHONES = frozenset({"sil", "sp", "spn", "pau", "<eps>", "O"})
_ONSET_TO_INTERNAL = {
    "b": "b", "p": "b", "d": "d", "t": "d", "g": "g", "k": "g",
    "ts": "z", "c": "z", "s": "s", "f": "f", "h": "h", "j": "j", "w": "w",
    "l": "l", "m": "m", "n": "n", "N": "ng", "ng": "ng", "ŋ": "ng",
    "ɡ": "g", "pʰ": "b", "tʰ": "d", "tsʰ": "z", "kʰ": "g",
    "kʷ": "g", "kʷʰ": "g", "v": "f", "r": "l", "ɻ": "l",
}
_NUCLEUS_TO_INTERNAL = {
    "a": "aa", "@": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "aː": "aa", "aːaː": "aa", "aːi": "ai", "aːu": "au",
    "ei": "ei", "eɪ": "ei", "iː": "i", "iːa": "ai", "iːu": "iu",
    "ou": "ou", "uː": "u", "uːi": "ui", "yː": "yu",
    "ă": "a", "ɐ": "a", "ɐi": "ai", "ɐu": "au", "œː": "oe",
    "ɔː": "o", "ɔːi": "oi", "ɛː": "e", "ɛːu": "eo",
    "ɪ": "i", "ɵ": "eo", "ɵy": "eoi", "ʊ": "u",
    "ˈi": "i", "ˈɛː": "e", "ˌe": "e", "ˌi": "i",
    "m": "m", "N": "ng", "ng": "ng", "ŋ": "ng",
    "m̩": "m", "m̍": "m", "ŋ̩": "ng", "ŋ̍": "ng",
}
_CODA_TO_INTERNAL = {
    "N": "ng", "ng": "ng", "ŋ": "ng", "m": "m", "n": "n", "p": "p", "t": "t", "k": "k",
}


class EvidenceFusionError(ValueError):
    """Evidence cannot satisfy the product alignment gate."""

    def __init__(self, error_kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


def build_pitch_skeleton_payload(
    *,
    base_payload: Mapping[str, Any],
    pitch_rows: Sequence[Mapping[str, Any]],
    phone_segments: Sequence[PhoneSegment],
    lyric_source: str,
    tempo_bpm: float,
    direct_overlap_threshold: float = 0.95,
    maximum_interpolation_ratio: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one note per pitch row with time-monotonic phone and lyric evidence."""

    if not isinstance(base_payload, Mapping):
        raise EvidenceFusionError("invalid_evidence_payload", "draft payload must be an object")
    if not isinstance(pitch_rows, Sequence) or isinstance(pitch_rows, (str, bytes)):
        raise EvidenceFusionError("invalid_pitch_evidence", "pitch evidence must be an ordered object list")
    if not isinstance(phone_segments, Sequence) or isinstance(phone_segments, (str, bytes)):
        raise EvidenceFusionError("invalid_phone_evidence", "phone evidence must be an ordered PhoneSegment list")
    if not pitch_rows:
        raise EvidenceFusionError("missing_pitch_evidence", "pitch evidence contains no note candidates")
    tempo_bpm, direct_overlap_threshold, maximum_interpolation_ratio = _validate_inputs(
        base_payload=base_payload,
        pitch_rows=pitch_rows,
        phone_segments=phone_segments,
        lyric_source=lyric_source,
        tempo_bpm=tempo_bpm,
        direct_overlap_threshold=direct_overlap_threshold,
        maximum_interpolation_ratio=maximum_interpolation_ratio,
    )
    syllable_notes = _syllable_notes(base_payload)
    phone_groups = _phone_groups(phone_segments)
    if not phone_groups:
        raise EvidenceFusionError("missing_phone_evidence", "phone evidence contains no vowel-nucleus groups")
    if len(syllable_notes) > len(phone_groups):
        raise EvidenceFusionError(
            "insufficient_phone_evidence",
            f"{len(syllable_notes)} lyric syllables cannot be assigned to {len(phone_groups)} phone groups",
        )
    render_pitch_rows, pitch_boundary_splits = _split_pitch_rows_for_group_coverage(
        pitch_rows, phone_groups
    )
    if len(syllable_notes) > len(render_pitch_rows):
        raise EvidenceFusionError(
            "insufficient_pitch_evidence",
            f"{len(syllable_notes)} lyric syllables cannot be represented by "
            f"{len(render_pitch_rows)} direct pitch spans",
        )
    aligned, interpolations = _align_pitch_rows(render_pitch_rows, phone_groups)
    represented_groups = {int(item["phone_group_index"]) for item in aligned}
    if represented_groups != set(range(len(phone_groups))):
        missing_groups = sorted(set(range(len(phone_groups))) - represented_groups)
        raise EvidenceFusionError(
            "incomplete_phone_alignment",
            f"pitch evidence does not cover phone groups: missing indices {missing_groups}",
        )
    represented_syllables = {
        _group_syllable_index(
            phone_groups[int(item["phone_group_index"])],
            int(item["phone_group_index"]),
            len(phone_groups),
            len(syllable_notes),
        )
        for item in aligned
    }
    if represented_syllables != set(range(len(syllable_notes))):
        missing = sorted(set(range(len(syllable_notes))) - represented_syllables)
        raise EvidenceFusionError(
            "incomplete_lyrics_alignment",
            f"pitch evidence does not cover lyric syllables: missing indices {missing}",
        )
    covered_phone_indices = _validate_phone_coverage(phone_segments, phone_groups)
    _validate_monotonic_syllable_groups(phone_groups, len(syllable_notes))
    _validate_monotonic_alignment(aligned, len(phone_groups))
    interpolation_ratio = len(interpolations) / len(render_pitch_rows)
    direct_overlap_count = sum(1 for item in aligned if bool(item["direct_overlap"]))
    direct_ratio = direct_overlap_count / len(render_pitch_rows)
    if direct_ratio + 1e-12 < direct_overlap_threshold:
        raise EvidenceFusionError(
            "pitch_phone_overlap_below_threshold",
            f"direct pitch/phone overlap ratio {direct_ratio:.6f} is below {direct_overlap_threshold:.6f}",
        )
    if interpolation_ratio - 1e-12 > maximum_interpolation_ratio:
        raise EvidenceFusionError(
            "pitch_phone_interpolation_limit_exceeded",
            f"pitch/phone interpolation ratio {interpolation_ratio:.6f} exceeds {maximum_interpolation_ratio:.6f}",
        )

    payload = deepcopy(dict(base_payload))
    ticks_per_quarter = int(payload.get("ticks_per_quarter", 480))
    ticks_per_second = ticks_per_quarter * float(tempo_bpm) / 60.0
    mapper = MikuV4CPhonemeMapper.for_cantonese()
    result_notes: list[dict[str, Any]] = []
    previous_syllable: int | None = None
    transpose = _pitch_transpose(render_pitch_rows)

    for index, (row, alignment) in enumerate(zip(render_pitch_rows, aligned, strict=True)):
        group_index = int(alignment["phone_group_index"])
        syllable_index = _group_syllable_index(
            phone_groups[group_index], group_index, len(phone_groups), len(syllable_notes)
        )
        source_note = syllable_notes[syllable_index]
        continuation = previous_syllable == syllable_index
        phone_group = phone_groups[group_index]
        acoustic_internal = list(phone_group["internal_phones"])
        acoustic_phonemes, mapping_notes = mapper.map_internal(acoustic_internal)
        lexical_phonemes = _lyric_phonemes(source_note, syllable_index)
        if continuation:
            # Continuations are driven by the phone-truth group.  Lyric/G2P
            # phonemes remain export evidence and must not override it.
            phonemes = _vowel_nucleus(acoustic_phonemes)
            lyric = "-"
        else:
            phonemes = lexical_phonemes
            lyric = str(source_note.get("lyric", "")).strip()
        if not lyric or lyric.casefold() == "la":
            raise EvidenceFusionError("missing_lyrics_evidence", f"note {index} has placeholder or empty lyric evidence")
        try:
            assert_miku_v4c_note_phonemes_playable(phonemes)
        except (TypeError, ValueError) as exc:
            raise EvidenceFusionError(
                "invalid_phoneme_evidence",
                f"note {index} has unsafe or malformed export phoneme evidence: {exc}",
            ) from exc
        start_sec = float(row["start_sec"])
        duration_sec = float(row["duration_sec"])
        raw_midi = int(row["midi"])
        note = deepcopy(source_note)
        note.update(
            id=f"pitch-note-{index + 1}",
            start_tick=round(start_sec * ticks_per_second),
            duration_ticks=max(1, round(duration_sec * ticks_per_second)),
            pitch=raw_midi + transpose,
            lyric=lyric,
            phonemes=phonemes,
            phoneme_source="phone_segment_truth+lyric_phoneme_lattice",
        )
        raw_provenance = note.get("provenance", {})
        if raw_provenance is None:
            raw_provenance = {}
        if not isinstance(raw_provenance, Mapping):
            raise EvidenceFusionError(
                "invalid_lyrics_evidence",
                f"lyric syllable {syllable_index} provenance must be an object",
            )
        provenance = dict(raw_provenance)
        provenance.update(
            {
                "truth_layer": "phones_ipa_articulatory_features",
                "lyric_source": lyric_source,
                "lyric_syllable_index": syllable_index,
                "continuation_note": continuation,
                "continuation_policy": "first_full_then_dash_vowel_nucleus",
                "phone_group_index": group_index,
                "phone_segment_indices": list(phone_group["member_segment_indices"]),
                # Keep the exact canonical IPA labels alongside the internal
                # approximation used by the Vocaloid export mapper.
                "phone_ipa": list(phone_group["ipa_phones"]),
                "phone_internal": acoustic_internal,
                "phone_mapping_notes": mapping_notes,
                "phone_direct_overlap": bool(alignment["direct_overlap"]),
                "phone_overlap_sec": alignment["overlap_sec"],
                "forced_alignment": dict(phone_group.get("alignment", {})),
                "pitch_candidate_id": row.get("note_id", f"pitch-{index + 1}"),
                "pitch_source_candidate_id": row.get(
                    "_source_pitch_candidate_id", row.get("note_id", f"pitch-{index + 1}")
                ),
                "pitch_boundary_split": row.get("_pitch_boundary_split"),
                "pitch_source": row.get("source"),
                "extracted_midi": raw_midi,
                "render_midi": raw_midi + transpose,
                "pitch_transpose_semitones": transpose,
            }
        )
        note["provenance"] = provenance
        result_notes.append(note)
        previous_syllable = syllable_index

    payload["tempo"] = float(tempo_bpm)
    payload["tracks"][0]["notes"] = result_notes
    fusion_summary = {
        "schema_version": SCHEMA_VERSION,
        "note_count": len(result_notes),
        "phone_group_count": len(phone_groups),
        "lyric_syllable_count": len(syllable_notes),
        "direct_overlap_count": direct_overlap_count,
        "interpolation_count": len(interpolations),
        "direct_overlap_ratio": round(direct_ratio, 6),
        "interpolation_ratio": round(interpolation_ratio, 6),
        "direct_overlap_threshold": direct_overlap_threshold,
        "maximum_interpolation_ratio": maximum_interpolation_ratio,
        "interpolations": interpolations,
        "pitch_boundary_split_count": len(pitch_boundary_splits),
        "pitch_boundary_splits": pitch_boundary_splits,
        "pitch_boundary_policy": "split_only_for_missing_direct_phone_group_coverage",
        "coverage": {
            "pitch_candidate_count": len(pitch_rows),
            "pitch_note_count": len(result_notes),
            "pitch_coverage_ratio": 1.0,
            "phone_group_count": len(phone_groups),
            "phone_group_covered_count": len(represented_groups),
            "phone_group_coverage_ratio": 1.0,
            "phone_segment_count": len(phone_segments),
            "phone_segment_covered_count": len(covered_phone_indices),
            "phone_segment_coverage_ratio": 1.0,
            "lyric_syllable_count": len(syllable_notes),
            "lyric_syllable_covered_count": len(represented_syllables),
            "lyric_coverage_ratio": 1.0,
            "monotonic": True,
        },
        "pitch_coverage_ratio": 1.0,
        "phone_group_coverage_ratio": 1.0,
        "phone_coverage_ratio": 1.0,
        "lyric_coverage_ratio": 1.0,
        "monotonic_alignment": True,
        "continuation_policy": "first_full_then_dash_vowel_nucleus",
        "pitch_skeleton": True,
    }
    payload.setdefault("provenance", {})["audio_evidence_fusion"] = fusion_summary
    payload["provenance"]["pitch_render_policy"] = {
        "enabled": True,
        "floor_midi": 64,
        "transpose_semitones": transpose,
    }
    return payload, fusion_summary


def _syllable_notes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    tracks = payload.get("tracks")
    if not isinstance(tracks, list) or not tracks or not isinstance(tracks[0], dict):
        raise EvidenceFusionError("missing_lyrics_evidence", "draft contains no lyric track")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        raise EvidenceFusionError("missing_lyrics_evidence", "draft provenance must be an object")
    notes = tracks[0].get("notes")
    if not isinstance(notes, list) or any(not isinstance(note, Mapping) for note in notes):
        raise EvidenceFusionError("missing_lyrics_evidence", "draft lyric notes must be an object list")
    result = [dict(note) for note in notes]
    if not result:
        raise EvidenceFusionError("missing_lyrics_evidence", "lyrics did not resolve to syllable evidence")
    for index, note in enumerate(result):
        lyric_value = note.get("lyric")
        if not isinstance(lyric_value, str):
            raise EvidenceFusionError(
                "invalid_lyrics_evidence",
                f"lyric syllable {index} lyric must be a string",
            )
        lyric = lyric_value.strip()
        if not lyric or lyric.casefold() == "la":
            raise EvidenceFusionError(
                "missing_lyrics_evidence",
                f"lyric syllable {index} has placeholder or empty lyric evidence",
            )
    return result


def _phone_groups(segments: Sequence[PhoneSegment]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, segment in enumerate(segments):
        if not isinstance(segment, PhoneSegment):
            raise EvidenceFusionError("invalid_phone_evidence", f"phone segment {index} is not a PhoneSegment")
        if not isinstance(segment.metadata, Mapping):
            raise EvidenceFusionError("invalid_phone_evidence", f"phone segment {index} metadata must be an object")
        nucleus = _phone_label(segment, index)
        if not _is_nucleus_candidate(segments, index, used):
            continue
        members = [index]
        ipa = [nucleus]
        internal = [_NUCLEUS_TO_INTERNAL.get(nucleus, nucleus)]
        if index > 0 and index - 1 not in used:
            onset = _phone_label(segments[index - 1], index - 1)
            onset_gap = segment.time_range.start_sec - segments[index - 1].time_range.end_sec
            if (
                onset in _ONSET_TO_INTERNAL
                and 0.0 <= onset_gap <= 0.18
                and _may_share_lyric_item(segments, index - 1, index)
            ):
                members.insert(0, index - 1)
                ipa.insert(0, onset)
                internal.insert(0, _ONSET_TO_INTERNAL[onset])
        if index + 1 < len(segments) and index + 1 not in used:
            coda = _phone_label(segments[index + 1], index + 1)
            coda_gap = segments[index + 1].time_range.start_sec - segment.time_range.end_sec
            if (
                coda in _CODA_TO_INTERNAL
                and 0.0 <= coda_gap <= 0.12
                and _may_share_lyric_item(segments, index, index + 1)
            ):
                members.append(index + 1)
                ipa.append(coda)
                internal.append(_CODA_TO_INTERNAL[coda])
        used.update(members)
        lyric_indices: set[int] = set()
        for member in members:
            raw_index = _lyric_item_index(segments[member], member)
            if raw_index is None:
                continue
            lyric_indices.add(raw_index)
        if len(lyric_indices) > 1:
            raise EvidenceFusionError(
                "invalid_phone_evidence", f"phone group at segment {index} spans multiple lyric items"
            )
        alignment_rows = [
            {
                key: segments[member].metadata.get(key)
                for key in (
                    "alignment_source", "lyric_item_index", "direct_overlap_sec",
                    "phone_overlap_ratio", "alignment_confidence", "model_hash", "interpolation_reason",
                )
            }
            for member in members
        ]
        groups.append(
            {
                "start_sec": segments[members[0]].time_range.start_sec,
                "end_sec": segments[members[-1]].time_range.end_sec,
                "member_segment_indices": members,
                "ipa_phones": ipa,
                "internal_phones": internal,
                "lyric_item_index": next(iter(lyric_indices)) if lyric_indices else None,
                "alignment": {"segments": alignment_rows} if lyric_indices else {},
            }
        )
    return groups


def _validate_phone_coverage(
    segments: Sequence[PhoneSegment], groups: Sequence[Mapping[str, Any]]
) -> set[int]:
    covered = {
        int(member)
        for group in groups
        for member in group.get("member_segment_indices", [])
    }
    unassigned = [
        index
        for index, segment in enumerate(segments)
        if index not in covered
        and _is_content_phone(_phone_label(segment, index))
    ]
    if unassigned:
        raise EvidenceFusionError(
            "incomplete_phone_alignment",
            f"phone evidence contains unassigned content segments: {unassigned}",
        )
    return covered


def _is_nucleus_candidate(
    segments: Sequence[PhoneSegment], index: int, used: set[int]
) -> bool:
    label = _phone_label(segments[index], index)
    if label in _VOWELS:
        return True
    # Combining syllabic marks are explicit MFA nucleus evidence.  Do not
    # reinterpret m̩/ŋ̩ as a preceding vowel's coda based on timing alone.
    if label in _EXPLICIT_SYLLABIC_NUCLEI:
        return True
    if label not in _SYLLABIC_NUCLEI:
        return False
    # m/ng are ordinary onsets before a vowel.  They are syllabic nuclei only
    # when neither adjacent phone can form the normal onset/coda attachment.
    if index + 1 < len(segments) and index + 1 not in used:
        following = _phone_label(segments[index + 1], index + 1)
        gap = segments[index + 1].time_range.start_sec - segments[index].time_range.end_sec
        if (
            following in _VOWELS
            and 0.0 <= gap <= 0.18
            and _may_share_lyric_item(segments, index, index + 1)
        ):
            return False
    if index > 0 and index - 1 in used:
        previous = _phone_label(segments[index - 1], index - 1)
        gap = segments[index].time_range.start_sec - segments[index - 1].time_range.end_sec
        if (
            previous in _VOWELS
            and 0.0 <= gap <= 0.12
            and _may_share_lyric_item(segments, index - 1, index)
        ):
            return False
    return True


def _may_share_lyric_item(
    segments: Sequence[PhoneSegment], left_index: int, right_index: int
) -> bool:
    left = _lyric_item_index(segments[left_index], left_index)
    right = _lyric_item_index(segments[right_index], right_index)
    return left is None or right is None or left == right


def _lyric_item_index(segment: PhoneSegment, index: int) -> int | None:
    if not isinstance(segment.metadata, Mapping):
        raise EvidenceFusionError(
            "invalid_phone_evidence", f"phone segment {index} metadata must be an object"
        )
    value = segment.metadata.get("lyric_item_index")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceFusionError(
            "invalid_phone_evidence",
            f"phone segment {index} lyric_item_index must be an integer",
        )
    return value


def _phone_label(segment: PhoneSegment, index: int) -> str:
    value = segment.normalized_ipa
    if not isinstance(value, str):
        raise EvidenceFusionError("invalid_phone_evidence", f"phone segment {index} normalized_ipa must be a string")
    return value.strip()


def _is_content_phone(label: str) -> bool:
    return label not in _NON_CONTENT_PHONES and label.casefold() not in {
        value.casefold() for value in _NON_CONTENT_PHONES if value != "O"
    }


def _validate_inputs(
    *,
    base_payload: Mapping[str, Any],
    pitch_rows: Sequence[Mapping[str, Any]],
    phone_segments: Sequence[PhoneSegment],
    lyric_source: str,
    tempo_bpm: float,
    direct_overlap_threshold: float,
    maximum_interpolation_ratio: float,
) -> tuple[float, float, float]:
    if not isinstance(lyric_source, str) or not lyric_source.strip():
        raise EvidenceFusionError("missing_lyrics_evidence", "lyric evidence source must be named")
    if any(
        isinstance(value, bool)
        for value in (tempo_bpm, direct_overlap_threshold, maximum_interpolation_ratio)
    ):
        raise EvidenceFusionError("invalid_evidence_threshold", "tempo and ratios must be numeric")
    try:
        normalized_tempo = float(tempo_bpm)
        normalized_direct = float(direct_overlap_threshold)
        normalized_interpolation = float(maximum_interpolation_ratio)
    except (OverflowError, TypeError, ValueError) as exc:
        raise EvidenceFusionError("invalid_evidence_threshold", "tempo and ratios must be numeric") from exc
    if not math.isfinite(normalized_tempo) or normalized_tempo <= 0.0:
        raise EvidenceFusionError("invalid_pitch_evidence", "tempo must be finite and positive")
    ticks_per_quarter = base_payload.get("ticks_per_quarter", 480)
    if isinstance(ticks_per_quarter, bool) or not isinstance(ticks_per_quarter, int) or ticks_per_quarter <= 0:
        raise EvidenceFusionError("invalid_pitch_evidence", "ticks_per_quarter must be a positive integer")
    if not math.isfinite(normalized_direct) or not 0.0 <= normalized_direct <= 1.0:
        raise EvidenceFusionError("invalid_evidence_threshold", "direct overlap threshold must be between zero and one")
    if not math.isfinite(normalized_interpolation) or not 0.0 <= normalized_interpolation <= 1.0:
        raise EvidenceFusionError("invalid_evidence_threshold", "interpolation ratio limit must be between zero and one")
    if normalized_direct < _MIN_DIRECT_OVERLAP_THRESHOLD:
        raise EvidenceFusionError(
            "invalid_evidence_threshold",
            f"direct overlap threshold cannot be below {_MIN_DIRECT_OVERLAP_THRESHOLD:.2f}",
        )
    if normalized_interpolation > _MAX_INTERPOLATION_RATIO:
        raise EvidenceFusionError(
            "invalid_evidence_threshold",
            f"interpolation ratio limit cannot exceed {_MAX_INTERPOLATION_RATIO:.2f}",
        )

    previous_start = -1.0
    for index, row in enumerate(pitch_rows):
        try:
            start = float(row["start_sec"])
            duration = float(row["duration_sec"])
            midi_value = row["midi"]
            midi_number = float(midi_value)
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise EvidenceFusionError("invalid_pitch_evidence", f"pitch candidate {index} is malformed") from exc
        if (
            isinstance(row["start_sec"], bool)
            or isinstance(row["duration_sec"], bool)
            or not math.isfinite(start)
            or not math.isfinite(duration)
            or not math.isfinite(start + duration)
            or start < 0.0
            or duration <= 0.0
            or isinstance(midi_value, bool)
            or not isinstance(midi_value, int)
            or not math.isfinite(midi_number)
            or not midi_number.is_integer()
            or not 0 <= midi_number <= 127
        ):
            raise EvidenceFusionError("invalid_pitch_evidence", f"pitch candidate {index} has invalid values")
        if start < previous_start:
            raise EvidenceFusionError("invalid_pitch_evidence", "pitch candidates must be ordered by start time")
        previous_start = start

    previous_start = -1.0
    previous_end = -1.0
    for index, segment in enumerate(phone_segments):
        if not isinstance(segment, PhoneSegment):
            raise EvidenceFusionError("invalid_phone_evidence", f"phone segment {index} is not a PhoneSegment")
        try:
            start = float(segment.time_range.start_sec)
            end = float(segment.time_range.end_sec)
        except (AttributeError, TypeError, ValueError) as exc:
            raise EvidenceFusionError("invalid_phone_evidence", f"phone segment {index} has invalid timing") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0.0
            or end <= start
            or start < previous_start
            or start < previous_end
        ):
            raise EvidenceFusionError(
                "invalid_phone_evidence", "phone segments must be ordered and non-overlapping"
            )
        previous_start = start
        previous_end = end
    return normalized_tempo, normalized_direct, normalized_interpolation


def _align_pitch_rows(
    rows: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aligned: list[dict[str, Any]] = []
    interpolations: list[dict[str, Any]] = []
    minimum_group = 0
    previous_group: int | None = None
    for index, row in enumerate(rows):
        start = float(row["start_sec"])
        end = start + float(row["duration_sec"])
        best_index: int | None = None
        best_overlap = 0.0
        for group_index in range(minimum_group, len(groups)):
            group = groups[group_index]
            overlap = max(0.0, min(end, float(group["end_sec"])) - max(start, float(group["start_sec"])))
            if overlap > best_overlap:
                best_index, best_overlap = group_index, overlap
        direct = best_index is not None and best_overlap > 0.0
        if best_index is None:
            if minimum_group >= len(groups):
                raise EvidenceFusionError(
                    "incomplete_phone_alignment",
                    f"pitch candidate {index} has no monotonic phone group available",
                )
            center = (start + end) / 2.0
            best_index = min(
                range(minimum_group, len(groups)),
                key=lambda value: abs(center - (float(groups[value]["start_sec"]) + float(groups[value]["end_sec"])) / 2.0),
            )
            group_center = (float(groups[best_index]["start_sec"]) + float(groups[best_index]["end_sec"])) / 2.0
            interpolations.append(
                {
                    "pitch_index": index,
                    "pitch_candidate_id": row.get("note_id", f"pitch-{index + 1}"),
                    "phone_group_index": best_index,
                    "distance_sec": round(abs(center - group_center), 6),
                    "reason": "no_direct_time_overlap_nearest_monotonic_phone_group",
                }
            )
        aligned.append(
            {
                "phone_group_index": best_index,
                "direct_overlap": direct,
                "overlap_sec": round(best_overlap, 6),
            }
        )
        if previous_group is not None and best_index < previous_group:
            raise EvidenceFusionError(
                "non_monotonic_alignment",
                "pitch candidates must map to non-decreasing phone groups",
            )
        previous_group = best_index
        minimum_group = best_index
    return aligned, interpolations


def _split_pitch_rows_for_group_coverage(
    rows: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = [dict(row) for row in rows]
    aligned, _ = _align_pitch_rows(source_rows, groups)
    represented = {int(item["phone_group_index"]) for item in aligned}
    missing = sorted(set(range(len(groups))) - represented)
    if not missing:
        return source_rows, []

    split_groups_by_row: dict[int, set[int]] = {}
    for group_index in missing:
        group = groups[group_index]
        group_start = float(group["start_sec"])
        group_end = float(group["end_sec"])
        candidates: list[tuple[float, int]] = []
        for row_index, row in enumerate(source_rows):
            row_start = float(row["start_sec"])
            row_end = row_start + float(row["duration_sec"])
            overlap = max(0.0, min(row_end, group_end) - max(row_start, group_start))
            if overlap > 0.0:
                candidates.append((overlap, row_index))
        if not candidates:
            continue
        _, selected_row = max(candidates, key=lambda item: (item[0], -item[1]))
        split_groups_by_row.setdefault(selected_row, set()).add(group_index)

    rendered: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row_index, row in enumerate(source_rows):
        trigger_groups = sorted(split_groups_by_row.get(row_index, set()))
        source_id = str(row.get("note_id", f"pitch-{row_index + 1}"))
        if not trigger_groups:
            rendered.append(row)
            continue
        row_start = float(row["start_sec"])
        row_end = row_start + float(row["duration_sec"])
        boundaries = {
            boundary
            for group_index in trigger_groups
            for boundary in (
                float(groups[group_index]["start_sec"]),
                float(groups[group_index]["end_sec"]),
            )
            if row_start < boundary < row_end
        }
        if not boundaries:
            rendered.append(row)
            continue
        points = [row_start, *sorted(boundaries), row_end]
        generated_ids: list[str] = []
        for part_index, (part_start, part_end) in enumerate(zip(points, points[1:])):
            if part_end <= part_start:
                continue
            part = dict(row)
            part_id = f"{source_id}:phone-split-{part_index + 1}"
            part.update(
                {
                    "note_id": part_id,
                    "start_sec": part_start,
                    "end_sec": part_end,
                    "duration_sec": part_end - part_start,
                    "_source_pitch_candidate_id": source_id,
                    "_pitch_boundary_split": {
                        "part_index": part_index,
                        "part_count": len(points) - 1,
                        "source_start_sec": row_start,
                        "source_end_sec": row_end,
                    },
                }
            )
            rendered.append(part)
            generated_ids.append(part_id)
        audit.append(
            {
                "source_pitch_index": row_index,
                "source_pitch_candidate_id": source_id,
                "source_start_sec": row_start,
                "source_end_sec": row_end,
                "trigger_phone_group_indices": trigger_groups,
                "split_boundaries_sec": sorted(boundaries),
                "generated_pitch_candidate_ids": generated_ids,
                "reason": "one_pitch_candidate_directly_overlaps_phone_groups_missing_from_primary_assignment",
                "interpolation": False,
            }
        )
    return rendered, audit


def _monotonic_syllable_index(group_index: int, group_count: int, syllable_count: int) -> int:
    if group_count <= 1 or syllable_count <= 1:
        return 0
    return min(syllable_count - 1, math.floor(group_index * syllable_count / group_count))


def _group_syllable_index(
    group: Mapping[str, Any], group_index: int, group_count: int, syllable_count: int
) -> int:
    explicit = group.get("lyric_item_index")
    if explicit is None:
        return _monotonic_syllable_index(group_index, group_count, syllable_count)
    if isinstance(explicit, bool) or not isinstance(explicit, int):
        raise EvidenceFusionError("invalid_phone_evidence", "phone group lyric_item_index must be an integer")
    try:
        value = int(explicit)
    except (TypeError, ValueError) as exc:
        raise EvidenceFusionError("invalid_phone_evidence", "phone group lyric_item_index must be an integer") from exc
    if not 0 <= value < syllable_count:
        raise EvidenceFusionError(
            "invalid_phone_evidence", f"phone group lyric_item_index {value} is outside the lyric timeline"
        )
    return value


def _validate_monotonic_syllable_groups(
    groups: Sequence[Mapping[str, Any]], syllable_count: int
) -> None:
    previous = -1
    seen: set[int] = set()
    for index, group in enumerate(groups):
        value = _group_syllable_index(group, index, len(groups), syllable_count)
        if value < previous:
            raise EvidenceFusionError(
                "non_monotonic_alignment",
                "phone groups must map to non-decreasing lyric syllables",
            )
        previous = value
        seen.add(value)
    if seen != set(range(syllable_count)):
        missing = sorted(set(range(syllable_count)) - seen)
        raise EvidenceFusionError(
            "incomplete_lyrics_alignment",
            f"phone groups do not cover lyric syllables: missing indices {missing}",
        )


def _validate_monotonic_alignment(
    aligned: Sequence[Mapping[str, Any]], group_count: int
) -> None:
    previous = -1
    for index, item in enumerate(aligned):
        try:
            value = int(item["phone_group_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceFusionError(
                "invalid_phone_evidence", f"alignment row {index} has no valid phone group index"
            ) from exc
        if not 0 <= value < group_count or value < previous:
            raise EvidenceFusionError(
                "non_monotonic_alignment",
                "pitch candidates must map to non-decreasing phone groups",
            )
        previous = value


def _lyric_phonemes(note: Mapping[str, Any], syllable_index: int) -> list[str]:
    raw = note.get("phonemes")
    if isinstance(raw, str):
        phonemes = [part for part in raw.split() if part.strip()]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if any(not isinstance(value, str) for value in raw):
            raise EvidenceFusionError(
                "invalid_phoneme_evidence",
                f"lyric syllable {syllable_index} phonemes must contain strings",
            )
        phonemes = [value.strip() for value in raw if value.strip()]
    else:
        phonemes = []
    if not phonemes:
        raise EvidenceFusionError(
            "missing_phone_evidence", f"lyric syllable {syllable_index} has no export phoneme evidence"
        )
    return phonemes


def _pitch_transpose(rows: Sequence[Mapping[str, Any]], floor_midi: int = 64) -> int:
    values = [int(row["midi"]) for row in rows]
    transpose = max(0, floor_midi - min(values))
    if max(values) + transpose > 127:
        raise EvidenceFusionError("invalid_pitch_evidence", "pitch safety transposition exceeds MIDI 127")
    return transpose


def _vowel_nucleus(acoustic: Sequence[str]) -> list[str]:
    consonants = {
        "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "z", "c", "s", "j", "w", "N", "ng", "ts", "x",
    }
    for token in acoustic:
        text = str(token).strip()
        if text and text not in consonants:
            return [text]
    if len(acoustic) == 1:
        text = str(acoustic[0]).strip()
        if text in _SYLLABIC_NUCLEI:
            return [text]
    raise EvidenceFusionError("missing_phone_evidence", "continuation note has no vowel nucleus")


__all__ = ["SCHEMA_VERSION", "EvidenceFusionError", "build_pitch_skeleton_payload"]

from __future__ import annotations

from dataclasses import asdict
import re

from any2xsampa.core.models import PhoneSegment, SyllableGroup
from any2xsampa.pipeline.future_aligner_sequence_hint import (
    future_aligner_sequence_hint_payload,
    resolve_future_aligner_sequence_hint_identity,
)


class SyllableGrouper:
    def __init__(
        self,
        vowels: set[str],
        pause_phones: set[str],
        nasal_codas: set[str],
        stop_codas: set[str],
    ) -> None:
        self.vowels = vowels
        self.pause_phones = pause_phones
        self.nasal_codas = nasal_codas
        self.stop_codas = stop_codas

    def group(self, segments: list[PhoneSegment]) -> list[SyllableGroup]:
        groups: list[SyllableGroup] = []
        index = 0
        group_number = 0
        while index < len(segments):
            ipa = self._ipa(segments[index])
            if ipa in self.pause_phones:
                group = SyllableGroup(
                    group_id=f"g{group_number}",
                    member_segment_indices=[index],
                    nucleus="pau",
                    nucleus_segment_index=index,
                    lyric_token=self._lyric_token(segments[index]),
                    grouping_notes=["pause/rest group"],
                    grouping_fallback_used=False,
                )
                groups.append(group)
                index += 1
                group_number += 1
                continue

            group, next_index = self._build_voiced_group(segments, index, group_number)
            groups.append(group)
            index = next_index
            group_number += 1

        self._annotate_segments(segments, groups)
        return groups

    def _build_voiced_group(
        self,
        segments: list[PhoneSegment],
        start_index: int,
        group_number: int,
    ) -> tuple[SyllableGroup, int]:
        notes: list[str] = []
        identity_notes: list[str] = []
        fallback_used = False
        lyric_token = self._lyric_token(segments[start_index])
        onset_indices: list[int] = []
        nucleus_index: int | None = None
        coda_index: int | None = None
        score_identity_hint_applied = False
        score_identity_boundary_hit = False
        grouping_identity_fallback_used = False
        identity_reference_index = start_index
        index = start_index

        while index < len(segments):
            ipa = self._ipa(segments[index])
            if ipa in self.pause_phones:
                break
            if index != start_index:
                identity_result = self._identity_hint(segments[identity_reference_index], segments[index])
                self._extend_unique(identity_notes, identity_result["notes"])
                score_identity_hint_applied = score_identity_hint_applied or bool(identity_result["hint_applied"])
                score_identity_boundary_hit = score_identity_boundary_hit or bool(identity_result["boundary_hit"])
                grouping_identity_fallback_used = grouping_identity_fallback_used or bool(
                    identity_result["fallback_used"]
                )
                if identity_result["boundary_hit"]:
                    break
            if nucleus_index is None and ipa not in self.vowels:
                if self._token_conflict(segments[index], lyric_token):
                    notes.append("stopped onset scan at lyric token boundary")
                    break
                onset_indices.append(index)
                identity_reference_index = self._preferred_identity_reference_index(
                    segments,
                    identity_reference_index,
                    index,
                )
                if self._is_coda(ipa):
                    notes.append("encountered coda-like phone before nucleus; using fallback grouping")
                    fallback_used = True
                    break
                index += 1
                continue
            if ipa in self.vowels:
                nucleus_index = index
                identity_reference_index = self._preferred_identity_reference_index(
                    segments,
                    identity_reference_index,
                    index,
                )
                index += 1
                break
            break

        if nucleus_index is None:
            member_indices = [start_index]
            notes.append("no stable nucleus found; fallback to single-segment group")
            grouping_identity_fallback_used = self._finalize_identity_fallback(
                segments,
                member_indices,
                grouping_identity_fallback_used,
                identity_notes,
            )
            return (
                SyllableGroup(
                    group_id=f"g{group_number}",
                    member_segment_indices=member_indices,
                    onset=self._join_phones(segments, member_indices) if onset_indices else self._ipa(segments[start_index]),
                    nucleus=None,
                    coda=None,
                    onset_segment_indices=list(member_indices),
                    nucleus_segment_index=None,
                    coda_segment_indices=[],
                    lyric_token=lyric_token,
                    grouping_notes=notes,
                    grouping_fallback_used=True,
                    score_identity_hint_applied=score_identity_hint_applied,
                    score_identity_boundary_hit=score_identity_boundary_hit,
                    grouping_identity_fallback_used=grouping_identity_fallback_used,
                    grouping_identity_notes=identity_notes,
                ),
                start_index + 1,
            )

        if index < len(segments):
            ipa = self._ipa(segments[index])
            if self._is_coda(ipa):
                identity_result = self._identity_hint(segments[nucleus_index], segments[index])
                self._extend_unique(identity_notes, identity_result["notes"])
                score_identity_hint_applied = score_identity_hint_applied or bool(identity_result["hint_applied"])
                score_identity_boundary_hit = score_identity_boundary_hit or bool(identity_result["boundary_hit"])
                grouping_identity_fallback_used = grouping_identity_fallback_used or bool(
                    identity_result["fallback_used"]
                )
                if not identity_result["boundary_hit"]:
                    coda_index = index
                    index += 1
                    if self._token_conflict(segments[coda_index], lyric_token):
                        notes.append("attached coda across lyric token boundary")
                        fallback_used = True
                    if ipa in self.nasal_codas:
                        notes.append("nasal coda attached to syllable")
                    else:
                        notes.append("stop coda attached to syllable")

        member_indices = [*onset_indices, nucleus_index]
        if coda_index is not None:
            member_indices.append(coda_index)
        if lyric_token is not None and len({self._lyric_token(segments[i]) for i in member_indices if self._lyric_token(segments[i])}) == 1:
            notes.append("grouping stabilized by shared lyric token")
        elif lyric_token is not None:
            notes.append("grouping crossed missing or differing lyric token metadata")
            fallback_used = True
        if onset_indices:
            notes.append("cv(c) grouping built from onset+nucleus context")
        else:
            notes.append("v(c) grouping built without onset")
        grouping_identity_fallback_used = self._finalize_identity_fallback(
            segments,
            member_indices,
            grouping_identity_fallback_used,
            identity_notes,
        )
        return (
            SyllableGroup(
                group_id=f"g{group_number}",
                member_segment_indices=member_indices,
                onset=self._join_phones(segments, onset_indices),
                nucleus=self._ipa(segments[nucleus_index]),
                coda=self._ipa(segments[coda_index]) if coda_index is not None else None,
                onset_segment_indices=list(onset_indices),
                nucleus_segment_index=nucleus_index,
                coda_segment_indices=[coda_index] if coda_index is not None else [],
                lyric_token=lyric_token,
                grouping_notes=notes,
                grouping_fallback_used=fallback_used,
                score_identity_hint_applied=score_identity_hint_applied,
                score_identity_boundary_hit=score_identity_boundary_hit,
                grouping_identity_fallback_used=grouping_identity_fallback_used,
                grouping_identity_notes=identity_notes,
            ),
            index,
        )

    def _annotate_segments(self, segments: list[PhoneSegment], groups: list[SyllableGroup]) -> None:
        for group in groups:
            payload = asdict(group)
            for index in group.member_segment_indices:
                segments[index].syllable_group = group
                segments[index].metadata["syllable_group"] = payload
                segments[index].metadata["syllable_group_id"] = group.group_id
                segments[index].metadata["syllable_grouping_notes"] = list(group.grouping_notes)
                segments[index].metadata["syllable_grouping_fallback_used"] = group.grouping_fallback_used
                segments[index].metadata["score_identity_hint_applied"] = group.score_identity_hint_applied
                segments[index].metadata["score_identity_boundary_hit"] = group.score_identity_boundary_hit
                segments[index].metadata["grouping_identity_fallback_used"] = group.grouping_identity_fallback_used
                segments[index].metadata["grouping_identity_notes"] = list(group.grouping_identity_notes)

    def _ipa(self, segment: PhoneSegment) -> str:
        return segment.normalized_ipa or "sil"

    def _lyric_token(self, segment: PhoneSegment) -> str | None:
        token = segment.metadata.get("lyric_token")
        return str(token) if token else None

    def _token_conflict(self, segment: PhoneSegment, lyric_token: str | None) -> bool:
        if lyric_token is None:
            return False
        other = self._lyric_token(segment)
        return other is not None and other != lyric_token

    def _is_coda(self, ipa: str) -> bool:
        return ipa in self.nasal_codas or ipa in self.stop_codas

    def _join_phones(self, segments: list[PhoneSegment], indices: list[int]) -> str | None:
        if not indices:
            return None
        return "+".join(self._ipa(segments[index]) for index in indices)

    def _identity_hint(self, reference: PhoneSegment, candidate: PhoneSegment) -> dict[str, object]:
        reference_identity = self._segment_identity(reference)
        candidate_identity = self._segment_identity(candidate)
        notes: list[str] = []
        hint_applied = False
        boundary_hit = False
        fallback_used = False
        self._extend_unique(notes, self._sequence_hint_notes(reference, candidate))

        if not self._identity_rank(reference_identity) and not self._identity_rank(candidate_identity):
            return {
                "hint_applied": False,
                "boundary_hit": False,
                "fallback_used": True,
                "notes": ["no normalized score identity available; kept time-ordered phonetic grouping"],
            }

        for field in ("track_id", "part_id"):
            reference_value = reference_identity[field]
            candidate_value = candidate_identity[field]
            if reference_value and candidate_value:
                hint_applied = True
                if reference_value != candidate_value:
                    boundary_hit = True
                    notes.append(
                        f"respected {field} boundary ({reference_value} -> {candidate_value})"
                    )
                    return {
                        "hint_applied": hint_applied,
                        "boundary_hit": boundary_hit,
                        "fallback_used": fallback_used,
                        "notes": notes,
                    }
                notes.append(f"shared {field}={reference_value}")
                continue
            if reference_value or candidate_value:
                fallback_used = True
                notes.append(f"{field} missing on one side; fell back to lyric/phonetic grouping")

        reference_note_id = reference_identity["note_id"]
        candidate_note_id = candidate_identity["note_id"]
        if reference_note_id and candidate_note_id:
            hint_applied = True
            if reference_note_id != candidate_note_id:
                boundary_hit = True
                if self._note_ids_are_adjacent(reference_note_id, candidate_note_id):
                    notes.append(
                        f"respected adjacent note_id boundary ({reference_note_id} -> {candidate_note_id})"
                    )
                else:
                    notes.append(f"respected note_id boundary ({reference_note_id} -> {candidate_note_id})")
                return {
                    "hint_applied": hint_applied,
                    "boundary_hit": boundary_hit,
                    "fallback_used": fallback_used,
                    "notes": notes,
                }
            notes.append(f"shared note_id={reference_note_id}")
        elif reference_note_id or candidate_note_id:
            fallback_used = True
            notes.append("note_id missing on one side; fell back to track/part or phonetic grouping")

        return {
            "hint_applied": hint_applied,
            "boundary_hit": boundary_hit,
            "fallback_used": fallback_used,
            "notes": notes,
        }

    def _segment_identity(self, segment: PhoneSegment) -> dict[str, str | None]:
        return resolve_future_aligner_sequence_hint_identity(segment, consumer="grouping")

    def _sequence_hint_notes(self, reference: PhoneSegment, candidate: PhoneSegment) -> list[str]:
        notes: list[str] = []
        for segment in (reference, candidate):
            payload = future_aligner_sequence_hint_payload(segment)
            if not payload:
                continue
            aligned_note_id = payload.get("aligned_note_id")
            if isinstance(aligned_note_id, str) and aligned_note_id:
                notes.append(f"grouping read future_aligner_sequence_hint aligned_note_id={aligned_note_id}")
                continue
            if bool(payload.get("aligner_fallback_used", False)):
                notes.append("grouping observed conservative future_aligner_sequence_hint fallback")
        return notes

    def _identity_rank(self, identity: dict[str, str | None]) -> int:
        return sum(1 for value in identity.values() if value)

    def _preferred_identity_reference_index(
        self,
        segments: list[PhoneSegment],
        current_index: int,
        candidate_index: int,
    ) -> int:
        current_rank = self._identity_rank(self._segment_identity(segments[current_index]))
        candidate_rank = self._identity_rank(self._segment_identity(segments[candidate_index]))
        if candidate_rank > current_rank:
            return candidate_index
        return current_index

    def _finalize_identity_fallback(
        self,
        segments: list[PhoneSegment],
        member_indices: list[int],
        grouping_identity_fallback_used: bool,
        identity_notes: list[str],
    ) -> bool:
        if grouping_identity_fallback_used:
            return True
        if any(self._identity_rank(self._segment_identity(segments[index])) for index in member_indices):
            return False
        self._extend_unique(
            identity_notes,
            ["no normalized score identity available; grouping stayed on legacy lyric/time-order hints"],
        )
        return True

    def _note_ids_are_adjacent(self, left: str, right: str) -> bool:
        left_match = re.search(r"^(.*?)(\d+)$", left)
        right_match = re.search(r"^(.*?)(\d+)$", right)
        if left_match is None or right_match is None:
            return False
        if left_match.group(1) != right_match.group(1):
            return False
        return abs(int(left_match.group(2)) - int(right_match.group(2))) == 1

    def _extend_unique(self, target: list[str], values: list[str]) -> None:
        for value in values:
            if value not in target:
                target.append(value)

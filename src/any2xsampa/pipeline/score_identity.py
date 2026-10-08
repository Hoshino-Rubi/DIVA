from __future__ import annotations

import re
from dataclasses import dataclass, field

from any2xsampa.core.models import PhoneSegment


@dataclass(slots=True)
class SegmentScoreIdentity:
    normalized_track_id: str | None
    normalized_part_id: str | None
    normalized_note_id: str | None
    score_identity_source: str | None
    score_identity_fallback_used: bool


@dataclass(slots=True)
class ScoreBridgeResult:
    applied: bool
    strategy: str
    match_confidence: float
    notes: list[str] = field(default_factory=list)
    identity_payload: dict[str, object] | None = None
    tiebreak: str = "none"
    candidate_count: int = 0
    adjacency_used: bool = False
    part_continuity_used: bool = False
    rank_used: bool = False
    rank_consistency: str = "unavailable"
    order_score: float = 0.0
    window_size: int = 1
    monotonic_applied: bool = False
    bidirectional_applied: bool = False
    backward_context_used: bool = False
    sequence_score: float = 0.0
    order_violation_penalty: float = 0.0


@dataclass(slots=True)
class ScoreBridgeCandidate:
    track_id: str
    part_id: str
    note_id: str
    start_sec: float | None
    end_sec: float | None
    tick: int | None
    beat: float | None
    part_local_note_rank: int | None = None
    part_local_rank_source: str | None = None
    part_local_rank_fallback_used: bool = False
    slots: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ScoreBridgeChoice:
    candidate: ScoreBridgeCandidate
    overlap_ratio: float
    total_score: float
    order_score: float
    tiebreak: str
    notes: list[str]
    adjacency_used: bool
    part_continuity_used: bool
    rank_used: bool = False
    rank_consistency: str = "unavailable"
    sequence_score: float = 0.0
    order_violation_penalty: float = 0.0
    monotonic_applied: bool = False
    bidirectional_applied: bool = False
    backward_context_used: bool = False
    window_size: int = 1
    selection_score: float = 0.0


@dataclass(slots=True)
class ScoreBridgePathState:
    choice: ScoreBridgeChoice
    sequence_score: float
    continuity_score: float
    order_violation_penalty: float
    sequence_notes: list[str]
    adjacency_used: bool
    part_continuity_used: bool
    backward_context_used: bool
    rank_used: bool
    rank_consistency: str
    tiebreak: str


class ScoreBridgeIdentityProducer:
    def __init__(
        self,
        overlap_threshold: float = 0.6,
        candidate_overlap_threshold: float = 0.15,
        max_window_size: int = 5,
        bidirectional_smoothing: bool = True,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.candidate_overlap_threshold = candidate_overlap_threshold
        self.max_window_size = max(3, max_window_size)
        self.bidirectional_smoothing = bidirectional_smoothing

    def apply(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        for index, segment in enumerate(segments):
            result = self._produce(segments, index)
            self._annotate(segment, result)
        return segments

    def _produce(self, segments: list[PhoneSegment], index: int) -> ScoreBridgeResult:
        segment = segments[index]
        metadata = segment.metadata
        if isinstance(metadata.get("score_bridge_identity"), dict):
            return ScoreBridgeResult(
                applied=True,
                strategy="preexisting_bridge_identity",
                match_confidence=1.0,
                notes=["preserved preexisting score_bridge_identity metadata"],
                identity_payload=dict(metadata["score_bridge_identity"]),
                tiebreak="preexisting_identity",
                candidate_count=1,
                rank_used=False,
                rank_consistency="unavailable",
            )
        score_identity = metadata.get("score_identity")
        if isinstance(score_identity, dict) and self._mapping_has_identity(score_identity):
            return ScoreBridgeResult(
                applied=False,
                strategy="preexisting_score_identity",
                match_confidence=1.0,
                notes=["segment already carries normalized score_identity metadata"],
                tiebreak="preexisting_identity",
                candidate_count=1,
                rank_used=False,
                rank_consistency="unavailable",
            )

        note_context = metadata.get("note_context")
        if not isinstance(note_context, dict) or not metadata.get("note_context_applied", False):
            return ScoreBridgeResult(
                applied=False,
                strategy="no_note_context",
                match_confidence=0.0,
                notes=["score bridge requires an attached note context"],
            )

        track_id = self._value(note_context, "normalized_track_id", "track_id")
        part_id = self._value(note_context, "normalized_part_id", "part_id")
        note_id = self._value(note_context, "normalized_note_id", "note_id")
        if not all((track_id, part_id, note_id)):
            return ScoreBridgeResult(
                applied=False,
                strategy="missing_note_identity",
                match_confidence=0.0,
                notes=["note context lacks stable normalized track/part/note identity"],
            )

        match_strategy = self._value(note_context, "match_strategy") or "unknown"
        if match_strategy == "segment_index":
            return self._bridge_success(
                track_id,
                part_id,
                note_id,
                strategy="segment_index_bridge",
                confidence=1.0,
                notes=["score bridge derived from segment_index-bound note context"],
                tiebreak="segment_index",
                candidate_count=1,
                adjacency_used=False,
                part_continuity_used=False,
                rank_used=False,
                rank_consistency="unavailable",
                order_score=0.0,
                window_size=1,
                monotonic_applied=False,
                bidirectional_applied=False,
                backward_context_used=False,
                sequence_score=1.0,
                order_violation_penalty=0.0,
            )

        if match_strategy != "time_overlap":
            return ScoreBridgeResult(
                applied=False,
                strategy="unsupported_note_binding",
                match_confidence=0.0,
                notes=[f"score bridge does not derive from note binding strategy={match_strategy}"],
                tiebreak="unsupported",
            )

        window_candidates = self._window_candidate_pool(segments, index)
        current_offset = next(
            (offset for offset, (window_index, _) in enumerate(window_candidates) if window_index == index),
            None,
        )
        candidates = window_candidates[current_offset][1] if window_candidates and current_offset is not None else []
        if not candidates:
            overlap_ratio = self._overlap_ratio(segment, note_context)
            return ScoreBridgeResult(
                applied=False,
                strategy="time_overlap_fallback",
                match_confidence=overlap_ratio,
                notes=[
                    f"time overlap ratio {overlap_ratio:.2f} below bridge threshold {self.overlap_threshold:.2f}",
                    "score bridge found no stable identity-compatible candidates",
                ],
                tiebreak="fallback_no_candidates",
                candidate_count=0,
                window_size=0,
                rank_used=False,
                rank_consistency="unavailable",
            )

        choice = self._choose_candidate(segments, index, window_candidates)
        if choice is None:
            best_overlap = max(self._overlap_ratio(segment, self._candidate_note_context(candidate)) for candidate in candidates)
            return ScoreBridgeResult(
                applied=False,
                strategy="time_overlap_fallback",
                match_confidence=best_overlap,
                notes=[
                    f"time overlap ratio={best_overlap:.2f}",
                    "score bridge kept fallback behavior to avoid unstable identity",
                    f"candidate pool size={len(candidates)}",
                ],
                tiebreak="fallback_ambiguous",
                candidate_count=len(candidates),
                window_size=len(window_candidates),
                rank_used=False,
                rank_consistency="unavailable",
            )

        notes = list(choice.notes)
        lyric_token = self._value(metadata, "lyric_token")
        if lyric_token:
            notes.append(f"lyric_token_hint={lyric_token}")
        if choice.adjacency_used and self._adjacent_same_note(segments, index, choice.candidate.note_id):
            notes.append("adjacent segment shares bridged note_id")
        if choice.monotonic_applied:
            notes.append(f"short-window monotonic binding stabilized {choice.window_size}-segment window")
        return self._bridge_success(
            choice.candidate.track_id,
            choice.candidate.part_id,
            choice.candidate.note_id,
            strategy="time_overlap_bridge",
            confidence=min(1.0, choice.overlap_ratio + max(0.0, choice.order_score)),
            notes=notes,
            tiebreak=choice.tiebreak,
            candidate_count=len(candidates),
            adjacency_used=choice.adjacency_used,
            part_continuity_used=choice.part_continuity_used,
            rank_used=choice.rank_used,
            rank_consistency=choice.rank_consistency,
            order_score=choice.order_score,
                window_size=choice.window_size,
                monotonic_applied=choice.monotonic_applied,
                bidirectional_applied=choice.bidirectional_applied,
                backward_context_used=choice.backward_context_used,
            sequence_score=choice.sequence_score,
            order_violation_penalty=choice.order_violation_penalty,
        )

    def _bridge_success(
        self,
        track_id: str,
        part_id: str,
        note_id: str,
        strategy: str,
        confidence: float,
        notes: list[str],
        tiebreak: str,
        candidate_count: int,
        adjacency_used: bool,
        part_continuity_used: bool,
        rank_used: bool,
        rank_consistency: str,
        order_score: float,
        window_size: int,
        monotonic_applied: bool,
        bidirectional_applied: bool,
        backward_context_used: bool,
        sequence_score: float,
        order_violation_penalty: float,
    ) -> ScoreBridgeResult:
        return ScoreBridgeResult(
            applied=True,
            strategy=strategy,
            match_confidence=confidence,
            notes=notes,
            tiebreak=tiebreak,
            candidate_count=candidate_count,
            adjacency_used=adjacency_used,
            part_continuity_used=part_continuity_used,
            rank_used=rank_used,
            rank_consistency=rank_consistency,
            order_score=order_score,
            window_size=window_size,
            monotonic_applied=monotonic_applied,
            bidirectional_applied=bidirectional_applied,
            backward_context_used=backward_context_used,
            sequence_score=sequence_score,
            order_violation_penalty=order_violation_penalty,
            identity_payload={
                "normalized_track_id": track_id,
                "normalized_part_id": part_id,
                "normalized_note_id": note_id,
                "track_id": track_id,
                "part_id": part_id,
                "note_id": note_id,
                "source": f"score_bridge:{strategy}",
                "fallback_used": False,
            },
        )

    def _annotate(self, segment: PhoneSegment, result: ScoreBridgeResult) -> None:
        segment.metadata["score_bridge_applied"] = result.applied
        segment.metadata["score_bridge_strategy"] = result.strategy
        segment.metadata["score_bridge_match_confidence"] = round(result.match_confidence, 3)
        segment.metadata["score_bridge_tiebreak"] = result.tiebreak
        segment.metadata["score_bridge_candidate_count"] = result.candidate_count
        segment.metadata["score_bridge_adjacency_used"] = result.adjacency_used
        segment.metadata["score_bridge_part_continuity_used"] = result.part_continuity_used
        segment.metadata["score_bridge_rank_used"] = result.rank_used
        segment.metadata["score_bridge_rank_consistency"] = result.rank_consistency
        segment.metadata["score_bridge_order_score"] = round(result.order_score, 3)
        segment.metadata["score_bridge_window_size"] = result.window_size
        segment.metadata["score_bridge_monotonic_applied"] = result.monotonic_applied
        segment.metadata["score_bridge_bidirectional_applied"] = result.bidirectional_applied
        segment.metadata["score_bridge_backward_context_used"] = result.backward_context_used
        segment.metadata["score_bridge_sequence_score"] = round(result.sequence_score, 3)
        segment.metadata["score_bridge_order_violation_penalty"] = round(result.order_violation_penalty, 3)
        segment.metadata["score_bridge_notes"] = list(result.notes)
        if result.identity_payload is not None:
            segment.metadata["score_bridge_identity"] = dict(result.identity_payload)

    def _window_candidate_pool(
        self,
        segments: list[PhoneSegment],
        index: int,
    ) -> list[tuple[int, list[ScoreBridgeCandidate]]]:
        if not self.bidirectional_smoothing:
            return self._forward_window_candidate_pool(segments, index)

        current_candidates = self._candidate_pool(segments, index)
        if not current_candidates:
            return []

        window_map: dict[int, list[ScoreBridgeCandidate]] = {index: current_candidates}
        radius = max(1, (self.max_window_size - 1) // 2)
        for distance in range(1, radius + 1):
            left_index = index - distance
            if left_index >= 0:
                left_candidates = self._candidate_pool(segments, left_index)
                if left_candidates:
                    window_map[left_index] = left_candidates
            right_index = index + distance
            if right_index < len(segments):
                right_candidates = self._candidate_pool(segments, right_index)
                if right_candidates:
                    window_map[right_index] = right_candidates
        return sorted(window_map.items(), key=lambda item: item[0])

    def _forward_window_candidate_pool(
        self,
        segments: list[PhoneSegment],
        index: int,
    ) -> list[tuple[int, list[ScoreBridgeCandidate]]]:
        window: list[tuple[int, list[ScoreBridgeCandidate]]] = []
        upper_bound = min(len(segments), index + self.max_window_size)
        for window_index in range(index, upper_bound):
            candidates = self._candidate_pool(segments, window_index)
            if window_index == index and not candidates:
                return []
            if not candidates:
                break
            window.append((window_index, candidates))
        return window

    def _candidate_pool(self, segments: list[PhoneSegment], index: int) -> list[ScoreBridgeCandidate]:
        candidate_map: dict[tuple[str, str, str], ScoreBridgeCandidate] = {}
        for slot, neighbor_index in (("current", index), ("previous", index - 1), ("next", index + 1)):
            if neighbor_index < 0 or neighbor_index >= len(segments):
                continue
            note_context = segments[neighbor_index].metadata.get("note_context")
            if not isinstance(note_context, dict):
                continue
            track_id = self._value(note_context, "normalized_track_id", "track_id")
            part_id = self._value(note_context, "normalized_part_id", "part_id")
            note_id = self._value(note_context, "normalized_note_id", "note_id")
            if not all((track_id, part_id, note_id)):
                continue
            candidate = ScoreBridgeCandidate(
                track_id=track_id,
                part_id=part_id,
                note_id=note_id,
                start_sec=self._float_value(note_context.get("start_sec")),
                end_sec=self._float_value(note_context.get("end_sec")),
                tick=self._int_value(note_context.get("tick")),
                beat=self._float_value(note_context.get("beat")),
                part_local_note_rank=self._int_value(note_context.get("part_local_note_rank")),
                part_local_rank_source=self._value(note_context, "part_local_rank_source"),
                part_local_rank_fallback_used=bool(note_context.get("part_local_rank_fallback_used", False)),
                slots={slot},
            )
            signature = (candidate.track_id, candidate.part_id, candidate.note_id)
            existing = candidate_map.get(signature)
            if existing is None:
                candidate_map[signature] = candidate
            else:
                existing.slots.add(slot)
                if existing.start_sec is None:
                    existing.start_sec = candidate.start_sec
                if existing.end_sec is None:
                    existing.end_sec = candidate.end_sec
                if existing.tick is None:
                    existing.tick = candidate.tick
                if existing.beat is None:
                    existing.beat = candidate.beat
                if existing.part_local_note_rank is None:
                    existing.part_local_note_rank = candidate.part_local_note_rank
                if existing.part_local_rank_source is None:
                    existing.part_local_rank_source = candidate.part_local_rank_source
                existing.part_local_rank_fallback_used = (
                    existing.part_local_rank_fallback_used and candidate.part_local_rank_fallback_used
                )

        segment = segments[index]
        filtered: list[ScoreBridgeCandidate] = []
        for candidate in candidate_map.values():
            overlap_ratio = self._overlap_ratio(segment, self._candidate_note_context(candidate))
            if overlap_ratio >= self.candidate_overlap_threshold:
                filtered.append(candidate)
        return filtered

    def _choose_candidate(
        self,
        segments: list[PhoneSegment],
        index: int,
        window_candidates: list[tuple[int, list[ScoreBridgeCandidate]]],
    ) -> ScoreBridgeChoice | None:
        if not window_candidates:
            return None
        current_offset = next(
            (offset for offset, (window_index, _) in enumerate(window_candidates) if window_index == index),
            None,
        )
        if current_offset is None:
            return None

        local_choices = self._local_window_choices(segments, window_candidates)
        current_choices = local_choices[current_offset]
        if len(current_choices) == 1 and len(window_candidates) == 1:
            return self._finalize_bridge_choice(list(current_choices.values()), window_size=1)

        prefix_states = self._prefix_states(segments, window_candidates, local_choices)
        suffix_states = self._suffix_states(segments, window_candidates, local_choices)

        combined_choices: list[ScoreBridgeChoice] = []
        for signature, local_choice in current_choices.items():
            prefix_state = prefix_states[current_offset].get(signature)
            suffix_state = suffix_states[current_offset].get(signature)
            if prefix_state is None or suffix_state is None:
                continue
            combined_choices.append(
                self._combined_window_choice(
                    local_choice,
                    prefix_state,
                    suffix_state,
                    current_offset=current_offset,
                    window_size=len(window_candidates),
                )
            )

        return self._finalize_bridge_choice(combined_choices, window_size=len(window_candidates))

    def _finalize_bridge_choice(self, choices: list[ScoreBridgeChoice], window_size: int) -> ScoreBridgeChoice | None:
        if not choices:
            return None
        choices = sorted(
            choices,
            key=lambda choice: (
                choice.selection_score or choice.total_score,
                choice.overlap_ratio,
                choice.order_score,
                self._candidate_order_value(choice.candidate),
            ),
            reverse=True,
        )
        best = choices[0]
        second = choices[1] if len(choices) > 1 else None
        best.window_size = window_size
        if best.overlap_ratio >= self.overlap_threshold:
            return best
        if len(choices) > 1 and best.overlap_ratio >= self.candidate_overlap_threshold and (
            best.adjacency_used or best.part_continuity_used or best.order_score > 0.08
        ):
            margin = (best.selection_score or best.total_score) - (
                second.selection_score if second is not None and second.selection_score else 0.0
            )
            if second is None or margin >= 0.04 or best.adjacency_used:
                best.notes.append("accepted continuity-aware bridge below pure overlap threshold")
                if best.bidirectional_applied:
                    best.notes.append("accepted bidirectional score-bridge smoothing below pure overlap threshold")
                return best
        return None

    def _local_window_choices(
        self,
        segments: list[PhoneSegment],
        window_candidates: list[tuple[int, list[ScoreBridgeCandidate]]],
    ) -> list[dict[tuple[str, str, str], ScoreBridgeChoice]]:
        choices_by_offset: list[dict[tuple[str, str, str], ScoreBridgeChoice]] = []
        left_anchor = self._previous_identity(segments, window_candidates[0][0])
        left_anchor_token = self._previous_lyric_token(segments, window_candidates[0][0])
        for offset, (segment_index, candidates) in enumerate(window_candidates):
            segment = segments[segment_index]
            current_token = self._value(segment.metadata, "lyric_token")
            offset_choices: dict[tuple[str, str, str], ScoreBridgeChoice] = {}
            for candidate in candidates:
                overlap_ratio = self._overlap_ratio(segment, self._candidate_note_context(candidate))
                notes = [f"candidate_overlap_ratio={overlap_ratio:.2f}"]
                order_score = 0.0
                adjacency_used = False
                part_continuity_used = False
                order_violation_penalty = 0.0
                tiebreak = "overlap_rank"
                anchor_notes: list[str] = []
                if offset == 0 and left_anchor is not None:
                    (
                        anchor_score,
                        anchor_notes,
                        adjacency_used,
                        part_continuity_used,
                        order_violation_penalty,
                        tiebreak,
                    ) = self._transition_from_reference(
                        left_anchor,
                        candidate,
                        previous_token=left_anchor_token,
                        current_token=current_token,
                    )
                    order_score += anchor_score
                    self._extend_unique(notes, anchor_notes)
                choice = ScoreBridgeChoice(
                    candidate=candidate,
                    overlap_ratio=overlap_ratio,
                    total_score=overlap_ratio + order_score,
                    order_score=order_score,
                    tiebreak=tiebreak,
                    notes=notes,
                    adjacency_used=adjacency_used,
                    part_continuity_used=part_continuity_used,
                    rank_used=self._transition_rank_used(anchor_notes),
                    rank_consistency=self._transition_rank_consistency(anchor_notes),
                    sequence_score=overlap_ratio + order_score,
                    order_violation_penalty=order_violation_penalty,
                    monotonic_applied=False,
                    bidirectional_applied=False,
                    backward_context_used=False,
                    window_size=1,
                    selection_score=overlap_ratio + order_score,
                )
                offset_choices[self._candidate_signature(candidate)] = choice
            choices_by_offset.append(offset_choices)
        return choices_by_offset

    def _prefix_states(
        self,
        segments: list[PhoneSegment],
        window_candidates: list[tuple[int, list[ScoreBridgeCandidate]]],
        local_choices: list[dict[tuple[str, str, str], ScoreBridgeChoice]],
    ) -> list[dict[tuple[str, str, str], ScoreBridgePathState]]:
        prefix: list[dict[tuple[str, str, str], ScoreBridgePathState]] = []
        for offset, current_choices in enumerate(local_choices):
            current_states: dict[tuple[str, str, str], ScoreBridgePathState] = {}
            if offset == 0:
                for signature, choice in current_choices.items():
                    current_states[signature] = ScoreBridgePathState(
                        choice=choice,
                        sequence_score=choice.total_score,
                        continuity_score=choice.order_score,
                        order_violation_penalty=choice.order_violation_penalty,
                        sequence_notes=[],
                        adjacency_used=choice.adjacency_used,
                        part_continuity_used=choice.part_continuity_used,
                        backward_context_used=False,
                        rank_used=choice.rank_used,
                        rank_consistency=choice.rank_consistency,
                        tiebreak=choice.tiebreak,
                    )
                prefix.append(current_states)
                continue

            previous_states = prefix[offset - 1]
            previous_index = window_candidates[offset - 1][0]
            current_index = window_candidates[offset][0]
            previous_token = self._value(segments[previous_index].metadata, "lyric_token")
            current_token = self._value(segments[current_index].metadata, "lyric_token")
            for signature, choice in current_choices.items():
                best_state: ScoreBridgePathState | None = None
                for previous_state in previous_states.values():
                    transition = self._transition_between_candidates(
                        previous_state.choice.candidate,
                        choice.candidate,
                        previous_token=previous_token,
                        current_token=current_token,
                    )
                    candidate_state = self._advance_prefix_state(previous_state, choice, transition)
                    if best_state is None or candidate_state.sequence_score > best_state.sequence_score:
                        best_state = candidate_state
                if best_state is not None:
                    current_states[signature] = best_state
            prefix.append(current_states)
        return prefix

    def _suffix_states(
        self,
        segments: list[PhoneSegment],
        window_candidates: list[tuple[int, list[ScoreBridgeCandidate]]],
        local_choices: list[dict[tuple[str, str, str], ScoreBridgeChoice]],
    ) -> list[dict[tuple[str, str, str], ScoreBridgePathState]]:
        suffix: list[dict[tuple[str, str, str], ScoreBridgePathState]] = [{} for _ in local_choices]
        for offset in range(len(local_choices) - 1, -1, -1):
            current_choices = local_choices[offset]
            current_states: dict[tuple[str, str, str], ScoreBridgePathState] = {}
            if offset == len(local_choices) - 1:
                for signature, choice in current_choices.items():
                    current_states[signature] = ScoreBridgePathState(
                        choice=choice,
                        sequence_score=choice.total_score,
                        continuity_score=choice.order_score,
                        order_violation_penalty=choice.order_violation_penalty,
                        sequence_notes=[],
                        adjacency_used=choice.adjacency_used,
                        part_continuity_used=choice.part_continuity_used,
                        backward_context_used=False,
                        rank_used=choice.rank_used,
                        rank_consistency=choice.rank_consistency,
                        tiebreak=choice.tiebreak,
                    )
                suffix[offset] = current_states
                continue

            next_states = suffix[offset + 1]
            current_index = window_candidates[offset][0]
            next_index = window_candidates[offset + 1][0]
            current_token = self._value(segments[current_index].metadata, "lyric_token")
            next_token = self._value(segments[next_index].metadata, "lyric_token")
            for signature, choice in current_choices.items():
                best_state: ScoreBridgePathState | None = None
                for next_state in next_states.values():
                    transition = self._transition_between_candidates(
                        choice.candidate,
                        next_state.choice.candidate,
                        previous_token=current_token,
                        current_token=next_token,
                    )
                    candidate_state = self._advance_suffix_state(choice_state=choice, next_state=next_state, transition=transition)
                    if best_state is None or candidate_state.sequence_score > best_state.sequence_score:
                        best_state = candidate_state
                if best_state is not None:
                    current_states[signature] = best_state
            suffix[offset] = current_states
        return suffix

    def _advance_prefix_state(
        self,
        previous_state: ScoreBridgePathState,
        choice: ScoreBridgeChoice,
        transition: tuple[float, list[str], bool, bool, float, str],
    ) -> ScoreBridgePathState:
        transition_score, transition_notes, adjacency_used, part_continuity_used, penalty, transition_tiebreak = transition
        return ScoreBridgePathState(
            choice=choice,
            sequence_score=previous_state.sequence_score + choice.total_score + transition_score,
            continuity_score=previous_state.continuity_score + choice.order_score + transition_score,
            order_violation_penalty=previous_state.order_violation_penalty + choice.order_violation_penalty + penalty,
            sequence_notes=previous_state.sequence_notes + list(transition_notes),
            adjacency_used=previous_state.adjacency_used or choice.adjacency_used or adjacency_used,
            part_continuity_used=previous_state.part_continuity_used or choice.part_continuity_used or part_continuity_used,
            backward_context_used=previous_state.backward_context_used,
            rank_used=previous_state.rank_used or choice.rank_used or self._transition_rank_used(transition_notes),
            rank_consistency=self._merge_rank_consistency(
                previous_state.rank_consistency,
                choice.rank_consistency,
                self._transition_rank_consistency(transition_notes),
            ),
            tiebreak=self._merge_tiebreak(previous_state.tiebreak, transition_tiebreak, False),
        )

    def _advance_suffix_state(
        self,
        choice_state: ScoreBridgeChoice,
        next_state: ScoreBridgePathState,
        transition: tuple[float, list[str], bool, bool, float, str],
    ) -> ScoreBridgePathState:
        transition_score, transition_notes, adjacency_used, part_continuity_used, penalty, transition_tiebreak = transition
        return ScoreBridgePathState(
            choice=choice_state,
            sequence_score=choice_state.total_score + next_state.sequence_score + transition_score,
            continuity_score=choice_state.order_score + next_state.continuity_score + transition_score,
            order_violation_penalty=choice_state.order_violation_penalty + next_state.order_violation_penalty + penalty,
            sequence_notes=list(transition_notes) + next_state.sequence_notes,
            adjacency_used=choice_state.adjacency_used or next_state.adjacency_used or adjacency_used,
            part_continuity_used=choice_state.part_continuity_used or next_state.part_continuity_used or part_continuity_used,
            backward_context_used=True,
            rank_used=choice_state.rank_used or next_state.rank_used or self._transition_rank_used(transition_notes),
            rank_consistency=self._merge_rank_consistency(
                choice_state.rank_consistency,
                next_state.rank_consistency,
                self._transition_rank_consistency(transition_notes),
            ),
            tiebreak=self._merge_tiebreak(choice_state.tiebreak, transition_tiebreak, True),
        )

    def _combined_window_choice(
        self,
        local_choice: ScoreBridgeChoice,
        prefix_state: ScoreBridgePathState,
        suffix_state: ScoreBridgePathState,
        current_offset: int,
        window_size: int,
    ) -> ScoreBridgeChoice:
        left_bonus = max(0.0, prefix_state.sequence_score - local_choice.total_score)
        right_bonus = max(0.0, suffix_state.sequence_score - local_choice.total_score)
        sequence_score = prefix_state.sequence_score + suffix_state.sequence_score - local_choice.total_score
        continuity_score = prefix_state.continuity_score + suffix_state.continuity_score - local_choice.order_score
        order_violation_penalty = (
            prefix_state.order_violation_penalty + suffix_state.order_violation_penalty - local_choice.order_violation_penalty
        )
        notes = list(local_choice.notes)
        self._extend_unique(notes, prefix_state.sequence_notes)
        self._extend_unique(notes, suffix_state.sequence_notes)
        bidirectional_applied = (
            self.bidirectional_smoothing and window_size >= 3 and current_offset > 0 and current_offset < window_size - 1
        )
        backward_context_used = bidirectional_applied and right_bonus > 0.01
        monotonic_applied = window_size >= 3 and (left_bonus > 0.01 or right_bonus > 0.01)
        if monotonic_applied:
            notes.append(f"window_monotonic_sequence_score={sequence_score:.2f}")
        if backward_context_used:
            notes.append("backward context stabilized the current bridge choice")
        tiebreak = self._normalize_window_tiebreak(
            self._merge_bidirectional_tiebreak(local_choice.tiebreak, prefix_state.tiebreak, suffix_state.tiebreak),
            bidirectional_applied=bidirectional_applied,
        )
        continuity_bias = self._continuity_bias(prefix_state.tiebreak, backward_context=False) + self._continuity_bias(
            suffix_state.tiebreak,
            backward_context=True,
        )
        if continuity_bias > 0.0:
            notes.append(f"continuity bias bonus={continuity_bias:.2f}")
        selection_score = (
            local_choice.total_score
            + (0.35 * left_bonus)
            + (0.35 * right_bonus)
            + continuity_bias
            - (0.5 * max(0.0, order_violation_penalty))
        )
        return ScoreBridgeChoice(
            candidate=local_choice.candidate,
            overlap_ratio=local_choice.overlap_ratio,
            total_score=local_choice.total_score,
            order_score=continuity_score,
            tiebreak=tiebreak,
            notes=notes,
            adjacency_used=prefix_state.adjacency_used or suffix_state.adjacency_used,
            part_continuity_used=prefix_state.part_continuity_used or suffix_state.part_continuity_used,
            rank_used=local_choice.rank_used or prefix_state.rank_used or suffix_state.rank_used,
            rank_consistency=self._merge_rank_consistency(
                local_choice.rank_consistency,
                prefix_state.rank_consistency,
                suffix_state.rank_consistency,
            ),
            sequence_score=sequence_score,
            order_violation_penalty=order_violation_penalty,
            monotonic_applied=monotonic_applied,
            bidirectional_applied=bidirectional_applied,
            backward_context_used=backward_context_used,
            window_size=window_size,
            selection_score=selection_score,
        )

    def _transition_from_reference(
        self,
        reference: dict[str, object],
        candidate: ScoreBridgeCandidate,
        previous_token: str | None,
        current_token: str | None,
    ) -> tuple[float, list[str], bool, bool, float, str]:
        return self._transition_score(
            reference_track_id=self._value(reference, "track_id"),
            reference_part_id=self._value(reference, "part_id"),
            reference_note_id=self._value(reference, "note_id"),
            reference_rank=self._int_value(reference.get("part_local_note_rank")),
            reference_order=self._int_value(reference.get("order_value")),
            candidate=candidate,
            previous_token=previous_token,
            current_token=current_token,
        )

    def _transition_between_candidates(
        self,
        previous_candidate: ScoreBridgeCandidate,
        candidate: ScoreBridgeCandidate,
        previous_token: str | None,
        current_token: str | None,
    ) -> tuple[float, list[str], bool, bool, float, str]:
        return self._transition_score(
            reference_track_id=previous_candidate.track_id,
            reference_part_id=previous_candidate.part_id,
            reference_note_id=previous_candidate.note_id,
            reference_rank=previous_candidate.part_local_note_rank,
            reference_order=self._candidate_order_value(previous_candidate),
            candidate=candidate,
            previous_token=previous_token,
            current_token=current_token,
        )

    def _transition_score(
        self,
        reference_track_id: str | None,
        reference_part_id: str | None,
        reference_note_id: str | None,
        reference_rank: int | None,
        reference_order: int | None,
        candidate: ScoreBridgeCandidate,
        previous_token: str | None,
        current_token: str | None,
    ) -> tuple[float, list[str], bool, bool, float, str]:
        score = 0.0
        notes: list[str] = []
        adjacency_used = False
        part_continuity_used = False
        order_violation_penalty = 0.0
        tiebreak = "overlap_rank"

        if reference_track_id is not None:
            if candidate.track_id == reference_track_id:
                score += 0.02
            else:
                score -= 0.35
                part_continuity_used = True
                notes.append("track continuity penalty")
                tiebreak = "track_continuity"

        if reference_part_id is not None:
            if candidate.part_id == reference_part_id:
                score += 0.1
                part_continuity_used = True
                notes.append("kept part continuity")
                if tiebreak == "overlap_rank":
                    tiebreak = "part_continuity"
            else:
                score -= 0.22
                part_continuity_used = True
                notes.append("cross-part candidate penalty")
                tiebreak = "part_continuity"

        if reference_note_id is not None:
            if candidate.note_id == reference_note_id:
                score += 0.18
                adjacency_used = True
                notes.append("kept same-note adjacency")
                tiebreak = "same_note_adjacency"
            else:
                delta, rank_used, rank_consistency = self._bridge_order_delta(
                    reference_part_id=reference_part_id,
                    reference_note_id=reference_note_id,
                    reference_rank=reference_rank,
                    candidate_note_id=candidate.note_id,
                    candidate_rank=candidate.part_local_note_rank,
                    reference_order=reference_order,
                    candidate_order=self._candidate_order_value(candidate),
                )
                if rank_used:
                    notes.append(f"used part-local note rank consistency={rank_consistency}")
                if delta == 1:
                    score += 0.12
                    adjacency_used = True
                    notes.append("preferred adjacent forward note")
                    if tiebreak in {"overlap_rank", "part_continuity"}:
                        tiebreak = "adjacent_note_progression"
                elif delta is not None and delta < 0:
                    penalty = 0.24
                    score -= penalty
                    order_violation_penalty += penalty
                    notes.append("penalized backward note jump")
                    tiebreak = "note_order"
                elif delta is not None and delta > 1:
                    penalty = min(0.05 * float(delta - 1), 0.2)
                    score -= penalty
                    order_violation_penalty += penalty
                    notes.append(f"penalized long forward jump ({delta})")
                    tiebreak = "note_order"

            if previous_token and current_token and previous_token == current_token and candidate.note_id == reference_note_id:
                score += 0.05
                adjacency_used = True
                notes.append("kept lyric-token continuity on same note")
                tiebreak = "lyric_continuity"

        return score, notes, adjacency_used, part_continuity_used, order_violation_penalty, tiebreak

    def _candidate_signature(self, candidate: ScoreBridgeCandidate) -> tuple[str, str, str]:
        return (candidate.track_id, candidate.part_id, candidate.note_id)

    def _merge_tiebreak(self, base_reason: str, transition_tiebreak: str, backward_context: bool) -> str:
        if transition_tiebreak == "lyric_continuity":
            return "window_bidirectional_lyric" if backward_context else "window_monotonic_lyric"
        if transition_tiebreak == "same_note_adjacency":
            return "window_bidirectional_same_note" if backward_context else "window_monotonic_same_note"
        if transition_tiebreak == "adjacent_note_progression":
            return "window_bidirectional_progression" if backward_context else "window_monotonic_progression"
        if transition_tiebreak in {"part_continuity", "track_continuity"}:
            return "window_bidirectional_part_continuity" if backward_context else "window_monotonic_part_continuity"
        return base_reason

    def _merge_bidirectional_tiebreak(self, local_reason: str, prefix_reason: str, suffix_reason: str) -> str:
        for reason in (suffix_reason, prefix_reason, local_reason):
            if reason.startswith("window_bidirectional_"):
                return reason
        for reason in (suffix_reason, prefix_reason, local_reason):
            if reason.startswith("window_monotonic_"):
                return reason
        return local_reason

    def _normalize_window_tiebreak(self, reason: str, bidirectional_applied: bool) -> str:
        if bidirectional_applied:
            return reason
        return reason.replace("window_bidirectional_", "window_monotonic_")

    def _extend_unique(self, notes: list[str], extras: list[str]) -> None:
        for note in extras:
            if note not in notes:
                notes.append(note)

    def _continuity_bias(self, reason: str, backward_context: bool) -> float:
        if "same_note" in reason:
            return 0.025 if backward_context else 0.07
        if "progression" in reason:
            return 0.015 if backward_context else 0.02
        if "part_continuity" in reason:
            return 0.02 if backward_context else 0.025
        return 0.0

    def _bridge_order_delta(
        self,
        reference_part_id: str | None,
        reference_note_id: str,
        reference_rank: int | None,
        candidate_note_id: str,
        candidate_rank: int | None,
        reference_order: int | None,
        candidate_order: int | None,
    ) -> tuple[int | None, bool, str]:
        if reference_part_id is not None and reference_rank is not None and candidate_rank is not None:
            if candidate_rank == reference_rank:
                return 0, True, "same_rank"
            if candidate_rank > reference_rank:
                return candidate_rank - reference_rank, True, "monotonic"
            return candidate_rank - reference_rank, True, "backward"
        reference_tail = self._note_id_numeric_tail(reference_note_id)
        candidate_tail = self._note_id_numeric_tail(candidate_note_id)
        if reference_tail is not None and candidate_tail is not None:
            return candidate_tail - reference_tail, False, "note_id_tail"
        if reference_order is None or candidate_order is None:
            return None, False, "unavailable"
        if candidate_order == reference_order:
            return 0, False, "same_order"
        return (1 if candidate_order > reference_order else -1), False, "time_order"

    def _transition_rank_used(self, notes: list[str]) -> bool:
        return any(note.startswith("used part-local note rank") for note in notes)

    def _transition_rank_consistency(self, notes: list[str]) -> str:
        for note in notes:
            if note.startswith("used part-local note rank consistency="):
                return note.split("=", 1)[1]
        return "unavailable"

    def _merge_rank_consistency(self, *values: str) -> str:
        for value in values:
            if value in {"same_rank", "monotonic", "backward"}:
                return value
        for value in values:
            if value not in {"", "unavailable", None}:
                return value
        return "unavailable"

    def _overlap_ratio(self, segment: PhoneSegment, note_context: dict[str, object]) -> float:
        start_sec = self._float_value(note_context.get("start_sec"))
        end_sec = self._float_value(note_context.get("end_sec"))
        if start_sec is None or end_sec is None or end_sec <= start_sec:
            return 0.0
        segment_start = segment.time_range.start_sec
        segment_end = segment.time_range.end_sec
        overlap = min(segment_end, end_sec) - max(segment_start, start_sec)
        if overlap <= 0:
            return 0.0
        duration = max(1e-6, segment_end - segment_start)
        return min(1.0, overlap / duration)

    def _adjacent_same_note(self, segments: list[PhoneSegment], index: int, note_id: str) -> bool:
        for neighbor_index in (index - 1, index + 1):
            if neighbor_index < 0 or neighbor_index >= len(segments):
                continue
            neighbor_note_id = self._identity_note_id(segments[neighbor_index])
            if neighbor_note_id == note_id:
                return True
        return False

    def _previous_identity(self, segments: list[PhoneSegment], index: int) -> dict[str, object] | None:
        if index <= 0:
            return None
        previous_segment = segments[index - 1]
        note_id = self._identity_note_id(previous_segment)
        part_id = self._identity_part_id(previous_segment)
        track_id = self._identity_track_id(previous_segment)
        if note_id is None or part_id is None or track_id is None:
            return None
        return {
            "track_id": track_id,
            "part_id": part_id,
            "note_id": note_id,
            "part_local_note_rank": self._segment_rank_value(previous_segment),
            "order_value": self._identity_order_value(previous_segment),
        }

    def _next_note_context(self, segments: list[PhoneSegment], index: int) -> dict[str, object] | None:
        if index + 1 >= len(segments):
            return None
        note_context = segments[index + 1].metadata.get("note_context")
        if not isinstance(note_context, dict):
            return None
        track_id = self._value(note_context, "normalized_track_id", "track_id")
        part_id = self._value(note_context, "normalized_part_id", "part_id")
        note_id = self._value(note_context, "normalized_note_id", "note_id")
        if not all((track_id, part_id, note_id)):
            return None
        return {
            "track_id": track_id,
            "part_id": part_id,
            "note_id": note_id,
            "order_value": self._context_order_value(note_context, note_id),
        }

    def _previous_lyric_token(self, segments: list[PhoneSegment], index: int) -> str | None:
        if index <= 0:
            return None
        return self._value(segments[index - 1].metadata, "lyric_token")

    def _order_delta(self, reference: dict[str, object], candidate: ScoreBridgeCandidate | dict[str, object]) -> int | None:
        reference_order = reference.get("order_value")
        candidate_order = (
            self._candidate_order_value(candidate)
            if isinstance(candidate, ScoreBridgeCandidate)
            else candidate.get("order_value")
        )
        if reference_order is None or candidate_order is None:
            return None
        return int(candidate_order - reference_order)

    def _candidate_order_value(self, candidate: ScoreBridgeCandidate) -> int | None:
        if candidate.part_local_note_rank is not None:
            return candidate.part_local_note_rank
        if candidate.tick is not None:
            return candidate.tick
        if candidate.start_sec is not None:
            return int(round(candidate.start_sec * 1000.0))
        numeric_tail = self._note_id_numeric_tail(candidate.note_id)
        return numeric_tail

    def _context_order_value(self, note_context: dict[str, object], note_id: str) -> int | None:
        rank = self._int_value(note_context.get("part_local_note_rank"))
        if rank is not None:
            return rank
        tick = self._int_value(note_context.get("tick"))
        if tick is not None:
            return tick
        start_sec = self._float_value(note_context.get("start_sec"))
        if start_sec is not None:
            return int(round(start_sec * 1000.0))
        return self._note_id_numeric_tail(note_id)

    def _candidate_note_context(self, candidate: ScoreBridgeCandidate) -> dict[str, object]:
        return {
            "start_sec": candidate.start_sec,
            "end_sec": candidate.end_sec,
        }

    def _segment_rank_value(self, segment: PhoneSegment) -> int | None:
        note_context = segment.metadata.get("note_context")
        if isinstance(note_context, dict):
            rank = self._int_value(note_context.get("part_local_note_rank"))
            if rank is not None:
                return rank
        return self._int_value(segment.metadata.get("part_local_note_rank"))

    def _identity_track_id(self, segment: PhoneSegment) -> str | None:
        score_identity = segment.metadata.get("score_identity")
        if isinstance(score_identity, dict):
            value = self._value(score_identity, "normalized_track_id")
            if value is not None:
                return value
        score_bridge = segment.metadata.get("score_bridge_identity")
        if isinstance(score_bridge, dict):
            return self._value(score_bridge, "normalized_track_id", "track_id")
        return None

    def _identity_part_id(self, segment: PhoneSegment) -> str | None:
        score_identity = segment.metadata.get("score_identity")
        if isinstance(score_identity, dict):
            value = self._value(score_identity, "normalized_part_id")
            if value is not None:
                return value
        score_bridge = segment.metadata.get("score_bridge_identity")
        if isinstance(score_bridge, dict):
            return self._value(score_bridge, "normalized_part_id", "part_id")
        return None

    def _identity_note_id(self, segment: PhoneSegment) -> str | None:
        score_identity = segment.metadata.get("score_identity")
        if isinstance(score_identity, dict):
            note_id = self._value(score_identity, "normalized_note_id")
            if note_id is not None:
                return note_id
        score_bridge = segment.metadata.get("score_bridge_identity")
        if isinstance(score_bridge, dict):
            return self._value(score_bridge, "normalized_note_id", "note_id")
        return None

    def _identity_order_value(self, segment: PhoneSegment) -> int | None:
        note_context = segment.metadata.get("note_context")
        if isinstance(note_context, dict):
            order_value = self._context_order_value(note_context, self._value(note_context, "normalized_note_id", "note_id") or "")
            if order_value is not None:
                return order_value
        note_id = self._identity_note_id(segment)
        return self._note_id_numeric_tail(note_id) if note_id is not None else None

    def _note_id_numeric_tail(self, note_id: str | None) -> int | None:
        if note_id is None:
            return None
        match = re.search(r"(\d+)$", note_id)
        if match is None:
            return None
        return int(match.group(1))

    def _value(self, payload: dict[str, object], *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value in (None, ""):
                continue
            return str(value)
        return None

    def _float_value(self, value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _int_value(self, value: object) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return None

    def _mapping_has_identity(self, payload: dict[str, object]) -> bool:
        return any(
            payload.get(key) not in (None, "")
            for key in ("normalized_track_id", "normalized_part_id", "normalized_note_id", "track_id", "part_id", "note_id")
        )


class SegmentScoreIdentityNormalizer:
    def apply(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        for segment in segments:
            identity = self._resolve(segment)
            self._annotate(segment, identity)
        return segments

    def _resolve(self, segment: PhoneSegment) -> SegmentScoreIdentity:
        metadata = segment.metadata
        for key, default_source in (
            ("score_identity", "metadata:score_identity"),
            ("score_bridge_identity", "metadata:score_bridge_identity"),
        ):
            candidate = metadata.get(key)
            if not isinstance(candidate, dict):
                continue
            identity = self._from_mapping(candidate, default_source=default_source)
            if self._has_identity(identity):
                return identity

        if not bool(metadata.get("note_context_applied", False)):
            identity = self._from_mapping(metadata, default_source="metadata:legacy_identity")
            if self._has_identity(identity):
                identity.score_identity_fallback_used = True
                return identity

        return SegmentScoreIdentity(
            normalized_track_id=None,
            normalized_part_id=None,
            normalized_note_id=None,
            score_identity_source=None,
            score_identity_fallback_used=True,
        )

    def _from_mapping(self, payload: dict[str, object], default_source: str) -> SegmentScoreIdentity:
        track_id = self._value(
            payload,
            "normalized_track_id",
            "track_id",
            "track",
        )
        part_id = self._value(
            payload,
            "normalized_part_id",
            "part_id",
            "part",
        )
        note_id = self._value(
            payload,
            "normalized_note_id",
            "note_id",
            "note",
        )
        source = self._value(
            payload,
            "score_identity_source",
            "source",
        ) or default_source
        fallback_used = self._bool_value(
            payload.get("score_identity_fallback_used", payload.get("fallback_used")),
            default=False,
        )
        return SegmentScoreIdentity(
            normalized_track_id=track_id,
            normalized_part_id=part_id,
            normalized_note_id=note_id,
            score_identity_source=source,
            score_identity_fallback_used=fallback_used,
        )

    def _annotate(self, segment: PhoneSegment, identity: SegmentScoreIdentity) -> None:
        segment.metadata["score_identity"] = {
            "normalized_track_id": identity.normalized_track_id,
            "normalized_part_id": identity.normalized_part_id,
            "normalized_note_id": identity.normalized_note_id,
            "score_identity_source": identity.score_identity_source,
            "score_identity_fallback_used": identity.score_identity_fallback_used,
        }
        segment.metadata["score_identity_source"] = identity.score_identity_source
        segment.metadata["score_identity_fallback_used"] = identity.score_identity_fallback_used
        if identity.normalized_track_id is not None and "normalized_track_id" not in segment.metadata:
            segment.metadata["normalized_track_id"] = identity.normalized_track_id
        if identity.normalized_part_id is not None and "normalized_part_id" not in segment.metadata:
            segment.metadata["normalized_part_id"] = identity.normalized_part_id
        if identity.normalized_note_id is not None and "normalized_note_id" not in segment.metadata:
            segment.metadata["normalized_note_id"] = identity.normalized_note_id

    def _has_identity(self, identity: SegmentScoreIdentity) -> bool:
        return any(
            value is not None
            for value in (
                identity.normalized_track_id,
                identity.normalized_part_id,
                identity.normalized_note_id,
            )
        )

    def _value(self, payload: dict[str, object], *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value in (None, ""):
                continue
            return str(value)
        return None

    def _bool_value(self, value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        return default

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from any2xsampa.core.io import load_json
from any2xsampa.core.models import PhoneSegment
from any2xsampa.lyrics.lexicon import DialectLexiconRepository, LexiconEntry
from any2xsampa.lyrics.parser import LyricsParser


@dataclass(slots=True)
class PhoneTarget:
    token_text: str
    token_index: int
    phone: str
    lexicon_hit: bool
    oov_fallback_used: bool
    placeholder: bool
    source: str
    constraint_weight: float


@dataclass(slots=True)
class AlignmentAction:
    operation: str
    segment_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    score: float
    notes: list[str]


class LyricsConstrainedDecoder:
    def __init__(self, config_root: Path) -> None:
        self.parser = LyricsParser()
        self.lexicon_repo = DialectLexiconRepository(config_root)
        self.aliases = load_json(config_root / "ipa_aliases.json")

    def apply(self, segments: list[PhoneSegment], lyrics: str, dialect: str) -> list[PhoneSegment]:
        tokens = self.parser.parse(lyrics)
        if not tokens:
            for segment in segments:
                metadata = segment.metadata
                metadata["lyric_token"] = None
                metadata["lyric_constraint_applied"] = False
                metadata["dialect_lexicon_hit"] = False
                metadata["alignment_notes"] = ["no lyric tokens parsed; kept acoustic hypothesis"]
                metadata["alignment_operation"] = "no_lyrics"
                metadata["alignment_score"] = 0.0
                metadata["oov_fallback_used"] = False
                metadata["oov_fallback_source"] = None
                metadata["target_phone"] = None
            return segments

        entries = self.lexicon_repo.lookup_tokens(dialect, tokens)
        phone_targets = self._build_targets(entries)
        if not phone_targets:
            self._annotate_without_targets(segments, entries)
            return segments

        actions = self._align_segments_to_targets(segments, phone_targets)
        self._apply_actions(segments, phone_targets, actions)
        return segments

    def _build_targets(self, entries: list[LexiconEntry]) -> list[PhoneTarget]:
        targets: list[PhoneTarget] = []
        for entry in entries:
            if entry.hit:
                for phone in entry.phones:
                    targets.append(
                        PhoneTarget(
                            token_text=entry.token.text,
                            token_index=entry.token.index,
                            phone=phone,
                            lexicon_hit=True,
                            oov_fallback_used=False,
                            placeholder=False,
                            source=entry.source,
                            constraint_weight=1.0,
                        )
                    )
            elif entry.phones:
                for phone in entry.phones:
                    targets.append(
                        PhoneTarget(
                            token_text=entry.token.text,
                            token_index=entry.token.index,
                            phone=phone,
                            lexicon_hit=False,
                            oov_fallback_used=True,
                            placeholder=False,
                            source=entry.source,
                            constraint_weight=0.45,
                        )
                    )
            else:
                targets.append(
                    PhoneTarget(
                        token_text=entry.token.text,
                        token_index=entry.token.index,
                        phone=f"oov:{entry.token.text}",
                        lexicon_hit=False,
                        oov_fallback_used=True,
                        placeholder=True,
                        source=entry.source,
                        constraint_weight=0.2,
                    )
                )
        return targets

    def _annotate_without_targets(self, segments: list[PhoneSegment], entries: list[LexiconEntry]) -> None:
        for segment in segments:
            metadata = segment.metadata
            metadata["lyric_token"] = entries[0].token.text if entries else None
            metadata["lyric_constraint_applied"] = False
            metadata["dialect_lexicon_hit"] = False
            metadata["alignment_notes"] = ["lyrics parsed but no targets generated; kept acoustic hypothesis"]
            metadata["alignment_operation"] = "no_targets"
            metadata["alignment_score"] = 0.0
            metadata["oov_fallback_used"] = True
            metadata["oov_fallback_source"] = "placeholder_oov"
            metadata["target_phone"] = None

    def _align_segments_to_targets(
        self,
        segments: list[PhoneSegment],
        targets: list[PhoneTarget],
    ) -> list[AlignmentAction]:
        segment_count = len(segments)
        target_count = len(targets)
        negative_inf = float("-inf")
        dp = [[negative_inf] * (target_count + 1) for _ in range(segment_count + 1)]
        back: list[list[AlignmentAction | None]] = [[None] * (target_count + 1) for _ in range(segment_count + 1)]
        dp[0][0] = 0.0

        for segment_index in range(segment_count + 1):
            for target_index in range(target_count + 1):
                current = dp[segment_index][target_index]
                if current == negative_inf:
                    continue

                if segment_index < segment_count and target_index < target_count:
                    score, notes = self._score_alignment(
                        segments[segment_index : segment_index + 1],
                        targets[target_index : target_index + 1],
                        "match",
                    )
                    self._maybe_update(
                        dp,
                        back,
                        segment_index + 1,
                        target_index + 1,
                        current + score,
                        AlignmentAction("match", (segment_index,), (target_index,), score, notes),
                    )

                if segment_index + 1 < segment_count and target_index < target_count:
                    score, notes = self._score_alignment(
                        segments[segment_index : segment_index + 2],
                        targets[target_index : target_index + 1],
                        "merge_segments",
                    )
                    self._maybe_update(
                        dp,
                        back,
                        segment_index + 2,
                        target_index + 1,
                        current + score,
                        AlignmentAction(
                            "merge_segments",
                            (segment_index, segment_index + 1),
                            (target_index,),
                            score,
                            notes,
                        ),
                    )

                if segment_index < segment_count and target_index + 1 < target_count:
                    score, notes = self._score_alignment(
                        segments[segment_index : segment_index + 1],
                        targets[target_index : target_index + 2],
                        "split_segment",
                    )
                    self._maybe_update(
                        dp,
                        back,
                        segment_index + 1,
                        target_index + 2,
                        current + score,
                        AlignmentAction(
                            "split_segment",
                            (segment_index,),
                            (target_index, target_index + 1),
                            score,
                            notes,
                        ),
                    )

                if segment_index < segment_count:
                    score, notes = self._score_skip_segment(segments[segment_index])
                    self._maybe_update(
                        dp,
                        back,
                        segment_index + 1,
                        target_index,
                        current + score,
                        AlignmentAction("skip_segment", (segment_index,), tuple(), score, notes),
                    )

                if target_index < target_count:
                    score, notes = self._score_skip_target(targets[target_index])
                    self._maybe_update(
                        dp,
                        back,
                        segment_index,
                        target_index + 1,
                        current + score,
                        AlignmentAction("skip_target", tuple(), (target_index,), score, notes),
                    )

        actions: list[AlignmentAction] = []
        segment_index = segment_count
        target_index = target_count
        while segment_index > 0 or target_index > 0:
            action = back[segment_index][target_index]
            if action is None:
                break
            actions.append(action)
            segment_index -= len(action.segment_indices)
            target_index -= len(action.target_indices)
        actions.reverse()
        return actions

    def _maybe_update(
        self,
        dp: list[list[float]],
        back: list[list[AlignmentAction | None]],
        next_segment_index: int,
        next_target_index: int,
        score: float,
        action: AlignmentAction,
    ) -> None:
        if score > dp[next_segment_index][next_target_index]:
            dp[next_segment_index][next_target_index] = score
            back[next_segment_index][next_target_index] = action

    def _score_alignment(
        self,
        segments: list[PhoneSegment],
        targets: list[PhoneTarget],
        operation: str,
    ) -> tuple[float, list[str]]:
        candidate_maps = [self._candidate_scores(segment) for segment in segments]
        top_confidence = sum(segment.confidence for segment in segments) / len(segments)
        match_scores: list[float] = []
        lexicon_bonus = 0.0
        oov_penalty = 0.0
        approximation_penalty = 0.0

        for target in targets:
            if target.placeholder:
                match_scores.append(top_confidence * 0.45)
                oov_penalty += 0.18
            elif target.oov_fallback_used:
                target_match = max(candidate_map.get(target.phone, 0.0) for candidate_map in candidate_maps)
                match_scores.append(max(target_match, top_confidence * 0.3))
                if target_match == 0.0:
                    approximation_penalty += 0.08
                lexicon_bonus += 0.08 * target.constraint_weight
                oov_penalty += 0.05
            else:
                target_match = max(candidate_map.get(target.phone, 0.0) for candidate_map in candidate_maps)
                match_scores.append(target_match)
                if target_match == 0.0:
                    approximation_penalty += 0.12
                lexicon_bonus += 0.16 * target.constraint_weight

        acoustic_score = sum(match_scores) / max(1, len(match_scores))
        operation_penalty = {
            "match": 0.0,
            "merge_segments": 0.08,
            "split_segment": 0.1,
        }[operation]
        total = (0.72 * acoustic_score) + lexicon_bonus - oov_penalty - approximation_penalty - operation_penalty
        notes = [
            f"acoustic_score={acoustic_score:.3f}",
            f"lexicon_bonus={lexicon_bonus:.3f}",
            f"oov_penalty={oov_penalty:.3f}",
            f"approximation_penalty={approximation_penalty:.3f}",
            f"operation_penalty={operation_penalty:.3f}",
        ]
        return total, notes

    def _score_skip_segment(self, segment: PhoneSegment) -> tuple[float, list[str]]:
        penalty = 0.3 - (0.12 * min(segment.confidence, 1.0))
        score = -penalty
        return score, [f"skip_segment_penalty={penalty:.3f}", f"segment_confidence={segment.confidence:.3f}"]

    def _score_skip_target(self, target: PhoneTarget) -> tuple[float, list[str]]:
        penalty = 0.4 if target.lexicon_hit else 0.14
        score = -penalty
        return score, [f"skip_target_penalty={penalty:.3f}", f"lexicon_hit={target.lexicon_hit}"]

    def _apply_actions(
        self,
        segments: list[PhoneSegment],
        targets: list[PhoneTarget],
        actions: list[AlignmentAction],
    ) -> None:
        pending_skipped_targets: list[PhoneTarget] = []
        visited_segments: set[int] = set()

        for action in actions:
            if action.operation == "skip_target":
                pending_skipped_targets.extend(targets[index] for index in action.target_indices)
                continue

            if action.operation == "skip_segment":
                segment = segments[action.segment_indices[0]]
                visited_segments.add(action.segment_indices[0])
                self._annotate_skipped_segment(segment, action, pending_skipped_targets)
                pending_skipped_targets = []
                continue

            aligned_targets = [targets[index] for index in action.target_indices]
            skipped_before = pending_skipped_targets
            pending_skipped_targets = []
            for segment_index in action.segment_indices:
                visited_segments.add(segment_index)
                self._apply_aligned_targets(
                    segment=segments[segment_index],
                    targets=aligned_targets,
                    action=action,
                    skipped_before=skipped_before,
                )

        for index, segment in enumerate(segments):
            if index not in visited_segments:
                self._annotate_skipped_segment(
                    segment,
                    AlignmentAction("skip_segment", (index,), tuple(), -0.25, ["unvisited segment fallback"]),
                    pending_skipped_targets,
                )

    def _annotate_skipped_segment(
        self,
        segment: PhoneSegment,
        action: AlignmentAction,
        skipped_before: list[PhoneTarget],
    ) -> None:
        metadata = segment.metadata
        metadata["lyric_token"] = self._format_tokens(skipped_before) if skipped_before else None
        metadata["lyric_constraint_applied"] = False
        metadata["dialect_lexicon_hit"] = False
        metadata["alignment_operation"] = action.operation
        metadata["alignment_score"] = round(action.score, 3)
        metadata["oov_fallback_used"] = False
        metadata["oov_fallback_source"] = None
        metadata["target_phone"] = None
        notes = list(action.notes)
        if skipped_before:
            notes.append(f"skipped_targets_before={self._format_target_phones(skipped_before)}")
        notes.append("kept acoustic hypothesis")
        metadata["alignment_notes"] = notes

    def _apply_aligned_targets(
        self,
        segment: PhoneSegment,
        targets: list[PhoneTarget],
        action: AlignmentAction,
        skipped_before: list[PhoneTarget],
    ) -> None:
        candidate_scores = self._candidate_scores(segment)
        acoustic_phone = segment.normalized_ipa or "sil"
        target_phones = [target.phone for target in targets if not target.placeholder]
        phone_pool = set(candidate_scores) | {acoustic_phone} | set(target_phones)

        best_phone = acoustic_phone
        best_score = 0.78 * candidate_scores.get(acoustic_phone, segment.confidence)
        for phone in phone_pool:
            audio_score = candidate_scores.get(phone, 0.0)
            lyric_match_bonus = 0.24 if phone in target_phones else 0.0
            lexicon_hit_bonus = 0.1 if any(target.lexicon_hit and phone == target.phone for target in targets) else 0.0
            oov_penalty = 0.1 if any(target.placeholder for target in targets) else 0.04 if any(
                target.oov_fallback_used for target in targets
            ) else 0.0
            approximation_penalty = 0.06 if phone in target_phones and audio_score == 0.0 else 0.0
            combined = (0.78 * audio_score) + lyric_match_bonus + lexicon_hit_bonus - oov_penalty - approximation_penalty
            if combined > best_score:
                best_score = combined
                best_phone = phone

        metadata = segment.metadata
        previous_phone = acoustic_phone
        # Lyrics are downstream evidence. Keep the recognizer's normalized IPA
        # untouched and expose the lyric-derived choice as a separate view.
        metadata["lyric_candidate_phone"] = best_phone
        metadata["lyric_candidate_role"] = "downstream_constraint_candidate"
        metadata["lyric_candidate_replaces_recognizer_truth"] = False
        metadata["lyric_token"] = self._format_tokens(targets)
        substantive_constraint = any(not target.placeholder for target in targets)
        metadata["lyric_constraint_applied"] = substantive_constraint
        metadata["dialect_lexicon_hit"] = any(target.lexicon_hit for target in targets)
        metadata["alignment_operation"] = action.operation
        metadata["alignment_score"] = round(action.score, 3)
        metadata["oov_fallback_used"] = any(target.oov_fallback_used for target in targets)
        metadata["oov_fallback_source"] = self._format_sources(targets)
        metadata["target_phone"] = self._format_target_phones(targets)
        notes = list(action.notes)
        if skipped_before:
            notes.append(f"skipped_targets_before={self._format_target_phones(skipped_before)}")
        notes.extend(
            [
                f"target_phone={metadata['target_phone']}",
                f"acoustic_phone={previous_phone}",
                f"lyric_candidate_phone={best_phone}",
                "recognizer_phone_preserved=true",
            ]
        )
        metadata["alignment_notes"] = notes
        if best_phone != previous_phone and substantive_constraint:
            constraint_note = (
                f"lyrics candidate: {previous_phone} -> {best_phone}; recognizer IPA preserved"
            )
            metadata.setdefault("downstream_constraint_notes", []).append(constraint_note)
            # Keep the historical metadata key readable for downstream JSON
            # consumers while making its non-truth role explicit.
            metadata.setdefault("normalization_notes", []).append(constraint_note)

    def _format_tokens(self, targets: list[PhoneTarget]) -> str | None:
        if not targets:
            return None
        ordered: list[str] = []
        for target in targets:
            if target.token_text not in ordered:
                ordered.append(target.token_text)
        return "/".join(ordered)

    def _format_target_phones(self, targets: list[PhoneTarget]) -> str:
        if not targets:
            return ""
        return "+".join(target.phone for target in targets)

    def _format_sources(self, targets: list[PhoneTarget]) -> str | None:
        if not targets:
            return None
        ordered: list[str] = []
        for target in targets:
            if target.source not in ordered:
                ordered.append(target.source)
        return "/".join(ordered)

    def _candidate_scores(self, segment: PhoneSegment) -> dict[str, float]:
        scores: dict[str, float] = {}
        for candidate in segment.candidate_phones:
            normalized = self.aliases.get(candidate.symbol, candidate.symbol)
            scores[normalized] = max(scores.get(normalized, 0.0), candidate.confidence)
        return scores

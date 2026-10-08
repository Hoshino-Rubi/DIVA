from __future__ import annotations

import re
from collections import OrderedDict

from any2xsampa.core.models import PhoneSegment
from any2xsampa.pipeline.future_aligner_debug import (
    annotate_future_aligner_metadata,
    phrase_boundary_blocker_summary,
    provenance_summary,
)
from any2xsampa.pipeline.future_aligner_inputs import (
    future_aligner_acoustic_evidence_from_segment,
    future_aligner_float,
    future_aligner_hint_from_segment,
    future_aligner_int,
    future_aligner_text,
    normalize_hint_ownership_bindings,
)
from any2xsampa.pipeline.future_aligner_models import (
    FutureAlignerAcousticEvidence,
    FutureAlignerCandidateScore,
    FutureAlignerEvidenceItem,
    FutureAlignerHint,
    FutureAlignerResult,
    future_aligner_output_payload,
)
from any2xsampa.pipeline.future_aligner_phrase_boundary import build_phrase_boundary_candidate_debug
from any2xsampa.pipeline.future_aligner_reporting import (
    build_competition_summary,
    build_future_aligner_report,
    build_ownership_conflict_candidates,
    build_ownership_conflict_summary,
    build_ownership_resolution_basis,
    build_release_explanation,
    build_release_risk_summary,
    build_selective_release_safety,
)


class ScoreHintAcousticAligner:
    """Apply score-hint continuity with conservative acoustic guardrails.

    The original implementation was lost to a block overwrite.  This recovery
    keeps its public contract and metadata surface while using a small deterministic
    window scorer that can run without a model backend.
    """

    def __init__(
        self,
        sequence_window_size: int = 5,
        enable_sequence: bool = True,
        use_acoustic_evidence: bool = True,
        enable_part_aware_note_spans: bool = True,
    ) -> None:
        self.sequence_window_size = max(1, min(5, int(sequence_window_size)))
        self.enable_sequence = bool(enable_sequence)
        self.use_acoustic_evidence = bool(use_acoustic_evidence)
        self.enable_part_aware_note_spans = bool(enable_part_aware_note_spans)

    def apply(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        hints = [self._hint(segment) for segment in segments]
        self._normalize_hint_ownership_bindings(segments, hints)
        acoustics = [self._acoustic_evidence(segment) for segment in segments]
        for index, segment in enumerate(segments):
            result = self._align(segments, index, hints, acoustics)
            self._annotate(segment, result)
        return segments

    def _align(
        self,
        segments: list[PhoneSegment],
        index: int,
        hints: list[FutureAlignerHint | None],
        acoustics: list[FutureAlignerAcousticEvidence | None],
    ) -> FutureAlignerResult:
        current = hints[index]
        acoustic = acoustics[index] if self.use_acoustic_evidence else None
        window_indices = self._window_indices(len(segments), index)
        candidates = self._candidate_scores(segments, index, window_indices, hints, acoustics)
        if current is None:
            result = self._from_legacy_identity(segments[index])
        else:
            result = self._from_hint(current, acoustic)
        result.future_aligner_window_size = len(window_indices)
        if self.enable_sequence and current is not None and len(candidates) >= 2:
            self._apply_sequence_choice(
                result=result,
                current=current,
                candidates=candidates,
                acoustic=acoustic,
            )
        self._attach_v2_fields(
            result,
            segments[index],
            acoustic,
            candidate_scores=candidates,
            current=current,
        )
        return result

    def _apply_conservative_fallback(
        self,
        result: FutureAlignerResult,
        current: FutureAlignerHint | None,
        candidate_scores: list[FutureAlignerCandidateScore],
    ) -> bool:
        """Reject weak time-overlap ownership instead of exposing false certainty."""
        if current is None or not candidate_scores:
            return False
        if current.note_identity_match or current.note_binding_strategy != "time_overlap":
            return False
        ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
        top = ordered[0]
        if current.score_hint_confidence > 0.5 or top.note_span_overlap_ratio >= 0.5:
            return False

        result.aligned_note_id = None
        result.aligner_strategy = "conservative_part_track_only"
        result.aligner_confidence = min(result.aligner_confidence, 0.37)
        result.aligner_fallback_used = True
        result.same_note_support = False
        result.order_support = False
        result.part_continuity_support = False
        result.neighbor_support_count = 0
        result.left_neighbor_support = False
        result.right_neighbor_support = False
        result.future_aligner_continuity_used = False
        result.continuity_evidence = [
            "insufficient neighboring same-note support for note-level ownership"
        ]
        result.monotonic_evidence = [
            "no stable monotonic in-part order support was available"
        ]
        result.aligner_notes.insert(
            0,
            "future aligner kept only track/part ownership because note ownership was not stable enough",
        )
        result.evidence_items = [
            self._evidence_item(
                "fallback",
                None,
                None,
                0.0,
                "window",
                False,
                "insufficient_same_note_support",
                ["insufficient neighboring same-note support for note-level ownership"],
            ),
            self._evidence_item(
                "fallback",
                None,
                None,
                0.0,
                "window",
                False,
                "insufficient_order_support",
                ["no stable monotonic in-part order support was available"],
            ),
        ]
        result.future_aligner_provenance_applied = True
        result.future_aligner_provenance_summary = provenance_summary(result.evidence_items)
        return True

    def _hint(self, segment: PhoneSegment) -> FutureAlignerHint | None:
        return future_aligner_hint_from_segment(segment)

    def _acoustic_evidence(self, segment: PhoneSegment) -> FutureAlignerAcousticEvidence | None:
        return future_aligner_acoustic_evidence_from_segment(segment)

    def _normalize_hint_ownership_bindings(
        self,
        segments: list[PhoneSegment],
        hints: list[FutureAlignerHint | None],
    ) -> None:
        normalize_hint_ownership_bindings(segments, hints)

    def _from_hint(
        self,
        hint: FutureAlignerHint,
        acoustic: FutureAlignerAcousticEvidence | None,
    ) -> FutureAlignerResult:
        confidence = hint.score_hint_confidence
        if acoustic is not None and acoustic.available:
            confidence = 0.55 * confidence + 0.45 * acoustic.acoustic_confidence
        notes = list(hint.score_hint_notes)
        return FutureAlignerResult(
            aligned_track_id=hint.normalized_track_id,
            aligned_part_id=hint.normalized_part_id,
            aligned_note_id=hint.normalized_note_id,
            aligner_strategy="score_aware_aligner_v2",
            aligner_confidence=max(0.0, min(1.0, confidence)),
            aligner_fallback_used=False,
            aligner_notes=notes,
            future_aligner_continuity_used=False,
            source_score_hint_note_id=hint.source_normalized_note_id,
            source_score_hint_note_identity_match=hint.source_note_identity_match,
            source_score_hint_binding_strategy=hint.source_note_binding_strategy,
            effective_score_hint_note_id=hint.normalized_note_id,
            effective_score_hint_note_identity_match=hint.note_identity_match,
            effective_score_hint_binding_strategy=hint.note_binding_strategy,
            ownership_binding_adjustment={
                "applied": hint.ownership_binding_adjusted,
                "reason": hint.ownership_binding_reason,
                "source_note_id": hint.source_normalized_note_id,
                "effective_note_id": hint.normalized_note_id,
            },
        )

    def _from_legacy_identity(self, segment: PhoneSegment) -> FutureAlignerResult:
        metadata = segment.metadata
        track_id = self._text(metadata.get("aligned_track_id"))
        part_id = self._text(metadata.get("aligned_part_id"))
        note_id = self._text(metadata.get("aligned_note_id"))
        confidence = self._float(metadata.get("aligner_confidence"))
        if confidence is None:
            confidence = float(segment.confidence)
        return FutureAlignerResult(
            aligned_track_id=track_id,
            aligned_part_id=part_id,
            aligned_note_id=note_id,
            aligner_strategy="score_aware_aligner_v2",
            aligner_confidence=max(0.0, min(1.0, confidence)),
            aligner_fallback_used=True,
            aligner_notes=["no score hint available; retained conservative segment identity"],
            aligner_v2_fallback_used=True,
        )

    def _apply_sequence_choice(
        self,
        *,
        result: FutureAlignerResult,
        current: FutureAlignerHint,
        candidates: list[FutureAlignerCandidateScore],
        acoustic: FutureAlignerAcousticEvidence | None,
    ) -> None:
        ordered = sorted(candidates, key=lambda candidate: candidate.total_score, reverse=True)
        winner = ordered[0]
        current_candidate = next(
            (candidate for candidate in ordered if candidate.note_id == current.normalized_note_id),
            None,
        )
        if current_candidate is None:
            current_candidate = FutureAlignerCandidateScore(
                note_id=current.normalized_note_id or "",
                rank=current.part_local_note_rank,
                support_score=0.0,
                continuity_score=0.0,
                order_penalty=1.0,
                same_note_neighbors=0,
                left_neighbor_match=False,
                right_neighbor_match=False,
            )
        gap = winner.total_score - current_candidate.total_score
        boundary_pressure = acoustic.boundary_pressure if acoustic else 0.0
        boundary_conflict = acoustic.boundary_conflict_likelihood if acoustic else 0.0
        should_switch = (
            winner.note_id != current.normalized_note_id
            and not current.note_identity_match
            and current.score_hint_confidence <= 0.35
            and gap >= 0.08
            and (boundary_pressure >= 0.35 or boundary_conflict >= 0.35 or winner.note_span_score > current_candidate.note_span_score)
        )
        if should_switch:
            result.aligned_note_id = winner.note_id
            result.aligner_confidence = max(result.aligner_confidence, min(0.99, 0.58 + 0.2 * gap))
            result.future_aligner_sequence_applied = True
            result.future_aligner_sequence_score = winner.total_score
            result.future_aligner_order_penalty = winner.order_penalty
            result.future_aligner_continuity_used = winner.continuity_used
            result.same_note_support = winner.same_note_neighbors > 0
            result.order_support = winner.order_support
            result.part_continuity_support = winner.part_continuity_support
            result.neighbor_support_count = winner.same_note_neighbors
            result.left_neighbor_support = winner.left_neighbor_match
            result.right_neighbor_support = winner.right_neighbor_match
            result.continuity_evidence.extend(winner.continuity_evidence)
            result.monotonic_evidence.extend(winner.monotonic_evidence)
            result.part_boundary_evidence.extend(winner.part_boundary_evidence)
            result.aligner_notes.append(
                f"future aligner reassigned low-confidence hint to sequence winner {winner.note_id}"
            )
            result.effective_score_hint_note_id = winner.note_id
            result.effective_score_hint_note_identity_match = False
            result.effective_score_hint_binding_strategy = "local_sequence_rebind"
        else:
            result.same_note_support = current_candidate.same_note_neighbors > 0
            result.order_support = current_candidate.order_support
            result.part_continuity_support = current_candidate.part_continuity_support
            result.neighbor_support_count = current_candidate.same_note_neighbors
            result.left_neighbor_support = current_candidate.left_neighbor_match
            result.right_neighbor_support = current_candidate.right_neighbor_match
            result.future_aligner_continuity_used = current_candidate.continuity_used
            result.continuity_evidence.extend(current_candidate.continuity_evidence)
            result.monotonic_evidence.extend(current_candidate.monotonic_evidence)
            result.part_boundary_evidence.extend(current_candidate.part_boundary_evidence)

    def _candidate_scores(
        self,
        segments: list[PhoneSegment],
        index: int,
        window_indices: list[int],
        hints: list[FutureAlignerHint | None],
        acoustics: list[FutureAlignerAcousticEvidence | None],
    ) -> list[FutureAlignerCandidateScore]:
        candidates: "OrderedDict[str, FutureAlignerCandidateScore]" = OrderedDict()
        current = hints[index]
        for source_index in window_indices:
            hint = hints[source_index]
            if hint is None or hint.normalized_note_id is None:
                continue
            note_id = hint.normalized_note_id
            candidate = candidates.get(note_id)
            if candidate is None:
                candidate = FutureAlignerCandidateScore(
                    note_id=note_id,
                    rank=hint.part_local_note_rank,
                    support_score=0.0,
                    continuity_score=0.0,
                    order_penalty=0.0,
                    same_note_neighbors=0,
                    left_neighbor_match=False,
                    right_neighbor_match=False,
                )
                candidates[note_id] = candidate
            distance = abs(source_index - index)
            weight = self._distance_weight(distance)
            candidate.support_score += weight * self._hint_support(hint, source_index == index)
            direction = "current" if source_index == index else ("left" if source_index < index else "right")
            candidate.evidence_items.append(
                self._evidence_item(
                    "same_note_continuity",
                    source_index,
                    hint,
                    weight * self._hint_support(hint, source_index == index),
                    direction,
                    True,
                    notes=["neighbor supported the candidate note inside the local window"],
                )
            )
            if source_index < index:
                candidate.left_neighbor_match = True
                candidate.continuity_evidence.append(f"left segment {source_index} supports {note_id}")
                candidate.monotonic_evidence.append(f"left order rank={hint.part_local_note_rank}")
            elif source_index > index:
                candidate.right_neighbor_match = True
                candidate.continuity_evidence.append(f"right segment {source_index} supports {note_id}")
                candidate.monotonic_evidence.append(f"right order rank={hint.part_local_note_rank}")
            if source_index != index:
                candidate.same_note_neighbors += 1
            source_acoustic = acoustics[source_index] if self.use_acoustic_evidence else None
            candidate.acoustic_score += weight * self._acoustic_score(source_acoustic, hint, source_index == index)
            span_score, overlap_ratio, interior_ratio = self._note_span_metrics(segments[index], hint)
            candidate.note_span_score = max(candidate.note_span_score, weight * span_score)
            candidate.note_span_overlap_ratio = max(candidate.note_span_overlap_ratio, overlap_ratio)
            candidate.note_span_interior_ratio = max(candidate.note_span_interior_ratio, interior_ratio)
            candidate.note_span_anchor_count = max(candidate.note_span_anchor_count, 1 if hint.note_identity_match else 0)
            candidate.part_aware_note_span_used = candidate.part_aware_note_span_used or (
                self.enable_part_aware_note_spans and hint.note_start_sec is not None and hint.note_end_sec is not None
            )
            candidate.note_span_summary = f"{note_id}:overlap={overlap_ratio:.3f},interior={interior_ratio:.3f}"
            if source_index != index and hint.normalized_part_id and current and hint.normalized_part_id == current.normalized_part_id:
                candidate.part_continuity_support = True
        values = list(candidates.values())
        for candidate in values:
            if candidate.left_neighbor_match or candidate.right_neighbor_match:
                candidate.continuity_score += 0.15 * candidate.same_note_neighbors
            if candidate.left_neighbor_match and candidate.right_neighbor_match:
                candidate.continuity_score += 0.15
            if current is not None and current.part_local_note_rank is not None and candidate.rank is not None:
                candidate.order_penalty = min(1.0, abs(candidate.rank - current.part_local_note_rank) * 0.04)
            candidate.order_support = candidate.order_penalty <= 0.08
            candidate.support_score = min(1.5, candidate.support_score)
            candidate.acoustic_score = min(1.5, candidate.acoustic_score)
            self._dedupe_notes(candidate.continuity_evidence)
            self._dedupe_notes(candidate.monotonic_evidence)
        values.sort(key=lambda candidate: candidate.total_score, reverse=True)
        for rank, candidate in enumerate(values, start=1):
            candidate.rank = candidate.rank if candidate.rank is not None else rank
        return values

    def _sequence_result(self, *args: object, **kwargs: object) -> FutureAlignerResult | None:
        """Compatibility hook retained for callers of the lost implementation."""
        return None

    def _phrase_boundary_interior_advantage(
        self,
        candidate_scores: list[FutureAlignerCandidateScore],
    ) -> float:
        ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
        if len(ordered) < 2:
            return 0.0
        return ordered[0].note_span_interior_ratio - ordered[1].note_span_interior_ratio

    def _phrase_boundary_overlap_advantage(
        self,
        candidate_scores: list[FutureAlignerCandidateScore],
    ) -> float:
        ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
        if len(ordered) < 2:
            return 0.0
        return ordered[0].note_span_overlap_ratio - ordered[1].note_span_overlap_ratio

    def _phrase_window_pressure(
        self,
        candidate_scores: list[FutureAlignerCandidateScore],
        acoustic: FutureAlignerAcousticEvidence | None,
    ) -> float:
        if acoustic is None:
            return 0.0
        return max(
            acoustic.boundary_pressure,
            acoustic.boundary_conflict_likelihood,
            acoustic.ownership_ambiguity,
        )

    def _phrase_boundary_mode(
        self,
        candidate_scores: list[FutureAlignerCandidateScore],
        acoustic: FutureAlignerAcousticEvidence | None,
    ) -> str:
        if len(candidate_scores) < 2:
            return "none"
        top, runner = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)[:2]
        if acoustic is None or not acoustic.available:
            return "none"
        if acoustic.boundary_conflict_likelihood >= 0.75 and top.note_span_interior_ratio >= 0.4:
            if top.same_note_neighbors >= 2 and runner.same_note_neighbors == 1:
                return "asymmetric_chain_anchor_candidate"
            return "asymmetric_dual_touch_candidate"
        if top.same_note_neighbors >= 2 and runner.same_note_neighbors == 1:
            return "asymmetric_chain_anchor_candidate"
        if top.left_neighbor_match != top.right_neighbor_match and runner.left_neighbor_match != runner.right_neighbor_match:
            return "asymmetric_interior_candidate"
        return "none"

    def _attach_v2_fields(
        self,
        result: FutureAlignerResult,
        segment: PhoneSegment,
        acoustic: FutureAlignerAcousticEvidence | None,
        *,
        candidate_scores: list[FutureAlignerCandidateScore],
        current: FutureAlignerHint | None,
    ) -> None:
        self._apply_conservative_fallback(result, current, candidate_scores)
        result.aligner_v2_applied = True
        result.aligner_v2_strategy = "score_aware_aligner_v2"
        result.aligner_v2_confidence = result.aligner_confidence
        result.aligner_v2_fallback_used = result.aligner_fallback_used
        result.aligner_v2_notes = list(result.aligner_notes)
        if acoustic is not None:
            result.acoustic_evidence_applied = acoustic.available
            result.acoustic_evidence_summary = acoustic.summary
            result.boundary_uncertainty = dict(acoustic.boundary_uncertainty)
            result.boundary_conflict_likely = bool(
                acoustic.boundary_conflict_likelihood >= 0.68
                or acoustic.ownership_conflict_risk >= 0.72
            )
        if candidate_scores:
            ordered = sorted(candidate_scores, key=lambda candidate: candidate.total_score, reverse=True)
            top = ordered[0]
            runner = ordered[1] if len(ordered) > 1 else None
            result.note_span_evidence_score = top.note_span_score
            result.note_span_overlap_ratio = top.note_span_overlap_ratio
            result.note_span_interior_ratio = top.note_span_interior_ratio
            result.note_span_anchor_count = top.note_span_anchor_count
            result.part_aware_note_span_used = top.part_aware_note_span_used
            result.note_span_summary = top.note_span_summary
            result.ownership_conflict_gap = top.total_score - (runner.total_score if runner else 0.0)
            result.ownership_conflict_threshold = 0.15
            result.ownership_stability_margin = result.ownership_conflict_gap - result.ownership_conflict_threshold
            result.note_ownership_conflict = bool(runner and result.ownership_stability_margin < 0.0)
            phrase_mode = self._phrase_boundary_mode(candidate_scores, acoustic)
            phrase_pressure = self._phrase_window_pressure(candidate_scores, acoustic)
            overlap_advantage = self._phrase_boundary_overlap_advantage(candidate_scores)
            interior_advantage = self._phrase_boundary_interior_advantage(candidate_scores)
            guarded = bool(acoustic and acoustic.boundary_uncertain and phrase_pressure >= 0.7)
            result.ownership_conflict_candidates = build_ownership_conflict_candidates(candidate_scores)
            result.ownership_conflict_summary = build_ownership_conflict_summary(
                candidate_scores,
                acoustic,
                conflict_threshold=result.ownership_conflict_threshold,
                phrase_boundary_mode=phrase_mode,
                trained_phrase_ready=False,
            )
            result.ownership_resolution_basis = build_ownership_resolution_basis(
                result,
                acoustic,
                candidate_scores,
                phrase_boundary_mode=phrase_mode,
                phrase_boundary_guarded=guarded,
                trained_one_sided_ready=False,
                trained_phrase_ready=False,
            )
            competition_guardrails: list[str] = []
            if result.note_ownership_conflict:
                competition_guardrails.append("ownership_conflict")
            if guarded:
                competition_guardrails.append("phrase_window_guardrail")
            evidence_mode = "posterior+logit" if acoustic and acoustic.posterior_like_available and acoustic.logit_like_available else (
                "posterior" if acoustic and acoustic.posterior_like_available else "logit" if acoustic and acoustic.logit_like_available else "none"
            )
            result.alignment_competition_summary = build_competition_summary(
                candidate_scores,
                acoustic,
                conflict_threshold=result.ownership_conflict_threshold,
                evidence_mode=evidence_mode,
                competition_guardrails=competition_guardrails,
                phrase_window_pressure=phrase_pressure,
                phrase_boundary_mode=phrase_mode,
                phrase_boundary_interior_advantage=interior_advantage,
                phrase_boundary_overlap_advantage=overlap_advantage,
            )
            result.phrase_boundary_candidate_debug = build_phrase_boundary_candidate_debug(
                segment=segment,
                current=current,
                ordered_candidates=ordered,
                acoustic=acoustic,
                mode=phrase_mode,
                guarded=guarded,
                boundary_regime_debug={
                    "ready": bool(acoustic and acoustic.boundary_transition_strength >= 0.35),
                    "boundary_uncertain": bool(acoustic and acoustic.boundary_uncertain),
                    "effective_boundary_uncertain": bool(acoustic and acoustic.boundary_uncertain),
                    "boundary_pressure": acoustic.boundary_pressure if acoustic else 0.0,
                    "boundary_transition_strength": acoustic.boundary_transition_strength if acoustic else 0.0,
                    "boundary_conflict_likelihood": acoustic.boundary_conflict_likelihood if acoustic else 0.0,
                    "primary_blocker": "none" if acoustic else "missing_acoustic_evidence",
                    "uncertainty_normalization": {"applied": False},
                },
                overlap_advantage=overlap_advantage,
                interior_advantage=interior_advantage,
                phrase_window_pressure=phrase_pressure,
                current_candidate=next((item for item in ordered if current and item.note_id == current.normalized_note_id), None),
            )
            result.phrase_boundary_blocker_summary = phrase_boundary_blocker_summary(
                result.phrase_boundary_candidate_debug,
                segment_start_sec=float(segment.time_range.start_sec),
                segment_end_sec=float(segment.time_range.end_sec),
            )
            result.selective_release_safety = build_selective_release_safety(
                result,
                current,
                candidate_scores,
                acoustic,
                phrase_boundary_mode=phrase_mode,
                phrase_boundary_guarded=guarded,
                trained_one_sided_ready=False,
                trained_phrase_ready=False,
            )
            result.release_explanation = build_release_explanation(result, result.selective_release_safety)
            result.release_risk_summary = build_release_risk_summary(result, result.selective_release_safety)
            evidence: list[FutureAlignerEvidenceItem] = (
                list(result.evidence_items)
                if result.aligner_fallback_used and result.evidence_items
                else list(top.evidence_items)
            )
            result.evidence_items = evidence
            result.future_aligner_provenance_applied = bool(evidence)
            result.future_aligner_provenance_summary = provenance_summary(evidence)
            result.future_aligner_report = build_future_aligner_report(
                result,
                segment=segment,
                current=current,
                phrase_boundary_mode=phrase_mode,
                phrase_boundary_guarded=guarded,
                phrase_window_pressure=phrase_pressure,
                evidence_mode=evidence_mode,
                competition_guardrails=competition_guardrails,
            )
        elif current is not None:
            result.future_aligner_provenance_applied = False
            result.future_aligner_provenance_summary = "accepted=0,rejected=0,left=0,right=0,types=none"
            result.future_aligner_report = build_future_aligner_report(
                result,
                segment=segment,
                current=current,
                phrase_boundary_mode="none",
                phrase_boundary_guarded=False,
                phrase_window_pressure=0.0,
                evidence_mode="none",
                competition_guardrails=[],
            )

    def _note_span_metrics(self, segment: PhoneSegment, hint: FutureAlignerHint) -> tuple[float, float, float]:
        start = hint.note_start_sec
        end = hint.note_end_sec
        if start is None or end is None or end <= start:
            return 0.0, 0.0, 0.0
        seg_start = float(segment.time_range.start_sec)
        seg_end = float(segment.time_range.end_sec)
        overlap = max(0.0, min(seg_end, end) - max(seg_start, start))
        seg_duration = max(0.0, seg_end - seg_start)
        note_duration = max(0.0, end - start)
        overlap_ratio = overlap / seg_duration if seg_duration else 0.0
        interior_start = max(seg_start, start)
        interior_end = min(seg_end, end)
        interior_ratio = max(0.0, interior_end - interior_start) / note_duration if note_duration else 0.0
        return overlap_ratio + interior_ratio, overlap_ratio, interior_ratio

    def _acoustic_score(
        self,
        acoustic: FutureAlignerAcousticEvidence | None,
        hint: FutureAlignerHint,
        is_current: bool,
    ) -> float:
        if acoustic is None or not acoustic.available:
            return 0.0
        value = 0.18 * acoustic.acoustic_confidence + 0.12 * acoustic.timing_reliability
        value += 0.12 * acoustic.candidate_lead
        value += 0.14 * acoustic.competition_separation
        if is_current:
            value += 0.08 * acoustic.same_note_stability
        elif not hint.note_identity_match:
            value += 0.1 * acoustic.weak_overlap_recovery
        return value

    def _window_indices(self, total_segments: int, index: int) -> list[int]:
        size = min(self.sequence_window_size, total_segments)
        radius = size // 2
        start = max(0, index - radius)
        end = min(total_segments, start + size)
        start = max(0, end - size)
        return list(range(start, end))

    def _default_window_size(self, total_segments: int, index: int) -> int:
        return len(self._window_indices(total_segments, index))

    def _distance_weight(self, distance: int) -> float:
        if distance <= 0:
            return 1.0
        if distance == 1:
            return 0.85
        if distance == 2:
            return 0.55
        return 0.35

    def _hint_support(self, hint: FutureAlignerHint, is_current: bool) -> float:
        support = 0.22 + (0.4 * hint.score_hint_confidence)
        if hint.note_identity_match:
            support += 0.2
        if hint.note_binding_strategy == "identity":
            support += 0.14
        elif hint.note_binding_strategy == "identity_partial":
            support += 0.08
        elif hint.note_binding_strategy == "segment_index":
            support += 0.05
        elif hint.note_binding_strategy == "time_overlap":
            support -= 0.03
        if hint.score_bridge_strategy not in {None, "", "time_overlap_fallback"}:
            support += 0.05
        if is_current and hint.score_hint_confidence < 0.55:
            support -= 0.08
        return max(0.0, support)

    def _order_value(self, rank: int | None, note_id: str | None) -> int | None:
        if rank is not None:
            return rank
        if note_id is None:
            return None
        match = re.search(r"(\d+)$", note_id)
        return int(match.group(1)) if match else None

    def _dedupe_notes(self, notes: list[str]) -> None:
        deduped: list[str] = []
        for note in notes:
            if note not in deduped:
                deduped.append(note)
        notes[:] = deduped

    def _annotate(self, segment: PhoneSegment, result: FutureAlignerResult) -> None:
        annotate_future_aligner_metadata(segment, result)

    def _evidence_item(
        self,
        evidence_type: str,
        source_segment_index: int | None,
        source_hint: FutureAlignerHint | None,
        support_score: float,
        support_direction: str,
        accepted: bool,
        rejection_reason: str | None = None,
        notes: list[str] | None = None,
    ) -> FutureAlignerEvidenceItem:
        return FutureAlignerEvidenceItem(
            evidence_type=evidence_type,
            source_segment_index=source_segment_index,
            source_note_id=source_hint.normalized_note_id if source_hint is not None else None,
            source_track_id=source_hint.normalized_track_id if source_hint is not None else None,
            source_part_id=source_hint.normalized_part_id if source_hint is not None else None,
            support_score=support_score,
            support_direction=support_direction,
            accepted=accepted,
            rejection_reason=rejection_reason,
            notes=list(notes or []),
        )

    def _text(self, value: object) -> str | None:
        return future_aligner_text(value)

    def _float(self, value: object) -> float | None:
        return future_aligner_float(value)

    def _int(self, value: object) -> int | None:
        return future_aligner_int(value)


__all__ = ["ScoreHintAcousticAligner", "future_aligner_output_payload"]

from __future__ import annotations

from dataclasses import dataclass, field

from any2xsampa.core.models import PhoneSegment


@dataclass(slots=True)
class ScoreToSegmentHint:
    normalized_track_id: str | None
    normalized_part_id: str | None
    normalized_note_id: str | None
    part_local_note_rank: int | None
    note_midi: int | None
    note_name: str | None
    duration_ms: float | None
    start_sec: float | None
    end_sec: float | None
    release_context: str | None
    note_context_source: str | None
    note_context_adapter: str | None
    score_bridge_strategy: str | None
    score_bridge_match_confidence: float | None
    note_binding_strategy: str | None
    note_binding_fallback_used: bool
    note_identity_match: bool
    note_identity_partial_match: bool
    note_context_applied: bool
    note_context_fallback_used: bool
    note_identity_source: str | None
    note_identity_fallback_used: bool
    score_bridge_applied: bool
    score_identity_source: str | None
    score_identity_fallback_used: bool
    score_identity_hint_applied: bool
    score_hint_layer_applied: bool
    score_hint_source_summary: str
    score_hint_confidence: float
    score_hint_consumed: bool = False
    score_hint_consumer: list[str] = field(default_factory=list)
    score_hint_fallback_used: bool = False
    score_hint_consumer_notes: list[str] = field(default_factory=list)
    score_hint_notes: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, object]:
        return {
            "normalized_track_id": self.normalized_track_id,
            "normalized_part_id": self.normalized_part_id,
            "normalized_note_id": self.normalized_note_id,
            "part_local_note_rank": self.part_local_note_rank,
            "note_midi": self.note_midi,
            "note_name": self.note_name,
            "duration_ms": self.duration_ms,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "release_context": self.release_context,
            "note_context_source": self.note_context_source,
            "note_context_adapter": self.note_context_adapter,
            "score_bridge_strategy": self.score_bridge_strategy,
            "score_bridge_match_confidence": self.score_bridge_match_confidence,
            "note_binding_strategy": self.note_binding_strategy,
            "note_binding_fallback_used": self.note_binding_fallback_used,
            "note_identity_match": self.note_identity_match,
            "note_identity_partial_match": self.note_identity_partial_match,
            "note_context_applied": self.note_context_applied,
            "note_context_fallback_used": self.note_context_fallback_used,
            "note_identity_source": self.note_identity_source,
            "note_identity_fallback_used": self.note_identity_fallback_used,
            "score_bridge_applied": self.score_bridge_applied,
            "score_identity_source": self.score_identity_source,
            "score_identity_fallback_used": self.score_identity_fallback_used,
            "score_identity_hint_applied": self.score_identity_hint_applied,
            "score_hint_layer_applied": self.score_hint_layer_applied,
            "score_hint_source_summary": self.score_hint_source_summary,
            "score_hint_confidence": round(self.score_hint_confidence, 3),
            "score_hint_consumed": self.score_hint_consumed,
            "score_hint_consumer": list(self.score_hint_consumer),
            "score_hint_fallback_used": self.score_hint_fallback_used,
            "score_hint_consumer_notes": list(self.score_hint_consumer_notes),
            "score_hint_notes": list(self.score_hint_notes),
        }


@dataclass(slots=True)
class FutureAlignerScoreHint:
    normalized_track_id: str | None
    normalized_part_id: str | None
    normalized_note_id: str | None
    part_local_note_rank: int | None
    duration_ms: float | None
    start_sec: float | None
    end_sec: float | None
    release_context: str | None
    note_binding_strategy: str | None
    note_identity_match: bool
    score_bridge_strategy: str | None
    score_hint_confidence: float
    score_hint_notes: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, object]:
        return {
            "normalized_track_id": self.normalized_track_id,
            "normalized_part_id": self.normalized_part_id,
            "normalized_note_id": self.normalized_note_id,
            "part_local_note_rank": self.part_local_note_rank,
            "duration_ms": self.duration_ms,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "release_context": self.release_context,
            "note_binding_strategy": self.note_binding_strategy,
            "note_identity_match": self.note_identity_match,
            "score_bridge_strategy": self.score_bridge_strategy,
            "score_hint_confidence": round(self.score_hint_confidence, 3),
            "score_hint_notes": list(self.score_hint_notes),
        }


def score_hint_payload(segment: PhoneSegment) -> dict[str, object]:
    payload = segment.metadata.get("score_hint")
    return payload if isinstance(payload, dict) else {}


def future_aligner_hint_payload(segment: PhoneSegment) -> dict[str, object]:
    payload = segment.metadata.get("future_aligner_hint")
    if isinstance(payload, dict):
        record_score_hint_consumption(
            segment,
            consumer="future_aligner",
            consumed=True,
            fallback_used=False,
            notes=["future_aligner consumed future_aligner_hint derived from score_hint"],
        )
        return payload

    hint = score_hint_payload(segment)
    if hint:
        derived = FutureAlignerScoreHint(
            normalized_track_id=_hint_text(hint.get("normalized_track_id")),
            normalized_part_id=_hint_text(hint.get("normalized_part_id")),
            normalized_note_id=_hint_text(hint.get("normalized_note_id")),
            part_local_note_rank=_hint_int(hint.get("part_local_note_rank")),
            duration_ms=_hint_float(hint.get("duration_ms")),
            start_sec=_hint_float(hint.get("start_sec")),
            end_sec=_hint_float(hint.get("end_sec")),
            release_context=_hint_text(hint.get("release_context")),
            note_binding_strategy=_hint_text(hint.get("note_binding_strategy")),
            note_identity_match=bool(hint.get("note_identity_match", False)),
            score_bridge_strategy=_hint_text(hint.get("score_bridge_strategy")),
            score_hint_confidence=_hint_float(hint.get("score_hint_confidence")) or 0.0,
            score_hint_notes=list(hint.get("score_hint_notes", [])) if isinstance(hint.get("score_hint_notes"), list) else [],
        ).to_metadata()
        segment.metadata["future_aligner_hint"] = derived
        record_score_hint_consumption(
            segment,
            consumer="future_aligner",
            consumed=True,
            fallback_used=False,
            notes=["future_aligner derived its input directly from score_hint"],
        )
        return derived

    record_score_hint_consumption(
        segment,
        consumer="future_aligner",
        consumed=False,
        fallback_used=True,
        notes=["future_aligner could not find score_hint; no direct aligner hint available"],
    )
    return {}


def record_score_hint_consumption(
    segment: PhoneSegment,
    consumer: str,
    consumed: bool,
    fallback_used: bool,
    notes: list[str],
) -> None:
    metadata = segment.metadata
    hint = score_hint_payload(segment)
    consumers = metadata.get("score_hint_consumer")
    if not isinstance(consumers, list):
        consumers = list(hint.get("score_hint_consumer", [])) if isinstance(hint.get("score_hint_consumer"), list) else []
    consumer_notes = metadata.get("score_hint_consumer_notes")
    if not isinstance(consumer_notes, list):
        consumer_notes = (
            list(hint.get("score_hint_consumer_notes", []))
            if isinstance(hint.get("score_hint_consumer_notes"), list)
            else []
        )

    if consumer not in consumers:
        consumers.append(consumer)
    if consumed:
        metadata["score_hint_consumed"] = True
    else:
        metadata["score_hint_consumed"] = bool(metadata.get("score_hint_consumed", False))
    metadata["score_hint_consumer"] = consumers
    metadata["score_hint_fallback_used"] = bool(metadata.get("score_hint_fallback_used", False) or fallback_used)
    for note in notes:
        if note not in consumer_notes:
            consumer_notes.append(note)
    metadata["score_hint_consumer_notes"] = consumer_notes

    if hint:
        hint["score_hint_consumed"] = metadata["score_hint_consumed"]
        hint["score_hint_consumer"] = list(consumers)
        hint["score_hint_fallback_used"] = metadata["score_hint_fallback_used"]
        hint["score_hint_consumer_notes"] = list(consumer_notes)
        metadata["score_hint"] = hint


def resolve_score_hint_identity(segment: PhoneSegment, consumer: str) -> dict[str, str | None]:
    hint = score_hint_payload(segment)
    identity = {
        "track_id": _hint_text(hint.get("normalized_track_id")),
        "part_id": _hint_text(hint.get("normalized_part_id")),
        "note_id": _hint_text(hint.get("normalized_note_id")),
    }
    consumed = any(value is not None for value in identity.values())
    fallback_used = False
    notes: list[str] = []
    if consumed:
        notes.append(f"{consumer} consumed score_hint identity")
    else:
        score_identity = segment.metadata.get("score_identity")
        score_identity_map = score_identity if isinstance(score_identity, dict) else {}
        note_context = segment.metadata.get("note_context")
        note_context_map = note_context if isinstance(note_context, dict) else {}
        identity = {
            "track_id": _legacy_identity_value(segment, score_identity_map, note_context_map, "normalized_track_id"),
            "part_id": _legacy_identity_value(segment, score_identity_map, note_context_map, "normalized_part_id"),
            "note_id": _legacy_identity_value(segment, score_identity_map, note_context_map, "normalized_note_id"),
        }
        fallback_used = any(value is not None for value in identity.values())
        notes.append(f"{consumer} fell back to legacy identity fields")
    record_score_hint_consumption(segment, consumer, consumed=consumed, fallback_used=fallback_used, notes=notes)
    return identity


def resolve_score_hint_context(segment: PhoneSegment, consumer: str) -> dict[str, object]:
    hint = score_hint_payload(segment)
    note_context = segment.metadata.get("note_context")
    note_context_map = note_context if isinstance(note_context, dict) else {}
    values = {
        "normalized_track_id": _hint_text(hint.get("normalized_track_id")),
        "normalized_part_id": _hint_text(hint.get("normalized_part_id")),
        "normalized_note_id": _hint_text(hint.get("normalized_note_id")),
        "part_local_note_rank": _hint_int(hint.get("part_local_note_rank")),
        "note_midi": _hint_int(hint.get("note_midi")),
        "note_name": _hint_text(hint.get("note_name")),
        "duration_ms": _hint_float(hint.get("duration_ms")),
        "start_sec": _hint_float(hint.get("start_sec")),
        "end_sec": _hint_float(hint.get("end_sec")),
        "release_context": _hint_text(hint.get("release_context")),
        "note_context_applied": _hint_bool(hint.get("note_context_applied")),
        "note_context_source": _hint_text(hint.get("note_context_source")),
        "note_context_adapter": _hint_text(hint.get("note_context_adapter")),
        "note_context_fallback_used": _hint_bool(hint.get("note_context_fallback_used")),
        "note_identity_source": _hint_text(hint.get("note_identity_source")),
        "note_identity_fallback_used": _hint_bool(hint.get("note_identity_fallback_used")),
        "note_binding_strategy": _hint_text(hint.get("note_binding_strategy")),
        "note_identity_match": _hint_bool(hint.get("note_identity_match")),
        "score_bridge_strategy": _hint_text(hint.get("score_bridge_strategy")),
        "score_hint_confidence": _hint_float(hint.get("score_hint_confidence")),
        "score_hint_notes": list(hint.get("score_hint_notes", [])) if isinstance(hint.get("score_hint_notes"), list) else [],
    }
    consumed = any(
        values[key] not in (None, [])
        for key in (
            "normalized_track_id",
            "normalized_part_id",
            "normalized_note_id",
            "note_context_applied",
            "note_context_source",
            "note_identity_source",
            "score_bridge_strategy",
        )
    )
    fallback_notes: list[str] = []
    for field_name, legacy_value in (
        ("normalized_track_id", _hint_text(note_context_map.get("normalized_track_id"))),
        ("normalized_part_id", _hint_text(note_context_map.get("normalized_part_id"))),
        ("normalized_note_id", _hint_text(note_context_map.get("normalized_note_id"))),
        ("part_local_note_rank", _hint_int(note_context_map.get("part_local_note_rank"))),
        ("note_midi", _hint_int(note_context_map.get("note_midi"))),
        ("note_name", _hint_text(note_context_map.get("note_name"))),
        ("duration_ms", _hint_float(note_context_map.get("duration_ms"))),
        ("start_sec", _hint_float(note_context_map.get("start_sec"))),
        ("end_sec", _hint_float(note_context_map.get("end_sec"))),
        ("release_context", _hint_text(note_context_map.get("release_context")) or _hint_text(note_context_map.get("release"))),
        ("note_context_source", _hint_text(segment.metadata.get("note_context_source"))),
        ("note_context_adapter", _hint_text(segment.metadata.get("note_context_adapter"))),
        ("note_identity_source", _hint_text(note_context_map.get("note_identity_source"))),
    ):
        if values[field_name] is None and legacy_value is not None:
            values[field_name] = legacy_value
            fallback_notes.append(f"{consumer} fell back to legacy {field_name}")
    if values["note_context_applied"] is None:
        values["note_context_applied"] = bool(segment.metadata.get("note_context_applied", False))
        if values["note_context_applied"]:
            fallback_notes.append(f"{consumer} fell back to legacy note_context_applied")
    if values["note_context_fallback_used"] is None:
        values["note_context_fallback_used"] = bool(segment.metadata.get("note_context_fallback_used", True))
    if values["note_identity_fallback_used"] is None:
        values["note_identity_fallback_used"] = bool(note_context_map.get("note_identity_fallback_used", False))
    fallback_used = bool(fallback_notes)
    notes = [f"{consumer} consumed score_hint-first context"] if consumed else [f"{consumer} used legacy score-aware context"]
    notes.extend(fallback_notes)
    record_score_hint_consumption(segment, consumer, consumed=consumed, fallback_used=fallback_used, notes=notes)
    return values


class ScoreHintLayerBuilder:
    def apply(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        for segment in segments:
            hint = self._build(segment)
            payload = hint.to_metadata()
            segment.metadata["score_hint"] = payload
            segment.metadata["future_aligner_hint"] = self._future_aligner_hint(hint).to_metadata()
            segment.metadata["score_hint_layer_applied"] = hint.score_hint_layer_applied
            segment.metadata["score_hint_source_summary"] = hint.score_hint_source_summary
            segment.metadata["score_hint_confidence"] = payload["score_hint_confidence"]
            segment.metadata["score_hint_notes"] = list(hint.score_hint_notes)
            segment.metadata["score_hint_consumed"] = hint.score_hint_consumed
            segment.metadata["score_hint_consumer"] = list(hint.score_hint_consumer)
            segment.metadata["score_hint_fallback_used"] = hint.score_hint_fallback_used
            segment.metadata["score_hint_consumer_notes"] = list(hint.score_hint_consumer_notes)
        return segments

    def _build(self, segment: PhoneSegment) -> ScoreToSegmentHint:
        metadata = segment.metadata
        note_context = metadata.get("note_context")
        note_context_map = note_context if isinstance(note_context, dict) else {}
        score_identity = metadata.get("score_identity")
        score_identity_map = score_identity if isinstance(score_identity, dict) else {}
        score_bridge_identity = metadata.get("score_bridge_identity")
        score_bridge_map = score_bridge_identity if isinstance(score_bridge_identity, dict) else {}

        normalized_track_id = self._first_value(
            score_identity_map,
            score_bridge_map,
            metadata,
            note_context_map,
            keys=("normalized_track_id", "track_id"),
        )
        normalized_part_id = self._first_value(
            score_identity_map,
            score_bridge_map,
            metadata,
            note_context_map,
            keys=("normalized_part_id", "part_id"),
        )
        normalized_note_id = self._first_value(
            score_identity_map,
            score_bridge_map,
            metadata,
            note_context_map,
            keys=("normalized_note_id", "note_id"),
        )
        part_local_note_rank = self._first_int(
            metadata.get("part_local_note_rank"),
            note_context_map.get("part_local_note_rank"),
        )
        note_midi = self._first_int(note_context_map.get("note_midi"), metadata.get("note_midi"))
        note_name = self._first_value(note_context_map, metadata, keys=("note_name", "note"))
        duration_ms = self._first_float(note_context_map.get("duration_ms"), metadata.get("duration_ms"))
        start_sec = self._first_float(note_context_map.get("start_sec"), metadata.get("start_sec"))
        end_sec = self._first_float(note_context_map.get("end_sec"), metadata.get("end_sec"))
        release_context = self._first_value(note_context_map, metadata, keys=("release_context", "release"))

        note_context_source = self._text(metadata.get("note_context_source"))
        note_context_adapter = self._text(metadata.get("note_context_adapter"))
        score_bridge_strategy = self._text(metadata.get("score_bridge_strategy"))
        score_bridge_match_confidence = self._float_value(metadata.get("score_bridge_match_confidence"))
        note_binding_strategy = self._text(metadata.get("note_binding_strategy"))
        note_binding_fallback_used = bool(metadata.get("note_binding_fallback_used", False))
        note_identity_match = bool(metadata.get("note_identity_match", False))
        note_identity_partial_match = bool(metadata.get("note_identity_partial_match", False))
        note_context_applied = bool(metadata.get("note_context_applied", False))
        note_context_fallback_used = bool(metadata.get("note_context_fallback_used", True))
        note_identity_source = self._text(metadata.get("note_identity_source"))
        note_identity_fallback_used = bool(metadata.get("note_identity_fallback_used", False))
        score_bridge_applied = bool(metadata.get("score_bridge_applied", False))
        score_identity_source = self._text(metadata.get("score_identity_source"))
        score_identity_fallback_used = bool(metadata.get("score_identity_fallback_used", True))

        score_identity_hint_applied = bool(
            normalized_track_id or normalized_part_id or normalized_note_id
        ) and (
            note_identity_match
            or note_context_applied
            or score_bridge_applied
            or score_identity_source is not None
        )
        score_hint_layer_applied = bool(
            normalized_track_id
            or normalized_part_id
            or normalized_note_id
            or note_context_applied
            or score_bridge_applied
        )

        source_summary = self._source_summary(
            score_identity_source=score_identity_source,
            note_context_adapter=note_context_adapter,
            note_context_applied=note_context_applied,
            score_bridge_strategy=score_bridge_strategy,
            note_binding_strategy=note_binding_strategy,
        )
        notes = self._notes(
            note_identity_match=note_identity_match,
            note_identity_partial_match=note_identity_partial_match,
            note_context_applied=note_context_applied,
            score_bridge_applied=score_bridge_applied,
            score_bridge_strategy=score_bridge_strategy,
            note_binding_strategy=note_binding_strategy,
            part_local_note_rank=part_local_note_rank,
            score_hint_layer_applied=score_hint_layer_applied,
            note_binding_notes=metadata.get("note_binding_notes"),
            score_bridge_notes=metadata.get("score_bridge_notes"),
        )
        confidence = self._confidence(
            normalized_track_id=normalized_track_id,
            normalized_part_id=normalized_part_id,
            normalized_note_id=normalized_note_id,
            part_local_note_rank=part_local_note_rank,
            note_context_adapter=note_context_adapter,
            note_context_applied=note_context_applied,
            note_binding_strategy=note_binding_strategy,
            note_binding_fallback_used=note_binding_fallback_used,
            note_identity_match=note_identity_match,
            note_identity_partial_match=note_identity_partial_match,
            score_bridge_applied=score_bridge_applied,
            score_bridge_match_confidence=score_bridge_match_confidence,
            score_identity_fallback_used=score_identity_fallback_used,
        )

        return ScoreToSegmentHint(
            normalized_track_id=normalized_track_id,
            normalized_part_id=normalized_part_id,
            normalized_note_id=normalized_note_id,
            part_local_note_rank=part_local_note_rank,
            note_midi=note_midi,
            note_name=note_name,
            duration_ms=duration_ms,
            start_sec=start_sec,
            end_sec=end_sec,
            release_context=release_context,
            note_context_source=note_context_source,
            note_context_adapter=note_context_adapter,
            score_bridge_strategy=score_bridge_strategy,
            score_bridge_match_confidence=score_bridge_match_confidence,
            note_binding_strategy=note_binding_strategy,
            note_binding_fallback_used=note_binding_fallback_used,
            note_identity_match=note_identity_match,
            note_identity_partial_match=note_identity_partial_match,
            note_context_applied=note_context_applied,
            note_context_fallback_used=note_context_fallback_used,
            note_identity_source=note_identity_source,
            note_identity_fallback_used=note_identity_fallback_used,
            score_bridge_applied=score_bridge_applied,
            score_identity_source=score_identity_source,
            score_identity_fallback_used=score_identity_fallback_used,
            score_identity_hint_applied=score_identity_hint_applied,
            score_hint_layer_applied=score_hint_layer_applied,
            score_hint_source_summary=source_summary,
            score_hint_confidence=confidence,
            score_hint_notes=notes,
        )

    def _future_aligner_hint(self, hint: ScoreToSegmentHint) -> FutureAlignerScoreHint:
        return FutureAlignerScoreHint(
            normalized_track_id=hint.normalized_track_id,
            normalized_part_id=hint.normalized_part_id,
            normalized_note_id=hint.normalized_note_id,
            part_local_note_rank=hint.part_local_note_rank,
            duration_ms=hint.duration_ms,
            start_sec=hint.start_sec,
            end_sec=hint.end_sec,
            release_context=hint.release_context,
            note_binding_strategy=hint.note_binding_strategy,
            note_identity_match=hint.note_identity_match,
            score_bridge_strategy=hint.score_bridge_strategy,
            score_hint_confidence=hint.score_hint_confidence,
            score_hint_notes=list(hint.score_hint_notes),
        )

    def _source_summary(
        self,
        score_identity_source: str | None,
        note_context_adapter: str | None,
        note_context_applied: bool,
        score_bridge_strategy: str | None,
        note_binding_strategy: str | None,
    ) -> str:
        tokens: list[str] = []
        if score_identity_source is not None:
            tokens.append(f"score_identity:{score_identity_source}")
        if note_context_adapter is not None:
            context_state = "bound" if note_context_applied else "fallback"
            tokens.append(f"note_context:{note_context_adapter}:{context_state}")
        if score_bridge_strategy is not None:
            tokens.append(f"score_bridge:{score_bridge_strategy}")
        if note_binding_strategy is not None:
            tokens.append(f"note_binding:{note_binding_strategy}")
        return ",".join(tokens) if tokens else "none"

    def _notes(
        self,
        note_identity_match: bool,
        note_identity_partial_match: bool,
        note_context_applied: bool,
        score_bridge_applied: bool,
        score_bridge_strategy: str | None,
        note_binding_strategy: str | None,
        part_local_note_rank: int | None,
        score_hint_layer_applied: bool,
        note_binding_notes: object,
        score_bridge_notes: object,
    ) -> list[str]:
        notes: list[str] = []
        if note_identity_match:
            if note_identity_partial_match:
                notes.append("hint anchored by partial identity-first note binding")
            else:
                notes.append("hint anchored by identity-first note binding")
        elif score_bridge_applied and score_bridge_strategy is not None:
            notes.append(f"hint anchored by score bridge strategy={score_bridge_strategy}")
        elif note_context_applied and note_binding_strategy is not None:
            notes.append(f"hint anchored by note binding strategy={note_binding_strategy}")
        if part_local_note_rank is not None:
            notes.append(f"part_local_note_rank={part_local_note_rank}")
        self._extend_unique(notes, note_binding_notes)
        self._extend_unique(notes, score_bridge_notes)
        if not score_hint_layer_applied:
            notes.append("no reusable score-to-segment hint could be derived")
        return notes

    def _confidence(
        self,
        normalized_track_id: str | None,
        normalized_part_id: str | None,
        normalized_note_id: str | None,
        part_local_note_rank: int | None,
        note_context_adapter: str | None,
        note_context_applied: bool,
        note_binding_strategy: str | None,
        note_binding_fallback_used: bool,
        note_identity_match: bool,
        note_identity_partial_match: bool,
        score_bridge_applied: bool,
        score_bridge_match_confidence: float | None,
        score_identity_fallback_used: bool,
    ) -> float:
        if note_identity_match:
            confidence = 0.78 if note_identity_partial_match else 0.95
        elif score_bridge_applied:
            bridge_confidence = score_bridge_match_confidence or 0.0
            confidence = 0.35 + (0.45 * bridge_confidence)
            if note_binding_strategy == "identity":
                confidence = max(confidence, 0.82)
        elif note_context_applied:
            if note_binding_strategy == "segment_index":
                confidence = 0.8
            elif note_binding_strategy == "time_overlap":
                confidence = 0.45
            elif note_binding_strategy == "unmatched":
                confidence = 0.05
            else:
                confidence = 0.2
        elif normalized_note_id is not None:
            confidence = 0.4
        elif normalized_part_id is not None or normalized_track_id is not None:
            confidence = 0.22
        else:
            confidence = 0.0

        if part_local_note_rank is not None and confidence > 0.0:
            confidence += 0.05
        if note_context_adapter is not None and confidence > 0.0 and score_bridge_applied:
            confidence += 0.05
        if note_binding_fallback_used and not note_identity_match:
            confidence -= 0.08
        if score_identity_fallback_used and not note_identity_match and normalized_note_id is not None:
            confidence -= 0.05
        return max(0.0, min(1.0, confidence))

    def _first_value(
        self,
        *payloads: dict[str, object],
        keys: tuple[str, ...],
    ) -> str | None:
        for payload in payloads:
            for key in keys:
                value = payload.get(key)
                if value in (None, ""):
                    continue
                return str(value)
        return None

    def _first_int(self, *values: object) -> int | None:
        for value in values:
            normalized = self._int_value(value)
            if normalized is not None:
                return normalized
        return None

    def _first_float(self, *values: object) -> float | None:
        for value in values:
            normalized = self._float_value(value)
            if normalized is not None:
                return normalized
        return None

    def _text(self, value: object) -> str | None:
        if value in (None, ""):
            return None
        return str(value)

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

    def _extend_unique(self, target: list[str], values: object) -> None:
        if not isinstance(values, list):
            return
        for value in values:
            text = self._text(value)
            if text is not None and text not in target:
                target.append(text)


def _legacy_identity_value(
    segment: PhoneSegment,
    score_identity: dict[str, object],
    note_context: dict[str, object],
    key: str,
) -> str | None:
    value = score_identity.get(key)
    if value not in (None, ""):
        return str(value)
    value = segment.metadata.get(key)
    if value is None:
        value = note_context.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _hint_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _hint_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hint_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _hint_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None

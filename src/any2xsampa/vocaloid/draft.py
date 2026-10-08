from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


VOCAL_TRACK_DRAFT_SCHEMA_VERSION = "vocal_track_draft_v1"
DEFAULT_SINGER_TARGET = "miku_v4c"
DEFAULT_TICKS_PER_QUARTER = 480
DEFAULT_NOTE_DURATION_TICKS = 480


@dataclass(slots=True)
class VocalNoteDraft:
    id: str
    start_tick: int
    duration_ticks: int
    pitch: int
    lyric: str
    phonemes: list[str] = field(default_factory=list)
    phoneme_source: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "VocalNoteDraft":
        return cls(
            id=str(payload.get("id", "note")),
            start_tick=int(payload.get("start_tick", 0)),
            duration_ticks=int(payload.get("duration_ticks", DEFAULT_NOTE_DURATION_TICKS)),
            pitch=int(payload.get("pitch", 60)),
            lyric=str(payload.get("lyric", "la")),
            phonemes=[str(value) for value in payload.get("phonemes", [])],
            phoneme_source=(str(payload["phoneme_source"]) if payload.get("phoneme_source") is not None else None),
            provenance=dict(payload.get("provenance", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "start_tick": self.start_tick,
            "duration_ticks": self.duration_ticks,
            "pitch": self.pitch,
            "lyric": self.lyric,
            "phonemes": list(self.phonemes),
        }
        if self.phoneme_source is not None:
            payload["phoneme_source"] = self.phoneme_source
        if self.provenance:
            payload["provenance"] = dict(self.provenance)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class VocalTrackDraftTrack:
    id: str
    name: str
    singer_target: str
    notes: list[VocalNoteDraft] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "VocalTrackDraftTrack":
        notes_payload = payload.get("notes", [])
        notes = [VocalNoteDraft.from_mapping(item) for item in notes_payload if isinstance(item, dict)]
        return cls(
            id=str(payload.get("id", "track-1")),
            name=str(payload.get("name", "Any2Xsampa track")),
            singer_target=str(payload.get("singer_target", DEFAULT_SINGER_TARGET)),
            notes=notes,
            provenance=dict(payload.get("provenance", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "singer_target": self.singer_target,
            "notes": [note.to_mapping() for note in self.notes],
        }
        if self.provenance:
            payload["provenance"] = dict(self.provenance)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class VocalTrackDraft:
    schema_version: str = VOCAL_TRACK_DRAFT_SCHEMA_VERSION
    tempo: float = 120.0
    time_signature: dict[str, int] = field(
        default_factory=lambda: {"numerator": 4, "denominator": 4}
    )
    ticks_per_quarter: int = DEFAULT_TICKS_PER_QUARTER
    singer_target: str = DEFAULT_SINGER_TARGET
    tracks: list[VocalTrackDraftTrack] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "VocalTrackDraft":
        schema_version = str(payload.get("schema_version", VOCAL_TRACK_DRAFT_SCHEMA_VERSION))
        if schema_version != VOCAL_TRACK_DRAFT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported vocal track draft schema: {schema_version}")
        tracks_payload = payload.get("tracks", [])
        tracks = [VocalTrackDraftTrack.from_mapping(item) for item in tracks_payload if isinstance(item, dict)]
        if not tracks:
            raise ValueError("vocal track draft must contain at least one track")
        signature = payload.get("time_signature", {"numerator": 4, "denominator": 4})
        return cls(
            schema_version=schema_version,
            tempo=float(payload.get("tempo", 120.0)),
            time_signature={
                "numerator": int(signature.get("numerator", 4)),
                "denominator": int(signature.get("denominator", 4)),
            },
            ticks_per_quarter=int(payload.get("ticks_per_quarter", DEFAULT_TICKS_PER_QUARTER)),
            singer_target=str(payload.get("singer_target", DEFAULT_SINGER_TARGET)),
            tracks=tracks,
            provenance=dict(payload.get("provenance", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tempo": self.tempo,
            "time_signature": dict(self.time_signature),
            "ticks_per_quarter": self.ticks_per_quarter,
            "singer_target": self.singer_target,
            "tracks": [track.to_mapping() for track in self.tracks],
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }


def load_vocal_track_draft(path: Path | str) -> VocalTrackDraft:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("vocal track draft must be a JSON object")
    return VocalTrackDraft.from_mapping(payload)


__all__ = [
    "DEFAULT_NOTE_DURATION_TICKS",
    "DEFAULT_SINGER_TARGET",
    "DEFAULT_TICKS_PER_QUARTER",
    "VOCAL_TRACK_DRAFT_SCHEMA_VERSION",
    "VocalNoteDraft",
    "VocalTrackDraft",
    "VocalTrackDraftTrack",
    "load_vocal_track_draft",
]

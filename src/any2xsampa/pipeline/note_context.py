from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from any2xsampa.core.models import PhoneSegment


_NOTE_OFFSETS = {
    "c": 0,
    "c#": 1,
    "db": 1,
    "d": 2,
    "d#": 3,
    "eb": 3,
    "e": 4,
    "f": 5,
    "f#": 6,
    "gb": 6,
    "g": 7,
    "g#": 8,
    "ab": 8,
    "a": 9,
    "a#": 10,
    "bb": 10,
    "b": 11,
}


@dataclass(slots=True)
class NoteContextEntry:
    note_midi: int | None = None
    note_name: str | None = None
    duration_ms: float | None = None
    beat: float | None = None
    tick: int | None = None
    release: str | None = None
    segment_index: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    track_id: str | None = None
    part_id: str | None = None
    note_id: str | None = None
    note_identity_source: str | None = None
    note_identity_fallback_used: bool = False
    part_local_note_rank: int | None = None
    part_local_rank_source: str | None = None
    part_local_rank_fallback_used: bool = False


@dataclass(slots=True)
class TempoPoint:
    position: int
    bpm: float


@dataclass(slots=True)
class NoteContextSource:
    text: str
    json_payload: object | None = None


@dataclass(slots=True)
class NoteBindingResult:
    entry: NoteContextEntry | None
    strategy: str
    identity_match: bool = False
    identity_partial_match: bool = False
    fallback_used: bool = False
    notes: list[str] = field(default_factory=list)


def normalize_note_midi(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        pass
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text)
    if match is None:
        return None
    note_name = f"{match.group(1).lower()}{match.group(2).lower()}"
    offset = _NOTE_OFFSETS.get(note_name)
    if offset is None:
        return None
    octave = int(match.group(3))
    return ((octave + 1) * 12) + offset


def normalize_release(value: object) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return None
    if text in {"short", "tight", "clipped", "staccato"}:
        return "short"
    if text in {"long", "tail", "linger", "sustain", "sustained", "legato"}:
        return "long"
    return text


def _normalized_duration_ms(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        duration_ms = float(value)
    except (TypeError, ValueError):
        return None
    if duration_ms < 0:
        return None
    return duration_ms


def _normalized_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _normalized_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _normalized_identifier(value: object) -> str | None:
    text = _normalized_text(value)
    return text if text else None


def _load_note_context_source(note_context_path: Path) -> NoteContextSource:
    text = note_context_path.read_text(encoding="utf-8")
    json_payload: object | None = None
    if note_context_path.suffix.lower() in {".json", ".ustx"}:
        json_payload = json.loads(text)
    else:
        stripped = text.lstrip("\ufeff").strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json_payload = json.loads(text)
            except json.JSONDecodeError:
                json_payload = None
    return NoteContextSource(text=text, json_payload=json_payload)


def _note_name_from_midi(note_midi: int | None) -> str | None:
    if note_midi is None:
        return None
    note_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    octave = (note_midi // 12) - 1
    return f"{note_names[note_midi % 12]}{octave}"


def _stable_key(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-") or "x"


def _time_token(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.3f}".replace(".", "p")


def _first_identifier(*values: object) -> str | None:
    for value in values:
        identifier = _normalized_identifier(value)
        if identifier:
            return identifier
    return None


def _json_identity(
    raw_entry: dict[str, object],
    entry_index: int,
) -> tuple[str, str, str, str, bool]:
    track_id = _first_identifier(raw_entry.get("track_id"), raw_entry.get("track"))
    part_id = _first_identifier(raw_entry.get("part_id"), raw_entry.get("part"))
    note_id = _first_identifier(raw_entry.get("note_id"), raw_entry.get("id"))
    fallback_used = False
    source_tokens: list[str] = []

    if track_id is None:
        track_id = "json-track-0"
        fallback_used = True
        source_tokens.append("track=default")
    else:
        source_tokens.append("track=explicit")
    if part_id is None:
        part_id = f"{track_id}:part-0"
        fallback_used = True
        source_tokens.append("part=default")
    else:
        source_tokens.append("part=explicit")
    if note_id is None:
        if (segment_index := _normalized_int(raw_entry.get("segment_index"))) is not None:
            note_id = f"{part_id}:segment-{segment_index}"
            source_tokens.append("note=segment_index")
        elif (tick := _normalized_int(raw_entry.get("tick"))) is not None:
            note_id = f"{part_id}:tick-{tick}"
            source_tokens.append("note=tick")
        else:
            start_sec = _normalized_float(raw_entry.get("start_sec"))
            end_sec = _normalized_float(raw_entry.get("end_sec"))
            if start_sec is not None or end_sec is not None:
                start_token = _time_token(start_sec) or "na"
                end_token = _time_token(end_sec) or "na"
                note_id = f"{part_id}:span-{start_token}-{end_token}"
                source_tokens.append("note=time_span")
            else:
                note_id = f"{part_id}:entry-{entry_index}"
                source_tokens.append("note=entry_index")
        fallback_used = True
    else:
        source_tokens.append("note=explicit")
    source = "json:" + ",".join(source_tokens)
    return (track_id, part_id, note_id, source, fallback_used)


def _with_identity(
    entry: NoteContextEntry,
    track_id: str,
    part_id: str,
    note_id: str,
    source: str,
    fallback_used: bool,
) -> NoteContextEntry:
    entry.track_id = track_id
    entry.part_id = part_id
    entry.note_id = note_id
    entry.note_identity_source = source
    entry.note_identity_fallback_used = fallback_used
    return entry


def _part_local_primary(entry: NoteContextEntry, entry_index: int) -> tuple[int, object, str, bool]:
    if entry.tick is not None:
        return (0, entry.tick, "tick", False)
    if entry.start_sec is not None:
        return (1, int(round(entry.start_sec * 1000.0)), "start_sec", False)
    if entry.duration_ms is not None:
        return (2, int(round(entry.duration_ms)), "duration", True)
    if entry.note_id is not None:
        return (3, entry.note_id, "note_id", True)
    return (4, entry_index, "entry_order", True)


def _assign_part_local_note_ranks(entries: list[NoteContextEntry]) -> list[NoteContextEntry]:
    part_groups: dict[str, list[tuple[int, NoteContextEntry]]] = {}
    for entry_index, entry in enumerate(entries):
        part_key = entry.part_id or entry.track_id or "part:unknown"
        part_groups.setdefault(part_key, []).append((entry_index, entry))

    for grouped_entries in part_groups.values():
        ordered = sorted(
            grouped_entries,
            key=lambda item: (
                *_part_local_primary(item[1], item[0])[:2],
                int(round(item[1].end_sec * 1000.0)) if item[1].end_sec is not None else 10**12,
                int(round(item[1].duration_ms)) if item[1].duration_ms is not None else 10**12,
                item[1].note_midi if item[1].note_midi is not None else 10**6,
                item[1].note_id or "",
                item[0],
            ),
        )
        for rank, (entry_index, entry) in enumerate(ordered):
            _, _, source, fallback_used = _part_local_primary(entry, entry_index)
            entry.part_local_note_rank = rank
            entry.part_local_rank_source = source
            entry.part_local_rank_fallback_used = fallback_used
    return entries


def _build_note_context_entry(raw_entry: dict[str, object], entry_index: int = 0) -> NoteContextEntry | None:
    entry = NoteContextEntry(
        note_midi=normalize_note_midi(raw_entry.get("note_midi", raw_entry.get("midi"))),
        note_name=_normalized_text(raw_entry.get("note_name", raw_entry.get("note"))),
        duration_ms=_normalized_duration_ms(raw_entry.get("duration_ms", raw_entry.get("duration"))),
        beat=_normalized_float(raw_entry.get("beat")),
        tick=_normalized_int(raw_entry.get("tick")),
        release=normalize_release(raw_entry.get("release", raw_entry.get("note_end_context"))),
        segment_index=_normalized_int(raw_entry.get("segment_index")),
        start_sec=_normalized_float(raw_entry.get("start_sec")),
        end_sec=_normalized_float(raw_entry.get("end_sec")),
    )
    if (
        entry.note_midi is None
        and entry.note_name is None
        and entry.duration_ms is None
        and entry.beat is None
        and entry.tick is None
        and entry.release is None
    ):
        return None
    if entry.duration_ms is None and entry.start_sec is not None and entry.end_sec is not None:
        entry.duration_ms = max(0.0, (entry.end_sec - entry.start_sec) * 1000.0)
    track_id, part_id, note_id, source, fallback_used = _json_identity(raw_entry, entry_index)
    return _with_identity(entry, track_id, part_id, note_id, source, fallback_used)


class ScoreNoteContextAdapter(ABC):
    name: str

    @abstractmethod
    def supports(self, note_context_path: Path, source: NoteContextSource) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self, note_context_path: Path, source: NoteContextSource) -> list[NoteContextEntry]:
        raise NotImplementedError


class UstxScoreProjectAdapter(ScoreNoteContextAdapter):
    name = "ustx"

    def supports(self, note_context_path: Path, source: NoteContextSource) -> bool:
        if note_context_path.suffix.lower() == ".ustx":
            return True
        payload = source.json_payload
        if not isinstance(payload, dict):
            return False
        return isinstance(payload.get("voice_parts"), list) or self._has_track_parts(payload)

    def load(self, note_context_path: Path, source: NoteContextSource) -> list[NoteContextEntry]:
        payload = source.json_payload
        if not isinstance(payload, dict):
            raise ValueError(f"USTX note context must be a JSON object: {note_context_path}")
        resolution = _normalized_int(payload.get("resolution")) or 480
        tempo_map = self._tempo_map(payload)
        entries: list[NoteContextEntry] = []
        for track_index, track, part_index, part in self._voice_parts(payload):
            part_position = _normalized_int(part.get("position")) or 0
            track_id, track_fallback = self._ustx_track_id(track, track_index)
            part_id, part_fallback = self._ustx_part_id(part, track_id, part_index)
            raw_notes = part.get("notes", [])
            if not isinstance(raw_notes, list):
                continue
            for note_index, raw_note in enumerate(raw_notes):
                if not isinstance(raw_note, dict):
                    continue
                duration_ticks = _normalized_int(raw_note.get("duration"))
                if duration_ticks is None or duration_ticks <= 0:
                    continue
                note_position = _normalized_int(raw_note.get("position")) or 0
                absolute_tick = part_position + note_position
                note_midi = normalize_note_midi(
                    raw_note.get("tone", raw_note.get("note_midi", raw_note.get("midi")))
                )
                note_id, note_fallback = self._ustx_note_id(raw_note, part_id, absolute_tick, duration_ticks, note_index)
                start_sec = self._ticks_to_seconds(absolute_tick, tempo_map, resolution)
                end_sec = self._ticks_to_seconds(absolute_tick + duration_ticks, tempo_map, resolution)
                entry = NoteContextEntry(
                    note_midi=note_midi,
                    note_name=_note_name_from_midi(note_midi),
                    duration_ms=max(0.0, (end_sec - start_sec) * 1000.0),
                    beat=absolute_tick / float(resolution),
                    tick=absolute_tick,
                    release=normalize_release(
                        raw_note.get("release", raw_note.get("note_end_context", raw_note.get("tail")))
                    ),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    track_id=track_id,
                    part_id=part_id,
                    note_id=note_id,
                    note_identity_source=self._ustx_identity_source(track_fallback, part_fallback, note_fallback),
                    note_identity_fallback_used=track_fallback or part_fallback or note_fallback,
                )
                entries.append(entry)
        if not entries:
            raise ValueError(f"No usable USTX note entries found: {note_context_path}")
        return entries

    def _has_track_parts(self, payload: dict[str, object]) -> bool:
        tracks = payload.get("tracks")
        if not isinstance(tracks, list):
            return False
        for track in tracks:
            if not isinstance(track, dict):
                continue
            if isinstance(track.get("parts"), list) or isinstance(track.get("voice_parts"), list):
                return True
        return False

    def _voice_parts(self, payload: dict[str, object]) -> list[tuple[int, dict[str, object], int, dict[str, object]]]:
        parts: list[tuple[int, dict[str, object], int, dict[str, object]]] = []
        voice_parts = payload.get("voice_parts")
        if isinstance(voice_parts, list):
            synthetic_track: dict[str, object] = {"track_id": "ustx-track-0"}
            for part_index, part in enumerate(voice_parts):
                if isinstance(part, dict):
                    parts.append((0, synthetic_track, part_index, part))
        tracks = payload.get("tracks")
        if isinstance(tracks, list):
            for track_index, track in enumerate(tracks):
                if not isinstance(track, dict):
                    continue
                raw_parts = track.get("parts", track.get("voice_parts", []))
                if not isinstance(raw_parts, list):
                    continue
                for part_index, part in enumerate(raw_parts):
                    if isinstance(part, dict):
                        parts.append((track_index, track, part_index, part))
        return parts

    def _ustx_track_id(self, track: dict[str, object], track_index: int) -> tuple[str, bool]:
        track_id = _first_identifier(track.get("track_id"), track.get("id"))
        if track_id is not None:
            return (track_id, False)
        return (f"ustx-track-{track_index}", True)

    def _ustx_part_id(self, part: dict[str, object], track_id: str, part_index: int) -> tuple[str, bool]:
        part_id = _first_identifier(part.get("part_id"), part.get("id"))
        if part_id is not None:
            return (part_id, False)
        return (f"{track_id}:part-{part_index}", True)

    def _ustx_note_id(
        self,
        raw_note: dict[str, object],
        part_id: str,
        absolute_tick: int,
        duration_ticks: int,
        note_index: int,
    ) -> tuple[str, bool]:
        note_id = _first_identifier(raw_note.get("note_id"), raw_note.get("id"))
        if note_id is not None:
            return (note_id, False)
        note_midi = normalize_note_midi(raw_note.get("tone", raw_note.get("note_midi", raw_note.get("midi"))))
        midi_token = note_midi if note_midi is not None else "x"
        return (f"{part_id}:tick-{absolute_tick}:dur-{duration_ticks}:midi-{midi_token}:idx-{note_index}", True)

    def _ustx_identity_source(self, track_fallback: bool, part_fallback: bool, note_fallback: bool) -> str:
        tokens = [
            "track=fallback" if track_fallback else "track=explicit",
            "part=fallback" if part_fallback else "part=explicit",
            "note=fallback" if note_fallback else "note=explicit",
        ]
        return "ustx:" + ",".join(tokens)

    def _tempo_map(self, payload: dict[str, object]) -> list[TempoPoint]:
        raw_tempos = payload.get("tempos")
        if not isinstance(raw_tempos, list):
            return [TempoPoint(position=0, bpm=120.0)]
        tempos: list[TempoPoint] = []
        for raw_tempo in raw_tempos:
            if not isinstance(raw_tempo, dict):
                continue
            position = _normalized_int(raw_tempo.get("position"))
            bpm = _normalized_float(raw_tempo.get("bpm"))
            if position is None or bpm is None or bpm <= 0:
                continue
            tempos.append(TempoPoint(position=position, bpm=bpm))
        if not tempos:
            return [TempoPoint(position=0, bpm=120.0)]
        tempos.sort(key=lambda tempo: tempo.position)
        if tempos[0].position != 0:
            tempos.insert(0, TempoPoint(position=0, bpm=tempos[0].bpm))
        return tempos

    def _ticks_to_seconds(self, tick: int, tempos: list[TempoPoint], resolution: int) -> float:
        seconds = 0.0
        current_tick = 0
        current_bpm = tempos[0].bpm
        for tempo in tempos[1:]:
            if tick <= tempo.position:
                break
            span_ticks = max(0, tempo.position - current_tick)
            seconds += self._span_seconds(span_ticks, current_bpm, resolution)
            current_tick = tempo.position
            current_bpm = tempo.bpm
        seconds += self._span_seconds(max(0, tick - current_tick), current_bpm, resolution)
        return seconds

    def _span_seconds(self, tick_span: int, bpm: float, resolution: int) -> float:
        seconds_per_tick = 60.0 / (bpm * float(resolution))
        return tick_span * seconds_per_tick


class UstScoreProjectAdapter(ScoreNoteContextAdapter):
    name = "ust"

    def supports(self, note_context_path: Path, source: NoteContextSource) -> bool:
        if note_context_path.suffix.lower() == ".ust":
            return True
        text = source.text.lstrip("\ufeff")
        return "[#SETTING]" in text and "Length=" in text

    def load(self, note_context_path: Path, source: NoteContextSource) -> list[NoteContextEntry]:
        sections = self._sections(source.text)
        default_tempo = self._default_tempo(sections)
        resolution = 480
        current_tick = 0
        current_time_sec = 0.0
        current_bpm = default_tempo
        entries: list[NoteContextEntry] = []
        track_id = "ust-track-0"
        part_id = "ust-track-0:part-main"

        for section_name, fields in sections:
            if not self._is_note_section(section_name):
                continue
            length_ticks = _normalized_int(fields.get("Length", fields.get("length")))
            if length_ticks is None or length_ticks <= 0:
                continue
            note_bpm = _normalized_float(fields.get("Tempo", fields.get("tempo"))) or current_bpm
            note_midi = normalize_note_midi(fields.get("NoteNum", fields.get("notenum")))
            duration_sec = self._span_seconds(length_ticks, note_bpm, resolution)
            start_sec = current_time_sec
            end_sec = current_time_sec + duration_sec
            release = normalize_release(
                fields.get(
                    "Release",
                    fields.get("release", fields.get("NoteEndContext", fields.get("NoteEnd", fields.get("Tail")))),
                )
            )
            entry = NoteContextEntry(
                note_midi=note_midi,
                note_name=_note_name_from_midi(note_midi),
                duration_ms=max(0.0, duration_sec * 1000.0),
                beat=current_tick / float(resolution),
                tick=current_tick,
                release=release,
                start_sec=start_sec,
                end_sec=end_sec,
                track_id=track_id,
                part_id=part_id,
                note_id=f"{part_id}:{section_name.lower()}",
                note_identity_source="ust:section_name_fallback",
                note_identity_fallback_used=True,
            )
            if (
                entry.note_midi is None
                and entry.note_name is None
                and entry.duration_ms is None
                and entry.beat is None
                and entry.tick is None
                and entry.release is None
            ):
                current_tick += length_ticks
                current_time_sec = end_sec
                current_bpm = note_bpm
                continue
            entries.append(entry)
            current_tick += length_ticks
            current_time_sec = end_sec
            current_bpm = note_bpm

        if not entries:
            raise ValueError(f"No usable UST note entries found: {note_context_path}")
        return entries

    def _sections(self, text: str) -> list[tuple[str, dict[str, str]]]:
        sections: list[tuple[str, dict[str, str]]] = []
        current_name: str | None = None
        current_fields: dict[str, str] = {}
        for raw_line in text.lstrip("\ufeff").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                if current_name is not None:
                    sections.append((current_name, current_fields))
                current_name = line[1:-1].strip()
                current_fields = {}
                continue
            if current_name is None or "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            current_fields[key.strip()] = value.strip()
        if current_name is not None:
            sections.append((current_name, current_fields))
        return sections

    def _default_tempo(self, sections: list[tuple[str, dict[str, str]]]) -> float:
        for section_name, fields in sections:
            if section_name.upper() != "#SETTING":
                continue
            tempo = _normalized_float(fields.get("Tempo", fields.get("tempo")))
            if tempo is not None and tempo > 0:
                return tempo
        return 120.0

    def _is_note_section(self, section_name: str) -> bool:
        return re.fullmatch(r"#\d+", section_name.strip()) is not None

    def _span_seconds(self, tick_span: int, bpm: float, resolution: int) -> float:
        seconds_per_tick = 60.0 / (bpm * float(resolution))
        return tick_span * seconds_per_tick


class JsonNoteContextAdapter(ScoreNoteContextAdapter):
    name = "json"

    def supports(self, note_context_path: Path, source: NoteContextSource) -> bool:
        payload = source.json_payload
        if isinstance(payload, list):
            return True
        if not isinstance(payload, dict):
            return False
        raw_entries: object
        raw_entries = payload.get("segments", payload.get("spans", payload.get("entries", [])))
        return isinstance(raw_entries, list)

    def load(self, note_context_path: Path, source: NoteContextSource) -> list[NoteContextEntry]:
        payload = source.json_payload
        raw_entries: object
        if isinstance(payload, list):
            raw_entries = payload
        elif isinstance(payload, dict):
            raw_entries = payload.get("segments", payload.get("spans", payload.get("entries", [])))
        else:
            raw_entries = []
        if not isinstance(raw_entries, list):
            raise ValueError(f"Note context JSON must contain a list under 'segments', 'spans', or 'entries': {note_context_path}")

        entries: list[NoteContextEntry] = []
        for entry_index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict):
                continue
            entry = _build_note_context_entry(raw_entry, entry_index=entry_index)
            if entry is None:
                continue
            entries.append(entry)
        if not entries:
            raise ValueError(f"No usable note context entries found: {note_context_path}")
        return entries


class ScoreNoteContextLoader:
    def __init__(self, adapters: list[ScoreNoteContextAdapter] | None = None) -> None:
        self.adapters = adapters or [UstScoreProjectAdapter(), UstxScoreProjectAdapter(), JsonNoteContextAdapter()]

    def load(
        self,
        note_context_path: Path,
        note_context_format: str | None = None,
    ) -> tuple[str, list[NoteContextEntry]]:
        source = _load_note_context_source(note_context_path)
        if note_context_format and note_context_format != "auto":
            adapter = self._adapter_by_name(note_context_format)
            return adapter.name, _assign_part_local_note_ranks(adapter.load(note_context_path, source))
        for adapter in self.adapters:
            if adapter.supports(note_context_path, source):
                return adapter.name, _assign_part_local_note_ranks(adapter.load(note_context_path, source))
        raise ValueError(f"Unsupported note/score context format: {note_context_path}")

    def _adapter_by_name(self, adapter_name: str) -> ScoreNoteContextAdapter:
        normalized = str(adapter_name).strip().lower()
        for adapter in self.adapters:
            if adapter.name == normalized:
                return adapter
        available = ", ".join(adapter.name for adapter in self.adapters)
        raise ValueError(f"Unknown note context format '{adapter_name}'. Available formats: {available}")


class NoteContextApplier:
    def __init__(self) -> None:
        self.loader = ScoreNoteContextLoader()

    def apply(
        self,
        segments: list[PhoneSegment],
        note_context_path: Path | None,
        note_context_format: str | None = None,
    ) -> list[str]:
        if note_context_path is None:
            for segment in segments:
                self._mark_no_context(segment)
            return []

        adapter_name, entries = self.loader.load(note_context_path, note_context_format=note_context_format)
        entry_by_index = {
            entry.segment_index: entry
            for entry in entries
            if entry.segment_index is not None
        }
        span_entries = [
            entry
            for entry in entries
            if entry.segment_index is None and entry.start_sec is not None and entry.end_sec is not None
        ]

        unmatched = 0
        source = str(note_context_path)
        for index, segment in enumerate(segments):
            binding = self._resolve_binding(index, segment, entries, entry_by_index, span_entries)
            if binding.entry is None:
                unmatched += 1
                self._mark_unmatched(segment, source, adapter_name, binding)
                continue
            self._apply_entry(segment, binding.entry, source, binding, adapter_name)

        warnings: list[str] = []
        if unmatched:
            warnings.append(
                f"note context matched {len(segments) - unmatched}/{len(segments)} segments from {note_context_path.name}"
            )
        return warnings

    def _resolve_binding(
        self,
        segment_index: int,
        segment: PhoneSegment,
        entries: list[NoteContextEntry],
        entry_by_index: dict[int, NoteContextEntry],
        span_entries: list[NoteContextEntry],
    ) -> NoteBindingResult:
        identity_binding = self._identity_binding(segment_index, segment, entries, entry_by_index, span_entries)
        if identity_binding.entry is not None:
            return identity_binding

        matched_entry = entry_by_index.get(segment_index)
        if matched_entry is not None:
            notes = list(identity_binding.notes)
            notes.append("fell back to segment_index note binding")
            return NoteBindingResult(
                entry=matched_entry,
                strategy="segment_index",
                identity_match=False,
                identity_partial_match=False,
                fallback_used=True,
                notes=notes,
            )

        matched_entry = self._best_span_match(segment, span_entries)
        if matched_entry is not None:
            notes = list(identity_binding.notes)
            notes.append("fell back to time_overlap note binding")
            return NoteBindingResult(
                entry=matched_entry,
                strategy="time_overlap",
                identity_match=False,
                identity_partial_match=False,
                fallback_used=True,
                notes=notes,
            )

        notes = list(identity_binding.notes)
        if not notes:
            notes.append("no matching note-context entry found")
        return NoteBindingResult(
            entry=None,
            strategy="unmatched",
            identity_match=False,
            identity_partial_match=False,
            fallback_used=True,
            notes=notes,
        )

    def _identity_binding(
        self,
        segment_index: int,
        segment: PhoneSegment,
        entries: list[NoteContextEntry],
        entry_by_index: dict[int, NoteContextEntry],
        span_entries: list[NoteContextEntry],
    ) -> NoteBindingResult:
        segment_identity = self._segment_identity(segment)
        available_fields = [field for field in ("track_id", "part_id", "note_id") if segment_identity[field] is not None]
        if not available_fields:
            return NoteBindingResult(
                entry=None,
                strategy="identity_unavailable",
                identity_match=False,
                identity_partial_match=False,
                fallback_used=True,
                notes=["segment exposes no normalized score identity; using fallback note binding"],
            )

        candidates: list[tuple[NoteContextEntry, list[str], list[str], int]] = []
        for entry_index, entry in enumerate(entries):
            matched_fields: list[str] = []
            conflicting_fields: list[str] = []
            for field_name in available_fields:
                entry_value = getattr(entry, field_name)
                segment_value = segment_identity[field_name]
                if entry_value is None or segment_value is None:
                    continue
                if entry_value == segment_value:
                    matched_fields.append(field_name)
                else:
                    conflicting_fields.append(field_name)
            if conflicting_fields or not matched_fields:
                continue
            candidates.append((entry, matched_fields, conflicting_fields, entry_index))

        if not candidates:
            return NoteBindingResult(
                entry=None,
                strategy="identity_unmatched",
                identity_match=False,
                identity_partial_match=False,
                fallback_used=True,
                notes=[
                    "segment identity did not match any note-context entry; using fallback note binding",
                    f"identity fields available={','.join(available_fields)}",
                ],
            )

        best_match_count = max(len(matched_fields) for _, matched_fields, _, _ in candidates)
        best_candidates = [
            (entry, matched_fields, entry_index)
            for entry, matched_fields, _, entry_index in candidates
            if len(matched_fields) == best_match_count
        ]
        partial_match = best_match_count < 3
        notes = [f"identity binding matched fields={','.join(best_candidates[0][1])}"]
        if partial_match:
            missing_fields = [field for field in ("track_id", "part_id", "note_id") if field not in best_candidates[0][1]]
            notes.append(f"identity binding degraded; missing fields={','.join(missing_fields)}")

        chosen_entry: NoteContextEntry | None = None
        tie_break_notes: list[str] = []
        if len(best_candidates) == 1:
            chosen_entry = best_candidates[0][0]
        else:
            chosen_entry, tie_break_notes = self._identity_tiebreak(
                segment_index,
                segment,
                best_candidates,
                entry_by_index,
                span_entries,
            )
            notes.extend(tie_break_notes)

        if chosen_entry is None:
            notes.append("identity-compatible entries remained ambiguous after tie-break")
            return NoteBindingResult(
                entry=None,
                strategy="identity_unmatched",
                identity_match=False,
                identity_partial_match=False,
                fallback_used=True,
                notes=notes,
            )

        strategy = "identity_partial" if partial_match else "identity"
        return NoteBindingResult(
            entry=chosen_entry,
            strategy=strategy,
            identity_match=True,
            identity_partial_match=partial_match,
            fallback_used=partial_match,
            notes=notes,
        )

    def _best_span_match(self, segment: PhoneSegment, entries: list[NoteContextEntry]) -> NoteContextEntry | None:
        best_entry: NoteContextEntry | None = None
        best_overlap = 0.0
        segment_start = segment.time_range.start_sec
        segment_end = segment.time_range.end_sec
        for entry in entries:
            assert entry.start_sec is not None
            assert entry.end_sec is not None
            overlap = min(segment_end, entry.end_sec) - max(segment_start, entry.start_sec)
            if overlap <= 0:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_entry = entry
        return best_entry

    def _identity_tiebreak(
        self,
        segment_index: int,
        segment: PhoneSegment,
        candidates: list[tuple[NoteContextEntry, list[str], int]],
        entry_by_index: dict[int, NoteContextEntry],
        span_entries: list[NoteContextEntry],
    ) -> tuple[NoteContextEntry | None, list[str]]:
        notes: list[str] = [f"identity binding found {len(candidates)} compatible entries; applying fallback tie-break"]
        indexed_entry = entry_by_index.get(segment_index)
        if indexed_entry is not None:
            for entry, _, _ in candidates:
                if entry is indexed_entry:
                    notes.append("identity tie-break resolved by segment_index")
                    return entry, notes

        best_overlap_entry = self._best_span_match(segment, [entry for entry, _, _ in candidates if entry in span_entries])
        if best_overlap_entry is not None:
            notes.append("identity tie-break resolved by time_overlap")
            return best_overlap_entry, notes

        candidate_entries = sorted(candidates, key=lambda item: item[2])
        notes.append("identity tie-break resolved by stable note-context order")
        return candidate_entries[0][0], notes

    def _segment_identity(self, segment: PhoneSegment) -> dict[str, str | None]:
        score_identity = segment.metadata.get("score_identity")
        score_identity_map = score_identity if isinstance(score_identity, dict) else {}
        return {
            "track_id": self._segment_identity_value(
                score_identity_map,
                "normalized_track_id",
                "track_id",
                "track",
            ),
            "part_id": self._segment_identity_value(
                score_identity_map,
                "normalized_part_id",
                "part_id",
                "part",
            ),
            "note_id": self._segment_identity_value(
                score_identity_map,
                "normalized_note_id",
                "note_id",
                "note",
            ),
        }

    def _segment_identity_value(
        self,
        score_identity: dict[str, object],
        *keys: str,
    ) -> str | None:
        for key in keys:
            value = score_identity.get(key)
            if value not in (None, ""):
                return str(value)
        return None

    def _apply_entry(
        self,
        segment: PhoneSegment,
        entry: NoteContextEntry,
        source: str,
        binding: NoteBindingResult,
        adapter_name: str,
    ) -> None:
        note_context = {
            "note_midi": entry.note_midi,
            "note_name": entry.note_name,
            "duration_ms": entry.duration_ms,
            "beat": entry.beat,
            "tick": entry.tick,
            "release_context": entry.release,
            "start_sec": entry.start_sec,
            "end_sec": entry.end_sec,
            "track_id": entry.track_id,
            "part_id": entry.part_id,
            "note_id": entry.note_id,
            "normalized_track_id": entry.track_id,
            "normalized_part_id": entry.part_id,
            "normalized_note_id": entry.note_id,
            "note_identity_source": entry.note_identity_source,
            "note_identity_fallback_used": entry.note_identity_fallback_used,
            "part_local_note_rank": entry.part_local_note_rank,
            "part_local_rank_source": entry.part_local_rank_source,
            "part_local_rank_fallback_used": entry.part_local_rank_fallback_used,
            "match_strategy": binding.strategy,
            "note_binding_strategy": binding.strategy,
            "note_identity_match": binding.identity_match,
            "note_identity_partial_match": binding.identity_partial_match,
            "note_binding_fallback_used": binding.fallback_used,
            "note_binding_notes": list(binding.notes),
        }
        segment.metadata["note_context"] = {key: value for key, value in note_context.items() if value is not None}
        segment.metadata["note_context_adapter"] = adapter_name
        score_identity = self._score_identity(segment)
        segment.metadata["normalized_track_id"] = score_identity["normalized_track_id"] or entry.track_id
        segment.metadata["normalized_part_id"] = score_identity["normalized_part_id"] or entry.part_id
        segment.metadata["normalized_note_id"] = score_identity["normalized_note_id"] or entry.note_id
        segment.metadata["note_identity_source"] = entry.note_identity_source
        segment.metadata["note_identity_fallback_used"] = entry.note_identity_fallback_used
        segment.metadata["part_local_note_rank"] = entry.part_local_note_rank
        segment.metadata["part_local_rank_source"] = entry.part_local_rank_source
        segment.metadata["part_local_rank_fallback_used"] = entry.part_local_rank_fallback_used
        segment.metadata["note_binding_strategy"] = binding.strategy
        segment.metadata["note_identity_match"] = binding.identity_match
        segment.metadata["note_identity_partial_match"] = binding.identity_partial_match
        segment.metadata["note_binding_fallback_used"] = binding.fallback_used
        segment.metadata["note_binding_notes"] = list(binding.notes)
        segment.metadata["note_context_applied"] = True
        segment.metadata["note_context_source"] = source
        segment.metadata["note_context_fallback_used"] = False

    def _mark_unmatched(
        self,
        segment: PhoneSegment,
        source: str,
        adapter_name: str,
        binding: NoteBindingResult,
    ) -> None:
        segment.metadata["note_context"] = {}
        segment.metadata["note_context_adapter"] = adapter_name
        score_identity = self._score_identity(segment)
        segment.metadata["normalized_track_id"] = score_identity["normalized_track_id"]
        segment.metadata["normalized_part_id"] = score_identity["normalized_part_id"]
        segment.metadata["normalized_note_id"] = score_identity["normalized_note_id"]
        segment.metadata["note_identity_source"] = None
        segment.metadata["note_identity_fallback_used"] = False
        segment.metadata["part_local_note_rank"] = None
        segment.metadata["part_local_rank_source"] = None
        segment.metadata["part_local_rank_fallback_used"] = True
        segment.metadata["note_binding_strategy"] = binding.strategy
        segment.metadata["note_identity_match"] = binding.identity_match
        segment.metadata["note_identity_partial_match"] = binding.identity_partial_match
        segment.metadata["note_binding_fallback_used"] = binding.fallback_used
        segment.metadata["note_binding_notes"] = list(binding.notes)
        segment.metadata["note_context_applied"] = False
        segment.metadata["note_context_source"] = source
        segment.metadata["note_context_fallback_used"] = True

    def _mark_no_context(self, segment: PhoneSegment) -> None:
        segment.metadata["note_context"] = {}
        segment.metadata["note_context_adapter"] = None
        score_identity = self._score_identity(segment)
        segment.metadata["normalized_track_id"] = score_identity["normalized_track_id"]
        segment.metadata["normalized_part_id"] = score_identity["normalized_part_id"]
        segment.metadata["normalized_note_id"] = score_identity["normalized_note_id"]
        segment.metadata["note_identity_source"] = None
        segment.metadata["note_identity_fallback_used"] = False
        segment.metadata["part_local_note_rank"] = None
        segment.metadata["part_local_rank_source"] = None
        segment.metadata["part_local_rank_fallback_used"] = True
        segment.metadata["note_binding_strategy"] = "no_context"
        segment.metadata["note_identity_match"] = False
        segment.metadata["note_identity_partial_match"] = False
        segment.metadata["note_binding_fallback_used"] = True
        segment.metadata["note_binding_notes"] = ["no note-context input provided"]
        segment.metadata["note_context_applied"] = False
        segment.metadata["note_context_source"] = None
        segment.metadata["note_context_fallback_used"] = True

    def _score_identity(self, segment: PhoneSegment) -> dict[str, str | None]:
        raw = segment.metadata.get("score_identity")
        payload = raw if isinstance(raw, dict) else {}
        return {
            "normalized_track_id": self._score_identity_value(payload.get("normalized_track_id")),
            "normalized_part_id": self._score_identity_value(payload.get("normalized_part_id")),
            "normalized_note_id": self._score_identity_value(payload.get("normalized_note_id")),
        }

    def _score_identity_value(self, value: object) -> str | None:
        if value in (None, ""):
            return None
        return str(value)

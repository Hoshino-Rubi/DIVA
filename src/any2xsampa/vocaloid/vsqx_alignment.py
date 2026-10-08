"""Post-hoc timing comparison between PhoneSegment JSON and VSQX notes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .vsqx import VSQ4_NS

SCHEMA_VERSION = "vsqx_phone_timing_comparison_v1"


def _f(node: ET.Element | None, default: float = 0.0) -> float:
    try:
        return float((node.text or "").strip()) if node is not None else default
    except (TypeError, ValueError):
        return default


def _tempo_seconds(tick: float, tempos: list[tuple[float, float]], resolution: int) -> float:
    if not tempos:
        tempos = [(0.0, 120.0)]
    tempos = sorted(tempos)
    seconds = 0.0
    previous_tick, previous_bpm = tempos[0]
    if previous_tick > 0:
        previous_tick = 0.0
    for change_tick, bpm in tempos[1:]:
        if tick <= change_tick:
            break
        span = max(0.0, change_tick - previous_tick)
        seconds += span / resolution * 60.0 / previous_bpm
        previous_tick, previous_bpm = change_tick, bpm
    if tick > previous_tick:
        seconds += (tick - previous_tick) / resolution * 60.0 / previous_bpm
    return round(seconds, 6)


def parse_vsqx_timeline(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    root = ET.parse(source).getroot()
    namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else VSQ4_NS
    q = lambda name: f"{{{namespace}}}{name}"
    master = root.find(f".//{q('masterTrack')}")
    resolution = int(_f(master.find(q("resolution")) if master is not None else None, 480))
    tempos: list[tuple[float, float]] = []
    if master is not None:
        for node in master.findall(q("tempo")):
            tempo_tick = _f(node.find(q("t")))
            bpm = _f(node.find(q("v")), 12000.0) / 100.0
            if bpm > 0:
                tempos.append((tempo_tick, bpm))
    if not tempos:
        tempos = [(0.0, 120.0)]
    notes: list[dict[str, Any]] = []
    for part in root.findall(f".//{q('vsPart')}"):
        part_tick = _f(part.find(q("t")))
        for note in part.findall(q("note")):
            start_tick = int(part_tick + _f(note.find(q("t"))))
            duration = int(_f(note.find(q("dur"))))
            end_tick = start_tick + duration
            tokens = (note.findtext(q("p")) or "").split()
            notes.append({
                "start_tick": start_tick,
                "end_tick": end_tick,
                "duration_ticks": duration,
                "start_sec": _tempo_seconds(start_tick, tempos, resolution),
                "end_sec": _tempo_seconds(end_tick, tempos, resolution),
                "pitch": int(_f(note.find(q("n")))) if note.find(q("n")) is not None else None,
                "lyric": note.findtext(q("y")) or "",
                "phonemes": tokens,
            })
    notes.sort(key=lambda item: (item["start_tick"], item["end_tick"]))
    return {
        "schema_version": "vsqx_timeline_v1",
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "resolution": resolution,
        "tempos": [{"tick": tick, "bpm": bpm} for tick, bpm in sorted(tempos)],
        "note_count": len(notes),
        "notes": notes,
    }


def _segment_bounds(segment: dict[str, Any]) -> tuple[float, float, str, dict[str, Any]]:
    start = float(segment.get("start_sec", segment.get("start", 0.0)))
    end = float(segment.get("end_sec", segment.get("end", start)))
    phone = str(segment.get("normalized_ipa") or segment.get("phone") or "")
    return start, end, phone, segment


def compare_phone_segments_to_vsqx(payload: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any]:
    raw_segments = payload.get("segments", []) if isinstance(payload, dict) else []
    segments = [_segment_bounds(item) for item in raw_segments if isinstance(item, dict)]
    silence = {"", "sil", "sp", "<sil>", "SIL"}
    active = [item for item in segments if item[2] not in silence]
    notes = list(timeline.get("notes", []))
    phone_active_duration = sum(max(0.0, end - start) for start, end, _, _ in active)
    note_duration = sum(max(0.0, float(note["end_sec"]) - float(note["start_sec"])) for note in notes)
    overlap_total = 0.0
    per_note: list[dict[str, Any]] = []
    for note in notes:
        ns, ne = float(note["start_sec"]), float(note["end_sec"])
        overlaps: list[dict[str, Any]] = []
        for ps, pe, phone, _ in active:
            overlap = max(0.0, min(pe, ne) - max(ps, ns))
            if overlap:
                overlaps.append({"phone": phone, "overlap_duration_sec": round(overlap, 6), "start_sec": ps, "end_sec": pe})
                overlap_total += overlap
        per_note.append({**note, "phones_by_overlap_duration": overlaps})
    result = {
        "schema_version": "vsqx_phone_timing_comparison_v1",
        "phone_segment_count": len(segments),
        "phone_non_silence_segment_count": len(active),
        "vsqx_note_count": len(notes),
        "phone_non_silence_duration_sec": round(phone_active_duration, 6),
        "vsqx_note_active_duration_sec": round(note_duration, 6),
        "overlap_duration_sec": round(overlap_total, 6),
        "phone_duration_covered_by_notes_ratio": round(overlap_total / phone_active_duration, 6) if phone_active_duration else 0.0,
        "note_duration_covered_by_phones_ratio": round(overlap_total / note_duration, 6) if note_duration else 0.0,
        "per_note": per_note,
    }
    if active and notes:
        result["first_phone_minus_first_note_sec"] = round(active[0][0] - float(notes[0]["start_sec"]), 6)
        result["last_phone_minus_last_note_sec"] = round(active[-1][1] - float(notes[-1]["end_sec"]), 6)
    else:
        result["first_phone_minus_first_note_sec"] = None
        result["last_phone_minus_last_note_sec"] = None
    return result


def build_vsqx_phone_timing_report(phone_segments_path: str | Path, vsqx_path: str | Path) -> dict[str, Any]:
    phone_path = Path(phone_segments_path)
    payload = json.loads(phone_path.read_text(encoding="utf-8-sig"))
    timeline = parse_vsqx_timeline(vsqx_path)
    metadata: dict[str, Any] = {}
    for segment in payload.get("segments", []) if isinstance(payload, dict) else []:
        if isinstance(segment, dict) and isinstance(segment.get("metadata"), dict):
            metadata.update(segment["metadata"])
            break
    producer = metadata.get("producer_raw_model_output") if isinstance(metadata.get("producer_raw_model_output"), dict) else {}
    report = compare_phone_segments_to_vsqx(payload, timeline)
    report.update({
        "schema_version": SCHEMA_VERSION,
        "phone_segments": {
            "path": str(phone_path),
            "sha256": hashlib.sha256(phone_path.read_bytes()).hexdigest(),
            "producer_backend_kind": metadata.get("producer_backend_kind"),
            "raw_model_output_kind": producer.get("raw_model_output_kind"),
        },
        "vsqx": timeline,
        "recognizer_redecoded": False,
        "vsqx_used_as_recognizer_input": False,
        "lyrics_used_as_recognizer_input": False,
    })
    return report


__all__ = ["SCHEMA_VERSION", "parse_vsqx_timeline", "compare_phone_segments_to_vsqx", "build_vsqx_phone_timing_report"]

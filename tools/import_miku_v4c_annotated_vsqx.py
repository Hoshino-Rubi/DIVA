from __future__ import annotations

"""Import human-authored VSQX phoneme annotations as reference evidence."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET


CORPUS_SCHEMA_VERSION = "miku_v4c_annotated_vsqx_corpus_v1"
CHARACTER_MAP_SCHEMA_VERSION = "miku_v4c_annotated_character_map_v1"


def _text(value: object) -> str:
    return str(value or "").strip()


def _lyrics(value: object) -> list[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    return [char for char in _text(value) if not char.isspace()]


def _q(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}" if namespace else tag


def _namespace(root: ET.Element) -> str:
    return root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""


def _parse_segment(spec: Mapping[str, Any], manifest_dir: Path) -> dict[str, Any]:
    source = Path(_text(spec.get("vsqx")))
    if not source.is_absolute():
        source = manifest_dir / source
    root = ET.parse(source).getroot()
    namespace = _namespace(root)
    q = lambda tag: _q(namespace, tag)
    voice_name = root.findtext(f".//{q('vVoice')}/{q('name')}") or ""
    characters = _lyrics(spec.get("lyrics"))
    raw_indexes = spec.get("note_character_indexes")
    tracks = root.findall(q("vsTrack"))
    note_rows: list[tuple[int, int, ET.Element]] = []
    for track_index, track in enumerate(tracks):
        for part in track.findall(q("vsPart")):
            part_start = int(part.findtext(q("t")) or 0)
            for note in part.findall(q("note")):
                note_rows.append((track_index, part_start, note))
    if isinstance(raw_indexes, list):
        character_indexes = [int(value) for value in raw_indexes]
        alignment_mode = "explicit_note_character_indexes"
        if len(character_indexes) != len(note_rows):
            raise ValueError("note_character_indexes/note count mismatch")
    else:
        if len(characters) != len(note_rows):
            raise ValueError(
                f"lyric/note count mismatch for {spec.get('segment_id')}: "
                f"lyrics={len(characters)} notes={len(note_rows)}"
            )
        character_indexes = list(range(1, len(note_rows) + 1))
        alignment_mode = "one_note_per_character"
    if any(index < 1 or index > len(characters) for index in character_indexes):
        raise ValueError("note_character_indexes contains an out-of-range character slot")
    notes: list[dict[str, Any]] = []
    for note_index, ((track_index, part_start, note), character_index) in enumerate(
        zip(note_rows, character_indexes),
        start=1,
    ):
        phoneme = note.find(q("p"))
        phoneme_text = _text(phoneme.text if phoneme is not None else "")
        notes.append(
            {
                "note_index": note_index,
                "character_index": character_index,
                "lyric_character": characters[character_index - 1],
                "vsqx_lyric": note.findtext(q("y")) or "",
                "phoneme_text": phoneme_text,
                "phoneme_tokens": phoneme_text.split(),
                "phoneme_lock": bool(phoneme is not None and phoneme.get("lock") == "1"),
                "absolute_start_tick": part_start + int(note.findtext(q("t")) or 0),
                "duration_ticks": int(note.findtext(q("dur")) or 0),
                "pitch": int(note.findtext(q("n")) or 0),
                "track_index": track_index,
                "model_output": False,
                "training_data": False,
            }
        )
    return {
        "segment_id": _text(spec.get("segment_id")) or source.stem,
        "order": int(spec.get("order", 0) or 0),
        "variant_group": spec.get("variant_group"),
        "annotation_status": spec.get("annotation_status"),
        "source_vsqx": str(source),
        "voice_name": voice_name,
        "track_index": notes[0]["track_index"] if notes else 0,
        "lyrics": "".join(characters),
        "alignment_mode": alignment_mode,
        "note_character_indexes": character_indexes,
        "notes": notes,
        "model_output": False,
        "training_data": False,
        "truth_role": "human_vsqx_reference_not_recognizer_truth",
    }


def _character_map(segments: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        for note in segment["notes"]:
            grouped[note["lyric_character"]].append(
                {
                    "segment_id": segment["segment_id"],
                    "note_index": note["note_index"],
                    "phoneme_text": note["phoneme_text"],
                    "phoneme_tokens": list(note["phoneme_tokens"]),
                    "phoneme_lock": note["phoneme_lock"],
                    "source_vsqx": segment["source_vsqx"],
                }
            )
    characters = []
    for character in sorted(grouped):
        observations = grouped[character]
        counts = Counter(row["phoneme_text"] for row in observations)
        characters.append(
            {
                "character": character,
                "observation_count": len(observations),
                "phoneme_sequence_counts": dict(sorted(counts.items())),
                "observations": observations,
            }
        )
    return {
        "schema_version": CHARACTER_MAP_SCHEMA_VERSION,
        "observation_count": sum(len(rows) for rows in grouped.values()),
        "character_count": len(characters),
        "characters": characters,
        "model_output": False,
        "training_data": False,
    }


def run_import(manifest: Path, output: Path, character_map_output: Path, *, pretty: bool) -> int:
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    segments: list[dict[str, Any]] = []
    errors: list[str] = []
    raw_segments = payload.get("segments", []) if isinstance(payload, Mapping) else []
    for spec in raw_segments:
        if not isinstance(spec, Mapping):
            continue
        try:
            segments.append(_parse_segment(spec, manifest.parent))
        except Exception as exc:
            errors.append(str(exc))
    corpus = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "ok": not errors,
        "errors": errors,
        "corpus_id": payload.get("corpus_id") if isinstance(payload, Mapping) else None,
        "expected_voice": payload.get("expected_voice") if isinstance(payload, Mapping) else None,
        "segment_count": len(segments),
        "note_count": sum(len(segment["notes"]) for segment in segments),
        "segments": sorted(segments, key=lambda row: (row["order"], row["segment_id"])),
        "model_output": False,
        "training_data": False,
        "truth_role": "human_vsqx_reference_not_recognizer_truth",
    }
    indent = 2 if pretty else None
    output.parent.mkdir(parents=True, exist_ok=True)
    character_map_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    character_map_output.write_text(
        json.dumps(_character_map(segments), ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return 0 if not errors else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--character-map-output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    return run_import(args.manifest, args.output, args.character_map_output, pretty=args.pretty)


if __name__ == "__main__":
    raise SystemExit(main())

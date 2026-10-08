"""Conservative VSQX template-clone exporter."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
from typing import Any
from xml.etree import ElementTree as ET

VSQ4_NS = "http://www.yamaha.co.jp/vocaloid/schema/vsq4/"
MIKU_V4C_ENGINE_NAME = "Miku V4C"


def _namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def _tag(name: str, namespace: str) -> str:
    return f"{{{namespace}}}{name}" if namespace else name


def _text(node: ET.Element | None, default: str = "") -> str:
    return (node.text or default).strip() if node is not None else default


def _draft_tracks(draft: Any) -> list[Any]:
    tracks = getattr(draft, "tracks", None)
    if tracks is None and isinstance(draft, dict):
        tracks = draft.get("tracks", [])
    return list(tracks or [])


def _notes(track: Any) -> list[Any]:
    notes = getattr(track, "notes", None)
    if notes is None and isinstance(track, dict):
        notes = track.get("notes", [])
    return list(notes or [])


def _value(note: Any, key: str, default: Any = None) -> Any:
    if isinstance(note, dict):
        return note.get(key, default)
    return getattr(note, key, default)


def _ensure_minimal_note_style(node: ET.Element, namespace: str) -> None:
    """Add the required VSQ4 note fields for a non-template scaffold."""

    if node.find(_tag("v", namespace)) is None:
        ET.SubElement(node, _tag("v", namespace)).text = "64"
    if node.find(_tag("nStyle", namespace)) is not None:
        return
    style = ET.SubElement(node, _tag("nStyle", namespace))
    for name, value in (
        ("accent", "50"),
        ("bendDep", "0"),
        ("bendLen", "0"),
        ("decay", "50"),
        ("fallPort", "0"),
        ("opening", "127"),
        ("risePort", "0"),
        ("vibLen", "0"),
        ("vibType", "0"),
    ):
        ET.SubElement(style, _tag("v", namespace), {"id": name}).text = value


def build_vsqx_tree(draft: Any, *, template_path: str | Path | None = None) -> ET.ElementTree:
    if template_path is not None:
        tree = ET.parse(template_path)
        root = deepcopy(tree.getroot())
    else:
        root = ET.Element(_tag("vsq4", VSQ4_NS))
        ET.SubElement(root, _tag("masterTrack", VSQ4_NS))
        ET.SubElement(root, _tag("vsTrack", VSQ4_NS))
    namespace = _namespace(root.tag)
    tempo = float(_value(draft, "tempo", 120.0))
    if not math.isfinite(tempo) or tempo <= 0.0:
        raise ValueError("VSQX draft tempo must be finite and positive")
    master_track = root.find(_tag("masterTrack", namespace))
    if master_track is not None:
        tempo_node = master_track.find(_tag("tempo", namespace))
        if tempo_node is not None:
            tempo_value = tempo_node.find(_tag("v", namespace))
            if tempo_value is not None:
                tempo_value.text = str(round(tempo * 100.0))
    track_nodes = root.findall(f".//{_tag('vsTrack', namespace)}")
    if not track_nodes:
        track_nodes = [ET.SubElement(root, _tag("vsTrack", namespace))]
    track_node = track_nodes[0]
    part_nodes = track_node.findall(_tag("vsPart", namespace))
    if part_nodes:
        part = part_nodes[0]
    else:
        part = ET.SubElement(track_node, _tag("vsPart", namespace))
    note_template = next(
        (old for old in list(part) if old.tag == _tag("note", namespace)),
        None,
    )
    for old in list(part):
        if old.tag == _tag("note", namespace):
            part.remove(old)
    all_notes = [note for track in _draft_tracks(draft) for note in _notes(track)]
    normalized_notes: list[tuple[Any, int, int, int]] = []
    for index, note in enumerate(all_notes):
        start_tick = int(_value(note, "start_tick", 0))
        duration_ticks = int(_value(note, "duration_ticks", 480))
        pitch = int(_value(note, "pitch", 60))
        if start_tick < 0:
            raise ValueError(f"VSQX note {index} start_tick must be non-negative")
        if duration_ticks <= 0:
            raise ValueError(f"VSQX note {index} duration_ticks must be positive")
        if not 0 <= pitch <= 127:
            raise ValueError(f"VSQX note {index} pitch must be between 0 and 127")
        normalized_notes.append((note, start_tick, duration_ticks, pitch))
    # VocalNoteDraft.start_tick is relative to the cloned vsPart. Preserve the
    # template's project-level part position and write draft ticks unchanged.
    relative_notes = normalized_notes
    relative_end_tick = max(
        (start_tick + duration_ticks for _, start_tick, duration_ticks, _ in relative_notes),
        default=1,
    )
    play_time = part.find(_tag("playTime", namespace))
    if play_time is None:
        play_time = ET.SubElement(part, _tag("playTime", namespace))
    play_time.text = str(max(1, relative_end_tick))
    for note, start_tick, duration_ticks, pitch in relative_notes:
        node = deepcopy(note_template) if note_template is not None else ET.Element(_tag("note", namespace))
        for child_name in ("t", "dur", "n", "v", "y", "p"):
            child = node.find(_tag(child_name, namespace))
            if child is None:
                child = ET.SubElement(node, _tag(child_name, namespace))
            if child_name == "t":
                child.text = str(start_tick)
            elif child_name == "dur":
                child.text = str(duration_ticks)
            elif child_name == "n":
                child.text = str(pitch)
            elif child_name == "v":
                child.text = "64"
            elif child_name == "y":
                child.text = str(_value(note, "lyric", "la"))
            elif child_name == "p":
                phonemes = _value(note, "phonemes", []) or []
                child.text = phonemes if isinstance(phonemes, str) else " ".join(str(item) for item in phonemes)
        _ensure_minimal_note_style(node, namespace)
        plane = part.find(_tag("plane", namespace))
        if plane is None:
            part.append(node)
        else:
            part.insert(list(part).index(plane), node)
    return ET.ElementTree(root)


def export_vsqx_file(tree: ET.ElementTree, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def export_vsqx_from_template(draft: Any, output_path: str | Path, *, template_path: str | Path) -> Path:
    return export_vsqx_file(build_vsqx_tree(draft, template_path=template_path), output_path)


__all__ = ["VSQ4_NS", "MIKU_V4C_ENGINE_NAME", "build_vsqx_tree", "export_vsqx_file", "export_vsqx_from_template"]

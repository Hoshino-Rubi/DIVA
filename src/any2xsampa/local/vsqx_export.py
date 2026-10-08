"""Template-preserving VSQX export for local recognition results."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from any2xsampa.vocaloid.phoneme_policy import validate_miku_v4c_note_phonemes
from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline

ROOT = Path(__file__).resolve().parents[3]
POLICY = Path(__file__).resolve().parents[1] / "config/local_score_v1.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _namespace(root: ET.Element) -> str:
    return root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""


def _tag(name: str, namespace: str) -> str:
    return f"{{{namespace}}}{name}" if namespace else name


def _note_fingerprint(note: ET.Element, namespace: str, part_tick: int) -> dict[str, Any]:
    def text(name: str) -> str:
        return (note.findtext(_tag(name, namespace)) or "").strip()

    start = int(text("t") or 0) + part_tick
    duration = int(text("dur") or 0)
    return {
        "start_tick": start,
        "duration_ticks": duration,
        "end_tick": start + duration,
        "pitch": text("n"),
        "lyric": text("y"),
    }


def _rules() -> dict[str, dict[str, Any]]:
    payload = _read(POLICY)
    return {str(key): dict(value) for key, value in payload.get("v4c_phone_rules", {}).items()}


def _tokens_for_phone(phone: str, rules: dict[str, dict[str, Any]]) -> tuple[list[str], bool]:
    rule = rules.get(phone)
    if rule is None:
        return [], False
    tokens = rule.get("tokens", [])
    if not isinstance(tokens, list) or not all(isinstance(token, str) and token for token in tokens):
        return [], False
    return list(tokens), bool(rule.get("approximation", False))


def _segments_for_note(segments: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for segment in segments:
        try:
            seg_start = float(segment["start_sec"])
            seg_end = float(segment["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        midpoint = (seg_start + seg_end) / 2.0
        if start <= midpoint < end:
            selected.append(segment)
    return selected


def _suggestion_for_note(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
    rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = _segments_for_note(segments, start, end)
    tokens: list[str] = []
    phones: list[str] = []
    approximate = False
    unsupported: list[str] = []
    for segment in selected:
        phone = str(segment.get("normalized_ipa", ""))
        phones.append(phone)
        mapped, is_approximate = _tokens_for_phone(phone, rules)
        if not mapped:
            unsupported.append(phone)
            continue
        approximate = approximate or is_approximate
        tokens.extend(mapped)
    compact = tokens
    policy_issues = validate_miku_v4c_note_phonemes(compact) if compact else []
    return {
        "phones": phones,
        "tokens": compact,
        "approximation_applied": approximate,
        "unsupported_phones": unsupported,
        "policy_issues": [issue.to_dict() for issue in policy_issues],
        "selected_segment_count": len(selected),
        "start_sec": start,
        "end_sec": end,
    }


def export_vsqx_from_result(
    *,
    result_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
    result_sha256: str | None = None,
    use_draft: bool = False,
    apply_phone_suggestions: bool = False,
) -> dict[str, Any]:
    """Copy a VSQX template while preserving note pitch/timing fields exactly.

    The recognizer's CTC emission times never become note timing.  When enabled,
    machine phone suggestions may replace only ``note/p``; the source template,
    note ``t/dur/n/y``, tempo and track structure remain unchanged.
    """

    result_file = Path(result_path).resolve()
    template_file = Path(template_path).resolve()
    output_file = Path(output_path).resolve()
    if result_sha256 is not None and _sha(result_file) != result_sha256:
        raise ValueError("Result hash mismatch; refusing VSQX export")
    result = _read(result_file)
    if result.get("schema_version") != "local_task_result_v1":
        raise ValueError("VSQX export requires a local_task_result_v1 result")
    if result.get("scope") not in {None, "whole_file"}:
        raise ValueError("VSQX export currently requires a whole-file result")
    if not template_file.is_file():
        raise FileNotFoundError(f"VSQX template not found: {template_file}")
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite VSQX: {output_file}")

    tree = ET.parse(template_file)
    root = tree.getroot()
    namespace = _namespace(root)
    timeline = parse_vsqx_timeline(template_file)
    timeline_notes = list(timeline.get("notes", []))
    parts = root.findall(f".//{_tag('vsPart', namespace)}")
    note_nodes: list[tuple[ET.Element, int]] = []
    for part in parts:
        part_tick = int(float((part.findtext(_tag("t", namespace)) or "0").strip() or 0))
        note_nodes.extend((node, part_tick) for node in part.findall(_tag("note", namespace)))
    if len(note_nodes) != len(timeline_notes):
        raise ValueError("VSQX note structure could not be aligned without guessing")

    source_segments = list(result.get("segments", []))
    if not all(isinstance(row, dict) for row in source_segments):
        raise ValueError("Result segments must be objects")
    if use_draft:
        draft_file = result_file.parent / "review-draft.json"
        draft = _read(draft_file)
        if draft.get("result_sha256") != _sha(result_file):
            raise ValueError("Manual draft belongs to a different result")
        edits = draft.get("edits")
        if not isinstance(edits, list) or len(edits) != len(source_segments):
            raise ValueError("Manual draft does not preserve machine segment count")
        if any(edit.get("index") != index or not isinstance(edit.get("ipa"), str)
               or not edit["ipa"].strip() for index, edit in enumerate(edits)):
            raise ValueError("Invalid manual draft indices or IPA")
        source_segments = [dict(row, normalized_ipa=str(edits[index].get("ipa", ""))) for index, row in enumerate(source_segments)]

    rules = _rules()
    suggestions: list[dict[str, Any]] = []
    original_fingerprints: list[dict[str, Any]] = []
    output_fingerprints: list[dict[str, Any]] = []
    changed_p = 0
    skipped_policy = 0
    note_nodes.sort(key=lambda row: (int(row[0].findtext(_tag("t", namespace)) or 0) + row[1],
                                    int(row[0].findtext(_tag("dur", namespace)) or 0)))
    for index, ((node, part_tick), timeline_note) in enumerate(zip(note_nodes, timeline_notes, strict=True)):
        original_fingerprints.append(_note_fingerprint(node, namespace, part_tick))
        suggestion = _suggestion_for_note(
            source_segments,
            float(timeline_note["start_sec"]),
            float(timeline_note["end_sec"]),
            rules,
        )
        suggestion["note_index"] = index
        suggestion["original_phonemes"] = (node.findtext(_tag("p", namespace)) or "").strip().split()
        suggestion["applied"] = False
        if apply_phone_suggestions and suggestion["tokens"] and not suggestion["policy_issues"] and not suggestion["unsupported_phones"]:
            p_node = node.find(_tag("p", namespace))
            if p_node is None:
                p_node = ET.SubElement(node, _tag("p", namespace))
            p_node.text = " ".join(suggestion["tokens"])
            suggestion["applied"] = True
            changed_p += 1
        elif apply_phone_suggestions and suggestion["policy_issues"]:
            skipped_policy += 1
        suggestions.append(suggestion)
        output_fingerprints.append(_note_fingerprint(node, namespace, part_tick))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    semantic_before = [item for item in original_fingerprints]
    semantic_after = [item for item in output_fingerprints]
    if semantic_before != semantic_after:
        output_file.unlink(missing_ok=True)
        raise ValueError("VSQX export changed note pitch or timing")
    provenance_path = output_file.with_name(output_file.name + ".provenance.json")
    provenance = {
        "schema_version": "local_vsqx_export_v1",
        "result_path": str(result_file),
        "result_sha256": _sha(result_file),
        "template_path": str(template_file),
        "template_sha256": _sha(template_file),
        "output_path": str(output_file),
        "output_sha256": _sha(output_file),
        "note_count": len(note_nodes),
        "pitch_and_duration_preserved": True,
        "tempo_and_track_structure_preserved": True,
        "ctc_emission_times_used_as_note_timing": False,
        "apply_phone_suggestions": bool(apply_phone_suggestions),
        "use_manual_draft": bool(use_draft),
        "phone_suggestion_changed_note_p_count": changed_p,
        "phone_suggestion_skipped_policy_count": skipped_policy,
        "machine_model_forwards": 0,
        "suggestions": suggestions,
        "note_fingerprints": semantic_after,
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output_file),
        "provenance": str(provenance_path),
        "note_count": len(note_nodes),
        "changed_note_p_count": changed_p,
        "pitch_and_duration_preserved": True,
        "machine_model_forwards": 0,
    }

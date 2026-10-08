"""Assemble the two edge answers for each frozen clip into one full reference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, write_json_new  # noqa: E402
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay  # noqa: E402
from any2xsampa.training.segment_reference import validate_segment_reference  # noqa: E402


def _assemble_one(start: dict, end: dict, candidate: dict, song: dict, reference_hash: str) -> dict:
    note_map = {note["note_id"]: note for note in song["notes"]}
    note_ids = candidate["note_ids"]
    if not candidate.get("mapping_complete") or not note_ids:
        raise ValueError("machine candidate has no complete mapping")
    if not set(start["note_ids"]).issubset(note_ids) or not set(end["note_ids"]).issubset(note_ids):
        raise ValueError("human edge notes do not belong to the frozen candidate sequence")
    selected: dict[tuple[str, int], dict] = {}
    for note_id in note_ids:
        note = note_map[note_id]
        for index, phone in enumerate(note["model_target_ipa"]):
            selected[(note_id, index)] = {"note_id": note_id, "phone_index": index, "phone": phone, "audible": True, "edge_status": "full"}
    edge_keys: dict[str, set[tuple[str, int]]] = {}
    for edge, label in ((start, "truncated_start"), (end, "truncated_end")):
        if not edge["phones"]:
            raise ValueError(f"{label} edge answer has no explicit audible phones")
        touched = set(edge["note_ids"])
        edge_keys[label] = set()
        for key in list(selected):
            if key[0] in touched and not any(key in keys for keys in edge_keys.values()):
                del selected[key]
        for phone in edge["phones"]:
            key = (phone["note_id"], phone["phone_index"])
            if phone["note_id"] not in touched:
                raise ValueError("edge phone conflicts with its selected source note")
            if key in selected and selected[key]["phone"] != phone["phone"]:
                raise ValueError("start/end edge answers disagree on a phone")
            selected[key] = {**phone, "edge_status": phone.get("edge_status", label)}
            edge_keys[label].add(key)
    ordered = [selected[key] for note_id in note_ids for key in sorted(selected) if key[0] == note_id]
    if not ordered:
        raise ValueError("assembled clip has no phones")
    expected_internal = [note_id for note_id in note_ids[1:-1]]
    for note_id in expected_internal:
        expected = len(note_map[note_id]["model_target_ipa"])
        if sum(phone["note_id"] == note_id for phone in ordered) != expected:
            raise ValueError("assembled clip dropped an internal note phone")
    first_note = note_map[note_ids[0]]; last_note = note_map[note_ids[-1]]
    first_count = sum(phone["note_id"] == note_ids[0] for phone in ordered)
    last_count = sum(phone["note_id"] == note_ids[-1] for phone in ordered)
    first_full = first_count == len(first_note["model_target_ipa"])
    last_full = last_count == len(last_note["model_target_ipa"])
    boundary_state = "full" if first_full and last_full else ("truncated_both" if not first_full and not last_full else ("truncated_start" if not first_full else "truncated_end"))
    return {
        "schema_version": "original_segment_reference_v2", "example_id": start["source_example_id"], "source_example_id": start["source_example_id"],
        "review_id": f"assembled-{start['source_example_id']}", "song_id": start["song_id"], "split": start["split"], "audio_role": "original",
        "audio_path": start["audio_path"], "audio_sha256": start["audio_sha256"], "source_vsqx_sha256": start["source_vsqx_sha256"],
        "reference_manifest_sha256": reference_hash, "start_sec": start["start_sec"], "end_sec": start["end_sec"],
        "status": "human_confirmed", "boundary_state": boundary_state,
        "note_ids": note_ids, "phones": ordered, "reviewer": f"{start.get('reviewer') or ''}; {end.get('reviewer') or ''}".strip("; "),
        "review_statement": f"start: {start.get('review_statement') or ''}; end: {end.get('review_statement') or ''}",
        "edge_evidence": {"start_review_id": start["review_id"], "end_review_id": end["review_id"], "candidate_factor": candidate["factor"]},
        "source_phoneme_approval_preserved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine B01/B02-style edge evidence into complete clip references.")
    parser.add_argument("--edge-reference", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--mapping-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    edge = read_json(args.edge_reference); validate_segment_reference(edge)
    reference = read_json(args.reference); view, _ = apply_original_mapping_overlay(reference, args.mapping_evidence)
    evaluation = read_json(args.evaluation_manifest)
    by_example = {row["example_id"]: row for row in evaluation.get("examples", [])}
    songs = {song["song_id"]: song for song in view["songs"]}
    groups: dict[str, dict[str, dict]] = {}
    for row in edge["examples"]:
        if row["status"] != "human_confirmed":
            raise ValueError("all edge answers must be confirmed before assembly")
        groups.setdefault(row["source_example_id"], {})[row.get("side") or row["review_id"]] = row
    assembled = []
    for example_id, sides in groups.items():
        if "start" not in sides or "end" not in sides:
            raise ValueError(f"clip {example_id} needs both start and end evidence")
        proposal = by_example.get(example_id)
        if proposal is None or len(proposal.get("candidate_references", [])) < 1:
            raise ValueError("missing frozen correspondence proposal")
        candidate = proposal["candidate_references"][0]
        if len(proposal.get("candidate_references", [])) > 1 and proposal["candidate_references"][1].get("note_ids") != candidate.get("note_ids"):
            raise ValueError("correspondence resolutions disagree; no clip assembly")
        assembled.append(_assemble_one(sides["start"], sides["end"], candidate, songs[sides["start"]["song_id"]], sha256_file(args.reference)))
    result = {"schema_version": "original_segment_reference_v2", "source_reference_manifest_sha256": sha256_file(args.reference), "packet_sha256": sha256_object([edge["manifest_hash"], sha256_file(args.evaluation_manifest)]), "examples": assembled, "human_confirmed_count": len(assembled), "test_predictions_opened": False, "historical_sources_modified": False, "assembly_policy": {"edge_answers_required": True, "machine_note_sequence_used_only_as_candidate": True, "partial_edge_phones_not_dropped": True}}
    result["manifest_hash"] = sha256_object(result); validate_segment_reference(result)
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    digest = write_json_new(output, result)
    print(f"assembled_clips={len(assembled)}; sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

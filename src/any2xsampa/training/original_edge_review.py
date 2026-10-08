"""Explicit edge answers -> unchanged-window v2 references, never model guesses."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from any2xsampa.training.limited_data_artifacts import finite_number, require_digest, sha256_object
from any2xsampa.training.segment_reference import _phone_rows, validate_segment_reference

PACKET_SCHEMA = "original_edge_review_packet_v1"
ANSWER_SCHEMA = "original_edge_review_answers_v1"
OBSERVATION_SCHEMA = "phone_audibility_v2"
IDS = {f"B{i:02d}" for i in range(1, 7)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rows_by_key(rows: Any, key: str) -> dict[str, dict]:
    _require(isinstance(rows, list), f"{key} rows must be a list")
    result = {}
    for row in rows:
        _require(isinstance(row, dict), f"{key} row must be an object")
        value = row.get(key)
        _require(isinstance(value, str) and bool(value), f"invalid {key}")
        _require(value not in result, f"duplicate {key}: {value}")
        result[value] = row
    return result


def validate_packet(packet: dict) -> None:
    _require(packet.get("schema_version") == PACKET_SCHEMA, "unsupported edge packet schema")
    expected = sha256_object({k: v for k, v in packet.items() if k != "packet_sha256"})
    _require(packet.get("packet_sha256") == expected, "edge packet hash mismatch")
    clips = _rows_by_key(packet.get("clips"), "example_id")
    items = _rows_by_key(packet.get("items"), "review_id")
    _require(set(items) == IDS and len(clips) == 3, "expected six B edges and three dev clips")
    sides = defaultdict(list)
    for clip in clips.values():
        _require(clip.get("song_id") == "shalong" and clip.get("split") == "dev", "sealed or wrong song/split")
        start = finite_number(clip.get("start_sec"), "clip start", minimum=0)
        end = finite_number(clip.get("end_sec"), "clip end", minimum=0)
        _require(abs(end - start - 18) < 1e-6, "original dev window must remain 18 seconds")
        for field in ("audio_sha256", "source_vsqx_sha256", "reference_manifest_sha256"):
            require_digest(clip.get(field), field)
        notes = _rows_by_key(clip.get("notes"), "note_id")
        _require(bool(notes), "empty note catalog")
        for note in notes.values():
            _require(isinstance(note.get("phones"), list), "note phones must be a list")
            _require(all(isinstance(p, str) and p for p in note["phones"]), "invalid mapped phone")
    for item in items.values():
        example = item.get("example_id")
        _require(example in clips, "edge references unknown clip")
        side = item.get("side")
        _require(side in {"start", "end"}, "invalid edge side")
        sides[example].append(side)
        cut = finite_number(item.get("original_cut_sec"), "original cut", minimum=0)
        _require(abs(cut - clips[example][f"{side}_sec"]) < 1e-6, "edge cut differs from original window")
        duration = finite_number(item.get("duration_sec"), "preview duration", minimum=0)
        middle = finite_number(item.get("cut_in_preview_sec"), "preview cut", minimum=0)
        _require(0 < middle < duration, "preview cut must be inside preview")
        catalog = _rows_by_key(clips[example]["notes"], "note_id")
        nearby = _rows_by_key(item.get("notes"), "note_id")
        _require(bool(nearby), "empty edge note catalog")
        for nid, note in nearby.items():
            _require(nid in catalog, "edge note is outside clip catalog")
            _require(note["phones"] == catalog[nid]["phones"] and note.get("mapping_status") == catalog[nid].get("mapping_status"), "edge note mapping differs from clip")
    _require(all(sorted(v) == ["end", "start"] for v in sides.values()) and set(sides) == set(clips), "each clip needs exactly one start and one end")


def _validate_answer(item: dict, answer: dict, reviewer: str) -> None:
    _require(answer.get("observation_schema") in {None, OBSERVATION_SCHEMA}, "unknown audibility observation schema")
    observation = answer.get("observation_schema") == OBSERVATION_SCHEMA
    status = answer.get("status")
    _require(status in {"confirmed", "uncertain", "wrong_correspondence"}, "invalid answer status")
    statement = answer.get("statement")
    _require(isinstance(statement, str) and len(statement) <= 4000, "invalid answer statement")
    if status != "confirmed":
        _require(answer.get("note_id") is None and answer.get("phones") == [] and answer.get("audible_identity_confirmed") is False,
                 "unconfirmed answer cannot contain phone assertions")
        return
    _require(bool(reviewer.strip()) and bool(statement.strip()), "confirmed answer needs reviewer and statement")
    _require(answer.get("audible_identity_confirmed") is True, "audible identity needs explicit confirmation")
    notes = _rows_by_key(item["notes"], "note_id")
    nid = answer.get("note_id")
    _require(nid in notes, "selected note is outside this edge")
    note = notes[nid]
    _require(note.get("mapping_status") == "resolved_proxy" and bool(note["phones"]), "selected note has unresolved mapping")
    phones = answer.get("phones")
    _require(isinstance(phones, list) and bool(phones), "explicit audible phones required")
    indices = []
    for phone in phones:
        _require(isinstance(phone, dict), "invalid phone selection")
        index = phone.get("phone_index")
        _require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(note["phones"]), "phone_index out of range")
        _require(phone.get("edge_status") in ({"full", "partial", "absent", "uncertain"} if observation else {"full", "truncated"}), "invalid phone edge_status")
        indices.append(index)
    _require(len(set(indices)) == len(indices) and indices == sorted(indices), "duplicate or unordered phone selection")
    if observation:
        return
    side = item["side"]
    expected = list(range(indices[0], len(note["phones"]))) if side == "start" else list(range(indices[-1] + 1))
    _require(indices == expected, "start must select a suffix; end must select a prefix; no internal omissions")
    # The legacy format describes geometric edge truncation, not audibility observations.
    edge_index = indices[0] if side == "start" else indices[-1]
    _require(all(p["edge_status"] != "truncated" or p["phone_index"] == edge_index for p in phones),
             "legacy truncation must face the cut; use audibility v2 for heard absence/partial phones")


def validate_answers(packet: dict, payload: dict) -> dict[str, dict]:
    validate_packet(packet)
    _require(payload.get("schema_version") in {ANSWER_SCHEMA, "original_edge_review_answers_v2"}, "unsupported edge answers schema")
    _require(payload.get("packet_sha256") == packet["packet_sha256"], "answers belong to a different packet")
    reviewer = payload.get("reviewer")
    _require(isinstance(reviewer, str) and len(reviewer) <= 120, "invalid reviewer")
    answers = _rows_by_key(payload.get("answers"), "review_id")
    items = _rows_by_key(packet["items"], "review_id")
    _require(set(answers).issubset(items), "unknown answer ID")
    for key, answer in answers.items():
        _require(payload["schema_version"] != ANSWER_SCHEMA or "observation_schema" not in answer,
                 "legacy envelope cannot contain audibility v2")
        _validate_answer(items[key], answer, reviewer)
    return answers


def _clip_base(clip: dict, reviewer: str) -> dict:
    row = {k: deepcopy(clip[k]) for k in ("example_id", "song_id", "split", "audio_path", "audio_sha256", "source_vsqx_sha256", "reference_manifest_sha256", "start_sec", "end_sec")}
    row.update({"source_example_id": clip["example_id"], "audio_role": "original", "review_id": f"edge-pair-{clip['example_id']}",
                "status": "pending", "boundary_state": None, "note_ids": [], "phones": [], "reviewer": reviewer,
                "review_statement": "", "blocking_reasons": [], "training_eligible": False})
    return row


def _selection_range(clip: dict, first: dict, last: dict) -> tuple[list[dict], int, int]:
    notes = clip["notes"]
    positions = {note["note_id"]: i for i, note in enumerate(notes)}
    lo, hi = positions[first["note_id"]], positions[last["note_id"]]
    _require(lo <= hi, "selected start note follows end note")
    selected = notes[lo:hi + 1]
    _require(all(n["mapping_status"] == "resolved_proxy" and n["phones"] for n in selected), "whole selected window contains unresolved mapping")
    return selected, lo, hi


def _assemble_selected(clip: dict, first: dict, last: dict) -> tuple[list[str], list[dict], str]:
    notes, _, _ = _selection_range(clip, first, last)
    # Preserve observations in edge_evidence, but never silently delete a source
    # phone or infer which boundary caused it to be inaudible.
    for edge in (first, last):
        if edge.get("observation_schema") == OBSERVATION_SCHEMA:
            note = next(n for n in notes if n["note_id"] == edge["note_id"])
            _require(len(edge["phones"]) == len(note["phones"]) and
                     all(p["edge_status"] == "full" for p in edge["phones"]),
                     "audibility_observations_require_reference_reconciliation")
    by_id = {n["note_id"]: n for n in notes}
    selected = {(n["note_id"], i): {"note_id": n["note_id"], "phone_index": i, "phone": symbol,
                 "audible": True, "edge_status": "full"} for n in notes for i, symbol in enumerate(n["phones"])}
    for edge in (first, last):
        permitted = {p["phone_index"]: p for p in edge["phones"]}
        nid = edge["note_id"]
        for key in list(selected):
            if key[0] != nid:
                continue
            if key[1] not in permitted:
                del selected[key]
            elif permitted[key[1]]["edge_status"] == "truncated":
                selected[key]["edge_status"] = "truncated"
    _require(bool(selected), "start/end selections have no audible intersection")
    truncated = []
    for edge in (first, last):
        nid = edge["note_id"]
        truncated.append(len(edge["phones"]) != len(by_id[nid]["phones"]) or any(p["edge_status"] == "truncated" for p in edge["phones"]))
    boundary = ("truncated_both" if all(truncated) else "truncated_start" if truncated[0] else "truncated_end" if truncated[1] else "full")
    phones = list(selected.values())
    source = [{"note_id": n["note_id"], "mapping_status": n["mapping_status"], "model_target_ipa": n["phones"]} for n in clip["notes"]]
    ids = [n["note_id"] for n in notes]
    _phone_rows(source, ids, phones, boundary)
    return ids, phones, boundary


def assemble_references(packet: dict, payload: dict) -> dict[str, Any]:
    """Pending edges stay pending. Only explicit pairs plus bound sequence proceed."""
    answers = validate_answers(packet, payload)
    item_groups = defaultdict(dict)
    for item in packet["items"]:
        item_groups[item["example_id"]][item["side"]] = item
    rows = []
    for clip in packet["clips"]:
        row = _clip_base(clip, payload["reviewer"])
        edges = {side: answers.get(item["review_id"]) for side, item in item_groups[clip["example_id"]].items()}
        row["edge_evidence"] = deepcopy(edges)
        if any(e and e["status"] == "wrong_correspondence" for e in edges.values()):
            row["status"] = "rejected"
            row["blocking_reasons"] = ["human_rejected_specific_clip_correspondence"]
        elif any(e is None or e["status"] != "confirmed" for e in edges.values()):
            row["blocking_reasons"] = [f"{side}_edge_not_confirmed" for side, e in edges.items() if e is None or e["status"] != "confirmed"]
        elif clip.get("source_sequence_approval_bound") is not True:
            row["blocking_reasons"] = ["source_phone_sequence_approval_not_bound"]
        else:
            try:
                ids, phones, boundary = _assemble_selected(clip, edges["start"], edges["end"])
            except ValueError as error:
                row["blocking_reasons"] = [str(error)]
            else:
                row.update({"status": "human_confirmed", "note_ids": ids, "phones": phones, "boundary_state": boundary,
                            "review_statement": f"start: {edges['start']['statement']}; end: {edges['end']['statement']}",
                            "source_phoneme_approval_preserved": True,
                            "interior_phone_evidence": "inherited_source_sequence_between_explicit_edge_notes"})
        rows.append(row)
    result = {"schema_version": "original_segment_reference_v2", "packet_sha256": packet["packet_sha256"],
              "answers_sha256": sha256_object(payload), "examples": rows,
              "human_confirmed_count": sum(r["status"] == "human_confirmed" for r in rows),
              "test_predictions_opened": False, "historical_sources_modified": False,
              "assembly_policy": {"no_internal_note_omissions": True, "explicit_audible_edges_required": True,
                                  "inherited_source_sequence_required": True, "training_started": False}}
    result["manifest_hash"] = sha256_object(result)
    validate_segment_reference(result)
    return result

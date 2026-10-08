"""Phone-sequence references from explicit audibility, without inferred cut causes.

Separate from segment-reference v2: absence inside an edge note is a human
sequence correction, not a claim about which side of a timestamp clipped it.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from any2xsampa.training.limited_data_artifacts import sha256_object
from any2xsampa.training.original_edge_review import validate_answers, _selection_range

SCHEMA = "reviewed_audibility_reference_v1"


def _edge_states(answer: dict, note: dict) -> dict[int, str]:
    values = {p["phone_index"]: p["edge_status"] for p in answer["phones"]}
    if answer.get("observation_schema") == "phone_audibility_v2":
        if set(values) != set(range(len(note["phones"]))):
            raise ValueError("edge_has_unreviewed_phones")
        if "uncertain" in values.values():
            raise ValueError("edge_has_uncertain_phones")
        return values
    # Legacy selected phones are audible; unselected prefix/suffix is outside
    # the old explicitly confirmed audible extent, not newly inferred absence.
    return {i: ("partial" if values.get(i) == "truncated" else values.get(i, "legacy_outside_selection"))
            for i in range(len(note["phones"]))}


def build_reviewed_reference(packet: dict, payload: dict) -> dict[str, Any]:
    answers = validate_answers(packet, payload)
    groups: dict[str, dict] = defaultdict(dict)
    for item in packet["items"]:
        groups[item["example_id"]][item["side"]] = item
    rows = []
    for clip in packet["clips"]:
        edges = {side: answers.get(item["review_id"]) for side, item in groups[clip["example_id"]].items()}
        row = {k: deepcopy(clip[k]) for k in (
            "example_id", "song_id", "split", "audio_path", "audio_sha256",
            "source_vsqx_sha256", "reference_manifest_sha256", "start_sec", "end_sec")}
        row.update(status="pending", phones=[], omitted_phones=[], source_phones=[],
                   note_ids=[], blocking_reasons=[], edge_evidence=deepcopy(edges),
                   audio_role="original", reviewer=payload["reviewer"],
                   reference_nature="human_reviewed_edges_with_inherited_calibrated_source_ipa",
                   missing_phone_cause="not_assessed", training_eligible=False)
        rows.append(row)
        if any(e and e["status"] == "wrong_correspondence" for e in edges.values()):
            row.update(status="rejected", blocking_reasons=["human_rejected_correspondence"])
            continue
        if any(e is None or e["status"] != "confirmed" for e in edges.values()):
            row["blocking_reasons"] = ["incomplete_confirmed_edge_pair"]
            continue
        if clip.get("source_sequence_approval_bound") is not True:
            row["blocking_reasons"] = ["source_sequence_approval_not_bound"]
            continue
        try:
            notes, _, _ = _selection_range(clip, edges["start"], edges["end"])
            lookup = {n["note_id"]: n for n in notes}
            states: dict[str, dict] = {}
            review_ids: dict[str, list[str]] = defaultdict(list)
            for edge in edges.values():
                nid = edge["note_id"]
                state = _edge_states(edge, lookup[nid])
                if nid in states and state != states[nid]:
                    raise ValueError("same_note_edges_have_conflicting_audibility")
                if not any(v in {"full", "partial"} for v in state.values()):
                    raise ValueError("selected_edge_note_has_no_audible_phone")
                states[nid] = state
                review_ids[nid].append(edge["review_id"])
            source, retained, omitted = [], [], []
            for note in notes:
                for i, phone in enumerate(note["phones"]):
                    nid = note["note_id"]
                    state = states[nid][i] if nid in states else "full"
                    value = {"note_id": nid, "phone_index": i, "phone": phone,
                             "audibility": state, "review_ids": review_ids.get(nid, []),
                             "evidence_scope": "explicit_human_edge" if nid in states else "inherited_source_interior"}
                    source.append(value)
                    (retained if state in {"full", "partial"} else omitted).append(deepcopy(value))
        except ValueError as error:
            row["blocking_reasons"] = [str(error)]
            continue
        row.update(status="human_confirmed", phones=retained, omitted_phones=omitted,
                   source_phones=source, note_ids=[n["note_id"] for n in notes],
                   source_phone_count=len(source), reference_phone_count=len(retained),
                   partial_phone_count=sum(p["audibility"] == "partial" for p in retained),
                   review_statement={side: e["statement"] for side, e in edges.items()})
    confirmed = [r for r in rows if r["status"] == "human_confirmed"]
    result = {
        "schema_version": SCHEMA, "packet_sha256": packet["packet_sha256"],
        "answers_sha256": sha256_object(payload), "examples": rows,
        "coverage": {"confirmed_windows": len(confirmed), "total_windows": len(rows),
                     "confirmed_seconds": sum(r["end_sec"] - r["start_sec"] for r in confirmed),
                     "reference_phones": sum(len(r["phones"]) for r in confirmed),
                     "source_phones": sum(len(r["source_phones"]) for r in confirmed),
                     "omitted_phones": sum(len(r["omitted_phones"]) for r in confirmed),
                     "partial_phones": sum(r["partial_phone_count"] for r in confirmed)},
        "policy": {"internal_source_phones_unchanged": True, "partial_phones_counted_once": True,
                   "missing_cause_inferred": False, "reference_selected_using_predictions": False,
                   "full_song_timing_confirmed": False, "test_predictions_opened": False,
                   "training_started": False}}
    result["manifest_hash"] = sha256_object(result)
    validate_reviewed_reference(result)
    return result


def validate_reviewed_reference(value: dict) -> None:
    if value.get("schema_version") != SCHEMA:
        raise ValueError("unsupported reviewed audibility reference")
    unsigned = {k: v for k, v in value.items() if k != "manifest_hash"}
    if value.get("manifest_hash") != sha256_object(unsigned):
        raise ValueError("reviewed audibility reference hash mismatch")
    seen = set()
    for row in value["examples"]:
        if row["example_id"] in seen or row["song_id"] != "shalong" or row["split"] != "dev" or row["audio_role"] != "original":
            raise ValueError("duplicate, sealed or wrong-role reference")
        seen.add(row["example_id"])
        if row["status"] != "human_confirmed":
            if row["phones"]:
                raise ValueError("unconfirmed rows cannot contain reference phones")
            continue
        if not row["phones"] or row["end_sec"] <= row["start_sec"]:
            raise ValueError("empty confirmed reference")
        expected = [p for p in row["source_phones"] if p["audibility"] in {"full", "partial"}]
        missing = [p for p in row["source_phones"] if p["audibility"] in {"absent", "legacy_outside_selection"}]
        if row["phones"] != expected or row["omitted_phones"] != missing or len(expected) + len(missing) != len(row["source_phones"]):
            raise ValueError("reference dropped or changed an unaccounted phone")
        if any(p["evidence_scope"] != "explicit_human_edge" for p in missing):
            raise ValueError("internal source phone omission forbidden")
        keys = [(p["note_id"], p["phone_index"]) for p in row["source_phones"]]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate note/phone indices")

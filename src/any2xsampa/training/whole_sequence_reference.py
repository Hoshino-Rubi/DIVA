"""Whole-song sequence proposals with explicit mapping and reviewed-anchor checks."""
from __future__ import annotations
from typing import Any


def audit_song(song: dict, reviewed: list[dict], *, minimum_anchors: int = 3) -> dict[str, Any]:
    if song["split"] != "train":
        raise ValueError("whole-sequence proposals are limited to training songs")
    notes = song["notes"]
    index = {n["note_id"]: n for n in notes}
    if len(index) != len(notes) or any(n["note_index"] != i+1 for i,n in enumerate(notes)):
        raise ValueError("duplicate/non-contiguous source note sequence")
    reasons = []
    unresolved = [n["note_id"] for n in notes if n["mapping_status"] != "resolved_proxy" or not n["model_target_ids"]]
    if unresolved:
        reasons.append("complete_song_contains_unresolved_mapping")
    if song["source_claims"].get("phoneme_correctness") != "user_asserted_correct":
        reasons.append("source_phone_approval_not_bound")
    anchors = []
    prior_end, prior_note = -1.0, -1
    for row in sorted(reviewed, key=lambda r: r["start_sec"]):
        if row["song_id"] != song["song_id"]:
            continue
        if (row["status"] != "human_confirmed" or row["audio_role"] != "original"
                or row["audio_sha256"] != song["original_audio"]["sha256"]
                or row["source_vsqx_sha256"] != song["human_vsqx"]["sha256"]):
            raise ValueError("reviewed anchor source/role mismatch")
        ids = list(dict.fromkeys(p["note_id"] for p in row["source_phones"]))
        if any(n not in index for n in ids):
            raise ValueError("reviewed anchor references unknown note")
        lo, hi = index[ids[0]]["note_index"], index[ids[-1]]["note_index"]
        if [index[n]["note_index"] for n in ids] != list(range(lo, hi+1)):
            raise ValueError("reviewed anchor contains non-contiguous notes")
        for phone in row["source_phones"]:
            targets = index[phone["note_id"]]["model_target_ipa"]
            if phone["phone_index"] >= len(targets) or targets[phone["phone_index"]] != phone["phone"]:
                raise ValueError("reviewed phone differs from the existing mapped source")
        if row["start_sec"] < prior_end-1e-5 or lo < prior_note:
            reasons.append("reviewed_anchors_overlap_or_reverse_source_order")
        prior_end, prior_note = row["end_sec"], hi
        anchors.append({"example_id": row["example_id"], "original_start_sec": row["start_sec"],
            "original_end_sec": row["end_sec"], "first_note_id": ids[0], "last_note_id": ids[-1],
            "first_note_range": [index[ids[0]]["start_sec"], index[ids[0]]["end_sec"]],
            "last_note_range": [index[ids[-1]]["start_sec"], index[ids[-1]]["end_sec"]],
            "partial_or_absent_phones": sum(p["audibility"] != "full" for p in row["source_phones"])})
    if len(anchors) < minimum_anchors:
        reasons.append("insufficient_independent_reviewed_anchors")
    return {"song_id": song["song_id"], "mapping_complete": not unresolved,
        "unresolved_note_ids": unresolved,
        "unresolved_tokens": sorted({t for n in notes for t in n["unresolved_tokens"]}),
        "note_count": len(notes), "target_phone_count": sum(len(n["model_target_ids"]) for n in notes),
        "anchors": anchors, "source_structure_passed": not reasons,
        "exclusion_reasons": sorted(set(reasons)), "source_approval_scope": "VSQX phone sequence",
        "human_whole_song_original_timeline_claimed": False}


def check_clock_anchors(audit: dict, paths: list[dict], *, edge_tolerance: float = 2.0,
                        resolution_tolerance: float = 0.5) -> dict:
    from any2xsampa.training.original_reference_correspondence import map_clip
    checks, reasons = [], list(audit["exclusion_reasons"])
    for anchor in audit["anchors"]:
        mapped = [map_clip(p, anchor["original_start_sec"], anchor["original_end_sec"]) for p in paths]
        endpoint_ok = []
        for row in mapped:
            start = row["rendered_start_sec"]
            end = row["rendered_end_sec"]
            endpoint_ok.append(anchor["first_note_range"][0]-edge_tolerance <= start <= anchor["first_note_range"][1]+edge_tolerance
                and anchor["last_note_range"][0]-edge_tolerance <= end <= anchor["last_note_range"][1]+edge_tolerance)
        disagreement = max(max(r[k] for r in mapped)-min(r[k] for r in mapped)
                           for k in ("rendered_start_sec", "rendered_end_sec"))
        if not all(endpoint_ok):
            reasons.append("whole_song_path_conflicts_with_reviewed_anchor")
        if disagreement > resolution_tolerance:
            reasons.append("whole_song_path_resolution_disagreement")
        checks.append({**anchor, "mapped": mapped, "endpoint_support": endpoint_ok,
            "resolution_disagreement_sec": disagreement})
    return {**audit, "clock_checks": checks, "exclusion_reasons": sorted(set(reasons)),
        "machine_sequence_training_eligible": not reasons,
        "full_sequence_timing_human_confirmed": False,
        "phone_model_predictions_used_for_acceptance": False,
        "existing_reviewed_windows_modified": False}

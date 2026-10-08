"""Audio-correspondence driven original-song alignment proposals.

This module deliberately produces a *machine proxy* reference, not human truth.
It consumes the existing audio-only rendered/original correspondence evidence and
transfers the already-approved VSQX phone sequence to the original clip.  The
strict human-reference schema remains unchanged; downstream reports must keep
``machine_aligned`` separate from ``human_confirmed``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from any2xsampa.training.limited_data_artifacts import finite_number, sha256_object


SCHEMA = "original_segment_machine_alignment_v1"
STATUSES = {"machine_aligned", "needs_review", "rejected"}

ALIGNMENT_POLICY: dict[str, Any] = {
    "schema_version": "original_segment_machine_alignment_policy_v1",
    "evidence": "audio_only_rendered_original_correspondence",
    "max_endpoint_disagreement_sec": 0.20,
    "max_local_offset_disagreement_sec": 0.20,
    "minimum_local_similarity": 0.45,
    "human_review_required_for_official_evaluation": True,
    "machine_aligned_is_proxy_only": True,
    "phone_model_outputs_used_for_alignment": False,
}


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _candidate_reference(candidate: Mapping[str, Any]) -> tuple[list[str], list[Mapping[str, Any]]]:
    refs = candidate.get("candidate_references")
    if not isinstance(refs, list) or not refs:
        raise ValueError("candidate has no correspondence references")
    usable = [ref for ref in refs if isinstance(ref, Mapping)]
    if not usable:
        raise ValueError("candidate correspondence references are invalid")
    note_ids = usable[0].get("note_ids")
    if not isinstance(note_ids, list) or not note_ids or any(not isinstance(x, str) or not x for x in note_ids):
        raise ValueError("candidate reference has no note sequence")
    for ref in usable:
        if ref.get("note_ids") != note_ids:
            raise ValueError("correspondence resolutions disagree on note sequence")
        if ref.get("mapping_complete") is not True or ref.get("valid_rendered_range") is not True:
            raise ValueError("candidate correspondence is not complete")
        finite_number(ref.get("rendered_start_sec"), "rendered start", minimum=0)
        finite_number(ref.get("rendered_end_sec"), "rendered end", minimum=0)
        if ref["rendered_end_sec"] <= ref["rendered_start_sec"]:
            raise ValueError("candidate rendered interval is invalid")
    return list(note_ids), usable


def _phone_rows(note_ids: Sequence[str], note_map: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for note_id in note_ids:
        note = note_map.get(note_id)
        if note is None:
            raise ValueError(f"candidate references unknown note: {note_id}")
        if note.get("mapping_status") != "resolved_proxy":
            raise ValueError(f"candidate references unresolved note: {note_id}")
        phones = note.get("model_target_ipa")
        if not isinstance(phones, list) or not phones:
            raise ValueError(f"candidate note has no mapped phones: {note_id}")
        for index, phone in enumerate(phones):
            rows.append({
                "note_id": note_id,
                "phone_index": index,
                "phone": _require_text(phone, "mapped phone"),
                "audible": True,
                "edge_status": "full",
                "source": "vsqx_mapping_transferred_by_audio_correspondence",
            })
    return rows


def _local_similarity(correspondence: Mapping[str, Any], example_id: str) -> float | None:
    values: list[float] = []
    for probe in correspondence.get("probes", []):
        for row in probe.get("validation", []):
            if row.get("example_id") != example_id:
                continue
            local = row.get("local_best")
            if isinstance(local, Mapping) and isinstance(local.get("cosine_similarity"), (int, float)):
                values.append(float(local["cosine_similarity"]))
    return min(values) if values else None


def _confidence(
    candidate: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
    correspondence: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    starts = [float(ref["rendered_start_sec"]) for ref in refs]
    ends = [float(ref["rendered_end_sec"]) for ref in refs]
    endpoint_disagreement = max(max(starts) - min(starts), max(ends) - min(ends))
    local_similarity = _local_similarity(correspondence, str(candidate["example_id"]))
    clip = next((row for row in correspondence.get("clips", []) if row.get("example_id") == candidate.get("example_id")), None)
    evidence_disagreement = float(clip.get("resolution_disagreement_sec")) if isinstance(clip, Mapping) else endpoint_disagreement
    consistent = bool(clip.get("resolution_consistent")) if isinstance(clip, Mapping) else endpoint_disagreement <= float(policy["max_endpoint_disagreement_sec"])
    endpoint_score = max(0.0, 1.0 - endpoint_disagreement / max(float(policy["max_endpoint_disagreement_sec"]), 1e-9))
    similarity_score = 0.5 if local_similarity is None else max(0.0, min(1.0, (local_similarity - 0.35) / 0.45))
    score = 0.55 * endpoint_score + 0.45 * similarity_score
    return {
        "score": round(float(score), 6),
        "endpoint_disagreement_sec": endpoint_disagreement,
        "evidence_resolution_disagreement_sec": evidence_disagreement,
        "local_similarity_min": local_similarity,
        "resolution_consistent": consistent,
        "factors": sorted(ref.get("factor") for ref in refs if isinstance(ref.get("factor"), int)),
    }


def align_candidate(
    candidate: Mapping[str, Any],
    *,
    song: Mapping[str, Any],
    correspondence: Mapping[str, Any],
    reference_manifest_sha256: str,
    policy: Mapping[str, Any] = ALIGNMENT_POLICY,
) -> dict[str, Any]:
    """Create one machine proxy row without claiming human confirmation."""

    if song.get("split") == "test":
        raise ValueError("sealed test songs cannot be machine-aligned")
    if candidate.get("song_id") != song.get("song_id") or candidate.get("split", song.get("split")) != song.get("split"):
        raise ValueError("candidate song/split differs from reference song")
    start = finite_number(candidate.get("start_sec"), "original start", minimum=0)
    end = finite_number(candidate.get("end_sec"), "original end", minimum=0)
    if end <= start or end - start > 20.000001:
        raise ValueError("candidate duration must be in (0, 20] seconds")
    note_ids, refs = _candidate_reference(candidate)
    note_map = {note["note_id"]: note for note in song.get("notes", [])}
    phones = _phone_rows(note_ids, note_map)
    confidence = _confidence(candidate, refs, correspondence, policy)
    good = (
        confidence["resolution_consistent"]
        and confidence["endpoint_disagreement_sec"] <= float(policy["max_endpoint_disagreement_sec"])
        and (confidence["local_similarity_min"] is None or confidence["local_similarity_min"] >= float(policy["minimum_local_similarity"]))
    )
    status = "machine_aligned" if good else "needs_review"
    primary = refs[0]
    return {
        "example_id": str(candidate.get("candidate_id") or candidate.get("example_id")),
        "source_example_id": str(candidate.get("example_id")),
        "song_id": song["song_id"],
        "split": song["split"],
        "audio_role": "original",
        "audio_path": song["original_audio"]["path"],
        "audio_sha256": song["original_audio"]["sha256"],
        "source_vsqx_sha256": song["human_vsqx"]["sha256"],
        "reference_manifest_sha256": reference_manifest_sha256,
        "start_sec": start,
        "end_sec": end,
        "status": status,
        "reference_kind": "machine_aligned_proxy",
        "human_confirmed": False,
        "official_evaluation_eligible": False,
        "training_eligible": status == "machine_aligned" and song["split"] == "train",
        "note_ids": note_ids,
        "phones": phones,
        "rendered_interval": {"start_sec": primary["rendered_start_sec"], "end_sec": primary["rendered_end_sec"]},
        "confidence": confidence,
        "human_review_required": status != "machine_aligned",
        "selection_uses_model_outputs": False,
        "selection_uses_per": False,
        "source_phoneme_approval_preserved": True,
    }


def build_machine_alignment(
    *,
    reference: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    correspondences: Mapping[str, Mapping[str, Any]],
    source_bindings: Mapping[str, Any],
    policy: Mapping[str, Any] = ALIGNMENT_POLICY,
) -> dict[str, Any]:
    songs = {song["song_id"]: song for song in reference.get("songs", [])}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for candidate in candidates:
        song_id = candidate.get("song_id")
        try:
            song = songs[song_id]
            correspondence = correspondences[song_id]
            rows.append(align_candidate(candidate, song=song, correspondence=correspondence,
                                         reference_manifest_sha256=str(source_bindings["reference"]["sha256"]), policy=policy))
        except (KeyError, TypeError, ValueError) as exc:
            # A candidate-specific failure is an exception-queue item, not a
            # silent discard.  It cannot enter proxy training until the mapping
            # or correspondence evidence is repaired.
            errors.append({"candidate_id": candidate.get("candidate_id"), "song_id": song_id,
                           "status": "needs_review", "reason": str(exc),
                           "training_eligible": False, "official_evaluation_eligible": False})
    result = {
        "schema_version": SCHEMA,
        "policy": dict(policy),
        "source_bindings": dict(source_bindings),
        "examples": rows,
        "errors": errors,
        "counts": {
            "total": len(candidates),
            "machine_aligned": sum(row["status"] == "machine_aligned" for row in rows),
            "needs_review": sum(row["status"] == "needs_review" for row in rows)
            + sum(row["status"] == "needs_review" for row in errors),
            "rejected": sum(row["status"] == "rejected" for row in errors),
        },
        "training": {
            "machine_proxy_rows": sum(row.get("training_eligible", False) for row in rows),
            "human_confirmation_required_for_official_evaluation": True,
            "original_accuracy_claim_allowed": False,
        },
        "test_predictions_opened": False,
        "historical_sources_modified": False,
        "default_model_replaced": False,
    }
    result["manifest_hash"] = sha256_object(result)
    return result


def build_edge_review_queue(
    *,
    packet: Mapping[str, Any],
    reference: Mapping[str, Any],
    correspondence: Mapping[str, Mapping[str, Any]],
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Create machine suggestions for boundary reviews without promoting them."""

    songs = {song["song_id"]: song for song in reference.get("songs", [])}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in packet.get("items", []):
        review_id = item.get("review_id")
        try:
            example_id = _require_text(item.get("example_id"), "review example_id")
            song = songs["shalong"]
            note_ids = item.get("candidate_note_ids")
            if not item.get("candidate_reference_mapping_complete") or not isinstance(note_ids, list) or not note_ids:
                raise ValueError("boundary candidate has no complete note sequence")
            phones = _phone_rows(note_ids, {n["note_id"]: n for n in song.get("notes", [])})
            clip = next((row for row in correspondence["shalong"].get("clips", []) if row.get("example_id") == example_id), None)
            local_similarity = _local_similarity(correspondence["shalong"], example_id)
            disagreement = float(clip.get("resolution_disagreement_sec")) if isinstance(clip, Mapping) else None
            rows.append({
                "review_id": review_id,
                "example_id": example_id,
                "song_id": "shalong",
                "side": item.get("side"),
                "original_cut_sec": item.get("original_cut_sec"),
                "suggested_note_ids": list(note_ids),
                "suggested_phones": phones,
                "confidence": {"local_similarity_min": local_similarity, "resolution_disagreement_sec": disagreement,
                                "resolution_consistent": bool(clip.get("resolution_consistent")) if isinstance(clip, Mapping) else False},
                "status": "machine_suggested",
                "human_review_required": True,
                "official_evaluation_eligible": False,
            })
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"review_id": review_id, "status": "needs_review", "reason": str(exc)})
    result = {
        "schema_version": "shalong_machine_edge_suggestions_v1",
        "source_bindings": dict(source_bindings),
        "items": rows,
        "errors": errors,
        "counts": {"total": len(packet.get("items", [])), "machine_suggested": len(rows), "needs_review": len(errors)},
        "human_confirmation_required": True,
        "official_evaluation_eligible": False,
        "test_predictions_opened": False,
    }
    result["manifest_hash"] = sha256_object(result)
    return result

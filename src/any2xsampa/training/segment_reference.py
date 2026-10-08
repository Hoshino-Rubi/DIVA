"""Explicit, human-bound clip references with optional edge-phone evidence.

Version two is deliberately separate from the historical complete-note
reference.  It can represent a clip which begins or ends inside a note, but it
never infers which phones are audible from a timestamp alone.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from any2xsampa.training.limited_data_artifacts import (
    finite_number,
    require_digest,
    sha256_file,
    sha256_object,
    verify_file,
    write_json_new,
)


SCHEMA = "original_segment_reference_v2"
RESPONSE_SCHEMA = "original_segment_review_responses_v2"
STATUSES = {"pending", "human_confirmed", "rejected"}
BOUNDARY_STATES = {"full", "truncated_start", "truncated_end", "truncated_both"}


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _contiguous(notes: Sequence[Mapping[str, Any]], note_ids: Sequence[str]) -> None:
    if not note_ids or len(note_ids) != len(set(note_ids)):
        raise ValueError("note_ids must be a non-empty unique sequence")
    positions = {row["note_id"]: index for index, row in enumerate(notes)}
    if any(note_id not in positions for note_id in note_ids):
        raise ValueError("segment references an unknown note")
    indices = [positions[note_id] for note_id in note_ids]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("segment notes must be contiguous in source order")


def _note_map(notes: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {row["note_id"]: row for row in notes}
    if len(result) != len(notes):
        raise ValueError("duplicate source note IDs")
    return result


def _phone_rows(
    notes: Sequence[Mapping[str, Any]],
    note_ids: Sequence[str],
    phones: Sequence[Mapping[str, Any]],
    boundary_state: str,
) -> list[dict[str, Any]]:
    if boundary_state not in BOUNDARY_STATES:
        raise ValueError("unsupported boundary_state")
    lookup = _note_map(notes)
    _contiguous(notes, note_ids)
    expected: list[tuple[str, int, str]] = []
    for note_id in note_ids:
        row = lookup[note_id]
        targets = row.get("model_target_ipa", [])
        if row.get("mapping_status") != "resolved_proxy" or not isinstance(targets, list):
            raise ValueError("segment contains an unresolved mapped note")
        expected.extend((note_id, index, phone) for index, phone in enumerate(targets))
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for phone in phones:
        if not isinstance(phone, Mapping):
            raise ValueError("phone selection must be an object")
        note_id = _string(phone.get("note_id"), "phone.note_id")
        index = phone.get("phone_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("phone.phone_index must be a non-negative integer")
        key = (note_id, index)
        if key in by_key:
            raise ValueError("duplicate phone selection")
        by_key[key] = phone
    expected_map = {(note_id, index): symbol for note_id, index, symbol in expected}
    if set(by_key) - set(expected_map):
        raise ValueError("phone selection is outside the declared note sequence")
    selected: list[dict[str, Any]] = []
    for note_id, index, symbol in expected:
        if (note_id, index) not in by_key:
            continue
        row = by_key[(note_id, index)]
        if row.get("phone") != symbol:
            raise ValueError("phone selection symbol differs from the source mapping")
        audible = row.get("audible")
        if not isinstance(audible, bool) or not audible:
            raise ValueError("only explicitly audible phones may enter a confirmed reference")
        selected.append({"note_id": note_id, "phone_index": index, "phone": symbol,
                         "audible": True, "edge_status": row.get("edge_status", "full")})
    # A full note may not silently lose phones.  Edge notes may be partial only
    # when the reviewer explicitly enumerates the audible phones.
    first, last = note_ids[0], note_ids[-1]
    for note_id in note_ids[1:-1]:
        target_count = len(lookup[note_id]["model_target_ipa"])
        if sum(row["note_id"] == note_id for row in selected) != target_count:
            raise ValueError("internal notes must retain every mapped phone")
    for note_id in (first, last):
        target_count = len(lookup[note_id]["model_target_ipa"])
        selected_count = sum(row["note_id"] == note_id for row in selected)
        if note_id == first and note_id == last and boundary_state == "full" and selected_count != target_count:
            raise ValueError("a full one-note segment must retain every phone")
        if note_id == first and boundary_state in {"full", "truncated_end"} and selected_count != target_count:
            raise ValueError("a non-truncated first note must retain every phone")
        if note_id == last and boundary_state in {"full", "truncated_start"} and selected_count != target_count:
            raise ValueError("a non-truncated last note must retain every phone")
        if selected_count == 0:
            raise ValueError("a confirmed edge note needs at least one audible phone")
    return selected


def validate_segment_reference(reference: Mapping[str, Any], *, source_root: Path | None = None) -> None:
    if reference.get("schema_version") != SCHEMA:
        raise ValueError("unsupported segment reference schema")
    unsigned = {key: value for key, value in reference.items() if key != "manifest_hash"}
    if reference.get("manifest_hash") != sha256_object(unsigned):
        raise ValueError("segment reference hash mismatch")
    examples = reference.get("examples")
    if not isinstance(examples, list):
        raise ValueError("segment reference examples must be a list")
    seen: set[str] = set()
    for row in examples:
        example_id = _string(row.get("example_id"), "example_id")
        if example_id in seen:
            raise ValueError("duplicate segment example")
        seen.add(example_id)
        if row.get("status") not in STATUSES:
            raise ValueError("invalid segment status")
        for field in ("audio_sha256", "source_vsqx_sha256", "reference_manifest_sha256"):
            require_digest(row.get(field), field)
        start = finite_number(row.get("start_sec"), "segment start", minimum=0)
        end = finite_number(row.get("end_sec"), "segment end", minimum=0)
        if end <= start or end - start > 20.000001:
            raise ValueError("segment duration must be in (0, 20] seconds")
        notes = row.get("note_ids")
        if not isinstance(notes, list) or (not notes and row.get("status") not in {"pending", "rejected"}):
            raise ValueError("confirmed segment note_ids are required")
        phones = row.get("phones", [])
        if not isinstance(phones, list):
            raise ValueError("segment phones must be a list")
        if row.get("status") == "human_confirmed":
            if row.get("boundary_state") not in BOUNDARY_STATES:
                raise ValueError("confirmed segment requires boundary_state")
            if not _string(row.get("reviewer"), "reviewer") or not _string(row.get("review_statement"), "review_statement"):
                raise ValueError("confirmed segment requires human evidence")
            if not phones:
                raise ValueError("confirmed segment requires explicit phones")
        if row.get("status") == "pending" and row.get("phones"):
            raise ValueError("pending segment cannot contain confirmed phones")
        if row.get("status") == "rejected" and row.get("phones"):
            raise ValueError("rejected segment cannot contain reference phones")
        if source_root is not None:
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = source_root / audio_path
            verify_file(audio_path, row["audio_sha256"])


def import_responses(
    *,
    packet: Mapping[str, Any],
    responses: Mapping[str, Any],
    source_reference: Mapping[str, Any],
    source_reference_hash: str,
) -> dict[str, Any]:
    if responses.get("schema_version") != RESPONSE_SCHEMA:
        raise ValueError("unsupported segment response schema")
    if responses.get("packet_sha256") != packet.get("packet_sha256"):
        raise ValueError("responses are bound to a different review packet")
    if packet.get("test_songs_opened") is True:
        raise ValueError("test review items cannot be imported")
    songs = {row["song_id"]: row for row in source_reference["songs"]}
    by_review = {row["review_id"]: row for row in packet.get("items", [])}
    supplied = responses.get("responses")
    if not isinstance(supplied, list):
        raise ValueError("responses must be a list")
    answers = {row.get("review_id"): row for row in supplied}
    if len(answers) != len(supplied) or set(answers) != set(by_review):
        raise ValueError("responses must cover each packet item exactly once")
    rows: list[dict[str, Any]] = []
    for review_id, item in by_review.items():
        answer = answers[review_id]
        status = answer.get("status", "pending")
        if status not in STATUSES:
            raise ValueError("invalid response status")
        clip = next((c for c in source_reference["original_clips"] if c["example_id"] == item.get("example_id")), None)
        song_id = item.get("song_id") or (clip.get("song_id") if clip is not None else None)
        song = songs.get(song_id)
        if song is None or song["split"] == "test":
            raise ValueError("review item references a sealed or unknown song")
        note_map = _note_map(song["notes"])
        note_ids = answer.get("note_ids", [])
        if status in {"human_confirmed", "rejected"} and not isinstance(note_ids, list):
            raise ValueError("confirmed/rejected responses need note_ids list")
        if status == "human_confirmed":
            _contiguous(song["notes"], note_ids)
            phones = _phone_rows(song["notes"], note_ids, answer.get("phones", []), answer.get("boundary_state", "full"))
        else:
            phones = []
            if note_ids:
                _contiguous(song["notes"], note_ids)
        start = finite_number(item.get("original_cut_sec", item.get("start_sec")), "review start", minimum=0)
        end = finite_number(item.get("original_cut_sec", item.get("end_sec", start + 1)), "review end", minimum=0)
        if item.get("example_id") and "original_cut_sec" in item:
            # Boundary reviews describe one edge, not a new crop.  Bind the
            # unchanged old 18-second interval and keep the edge evidence.
            if clip is None:
                raise ValueError("boundary review references unknown clip")
            start, end = clip["start_sec"], clip["end_sec"]
        segment_id = item.get("segment_id") or item.get("candidate_id") or review_id
        row = {
            "review_id": review_id, "side": item.get("side"), "example_id": segment_id, "source_example_id": item.get("example_id"), "song_id": song_id,
            "split": song["split"], "audio_role": "original", "audio_path": song["original_audio"]["path"],
            "audio_sha256": song["original_audio"]["sha256"], "source_vsqx_sha256": song["human_vsqx"]["sha256"],
            "reference_manifest_sha256": source_reference_hash, "start_sec": start, "end_sec": end,
            "status": status, "boundary_state": answer.get("boundary_state", "full") if status == "human_confirmed" else None,
            "note_ids": list(note_ids), "phones": phones, "reviewer": answer.get("reviewer"),
            "review_statement": answer.get("review_statement"), "raw_response": answer,
            "source_phoneme_approval_preserved": True,
        }
        rows.append(row)
    result = {"schema_version": SCHEMA, "source_reference_manifest_sha256": source_reference_hash,
              "packet_sha256": packet["packet_sha256"], "examples": rows,
              "human_confirmed_count": sum(row["status"] == "human_confirmed" for row in rows),
              "test_predictions_opened": False, "historical_sources_modified": False}
    result["manifest_hash"] = sha256_object(result)
    validate_segment_reference(result)
    return result


def write_segment_reference(path: Path, reference: Mapping[str, Any]) -> str:
    validate_segment_reference(reference)
    return write_json_new(path, reference)

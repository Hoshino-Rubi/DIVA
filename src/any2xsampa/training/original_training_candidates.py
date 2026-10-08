"""Deterministic, audio-only proposals for bounded original-song review."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from any2xsampa.training.limited_data_artifacts import finite_number


def _notes_in(notes: Sequence[Mapping[str, Any]], start: float, end: float) -> list[Mapping[str, Any]]:
    return [note for note in notes if note["start_sec"] < end and note["end_sec"] > start]


def _candidate_reference(song: Mapping[str, Any], start: float, end: float, offset: float, factor: int) -> dict[str, Any]:
    rendered_start = round(start + offset, 6)
    rendered_end = round(end + offset, 6)
    duration = finite_number(song["rendered_audio"]["duration_sec"], "rendered duration", minimum=0)
    valid_range = 0 <= rendered_start < rendered_end <= duration
    selected = _notes_in(song["notes"], rendered_start, rendered_end) if valid_range else []
    unresolved = sorted({token for note in selected for token in note.get("unresolved_tokens", [])})
    mapping_complete = bool(selected) and not unresolved and all(note.get("mapping_status") == "resolved_proxy" for note in selected)
    return {
        "factor": factor,
        "offset_sec": offset,
        "rendered_start_sec": rendered_start,
        "rendered_end_sec": rendered_end,
        "note_ids": [note["note_id"] for note in selected],
        "model_target_ipa": [phone for note in selected for phone in note.get("model_target_ipa", [])] if mapping_complete else [],
        "model_target_ids": [phone for note in selected for phone in note.get("model_target_ids", [])] if mapping_complete else [],
        "unresolved_tokens": unresolved,
        "mapping_complete": mapping_complete,
        "valid_rendered_range": valid_range,
        "edge_notes_present": bool(selected and (selected[0]["start_sec"] < rendered_start or selected[-1]["end_sec"] > rendered_end)),
    }


def propose_training_candidates(
    *,
    contexts: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    correspondence: Mapping[str, Any],
    maximum: int = 20,
    window_sec: float = 10.0,
    step_sec: float = 0.5,
) -> dict[str, Any]:
    if maximum != 20 or window_sec != 10.0 or step_sec != 0.5:
        raise ValueError("this bounded proposal policy is fixed at 20 x 10s with 0.5s steps")
    if maximum <= 0 or window_sec <= 0 or step_sec <= 0:
        raise ValueError("proposal limits must be positive")
    songs = {song["song_id"]: song for song in reference["songs"] if song["split"] == "train"}
    reports = {row["song_id"]: row for row in correspondence.get("songs", [])}
    order = {song_id: index for index, song_id in enumerate(reference["split"]["train"])}
    pool: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for context in contexts:
        if context.get("split") != "train":
            continue
        song_id = context["song_id"]
        song = songs.get(song_id)
        report = reports.get(song_id)
        if song is None or report is None:
            rejection_counts["missing_train_source"] += 1
            continue
        old_example = next((row for row in report["clips"] if row["example_id"] == context["example_id"]), None)
        if old_example is None or not old_example.get("candidates"):
            rejection_counts["missing_audio_correspondence_candidate"] += 1
            continue
        if any("rendered_start_sec" not in candidate for candidate in old_example["candidates"]):
            rejection_counts["invalid_audio_correspondence_candidate"] += 1
            continue
        offsets = []
        for candidate in old_example["candidates"]:
            offsets.append((int(candidate["factor"]), finite_number(candidate["rendered_start_sec"], "candidate start") - finite_number(old_example["requested_original_start_sec"], "old start")))
        cursor = finite_number(context["start_sec"], "context start", minimum=0)
        last = finite_number(context["end_sec"], "context end", minimum=0) - window_sec
        while cursor <= last + 1e-7:
            start, end = round(cursor, 6), round(cursor + window_sec, 6)
            candidates = [_candidate_reference(song, start, end, offset, factor) for factor, offset in offsets]
            if not all(candidate["valid_rendered_range"] for candidate in candidates):
                rejection_counts["candidate_outside_rendered_audio"] += 1
            elif not all(candidate["mapping_complete"] for candidate in candidates):
                rejection_counts["whole_proposed_window_unresolved"] += 1
            elif not all(candidate["note_ids"] for candidate in candidates):
                rejection_counts["no_mapped_notes"] += 1
            else:
                pool.append({
                    "candidate_id": f"{song_id}-original-candidate-{len(pool) + 1:03d}",
                    "song_id": song_id, "split": "train", "context_id": context["context_id"],
                    "example_id": context["example_id"], "audio_role": "original",
                    "audio_path": context["audio_path"], "audio_sha256": context["audio_sha256"],
                    "start_sec": start, "end_sec": end, "duration_sec": window_sec,
                    "candidate_references": candidates,
                    "selection_uses_model_outputs": False, "selection_uses_per": False,
                    "human_status": "pending", "training_eligible": False,
                    "center_distance_sec": abs((start + end) / 2 - (finite_number(context["old_start_sec"], "old start") + finite_number(context["old_end_sec"], "old end")) / 2),
                })
            cursor = round(cursor + step_sec, 6)
    initial_pool_count = len(pool)
    selected: list[dict[str, Any]] = []
    song_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    while pool and len(selected) < maximum:
        row = min(pool, key=lambda candidate: (
            song_counts[candidate["song_id"]], context_counts[candidate["context_id"]],
            candidate["center_distance_sec"], order.get(candidate["song_id"], 999), candidate["start_sec"],
        ))
        selected.append(row)
        song_counts[row["song_id"]] += 1
        context_counts[row["context_id"]] += 1
        pool = [candidate for candidate in pool if not (
            candidate["song_id"] == row["song_id"] and candidate["start_sec"] < row["end_sec"] and row["start_sec"] < candidate["end_sec"]
        )]
    return {
        "schema_version": "original_training_candidate_proposals_v2",
        "policy": {"maximum_candidates": maximum, "window_sec": window_sec, "step_sec": step_sec,
                   "selection_order": ["song_count", "context_count", "center_distance", "fixed_song_order", "start_sec"],
                   "selection_uses_model_outputs": False, "selection_uses_per": False, "non_overlapping_within_song": True},
        "candidates": selected,
        "candidate_pool_before_selection": initial_pool_count,
        "song_counts": dict(song_counts), "context_counts": dict(context_counts),
        "unique_audio_seconds": sum(row["duration_sec"] for row in selected),
        "rejection_counts": dict(rejection_counts),
        "test_songs_opened": False, "human_confirmation_required": True,
    }

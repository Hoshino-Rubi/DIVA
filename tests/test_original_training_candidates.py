from __future__ import annotations

from any2xsampa.training.original_training_candidates import propose_training_candidates


def test_candidate_policy_is_bounded_and_balanced() -> None:
    notes = []
    for i in range(80):
        notes.append({"note_id": f"n{i}", "start_sec": i * .25, "end_sec": (i + 1) * .25, "mapping_status": "resolved_proxy", "unresolved_tokens": [], "model_target_ipa": ["a"], "model_target_ids": [1], "raw_v4c_tokens": ["a"], "raw_lyric": "字"})
    song = {"song_id": "freedom-flower", "split": "train", "notes": notes, "rendered_audio": {"duration_sec": 30}}
    contexts = [{"context_id": "c1", "example_id": "e1", "song_id": "freedom-flower", "split": "train", "start_sec": 0, "end_sec": 30, "old_start_sec": 6, "old_end_sec": 24, "audio_path": "x.wav", "audio_sha256": "a" * 64}]
    correspondence = {"songs": [{"song_id": "freedom-flower", "clips": [{"example_id": "e1", "requested_original_start_sec": 6, "requested_original_end_sec": 24, "candidates": [{"factor": 4, "rendered_start_sec": 6, "rendered_end_sec": 24}, {"factor": 8, "rendered_start_sec": 6, "rendered_end_sec": 24}]}]}]}
    reference = {"songs": [song], "split": {"train": ["freedom-flower"]}}
    result = propose_training_candidates(contexts=contexts, reference=reference, correspondence=correspondence)
    assert len(result["candidates"]) == 2
    assert result["policy"]["maximum_candidates"] == 20
    assert all(not row["selection_uses_model_outputs"] and not row["selection_uses_per"] for row in result["candidates"])

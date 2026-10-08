from __future__ import annotations

from tools.assemble_original_clip_references import _assemble_one


def _note(note_id: str, phones: list[str]) -> dict:
    return {"note_id": note_id, "mapping_status": "resolved_proxy", "model_target_ipa": phones}


def _edge(review_id: str, side: str, phones: list[dict]) -> dict:
    return {"review_id": review_id, "source_example_id": "clip", "song_id": "song", "split": "dev", "audio_path": "song.wav", "audio_sha256": "a" * 64, "source_vsqx_sha256": "b" * 64, "start_sec": 1.0, "end_sec": 11.0, "side": side, "note_ids": ["n1" if side == "start" else "n3"], "phones": phones, "reviewer": "u", "review_statement": side,}


def test_assembly_keeps_internal_note_and_explicit_edge_subsets() -> None:
    song = {"notes": [_note("n1", ["m", "a"]), _note("n2", ["t", "s"]), _note("n3", ["o", "n"])]}
    start = _edge("B01", "start", [{"note_id": "n1", "phone_index": 1, "phone": "a", "audible": True}])
    end = _edge("B02", "end", [{"note_id": "n3", "phone_index": 0, "phone": "o", "audible": True}])
    candidate = {"factor": 4, "mapping_complete": True, "note_ids": ["n1", "n2", "n3"]}
    result = _assemble_one(start, end, candidate, song, "c" * 64)
    assert result["example_id"] == "clip"
    assert [row["phone"] for row in result["phones"]] == ["a", "t", "s", "o"]
    assert result["boundary_state"] == "truncated_both"

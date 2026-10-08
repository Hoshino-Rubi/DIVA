from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


def module(name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def order_inputs() -> tuple[dict, dict]:
    song = {"human_vsqx": {"sha256": "a" * 64}, "original_audio": {"sha256": "b" * 64}}
    statement = "The full song order is confirmed; a leading offset exists."
    row = {"song_id": "shalong", "split": "dev", "status": "human_confirmed_order_only", "suggestion_only": True,
           "complete_sequence_confirmed": False, "reference_manifest_sha256": "c" * 64, "review_evidence_sha256": "d" * 64,
           "source_vsqx_sha256": "a" * 64, "original_audio_sha256": "b" * 64, "review_statement": statement}
    value = {"schema_version": "original_clip_binding_reviews_v1", "items": [{**row, "example_id": str(i)} for i in range(3)],
             "scope": "Misleading legacy exact-clip wording must not override order-only status.",
             "evidence": {"source": "direct_user_statement_in_current_conversation", "statement": statement,
                          "statement_sha256": hashlib.sha256(statement.encode("utf-8")).hexdigest()}}
    return value, song


def test_order_only_evidence_does_not_confirm_numeric_offset_or_clips() -> None:
    value, song = order_inputs()
    result = module("audit_shalong_clock").normalize_order_evidence(value, song, "c" * 64, "d" * 64)
    assert result["full_song_order_confirmed"]
    assert not result["numeric_offset_confirmed"]
    assert not result["machine_crops_confirmed"]
    assert "note_ids" not in result


@pytest.mark.parametrize("field,value", [("status", "human_confirmed"), ("split", "test"), ("song_id", "king-of-karaoke"), ("complete_sequence_confirmed", True), ("source_vsqx_sha256", "0" * 64)])
def test_overpromoted_or_different_order_evidence_rejected(field, value) -> None:
    payload, song = order_inputs()
    payload["items"][0][field] = value
    with pytest.raises(ValueError):
        module("audit_shalong_clock").normalize_order_evidence(payload, song, "c" * 64, "d" * 64)


def test_statement_hash_not_advisory() -> None:
    payload, song = order_inputs()
    payload["evidence"]["statement"] = "different"
    with pytest.raises(ValueError, match="hash"):
        module("audit_shalong_clock").normalize_order_evidence(payload, song, "c" * 64, "d" * 64)


def test_future_order_tool_scope_cannot_say_exact_clips_are_confirmed() -> None:
    source = (Path(__file__).resolve().parents[1] / "tools/confirm_shalong_clip_bindings.py").read_text(encoding="utf-8")
    assert "Exact Shalong fixed-clip sequence/order correspondence confirmed" not in source
    assert "numeric clock offset are not confirmed" in source


def test_foreign_descriptor_rejected_before_cache_read(tmp_path) -> None:
    from any2xsampa.training.limited_data_artifacts import write_json_new
    path = tmp_path / "report.json"
    write_json_new(path, {"song_id": "king-of-karaoke", "split": "test"})
    with pytest.raises(ValueError, match="different song descriptor"):
        module("audit_shalong_clock").read_descriptors(path, {}, {})


def test_boundary_review_never_calls_order_an_exact_clip_approval() -> None:
    from any2xsampa.training.limited_data_artifacts import sha256_object
    song = {"song_id": "shalong", "human_vsqx": {"sha256": "a"*64},
            "original_audio": {"sha256": "b"*64, "duration_sec": 100},
            "rendered_audio": {"sha256": "c"*64, "duration_sec": 100},
            "notes": [{"note_id": "n1", "raw_lyric": "a", "raw_v4c_tokens": ["k_h"], "start_sec": 10, "end_sec": 30}]}
    clips = [{"example_id": f"shalong-original-{i}", "song_id": "shalong", "start_sec": i*20, "end_sec": i*20+18} for i in range(1,4)]
    rule = {"schema_version": "original_reference_clock_rule_v1", "song_id": "shalong", "split": "dev",
            "source_bindings": {"original_audio": song["original_audio"], "rendered_audio": song["rendered_audio"], "source_vsqx": song["human_vsqx"]},
            "composed": {"offset_sec": 6.7}, "transforms": [{"offset_sec": 6.7}], "clips": clips}
    rule["rule_hash"] = sha256_object(rule)
    items = module("build_clock_boundary_review").boundary_items(rule, {"songs": [song], "original_clips": clips})
    assert len(items) == 6
    assert all(r["status"] == "pending" and r["human_response"] is None for r in items)
    assert all(r["cut_in_preview_sec"] == 1.25 for r in items)
    assert all(not r["numeric_offset_or_clip_approval_inferred"] for r in items)

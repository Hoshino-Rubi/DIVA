from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from any2xsampa.training.limited_data_artifacts import sha256_file, write_json_new


def worker():
    spec = importlib.util.spec_from_file_location("auxiliary_worker_test", Path("tools/run_auxiliary_worker.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_context() -> dict:
    return {"context_id": "freedom-flower-original-01-context", "example_id": "freedom-flower-original-01",
            "song_id": "freedom-flower", "split": "train", "audio_path": "source.wav", "audio_sha256": "a" * 64,
            "start_sec": 4., "end_sec": 34., "old_start_sec": 10., "old_end_sec": 28., "contains_labels_or_score_context": False}


def fixture_index(run: Path, *, stage: str = "separate") -> tuple[Path, list[dict]]:
    context = frozen_context()
    write_json_new(run / "execution-start.json", {"policy_sha256": "b" * 64})
    entries = []
    for role in (["mixed", "vocal"] if stage == "asr" else [None]):
        record = {"context": context, "clock": {"source": {"sha256": context["audio_sha256"]}}}
        if role:
            record["audio_role"] = role
        path = run / "stage" / f"{role}.json"
        digest = write_json_new(path, record)
        entry = {"context": context, "record_path": str(path), "record_sha256": digest}
        if role:
            entry["audio_role"] = role
        entries.append(entry)
    path = run / "stage/index.json"
    write_json_new(path, {"schema_version": "quality8h_auxiliary_index_v1", "stage": stage, "selection": "smoke8",
                        "policy_sha256": "b" * 64, "test_songs_loaded": False, "entries": entries})
    return path, [context]


def test_worker_only_consumes_hash_bound_complete_non_test_inputs(tmp_path: Path) -> None:
    path, contexts = fixture_index(tmp_path)
    result = worker().load_index(path, sha256_file(path), tmp_path, "separate", contexts)
    assert len(result["entries"]) == 1
    with pytest.raises(ValueError, match="SHA-256"):
        worker().load_index(path, "f" * 64, tmp_path, "separate", contexts)


def test_worker_detects_missing_role_even_with_a_new_outer_hash(tmp_path: Path) -> None:
    import json
    path, contexts = fixture_index(tmp_path, stage="asr")
    value = json.loads(path.read_text())
    value["entries"].pop()
    changed = tmp_path / "stage/changed.json"
    write_json_new(changed, value)
    with pytest.raises(ValueError, match="entire declared subset"):
        worker().load_index(changed, sha256_file(changed), tmp_path, "asr", contexts)


def test_worker_refuses_record_role_spoofing(tmp_path: Path) -> None:
    import json
    path, contexts = fixture_index(tmp_path, stage="asr")
    value = json.loads(path.read_text())
    value["entries"][0]["record_path"] = value["entries"][1]["record_path"]
    value["entries"][0]["record_sha256"] = value["entries"][1]["record_sha256"]
    changed = tmp_path / "stage/changed.json"
    write_json_new(changed, value)
    with pytest.raises(ValueError, match="record identity"):
        worker().load_index(changed, sha256_file(changed), tmp_path, "asr", contexts)

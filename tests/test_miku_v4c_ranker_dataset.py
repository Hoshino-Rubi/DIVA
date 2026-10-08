from __future__ import annotations

import json
from pathlib import Path

import pytest

from any2xsampa.training.ranker_dataset import DatasetBuildError, build_dataset, normalise_candidate


def _source() -> dict[str, str]:
    return {"source_id": "fixture", "revision": "r1", "sha256": "a" * 64}


def _pseudo() -> dict[str, object]:
    return {
        "kind": "pseudo",
        "model_id": "fixture-model",
        "model_revision": "r1",
        "generated_at_utc": "2026-08-11T00:00:00Z",
        "source": _source(),
    }


def _candidate(candidate_id: str, singer: str, song: str, phrase: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "singer_id": singer,
        "song_id": song,
        "phrase_id": phrase,
        "lyrics": "la",
        "phoneme_tokens": ["l", "a"],
        "ipa_tokens": ["l", "a"],
        "articulatory_features": [{"manner": "lateral"}, {"height": "open"}],
        "qwen_score": 0.99,
        "training_data_candidate": True,
        "rendered_wav": "fixture.wav",
        "label_provenance": _pseudo(),
    }


def test_ranker_candidate_fails_closed_without_identity_or_provenance() -> None:
    with pytest.raises(DatasetBuildError, match="identity"):
        normalise_candidate(
            {"candidate_id": "missing", "phoneme_tokens": ["a"]},
            split="train",
            positive_threshold=0.9,
        )
    row = normalise_candidate(
        {
            "candidate_id": "incomplete",
            "group_id": "s/song/p",
            "phoneme_tokens": ["a"],
            "qwen_score": 0.99,
            "rendered_wav": "fixture.wav",
        },
        split="train",
        positive_threshold=0.9,
    ).to_row(split_algorithm="test", seed=1, manifest_sha256="b" * 64)
    assert row["label"] == 0
    assert "training_data_candidate_missing" in row["positive_gate_failures"]
    assert "label_provenance_incomplete" in row["positive_gate_failures"]
    assert "phone_truth_ipa_incomplete" in row["positive_gate_failures"]
    assert "phone_truth_features_incomplete" in row["positive_gate_failures"]
    assert row["phones"][0]["ipa"] is None


def test_ranker_dataset_quarantines_incomplete_phone_truth(tmp_path: Path) -> None:
    valid = _candidate("valid", "s1", "song-1", "phrase-1")
    incomplete = _candidate("incomplete", "s2", "song-2", "phrase-2")
    del incomplete["ipa_tokens"]
    source = tmp_path / "candidates.json"
    output = tmp_path / "dataset"
    source.write_text(json.dumps({"scored_candidates": [valid, incomplete]}), encoding="utf-8")
    _, summary = build_dataset(source, output, val_ratio=0.0, test_ratio=0.0)
    assert summary["row_count"] == 1
    assert summary["quarantined_count"] == 1
    quarantined = json.loads((output / "quarantined.jsonl").read_text(encoding="utf-8"))
    assert quarantined["training_admissible"] is False
    assert "phone_truth_ipa_incomplete" in quarantined["quarantine_reasons"]


def test_ranker_dataset_keeps_shared_singer_in_one_split(tmp_path: Path) -> None:
    rows = [
        _candidate("c1", "same-singer", "song-a", "phrase-a"),
        _candidate("c2", "same-singer", "song-b", "phrase-b"),
        _candidate("c3", "other-singer", "song-c", "phrase-c"),
        _candidate("c4", "third-singer", "song-d", "phrase-d"),
    ]
    source = tmp_path / "candidates.json"
    output = tmp_path / "dataset"
    source.write_text(json.dumps({"scored_candidates": rows}), encoding="utf-8")
    rc, summary = build_dataset(source, output, val_ratio=0.25, test_ratio=0.25, seed=19)
    assert rc == 0
    all_rows = []
    for name in ("train", "val", "test"):
        all_rows.extend(json.loads(line) for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines())
    by_id = {row["candidate_id"]: row for row in all_rows}
    assert by_id["c1"]["split"] == by_id["c2"]["split"]
    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["leakage_guard"]["schema_version"] == "deterministic-split/v1"
    assert summary["split_manifest_sha256"] == manifest["manifest_sha256"]


def test_ranker_dataset_keeps_shared_source_hash_in_one_split(tmp_path: Path) -> None:
    rows = [
        _candidate("c1", "s1", "song-a", "phrase-a"),
        _candidate("c2", "s2", "song-b", "phrase-b"),
        _candidate("c3", "s3", "song-c", "phrase-c"),
    ]
    rows[0]["audio_sha256"] = "f" * 64
    rows[1]["audio_sha256"] = "f" * 64
    source = tmp_path / "candidates.json"
    output = tmp_path / "dataset"
    source.write_text(json.dumps({"scored_candidates": rows}), encoding="utf-8")
    build_dataset(source, output, val_ratio=1 / 3, test_ratio=0.0, seed=7)
    all_rows = [
        json.loads(line)
        for name in ("train", "val")
        for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["candidate_id"]: row for row in all_rows}
    assert by_id["c1"]["split"] == by_id["c2"]["split"]

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from any2xsampa.training.limited_data_artifacts import (
    read_json,
    sha256_file,
    sha256_object,
    write_json_new,
)
from any2xsampa.training.limited_data_reference import (
    DATASET_SCHEMA,
    SPLITS,
    clean_ipa,
    load_mapping,
    original_clip_rows,
    project_tokens,
    rendered_examples,
    training_gate,
    validate_dataset,
    verified_vsqx_source,
    require_rendered_source_role,
)


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "src/any2xsampa/config/v4c_reference_ipa_v4.json"


@pytest.fixture
def mapping() -> dict:
    return load_mapping(MAPPING)


@pytest.fixture
def vocabulary(mapping: dict) -> dict[str, int]:
    symbols = {
        mapping["model_token_aliases"].get(ipa, ipa)
        for entry in mapping["tokens"].values() for ipa in entry["ipa"]
    }
    return {"<blank>": 0, **{ipa: index for index, ipa in enumerate(sorted(symbols), 1)}}


def note(index: int, raw: list[str], mapping: dict, vocabulary: dict) -> dict:
    return {
        "note_id": f"freedom-flower-n{index:03d}",
        "note_index": index,
        "start_sec": float(index),
        "end_sec": index + 0.8,
        **project_tokens(raw, mapping, vocabulary),
    }


def song(notes: list[dict]) -> dict:
    return {
        "song_id": "freedom-flower",
        "split": "train",
        "notes": notes,
        "rendered_audio": {"path": "rendered.wav", "sha256": "a" * 64, "duration_sec": 40.0},
        "human_vsqx": {"sha256": "b" * 64},
        "source_claims": {"phoneme_correctness": "user_asserted_correct"},
    }


def dataset(rows: list[dict]) -> dict:
    value = {"schema_version": DATASET_SCHEMA, "split_sha256": sha256_object(SPLITS), "examples": rows}
    value["dataset_hash"] = sha256_object(value)
    return value


def test_exact_restoration_and_no_new_acoustic_truth(mapping: dict) -> None:
    parent = read_json(MAPPING.parent / mapping["evidence"]["source"])
    old = read_json(MAPPING.parent / "v4c_to_recognizer_contract_v3.json")
    assert len(mapping["tokens"]) == 40
    assert all(entry["ipa"] == parent["tokens"][token]["ipa"] for token, entry in mapping["tokens"].items())
    damaged = {token for token, entry in old["tokens"].items() if not all(clean_ipa(s) for s in entry["ipa"])}
    assert len(damaged) == 26
    assert len(damaged & mapping["tokens"].keys()) == 23
    assert mapping["acoustic_realization_verified"] is False


def test_aspiration_rounding_and_fricative_place_survive(mapping: dict, vocabulary: dict) -> None:
    result = project_tokens(["p_h", "t_h", "k_h", "ts_h", "f", "x", "y"], mapping, vocabulary)
    assert result["model_target_ipa"] == ["p\u02b0", "t\u02b0", "k\u02b0", "t\u0361s\u02b0", "f", "h", "y"]
    assert result["label_scope"] == "derived_engine_reference"
    assert result["acoustic_review_status"] == "not_reviewed"


@pytest.mark.parametrize("symbol", ["?", "k\u5906", "\ufffd", "a b", ""])
def test_damaged_ipa_rejected(symbol: str) -> None:
    assert not clean_ipa(symbol)


@pytest.mark.parametrize("token", ["7", "uAN", "iUN", "iE_r", "ua_n", "uo", "UN", "ts\\_h", "ts\u0060", "ts\u0060_h"])
def test_unresolved_tokens_preserved_without_partial_targets(token: str, mapping: dict, vocabulary: dict) -> None:
    result = project_tokens(["k", token, "i"], mapping, vocabulary)
    assert result["raw_v4c_tokens"] == ["k", token, "i"]
    assert result["unresolved_tokens"] == [token]
    assert result["model_target_ids"] == []
    assert result["mapping_status"] == "unresolved"


def test_whole_window_excluded_not_spliced(mapping: dict, vocabulary: dict) -> None:
    notes = [note(1, ["k", "i"], mapping, vocabulary), note(2, ["7"], mapping, vocabulary), note(3, ["m"], mapping, vocabulary)]
    before = copy.deepcopy(notes)
    rows = rendered_examples(song(notes))
    assert len(rows) == 1
    assert not rows[0]["eligible"]
    assert len(rows[0]["raw_v4c_tokens_by_note"]) == 3
    assert rows[0]["model_target_ids"] == []
    assert notes == before


def test_windows_do_not_duplicate_notes_or_cross_neighbor_context(mapping: dict, vocabulary: dict) -> None:
    notes = [note(i, ["i"], mapping, vocabulary) for i in range(1, 25)]
    rows = rendered_examples(song(notes))
    ids = [note_id for row in rows for note_id in row["note_ids"]]
    assert len(ids) == len(set(ids)) == 24
    assert all(a["end_sec"] <= b["start_sec"] for a, b in zip(rows, rows[1:]))


def test_unsupported_phone_excludes_window_without_collapsing(mapping: dict, vocabulary: dict) -> None:
    vocabulary.pop("y")
    result = project_tokens(["y"], mapping, vocabulary)
    assert result["unsupported_model_ipa"] == ["y"]
    assert result["model_target_ids"] == []


def test_clip_selection_audio_only_deterministic_and_sealed() -> None:
    audio = {"duration_sec": 222.355, "path": "original.wav", "sha256": "a" * 64}
    clips = original_clip_rows("king-of-karaoke", audio)
    assert clips == original_clip_rows("king-of-karaoke", audio)
    assert len(clips) == 3
    assert all(abs(row["end_sec"] - row["start_sec"] - 18) < 1e-6 for row in clips)
    assert all(row["test_outputs_sealed"] for row in clips)
    assert not any(row["training_eligible"] or row["evaluation_eligible"] for row in clips)
    assert not any(row["selection_uses_score_or_lyrics"] for row in clips)


def test_dataset_rejects_tampering_and_relabeling(mapping: dict, vocabulary: dict) -> None:
    rows = rendered_examples(song([note(1, ["i"], mapping, vocabulary)]))
    valid = dataset(rows)
    validate_dataset(valid)
    assert not training_gate(valid)["passed"]
    changed = copy.deepcopy(valid)
    changed["examples"][0]["start_sec"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_dataset(changed)
    rows[0]["audio_role"] = "original"
    with pytest.raises(ValueError, match="relabeled"):
        validate_dataset(dataset(rows))


def test_held_out_training_rejected_even_after_rehash(mapping: dict, vocabulary: dict) -> None:
    rows = rendered_examples(song([note(1, ["i"], mapping, vocabulary)]))
    rows[0].update(song_id="shalong", split="dev", training_eligible=True)
    with pytest.raises(ValueError, match="held-out"):
        validate_dataset(dataset(rows))


def test_json_integrity_and_immutable_writes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    digest = write_json_new(path, {"ipa": "\u0259\u014b"})
    assert digest == sha256_file(path)
    assert read_json(path) == {"ipa": "\u0259\u014b"}
    assert path.read_bytes().isascii()
    with pytest.raises(FileExistsError):
        write_json_new(path, {"replace": True})
    path = tmp_path / "duplicate.json"
    path.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)
    path.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_json(path)


def test_existing_normalization_requires_both_hashes_and_provenance(tmp_path: Path) -> None:
    source, normalized = tmp_path / "source.vsqx", tmp_path / "normalized.vsqx"
    source.write_bytes(b"unaltered historical source")
    normalized.write_bytes(b"previously normalized source")
    record = {
        "path": str(source), "sha256": sha256_file(source),
        "normalized_path": str(normalized), "normalized_sha256": sha256_file(normalized),
        "normalization_applied": True,
    }
    selected, binding = verified_vsqx_source(record)
    assert selected == normalized
    assert binding["sha256"] != binding["normalized_sha256"]
    assert binding["normalization_executed_this_run"] is False
    record["normalization_applied"] = False
    with pytest.raises(ValueError, match="normalization provenance"):
        verified_vsqx_source(record)
    record["normalization_applied"] = True
    normalized.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verified_vsqx_source(record)


def test_source_role_is_checked_against_triplet_not_just_a_label(tmp_path: Path) -> None:
    triplet = {
        "original_wav": {"path": str(tmp_path / "original.wav"), "sha256": "a" * 64},
        "rendered_wav": {"path": str(tmp_path / "rendered.wav"), "sha256": "b" * 64},
        "human_vsqx": {"sha256": "c" * 64},
    }
    row = {"audio_role": "rendered", "audio_path": triplet["rendered_wav"]["path"], "audio_sha256": "b" * 64, "source_vsqx_sha256": "c" * 64}
    require_rendered_source_role(row, triplet)
    row.update(audio_path=triplet["original_wav"]["path"], audio_sha256="a" * 64)
    with pytest.raises(ValueError, match="rendered role"):
        require_rendered_source_role(row, triplet)
    row["audio_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source path"):
        require_rendered_source_role(row, triplet)

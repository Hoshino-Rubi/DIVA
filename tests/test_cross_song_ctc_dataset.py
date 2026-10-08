from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_cross_song_ctc_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_cross_song_ctc_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _dataset(path: Path, group: str, digest: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "example_id": f"{group}-1",
                        "group_id": group,
                        "audio_sha256": digest,
                        "source_vsqx_sha256": digest,
                        "target_phone_tokens": ["a"],
                        "internal_ipa_tokens": ["ɐ"],
                        "articulatory_features": [{"class": "vowel"}],
                        "human_authored_annotation_evidence": True,
                        "label_provenance": {
                            "schema_version": "human_phone_label_v1",
                            "kind": "human",
                            "source_id": group,
                            "source_revision": digest,
                            "annotation_sha256": digest,
                            "policy_version": "test",
                            "annotator_id": "test",
                            "review_status": "approved"
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cross_song_builder_creates_both_leakage_safe_directions(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "a.json", "song-a", "a" * 64)
    second = _dataset(tmp_path / "b.json", "song-b", "b" * 64)
    payload = MODULE.build_dataset(first, second, tmp_path / "combined.json")
    assert len(payload["folds"]) == 2
    assert {fold["validation_group"] for fold in payload["folds"]} == {"song-a", "song-b"}
    assert all(fold["leakage_free"] is True for fold in payload["folds"])


def test_cross_song_builder_rejects_shared_source_evidence(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "a.json", "song-a", "a" * 64)
    second = _dataset(tmp_path / "b.json", "song-b", "a" * 64)
    try:
        MODULE.build_dataset(first, second, tmp_path / "combined.json")
    except ValueError as exc:
        assert "source evidence overlaps" in str(exc)
    else:
        raise AssertionError("shared evidence must be rejected")


def test_cross_song_builder_supports_three_song_leave_one_out_folds(tmp_path: Path) -> None:
    paths = [
        _dataset(tmp_path / "a.json", "song-a", "a" * 64),
        _dataset(tmp_path / "b.json", "song-b", "b" * 64),
        _dataset(tmp_path / "c.json", "song-c", "c" * 64),
    ]
    payload = MODULE.build_datasets(paths, tmp_path / "combined-three.json")
    assert len(payload["folds"]) == 3
    assert {fold["validation_group"] for fold in payload["folds"]} == {"song-a", "song-b", "song-c"}
    assert all(len(fold["train_groups"]) == 2 for fold in payload["folds"])
    assert all(set(fold["train_groups"]) | {fold["validation_group"]} == {"song-a", "song-b", "song-c"} for fold in payload["folds"])


def test_cross_song_builder_checks_evidence_against_every_group(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "a.json", "song-a", "a" * 64)
    second = _dataset(tmp_path / "b.json", "song-b", "b" * 64)
    third = _dataset(tmp_path / "c.json", "song-c", "a" * 64)
    try:
        MODULE.build_datasets([first, second, third], tmp_path / "overlap.json")
    except ValueError as exc:
        assert "source evidence overlaps" in str(exc)
    else:
        raise AssertionError("evidence overlap with a non-first group must be rejected")

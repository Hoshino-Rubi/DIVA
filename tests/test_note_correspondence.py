from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

from any2xsampa.music.note_correspondence import audit_local_lag, note_count_ceiling


def test_count_ceiling_counts_missing_even_with_perfect_prediction() -> None:
    row = note_count_ceiling(["n1", "n2", "n3"], 2)
    assert row["maximum_one_to_one_matches"] == 2
    assert row["minimum_false_negatives"] == 1
    assert row["maximum_recall"] == pytest.approx(2 / 3)
    assert not row["pitch_or_boundary_accuracy_inferred"]


@pytest.mark.parametrize("ids,count", [([], 0), (["n1", "n1"], 2), (["n1"], -1), (["n1"], True)])
def test_invalid_ceiling(ids: list[str], count: int) -> None:
    with pytest.raises(ValueError):
        note_count_ceiling(ids, count)


def test_fixed_audio_lag_withheld_validation() -> None:
    rng = np.random.default_rng(17)
    x = rng.random((12, 320))
    y = rng.random((12, 400)) * .005
    y[:, 23:343] = x
    row = audit_local_lag(x, y, start_sec=4, end_sec=20, expected_lag_sec=2.2, frame_sec=.1,
                          radius_sec=1)
    assert row["candidate_lag_sec"] == pytest.approx(2.3)
    assert row["withheld_best_lag_sec"] == pytest.approx(2.3)
    assert row["withheld_lag_disagreement_sec"] == 0
    assert not row["model_predictions_used"]
    assert not row["clock_approved"]


def test_lag_search_edge_is_reported() -> None:
    rng = np.random.default_rng(43)
    x = rng.random((12, 400))
    y = rng.random((12, 500)) * .005
    y[:, 30:430] = x
    row = audit_local_lag(x, y, start_sec=4, end_sec=20, expected_lag_sec=2, frame_sec=.1,
                          radius_sec=1)
    assert row["candidate_lag_sec"] == pytest.approx(3)
    assert row["at_search_edge"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_clock_argument_rejected(value: float) -> None:
    x = np.ones((12, 320))
    with pytest.raises(ValueError):
        audit_local_lag(x, x, start_sec=4, end_sec=20, expected_lag_sec=value, frame_sec=.1)


def test_preflight_rejects_test_song_and_hash_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    source = Path(__file__).resolve().parents[1] / "tools/audit_game_note_correspondence.py"
    spec = spec_from_file_location("audit_game_note_correspondence", source)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    songs = []
    examples = []
    jobs = []
    for i in range(5):
        sid, eid = f"s{i}", f"e{i}"
        songs.append({"song_id": sid, "split": "train", "original_audio": {"sha256": "a"},
                      "human_vsqx": {"sha256": "b"}})
        examples.append({"example_id": eid, "song_id": sid, "split": "train",
                         "status": "human_confirmed", "audio_sha256": "a",
                         "source_vsqx_sha256": "b", "reference_manifest_sha256": "c",
                         "note_ids": ["n1", "n2"], "edge_evidence": {
                             "start": {"status": "confirmed", "note_id": "n1", "audible_identity_confirmed": True},
                             "end": {"status": "confirmed", "note_id": "n2", "audible_identity_confirmed": True}}})
        jobs.append({"id": f"original-{sid}", "song_id": sid, "source_example_id": eid})
    reviewed = tmp_path / "reviewed.json"
    selected = tmp_path / "SELECTED_INPUTS.json"
    manifest = tmp_path / "manifest.json"
    reviewed.write_text(json.dumps({"examples": examples}))
    manifest.write_text(json.dumps({"songs": songs}))
    selected.write_text(json.dumps({"reference_file": str(reviewed), "reference_sha256": "x", "jobs": jobs}))
    monkeypatch.setattr(module, "sha", lambda p: "c" if p == manifest else "x")
    assert len(module.inputs(tmp_path, manifest)[0]["jobs"]) == 5
    songs[0]["split"] = "test"
    manifest.write_text(json.dumps({"songs": songs}))
    with pytest.raises(ValueError, match="song, split"):
        module.inputs(tmp_path, manifest)
    songs[0]["split"] = "train"
    manifest.write_text(json.dumps({"songs": songs}))
    selected.write_text(json.dumps({"reference_file": str(reviewed), "reference_sha256": "changed", "jobs": jobs}))
    with pytest.raises(ValueError, match="source hash changed"):
        module.inputs(tmp_path, manifest)

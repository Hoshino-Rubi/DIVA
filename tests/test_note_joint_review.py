"""Review answers are bound to source audio and never silently become formal truth."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("note_joint_review", Path(__file__).parents[1] / "tools/note_joint_review.py")
assert SPEC and SPEC.loader
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


@pytest.fixture
def packet() -> dict:
    return {"clips": [{"song_id": song, "example_id": example,
                        "source_audio_sha256": "a" * 64, "source_start_sample": 1_000_000 * index,
                        "source_end_sample_exclusive": 1_000_000 * index + 441000,
                        "source_sample_rate": 44100}
                       for index, (song, _, example) in enumerate(REVIEW.SONGS)]}


@pytest.fixture
def answers(packet: dict) -> dict:
    return {"schema_version": "original_note_review_v1", "packet_sha256": "b" * 64,
            "reviewer": "human", "raw_statement": "independently heard source",
            "clips": [{"song_id": clip["song_id"], "example_id": clip["example_id"],
                       "source_audio_sha256": clip["source_audio_sha256"],
                       "source_start_sample": clip["source_start_sample"],
                       "source_sample_rate": 44100, "review_state": "complete",
                       "all_notes_reviewed": True, "silent_confirmed": False, "raw_statement": "", 
                       "notes": [{"id": "note-0001", "start_sample_abs": clip["source_start_sample"] + 100,
                                  "end_sample_abs": clip["source_start_sample"] + 12000,
                                  "pitch_midi": 64, "review_state": "confirmed",
                                  "edge_status": "fully_audible", "raw_statement": "",
                                  "overlap_reason": "", "origin": "manual"}],
                       "uncertain_intervals": [], "edit_log": [], "next_id": 2}
                      for clip in packet["clips"]]}


def test_all_five_complete(packet: dict, answers: dict) -> None:
    coverage = REVIEW.validate_answers(packet, "b" * 64, answers)
    assert coverage["formal_windows"] == 5
    assert coverage["all_five_formal"]


def test_reject_wrong_sha_source_or_song(packet: dict, answers: dict) -> None:
    with pytest.raises(ValueError, match="SHA"):
        REVIEW.validate_answers(packet, "c" * 64, answers)
    answers["clips"][1]["song_id"] = "shalong"
    with pytest.raises(ValueError, match="source binding"):
        REVIEW.validate_answers(packet, "b" * 64, answers)


@pytest.mark.parametrize("key,value", [("start_sample_abs", 0), ("end_sample_abs", 100),
                                        ("pitch_midi", True), ("pitch_midi", 128),
                                        ("end_sample_abs", float("nan"))])
def test_reject_bad_note_samples_and_pitch(packet: dict, answers: dict, key: str, value: object) -> None:
    answers["clips"][1]["notes"][0][key] = value
    with pytest.raises(ValueError):
        REVIEW.validate_answers(packet, "b" * 64, answers)


def test_duplicate_notes_and_overlap_need_provenance(packet: dict, answers: dict) -> None:
    clip = answers["clips"][0]
    clip["notes"].append(copy.deepcopy(clip["notes"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        REVIEW.validate_answers(packet, "b" * 64, answers)
    clip["notes"][1]["id"] = "note-0002"
    with pytest.raises(ValueError, match="Overlapping"):
        REVIEW.validate_answers(packet, "b" * 64, answers)
    clip["notes"][1]["overlap_reason"] = "two overlapping voices"
    assert REVIEW.validate_answers(packet, "b" * 64, answers)["all_five_formal"]


def test_censor_and_uncertainty_not_formal(packet: dict, answers: dict) -> None:
    note = answers["clips"][0]["notes"][0]
    note["edge_status"] = "left_censored"
    with pytest.raises(ValueError, match="Left censor"):
        REVIEW.validate_answers(packet, "b" * 64, answers)
    note["start_sample_abs"] = packet["clips"][0]["source_start_sample"]
    result = REVIEW.validate_answers(packet, "b" * 64, answers)
    assert result["formal_windows"] == 4
    assert result["coverage"][0]["censored_or_uncertain_notes"] == 1
    note["edge_status"] = "fully_audible"
    answers["clips"][0]["uncertain_intervals"] = [{"start_sample_abs": 0,
                                                     "end_sample_abs": 1000, "reason": "unclear"}]
    assert REVIEW.validate_answers(packet, "b" * 64, answers)["formal_windows"] == 4


def test_pending_and_silence(packet: dict, answers: dict) -> None:
    answers["clips"][0]["review_state"] = "pending"
    assert REVIEW.validate_answers(packet, "b" * 64, answers)["formal_windows"] == 4
    clip = answers["clips"][0]
    clip["review_state"] = "complete"
    clip["silent_confirmed"] = True
    with pytest.raises(ValueError, match="silence"):
        REVIEW.validate_answers(packet, "b" * 64, answers)
    clip["notes"] = []
    assert REVIEW.validate_answers(packet, "b" * 64, answers)["all_five_formal"]


def test_one_to_one_and_pitch_sequence_diagnostic() -> None:
    from any2xsampa.music.note_metrics import NoteEvent, evaluate_notes

    ref = [NoteEvent(1, 1.2, 60), NoteEvent(1.2, 1.4, 60)]
    pred = [NoteEvent(1, 1.2, 60), NoteEvent(1, 1.2, 60)]
    assert REVIEW._lcs_pitch(ref, pred) == 2
    assert evaluate_notes(ref, pred, reference_clock_confirmed=True)["pitch_onset_offset"]["matches"] == 1


def test_prediction_inputs_are_relative_to_source_clock(tmp_path: Path) -> None:
    record = {"schema_version": "note_transcription_v1", "time_basis": "original_audio",
              "notes": [{"start_sec": .5, "end_sec": .7, "pitch_midi": 61}]}
    path = tmp_path / "prediction.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    actual = REVIEW._scored_predictions(path, {"source_start_sample": 44100 * 40, "source_rate": 44100})
    assert actual[0].start_sec == 40.5
    assert actual[0].end_sec == 40.7


def test_real_saved_job_bindings_without_model_forward() -> None:
    entries, results = REVIEW._checked_sources()
    assert len(entries) == 5
    assert sum(row["notes"] for row in results["rows"] if row["role"] == "vocal") == 113
    assert sum(row["notes"] for row in results["rows"] if row["role"] == "mixed") == 93
    assert all(entry["source"]["split"] == "train" for entry in entries)


def test_cross_input_agreement_is_not_claimed_as_accuracy(tmp_path: Path) -> None:
    result = REVIEW.diagnose_saved_pairs(tmp_path / "diagnostic.json")
    assert result["formal_joint_accuracy_measured"] is False
    assert result["evaluation_kind"] == "cross_input_consistency_not_accuracy"
    assert result["model_forwards"] == 0
    assert result["totals"]["mixed_notes"] == 93
    assert result["totals"]["vocal_notes"] == 113
    assert len(result["rows"]) == 5
    for row in result["rows"]:
        assert row["joint_agreements_pitch_onset_offset_50ms"] <= row["boundary_only_agreements_50ms"]
        assert row["joint_agreements_pitch_onset_offset_50ms"] <= row["pitch_onset_agreements_50ms"]
    with pytest.raises(FileExistsError):
        REVIEW.diagnose_saved_pairs(tmp_path / "diagnostic.json")

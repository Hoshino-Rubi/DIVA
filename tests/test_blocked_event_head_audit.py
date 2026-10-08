from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_blocked_event_head_screen import job_state, compare_common
from any2xsampa.training.limited_data_artifacts import write_json_new


def test_partial_epochs_do_not_mean_recoverable_checkpoint(tmp_path):
    write_json_new(tmp_path / "START.json", {})
    for i in range(1, 15):
        write_json_new(tmp_path / f"epoch-{i:02d}.json", {"epoch": i, "optimizer_steps": 22})
    state = job_state(tmp_path, 73)
    assert state["state"] == "stopped_without_final_checkpoint"
    assert state["logged_optimizer_steps"] == 308
    assert state["unlogged_optimizer_steps_unknown"]
    assert not state["completed"] and not state["final_checkpoint_exists"]


def test_running_job_cannot_be_audited_as_closed(tmp_path):
    write_json_new(tmp_path / "START.json", {})
    with pytest.raises(ValueError, match="still running"):
        job_state(tmp_path, None)


def test_unknown_recovery_is_not_accepted(tmp_path):
    folder = tmp_path / "E" / "promise"
    for i in range(1, 21):
        write_json_new(folder / f"epoch-{i:02d}.json", {"epoch": i, "optimizer_steps": 22})
    write_json_new(folder / "RESULT.json", {"complete": True, "selected_epoch": 20,
        "quality_training_restarted": False, "quality_process_exit_code": 73})
    with pytest.raises(ValueError, match="unapproved recovery"):
        job_state(folder, 73)


def test_absent_fold_is_not_a_zero_error_result(tmp_path):
    state = job_state(tmp_path, None)
    assert state["state"] == "not_started" and not state["completed"]


def test_partial_comparison_keeps_identical_denominator_without_selection():
    row = {"example_id": "one", "song_id": "a", "reference": ["i", "u"], "prediction": ["i"]}
    extra = {"example_id": "two", "song_id": "b", "reference": ["i"], "prediction": ["i"]}
    result = compare_common({"H": [row, extra], "E": [row]})
    assert result["songs"] == ["a"]
    assert result["metrics"]["H"]["reference_phones"] == 2
    assert result["metrics"]["H"] == result["metrics"]["E"]
    assert not result["selection_allowed"] and not result["screen_gates_evaluated"]
    with pytest.raises(ValueError, match="different evaluation references"):
        compare_common({"H": [row], "E": [{**row, "reference": ["u"]}]})

import pytest

from tools import compare_game_paired_inputs as paired


def test_count_gate_is_per_song_and_requires_precision_and_recall():
    reference = {"a": 27, "b": 27, "c": 22, "d": 17, "e": 13}
    assert paired.count_gate(reference, reference)["recall_necessary_pass"]
    missed = paired.count_gate(reference, {"a": 28, "b": 29, "c": 21, "d": 17, "e": 11})
    assert missed["maximum_one_to_one_matches"] == 103
    assert missed["recall_necessary_pass"] and missed["precision_necessary_pass"]
    sparse = paired.count_gate(reference, {"a": 26, "b": 25, "c": 17, "d": 14, "e": 10})
    assert sparse["maximum_one_to_one_matches"] == 92
    assert not sparse["recall_necessary_pass"]
    excess = paired.count_gate(reference, {"a": 31, "b": 31, "c": 25, "d": 20, "e": 15})
    assert excess["recall_necessary_pass"] and not excess["precision_necessary_pass"]
    with pytest.raises(ValueError, match="Song set"):
        paired.count_gate(reference, {"wrong": 106})


def test_requests_are_audio_only_and_limited_to_fixed_windows():
    audio = {"path": "pair.wav", "sha256": "a" * 64, "samples": 441000,
             "sample_rate": 44100}
    assert paired.request_for(audio, 8) == {
        "audio_path": "pair.wav", "audio_sha256": "a" * 64,
        "start_sec": 0, "end_sec": 8,
    }
    assert paired.request_for(audio, 10)["end_sec"] == 10
    with pytest.raises(ValueError, match="8/10"):
        paired.request_for(audio, 9)
    with pytest.raises(ValueError, match="frozen"):
        paired.request_for({**audio, "samples": 220500}, 10)


def test_launch_refuses_existing_job_before_any_subprocess(tmp_path, monkeypatch):
    from any2xsampa.local.common import write

    run = tmp_path / "run"
    run.mkdir()
    output = run / "jobs/smoke-8"
    output.mkdir(parents=True)
    jobs = [{"id": "smoke-8", "output": str(output)}] * 11
    write(run / "LAUNCH_REVIEW.json", {"jobs": jobs, "maximum_total_graph_calls": 121})
    monkeypatch.setattr(paired.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("Existing output must stop before subprocess"))
    with pytest.raises(ValueError, match="already started"):
        paired.launch(run)


def test_review_freeze_requires_passed_tests_and_frozen_budget(tmp_path):
    from any2xsampa.local.common import write

    run = tmp_path / "run"
    run.mkdir()
    write(run / "PRECHECK.json", {"pairs": [dict() for _ in range(5)]})
    write(run / "DIAGNOSIS.json", {})
    write(run / "JOBS.json", {"jobs": []})
    xml = run / "tests.xml"
    xml.write_text('<testsuites><testsuite tests="1" failures="0" errors="0"/></testsuites>', encoding="utf-8")
    with pytest.raises(ValueError, match="Request budget"):
        paired.freeze(run, xml)
    xml.write_text('<testsuites><testsuite tests="1" failures="1" errors="0"/></testsuites>', encoding="utf-8")
    with pytest.raises(ValueError, match="tests have not passed"):
        paired.freeze(run, xml)


def test_real_old_diagnosis_replays_raw_without_model_forward():
    from any2xsampa.local.common import read

    count = read(paired.COUNT)
    rows = paired.diagnose_old(count["rows"])
    assert len(rows) == 5
    assert sum(x["region_count"] for x in rows) == 106
    assert sum(x["rest_count"] for x in rows) == 14
    assert sum(x["predicted_note_count"] for x in rows) == 92
    assert all(len(x["boundary_counts_after_each_of_8_samples"]) == 8 for x in rows)


def test_real_pair_source_and_both_separation_record_schemas_are_bound():
    from any2xsampa.local.common import read

    rows = read(paired.COUNT)["rows"]
    verified = [paired.check_pair(paired.PAIRS / row["example_id"] / "pair.json", row)
                for row in rows]
    assert len(verified) == 5
    assert {item["song_id"] for item in verified} == {row["song_id"] for row in rows}
    for row in rows:
        with pytest.raises(ValueError, match="song/split"):
            paired.check_pair(paired.PAIRS / row["example_id"] / "pair.json",
                              {**row, "song_id": "test-song-leak"})

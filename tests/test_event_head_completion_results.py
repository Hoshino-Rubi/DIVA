from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from any2xsampa.training.limited_data_artifacts import write_json_new, read_json, sha256_file
from any2xsampa.training.phone_error_metrics import phone_metrics
from event_head_completion_results import resolve_sources, verify_ledger
from verify_event_head_completion import reused_evidence
from finish_event_head_completion import readiness, render

SONGS = ["freedom-flower", "half-moon-night-song", "like-you", "promise", "queen-road-east"]


def source_fixture(tmp_path):
    source = tmp_path / "source";run = tmp_path / "completion"
    auth = {"source_run": str(source), "freeze_sha256": "fixed",
            "allowed_jobs": ["E--promise", "E--queen-road-east"], "existing_completed_results": {}}
    spec = {"arms": ["H", "E"], "songs": SONGS}
    for arm in spec["arms"]:
        for song in SONGS:
            key = arm + "--" + song;new = key in auth["allowed_jobs"]
            folder = (run if new else source) / "screen" / arm / song
            folder.mkdir(parents=True);checkpoint = folder / "final.safetensors";checkpoint.write_bytes(key.encode())
            result = {"complete": True, "selected_epoch": 20, "seed": 17, "arm": arm, "held_song": song,
                      "freeze_sha256": "fixed", "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)}
            path = folder / "RESULT.json";write_json_new(path, result)
            if new:write_json_new(run / "commands" / (key + ".exit.json"), {"exit_code": 0, "elapsed_seconds": 1})
            else:auth["existing_completed_results"][key] = {"path": str(path), "sha256": sha256_file(path), "checkpoint_sha256": sha256_file(checkpoint)}
    return source, run, auth, spec


def test_resolve_exact_eight_old_plus_two_new_without_overwriting(tmp_path):
    source, run, auth, spec = source_fixture(tmp_path)
    original = {p: p.read_bytes() for p in source.rglob("RESULT.json")}
    rows = resolve_sources(run, auth, spec)
    assert len(rows) == 10 and sum(r["reused"] for r in rows) == 8
    assert original == {p: p.read_bytes() for p in original}
    (run / "screen/E/promise/RESULT.json").unlink()
    with pytest.raises(FileNotFoundError):resolve_sources(run, auth, spec)


def test_changed_old_result_is_rejected(tmp_path):
    source, run, auth, spec = source_fixture(tmp_path)
    p = source / "screen/H/promise/RESULT.json";p.write_bytes(b"changed")
    with pytest.raises(ValueError):resolve_sources(run, auth, spec)


def test_global_ledger_retains_both_old_failures(tmp_path):
    source, run, auth, spec = source_fixture(tmp_path)
    for arm in ("H", "E"):
        for song in SONGS:
            if arm == "E" and song == "queen-road-east":continue
            key = arm + "--" + song;failure = arm == "E" and song in ("promise", "freedom-flower")
            write_json_new(source / "commands" / (key + ".command.json"), {})
            write_json_new(source / "commands" / (key + ".exit.json"), {"exit_code": 73 if failure else 0, "elapsed_seconds": 1})
            write_json_new(source / "screen" / arm / song / "START.json", {"quality_training_start": True})
            if failure:write_json_new(source / "screen" / arm / song / "monitor/resource-failure.json",
                                     {"reason": "wall_clock_limit" if song == "freedom-flower" else "total_gpu_memory_limit"})
    for song, ordinal in (("promise", 10), ("queen-road-east", 11)):
        write_json_new(run / "commands" / ("E--" + song + ".command.json"), {"quality_start_ordinal": ordinal})
        write_json_new(run / "screen/E" / song / "START.json", {"quality_training_start": True, "quality_start_ordinal": ordinal})
    ledger = verify_ledger(run, auth)
    assert ledger["quality_starts"] == 11 and ledger["normal_quality_exits"] == 9
    assert ledger["in_training_resource_abort"] == 1
    assert ledger["post_training_timeout_recovered_without_retraining"] == 1


def test_codec_proof_reuse_requires_exact_file_hashes(tmp_path):
    evidence = tmp_path / "events.json";raw = tmp_path / "scores.safetensors"
    evidence.write_bytes(b"evidence");raw.write_bytes(b"scores")
    proof = {"checks": [{"arm": "E", "example_id": "one", "fixture_equal": True, "metric_equal": True}],
             "artifact_bindings": {str(evidence): sha256_file(evidence), str(raw): sha256_file(raw)}}
    assert reused_evidence({"example_id": "one"}, "E", evidence, raw, proof)
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError):reused_evidence({"example_id": "one"}, "E", evidence, raw, proof)


def test_new_failure_stops_postprocessing_without_retry(tmp_path):
    assert not readiness(tmp_path)["ready"]
    write_json_new(tmp_path / "commands/E--promise.exit.json", {"exit_code": 73})
    assert readiness(tmp_path)["blocked"]


def test_final_renderer_utf8_and_accounting(tmp_path):
    metric = phone_metrics([(["i", "u"], ["i"])])
    gate = {"passed": False, "failures": ["fixture_only"], "original": metric, "rendered": metric,
            "per_song": {"baseline": {s: metric for s in SONGS}, "current": {s: metric for s in SONGS}}}
    ledger = {"quality_starts": 11, "logged_epochs_including_failed_attempt": 214, "logged_optimizer_steps_including_failed_attempt": 4588}
    report = {"selected_arm": None, "ledger": ledger, "gates": {"H": gate, "E": gate},
        "effective_optimizer_updates": {"H": 2140, "E": 2140}, "historical_endpoint": {"original": metric, "rendered": metric},
        "note_phone_strata": [], "first14_logged_loss_maximum_difference": 0.0}
    write_json_new(tmp_path / "SCREEN_REPORT.json", report)
    write_json_new(tmp_path / "FINAL_VERIFICATION.json", {"passed": True, "regression": {"tests": 5, "skipped": 0}})
    for name in ("AUTHORIZATION.json", "RESOLVED_RESULTS.json", "CODE_REVIEW_training.json", "CODE_REVIEW_runtime_supplement.json", "CODE_REVIEW_postprocess.json", "CODE_REVIEW_replay.json", "collection/independent-replay/RESULT.json"):
        write_json_new(tmp_path / name, {"fixture": True})
    (tmp_path / "postprocess").mkdir();(tmp_path / "postprocess/regression.xml").write_text("<testsuites/>")
    render(tmp_path)
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "累计11次" in text and "追加授权" in text and "40份" in text
    status = read_json(tmp_path / "FINAL_STATUS.json")
    assert status["conditional_final_status"] == "not_run_screen_failed"
    assert status["ledger"]["quality_starts"] == 11

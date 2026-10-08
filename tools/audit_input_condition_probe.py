"""Independent CPU audit for the fixed x/v/h input probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.phone_error_metrics import phone_metrics

SONGS = {"freedom-flower", "half-moon-night-song", "like-you", "promise", "queen-road-east"}
CONDITIONS = {"x", "v", "h"}


def audit(run: Path) -> None:
    run = run.resolve()
    freeze = read_json(run / "FREEZE.json")
    if freeze["training_started"] or freeze["development_or_test_audio_opened"] or freeze["final_test_predictions_allowed"]:
        raise ValueError("input probe boundary was widened")
    conditions = read_json(run / "CONDITIONS.json")["rows"]
    requests = read_json(run / "INPUT_REQUESTS.json")["rows"]
    refs = read_json(run / "EVALUATION_REFERENCES.json")["rows"]
    cache = read_json(run / "CACHE.json")
    result = read_json(run / "RESULT.json")
    if len(conditions) != 60 or len(requests) != 60 or len(refs) != 20 or len(cache["entries"]) != 60:
        raise ValueError("fixed input-probe coverage changed")
    if {row["condition"] for row in conditions} != CONDITIONS or {row["song_id"] for row in refs} != SONGS:
        raise ValueError("condition or song coverage changed")
    if cache["labels_received"] or cache["all_parameters_frozen"] is not True:
        raise ValueError("feature cache provenance is invalid")
    verify_file(run / "INPUT_REQUESTS.json", cache["input_requests_sha256"])
    entries = {row["condition_id"]: row for row in cache["entries"]}
    for row in conditions:
        entry = entries[row["condition_id"]]
        if entry["audio_sha256"] != row["audio_sha256"] or entry["audio_role"] != row["audio_role"]:
            raise ValueError("condition/cache source binding mismatch")
        verify_file(Path(entry["path"]), entry["sha256"])
        verify_file(Path(row["audio_path"]), row["audio_sha256"])
    for row in read_json(run / "SOURCE_BINDINGS.json")["sources"].items():
        verify_file(Path(row[0]), row[1])
    if result["windows"] != 20 or result["reference_phones"] != 1235:
        raise ValueError("evaluation denominator changed")
    for condition in CONDITIONS:
        details = result["details"][condition]
        if len(details) != 20:
            raise ValueError("condition detail coverage changed")
        metric = phone_metrics([(row["reference"], row["prediction"]) for row in details])
        if metric != result["condition_metrics"][condition]:
            raise ValueError(f"condition metric mismatch: {condition}")
    if result["matched_x_identity"]["sequence_equal"] != 20 or not result["matched_x_identity"]["all_equal"]:
        raise ValueError("matched x replay changed")
    monitor_path = Path(result['resource_monitor_path']) if 'resource_monitor_path' in result else run / 'evaluation-monitor/resource-monitor.json'
    if 'resource_monitor_sha256' in result: verify_file(monitor_path, result['resource_monitor_sha256'])
    monitor = read_json(monitor_path)
    if not monitor["operation_succeeded"] or monitor["maximum_sampled_total_used_bytes"] > 14 * 1024**3:
        raise ValueError("input-probe evaluation resource monitor failed")
    audit_result = {
        "schema_version": "input_condition_probe_audit_v1",
        "run": str(run),
        "audited_files": {name: sha256_file(run / name) for name in ("FREEZE.json", "CONDITIONS.json", "CACHE.json", "RESULT.json")},
        "windows": 20,
        "condition_requests": 60,
        "reference_phones": 1235,
        "conditions": sorted(CONDITIONS),
        "matched_x_replay_windows": 20,
        "resource_monitor_succeeded": True,
        "development_opened": False,
        "final_test_opened": False,
        "training_started": False,
    }
    write_json_new(run / "INDEPENDENT_AUDIT.json", audit_result)
    print(json.dumps(audit_result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True); args = parser.parse_args(); audit(args.run)

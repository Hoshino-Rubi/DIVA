"""CPU audit for the V/MV screen manifest and resource-blocked jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new

SONGS = ("freedom-flower", "half-moon-night-song", "like-you", "promise", "queen-road-east")


def audit(run: Path) -> None:
    run = run.resolve()
    rows = read_json(run / "ROWS.json")["rows"]
    cache = read_json(run / "CACHE_BINDINGS.json")
    policy = read_json(run / "POLICY.json")
    freeze = read_json(run / "FREEZE.json")
    v2 = freeze.get('schema_version') == 'input_supervision_freeze_v2'
    if v2:
        from any2xsampa.training.input_supervision import load_manifest, policy as current_policy
        load_manifest(run)
    if len(rows) != 169 or sum(row["audio_kind"] == "original" for row in rows) != 40 or sum(row["audio_kind"] == "rendered" for row in rows) != 129:
        raise ValueError("V/MV manifest coverage changed")
    if cache["labels_received"] or cache["all_parameters_frozen"] is not True:
        raise ValueError("training cache provenance changed")
    expected_slots = current_policy()['slots'] if v2 else {"V": {"v": 4}, "MV": {"x": 2, "v": 2}}
    if policy["variants"] != ["V", "MV"] or policy["slots"] != expected_slots:
        raise ValueError("input-supervision policy changed")
    if freeze["development_or_final_test_opened"]:
        raise ValueError("freeze boundary changed")
    for entry in cache["entries"]:
        verify_file(Path(entry["path"]), entry["sha256"])
    jobs = []
    for variant in ("V", "MV"):
        for song in SONGS:
            job = run / "screen" / variant / song
            if not job.exists():
                continue
            result = job / "RESULT.json"
            failure = job / "monitor" / "resource-failure.json"
            jobs.append({
                "variant": variant,
                "held_song": song,
                "complete": result.exists() and read_json(result).get('complete') is True,
                "valid_for_selection": result.exists() and read_json(result).get('schema_version') == 'input_supervision_job_v2',
                "result_sha256": sha256_file(result) if result.exists() else None,
                "resource_failure": read_json(failure) if failure.exists() else None,
            })
    completed = [job for job in jobs if job["complete"]]
    blocked = [job for job in jobs if job["resource_failure"]]
    audit_result = {
        "schema_version": "input_supervision_audit_v1",
        "run": str(run),
        "rows": len(rows),
        "original_view_rows": 40,
        "rendered_rows": 129,
        "jobs_seen": len(jobs),
        "completed_jobs": len(completed),
        "resource_blocked_jobs": len(blocked),
        "jobs": jobs,
        "training_started": bool(jobs),
        "development_opened": False,
        "final_test_opened": False,
        "automatic_promotion": False,
        "screen_gate_available": len(completed) == 10 and all(j['valid_for_selection'] for j in completed),
        "budget": freeze.get('budget'),
    }
    write_json_new(run / "INDEPENDENT_AUDIT.json", audit_result)
    print(json.dumps(audit_result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True); args = parser.parse_args(); audit(args.run)

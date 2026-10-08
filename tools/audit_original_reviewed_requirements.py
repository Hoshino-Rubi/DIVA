"""Requirement-by-requirement audit for the response-gated continuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit implemented scope without promoting pending human work.")
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_root.resolve()
    shalong = read_json(run / "shalong-review-v2-final/STATUS.json")
    training = read_json(run / "training-candidate-review-v5/STATUS.json")
    proposal = read_json(run / "training-candidate-review-v5/candidate-proposals.json")
    preflight = read_json(run / "PREFLIGHT_STATUS.json")
    tests = read_json(run / "RUN_STATUS_FINAL10.json")
    requirements = [
        {"id": "segment_reference_v2", "status": "complete", "evidence": ["src/any2xsampa/training/segment_reference.py", "tests/test_segment_reference.py"]},
        {"id": "shalong_review_packet", "status": "waiting_human_confirmation" if shalong["confirmed_items"] == 0 else "partial", "evidence": ["shalong-review-v2-final/review-packet.json", "shalong-review-v2-final/index.html"]},
        {"id": "training_candidate_packet", "status": "waiting_human_confirmation" if training["candidate_count"] == 20 else "failed", "evidence": ["training-candidate-review-v5/candidate-proposals.json", "training-candidate-review-v5/index.html"], "candidate_count": training["candidate_count"], "song_counts": proposal["song_counts"]},
        {"id": "formal_original_evaluation", "status": "not_started_gate_pending", "evidence": ["tools/evaluate_segment_reference.py"], "confirmed_dev_windows": preflight["gate"]["confirmed_dev_windows"]},
        {"id": "conditional_original_adaptation", "status": "not_started_gate_pending", "evidence": ["src/any2xsampa/training/original_adaptation.py", "tools/train_original_adapter.py"], "confirmed_train_windows": preflight["gate"]["confirmed_train_windows"]},
        {"id": "test_sealing_and_preservation", "status": "complete", "evidence": ["DELIVERY_AUDIT.json", "PREFLIGHT_STATUS.json"], "test_predictions_opened": False, "source_wav_vsqx_modified": False},
        {"id": "regression", "status": "complete", "evidence": ["RUN_STATUS_FINAL10.json"], "passed": tests["tests"]["full_regression_passed"], "skipped": tests["tests"]["full_regression_skipped"]},
    ]
    result = {"schema_version": "original_reviewed_requirement_audit_v1", "audited_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(run), "requirements": requirements, "overall_goal_complete": False, "reason": "human responses required before formal evaluation and conditional training", "model_jobs_started": False, "final_test_predictions_opened": False}
    output = run / "REQUIREMENT_AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    digest = write_json_new(output, result)
    print(f"audit={output}; sha256={digest}; overall_goal_complete=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

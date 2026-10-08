"""Fail-closed preflight for the response-driven original-review pipeline.

This command never starts a model job.  It reports whether human responses are
available and which exact next phase is permitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, write_json_new  # noqa: E402
from any2xsampa.training.segment_reference import import_responses, validate_segment_reference  # noqa: E402


def _packet_state(packet_path: Path, response_path: Path | None, reference: dict, mapping_evidence: Path | None) -> dict:
    packet = read_json(packet_path)
    if response_path is None or not response_path.exists():
        return {"packet": str(packet_path.resolve()), "packet_sha256": sha256_file(packet_path), "status": "waiting_human_confirmation", "confirmed": 0, "pending": len(packet.get("items", [])), "rejected": 0}
    responses = read_json(response_path)
    view = reference
    if mapping_evidence is not None:
        from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
        view, _ = apply_original_mapping_overlay(reference, mapping_evidence)
    imported = import_responses(packet=packet, responses=responses, source_reference=view, source_reference_hash=sha256_file(Path(reference["bindings"]["triplets"]["path"])))
    validate_segment_reference(imported)
    return {"packet": str(packet_path.resolve()), "packet_sha256": sha256_file(packet_path), "responses": str(response_path.resolve()), "responses_sha256": sha256_file(response_path), "status": "imported", "confirmed": imported["human_confirmed_count"], "pending": sum(row["status"] == "pending" for row in imported["examples"]), "rejected": sum(row["status"] == "rejected" for row in imported["examples"]), "reference_manifest": imported}


def _discover_response(packet_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = []
    for path in packet_path.parent.glob("*.json"):
        if path.name in {"review-packet.json", "responses-template.json", "STATUS.json"} or path.name.endswith("-provenance.json"):
            continue
        try:
            if read_json(path).get("schema_version") == "original_segment_review_responses_v2":
                candidates.append(path)
        except (OSError, ValueError):
            continue
    if len(candidates) > 1:
        raise ValueError(f"multiple response files found beside {packet_path}; pass the response explicitly")
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Report the next legal phase without running inference/training.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--shalong-responses", type=Path)
    parser.add_argument("--training-responses", type=Path)
    parser.add_argument("--mapping-evidence", type=Path)
    parser.add_argument("--output", type=Path, help="new status path; defaults to a UTC-stamped file")
    args = parser.parse_args()
    run = args.run_root.resolve(); reference = read_json(args.reference)
    shalong_packet = run / "shalong-review-v2-final/review-packet.json"
    training_packet = run / "training-candidate-review-v8/review-packet.json"
    if not training_packet.exists():
        training_packet = run / "training-candidate-review-v6/review-packet.json"
    if not training_packet.exists():
        training_packet = run / "training-candidate-review-v5/review-packet.json"
    shalong = _packet_state(shalong_packet, _discover_response(shalong_packet, args.shalong_responses), reference, args.mapping_evidence)
    training = _packet_state(training_packet, _discover_response(training_packet, args.training_responses), reference, args.mapping_evidence)
    confirmed_train = training["confirmed"] if training["status"] == "imported" else 0
    confirmed_dev = shalong["confirmed"] if shalong["status"] == "imported" else 0
    next_phase = "waiting_human_confirmation"
    if confirmed_dev >= 3 and confirmed_train >= 20:
        next_phase = "assemble_shalong_then_build_audio_requests"
    elif confirmed_dev >= 3:
        next_phase = "waiting_training_candidate_confirmation"
    result = {"schema_version": "original_reviewed_preflight_v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "reference_path": str(args.reference.resolve()), "reference_sha256": sha256_file(args.reference), "shalong": {key: value for key, value in shalong.items() if key != "reference_manifest"}, "training": {key: value for key, value in training.items() if key != "reference_manifest"}, "gate": {"confirmed_train_windows": confirmed_train, "confirmed_dev_windows": confirmed_dev, "minimum_train_windows": 20, "minimum_dev_windows": 3, "next_phase": next_phase}, "model_jobs_started": False, "test_predictions_opened": False, "default_model_replaced": False}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or (run / f"PREFLIGHT_STATUS-{stamp}.json")).resolve()
    if output.exists(): raise FileExistsError(output)
    digest = write_json_new(output, result)
    print(f"next_phase={next_phase}; confirmed_train={confirmed_train}; confirmed_dev={confirmed_dev}; sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

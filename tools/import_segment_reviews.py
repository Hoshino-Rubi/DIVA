"""Import explicit human answers into an immutable segment-reference v2 file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new  # noqa: E402
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay  # noqa: E402
from any2xsampa.training.segment_reference import import_responses, validate_segment_reference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Import explicit clip-review answers without modifying sources.")
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mapping-evidence", type=Path, help="optional evidence-backed v4 mapping overlay; raw VSQX is never changed")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("output must be a new directory inside .state/runs")
    packet = read_json(args.packet)
    responses = read_json(args.responses)
    reference = read_json(args.reference)
    if args.mapping_evidence:
        reference, _ = apply_original_mapping_overlay(reference, args.mapping_evidence)
    result = import_responses(
        packet=packet,
        responses=responses,
        source_reference=reference,
        source_reference_hash=sha256_file(args.reference),
    )
    output.mkdir(parents=True)
    digest = write_json_new(output / "segment-reference-v2.json", result)
    status = {
        "stage": "human_reference_imported",
        "segment_reference_sha256": digest,
        "packet_sha256": packet.get("packet_sha256"),
        "confirmed_items": result["human_confirmed_count"],
        "rejected_items": sum(row["status"] == "rejected" for row in result["examples"]),
        "pending_items": sum(row["status"] == "pending" for row in result["examples"]),
        "test_predictions_opened": False,
        "historical_sources_modified": False,
    }
    write_json_new(output / "STATUS.json", status)
    print(f"confirmed={status['confirmed_items']}; rejected={status['rejected_items']}; pending={status['pending_items']}; sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

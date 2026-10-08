"""Merge independently reviewed v2 packets without changing their answers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, write_json_new  # noqa: E402
from any2xsampa.training.segment_reference import validate_segment_reference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge v2 reference packets with collision and source checks.")
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    values = [read_json(path) for path in args.input]
    for value in values:
        validate_segment_reference(value)
    source_hashes = {value["source_reference_manifest_sha256"] for value in values}
    if len(source_hashes) != 1:
        raise ValueError("all segment references must bind the same source manifest")
    rows = []
    seen = set()
    for value in values:
        for row in value["examples"]:
            if row["example_id"] in seen:
                raise ValueError(f"duplicate segment ID: {row['example_id']}")
            seen.add(row["example_id"]); rows.append(row)
    rows.sort(key=lambda row: (row["split"], row["song_id"], row["start_sec"], row["example_id"]))
    merged = {"schema_version": "original_segment_reference_v2", "source_reference_manifest_sha256": next(iter(source_hashes)), "packet_sha256": sha256_object([value["packet_sha256"] for value in values]), "examples": rows, "human_confirmed_count": sum(row["status"] == "human_confirmed" for row in rows), "test_predictions_opened": False, "historical_sources_modified": False}
    merged["manifest_hash"] = sha256_object(merged)
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    digest = write_json_new(output, merged)
    print(f"segments={len(rows)}; confirmed={merged['human_confirmed_count']}; sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build audio-only requests for human-confirmed v2 segments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.limited_data_reference import SPLITS  # noqa: E402
from any2xsampa.training.segment_reference import validate_segment_reference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline recognizer requests only for confirmed original segments.")
    parser.add_argument("--segment-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = read_json(args.segment_reference)
    validate_segment_reference(reference)
    confirmed = [row for row in reference["examples"] if row["status"] == "human_confirmed"]
    if not confirmed:
        raise ValueError("no human-confirmed segments; no inference request may be emitted")
    rows = []
    seen: set[str] = set()
    for row in confirmed:
        if row["split"] == "test" or row["example_id"] in seen:
            raise ValueError("sealed/duplicate confirmed segment")
        if row["split"] == "dev" and "edge_evidence" not in row:
            raise ValueError("Shalong edge answers must be assembled into one complete clip before inference")
        seen.add(row["example_id"])
        rows.append({
            "example_id": row["example_id"], "song_id": row["song_id"], "split": row["split"],
            "audio_role": "original", "audio_path": row["audio_path"], "audio_sha256": row["audio_sha256"],
            "start_sec": row["start_sec"], "end_sec": row["end_sec"],
        })
    request = {
        "schema_version": "phoneticxeus_audio_requests_v1", "purpose": "original-reviewed-v2",
        "source_dataset_hash": reference["source_reference_manifest_sha256"],
        "source_manifest_file_sha256": sha256_file(args.segment_reference),
        "split_sha256": sha256_object(SPLITS), "rows": rows,
        "contains_labels_or_score_context": False, "test_songs_opened": False,
        "segment_reference_sha256": sha256_file(args.segment_reference),
    }
    # Worker v1 deliberately accepts only its audio request schema. Keep the
    # provenance extension in a sidecar so the worker input stays unchanged.
    worker_request = {key: value for key, value in request.items() if key in {
        "schema_version", "purpose", "source_dataset_hash", "source_manifest_file_sha256", "split_sha256", "rows", "contains_labels_or_score_context"
    }}
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_json_new(output, worker_request)
    sidecar = output.with_name(output.stem + "-provenance.json")
    write_json_new(sidecar, {"schema_version": "original_review_request_provenance_v1", "request_sha256": digest,
                             "segment_reference_path": str(args.segment_reference.resolve()),
                             "segment_reference_sha256": sha256_file(args.segment_reference),
                             "confirmed_segment_ids": [row["example_id"] for row in confirmed],
                             "test_songs_opened": False})
    print(f"requests={len(rows)}; request_sha256={digest}; provenance={sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

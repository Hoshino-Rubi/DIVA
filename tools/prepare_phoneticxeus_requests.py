from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new  # noqa: E402
from any2xsampa.training.limited_data_reference import validate_dataset  # noqa: E402


AUDIO_KEYS = ("example_id", "song_id", "split", "audio_role", "audio_path", "audio_sha256", "start_sec", "end_sec")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strip labels and score context before launching the acoustic worker.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / ".state/runs"):
        parser.error("requests must remain under .state/runs")
    reference = read_json(args.reference)
    dataset = read_json(args.dataset)
    validate_dataset(dataset)
    original = [{key: row[key] for key in AUDIO_KEYS} for row in reference["original_clips"]]
    features = [
        {key: row[key] for key in AUDIO_KEYS}
        for row in dataset["examples"] if row["eligible"] and row["split"] in {"train", "dev"}
    ]
    base = next(row for row in original if row["song_id"] == "freedom-flower")
    for name, rows, source_hash in (
        ("smoke8", [{**base, "example_id": "freedom-flower-smoke8", "end_sec": base["start_sec"] + 8}], reference["manifest_hash"]),
        ("smoke20", [{**base, "example_id": "freedom-flower-smoke20", "end_sec": base["start_sec"] + 20}], reference["manifest_hash"]),
        ("original-baselines", original, reference["manifest_hash"]),
        ("rendered-features", features, dataset["dataset_hash"]),
    ):
        payload = {
            "schema_version": "phoneticxeus_audio_requests_v1", "purpose": name,
            "source_dataset_hash": source_hash,
            "source_manifest_file_sha256": sha256_file(args.dataset if name == "rendered-features" else args.reference),
            "split_sha256": dataset["split_sha256"],
            "rows": rows, "contains_labels_or_score_context": False,
        }
        path = args.output_dir / f"{name}.json"
        print(f"{name}: sha256={write_json_new(path, payload)}; rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the gate-bound manifest consumed by original residual adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.segment_reference import validate_segment_reference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze original training/evaluation inputs and report the start gate.")
    parser.add_argument("--segment-reference", required=True, type=Path)
    parser.add_argument("--original-feature-index", required=True, type=Path)
    parser.add_argument("--rendered-dataset", required=True, type=Path)
    parser.add_argument("--rendered-feature-index", required=True, type=Path)
    parser.add_argument("--seed43-checkpoint", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    segment = read_json(args.segment_reference); validate_segment_reference(segment)
    original_index = read_json(args.original_feature_index)
    rendered_dataset = read_json(args.rendered_dataset); rendered_index = read_json(args.rendered_feature_index)
    if original_index.get("purpose") != "original-reviewed-v2":
        raise ValueError("original feature index must be produced from original-reviewed-v2 requests")
    if rendered_index.get("purpose") != "rendered-features":
        raise ValueError("rendered protection cache must remain rendered-features")
    confirmed = [row for row in segment["examples"] if row["status"] == "human_confirmed"]
    train = [row for row in confirmed if row["split"] == "train"]
    dev = [row for row in confirmed if row["split"] == "dev"]
    if any("edge_evidence" not in row for row in dev):
        raise ValueError("development edge answers must be assembled into complete clips")
    overlaps = []
    for index, left in enumerate(train):
        for right in train[index + 1:]:
            if left["song_id"] == right["song_id"] and left["audio_sha256"] == right["audio_sha256"] and left["start_sec"] < right["end_sec"] and right["start_sec"] < left["end_sec"]:
                overlaps.append([left["example_id"], right["example_id"]])
    counts = {"confirmed_train_windows": len(train), "confirmed_train_songs": len({row["song_id"] for row in train}), "confirmed_dev_windows": len(dev), "confirmed_dev_reference_phones": sum(len(row["phones"]) for row in dev), "overlaps": overlaps}
    gate = {"passed": counts["confirmed_train_windows"] >= 20 and counts["confirmed_train_songs"] >= 3 and counts["confirmed_dev_windows"] >= 3 and counts["confirmed_dev_reference_phones"] >= 100 and not overlaps, "counts": counts, "minimums": {"train_windows": 20, "train_songs": 3, "dev_windows": 3, "dev_reference_phones": 100, "overlaps": 0}}
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "original_training_manifest_v1", "reference_mode": "confirmed_original_segments",
        "segment_reference": {"path": str(args.segment_reference.resolve()), "sha256": sha256_file(args.segment_reference)},
        "original_feature_index": {"path": str(args.original_feature_index.resolve()), "sha256": sha256_file(args.original_feature_index)},
        "rendered_dataset": {"path": str(args.rendered_dataset.resolve()), "sha256": sha256_file(args.rendered_dataset)},
        "rendered_feature_index": {"path": str(args.rendered_feature_index.resolve()), "sha256": sha256_file(args.rendered_feature_index)},
        "seed43_checkpoint": {"path": str(args.seed43_checkpoint.resolve()), "sha256": sha256_file(args.seed43_checkpoint)},
        "vocabulary": {"path": str(args.vocabulary.resolve()), "sha256": sha256_file(args.vocabulary)},
        "gate": gate, "test_songs_opened": False, "automatic_model_promotion": False,
    }
    manifest["manifest_hash"] = sha256_object(manifest)
    digest = write_json_new(output, manifest)
    print(f"gate_passed={gate['passed']}; train={len(train)}; songs={len({row['song_id'] for row in train})}; dev={len(dev)}; dev_refs={sum(len(row['phones']) for row in dev)}; sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

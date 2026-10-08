from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.audio_section_correspondence import MATCH_POLICY, match_song  # noqa: E402
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, write_json_new  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Propose original/rendered section correspondence after audio-only inference.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--baseline-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("use a new output directory under .state/runs")
    reference = read_json(args.reference)
    index = read_json(args.baseline_index)
    if index["purpose"] != "original-baselines" or index["source_dataset_hash"] != reference["manifest_hash"]:
        raise ValueError("a matching completed audio-only baseline must precede correspondence")
    if {row["example_id"] for row in index["entries"]} != {row["example_id"] for row in reference["original_clips"]}:
        raise ValueError("audio-only baseline does not cover the locked clips")
    output.mkdir(parents=True)
    write_json_new(output / "policy.json", MATCH_POLICY)
    rows = []
    for song in reference["songs"]:
        if song["split"] == "test":
            continue
        clips = [row for row in reference["original_clips"] if row["song_id"] == song["song_id"]]
        report = match_song(song, clips, output)
        path = output / f"{song['song_id']}.json"
        rows.append({"song_id": song["song_id"], "path": str(path), "sha256": write_json_new(path, report)})
        print(f"matched_song={song['song_id']}; clips={len(clips)}; elapsed_seconds={report['elapsed_seconds']:.2f}", flush=True)
    write_json_new(output / "index.json", {
        "schema_version": "limited_data_correspondence_index_v1",
        "reference_manifest_sha256": sha256_file(args.reference),
        "baseline_index_sha256": sha256_file(args.baseline_index),
        "policy_sha256": sha256_object(MATCH_POLICY),
        "songs": rows, "test_songs_opened": False,
        "human_confirmation_required": True, "original_accuracy_available": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

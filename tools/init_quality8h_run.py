"""Freeze the approved experiment before any new model observations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.limited_data_reference import SONG_SPLIT, SPLITS, training_gate

BASE = ROOT / ".state/runs/limited-data-20260906T050251"


def initialize(destination: Path, started: str) -> dict:
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT / ".state/runs"):
        raise ValueError("run must remain under .state/runs")
    parsed = datetime.fromisoformat(started)
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise ValueError("execution start must be an actual timezone-aware past timestamp")
    if (destination / "execution-start.json").exists():
        raise FileExistsError("experiment already initialized")
    policy_path = ROOT / "src/any2xsampa/config/quality8h_v1.json"
    policy = read_json(policy_path)
    reference_path = BASE / "reference/reference-manifest.json"
    verify_file(reference_path, "120b45d2970e98d0bccba61d827004b72d50e531b739cfc52ac9f19bc10751c5")
    reference = read_json(reference_path)
    dataset_path = BASE / "reference/rendered-proxy-dataset.json"
    dataset = read_json(dataset_path)
    gate = training_gate(dataset)
    if not gate["passed"]:
        raise ValueError("original training gate fails")
    counts = {split: sum(r["eligible"] and r["split"] == split for r in dataset["examples"]) for split in ("train", "dev")}
    if counts != {"train": 129, "dev": 25}:
        raise ValueError("fixed feature dataset size changed")
    sources = []
    for song in reference["songs"]:
        for role in ("original_audio", "rendered_audio", "human_vsqx"):
            source = song[role]
            verify_file(Path(source["path"]), source["sha256"])
            sources.append({"song_id": song["song_id"], "split": song["split"], "role": role, "path": source["path"], "sha256": source["sha256"]})
    artifacts = [reference_path, dataset_path, BASE / "rendered-features/index.json", BASE / "execution-report.json"]
    artifacts += [BASE / f"training/seed{seed}/adapter.safetensors" for seed in (17, 29, 43)]
    baseline = {str(p.relative_to(ROOT)): sha256_file(p) for p in artifacts}
    auxiliary = {}
    for role in ("asr", "timestamps"):
        spec = policy["models"][role]
        model_root = ROOT / spec["root"]
        verify_file(model_root / "model.safetensors", spec["weight_sha256"])
        auxiliary[role] = {"model": spec, "files": {p.name: sha256_file(p) for p in sorted(model_root.iterdir()) if p.is_file()}}
    by_song = {s["song_id"]: s for s in reference["songs"]}
    contexts = []
    for clip in reference["original_clips"]:
        if clip["split"] == "test":
            continue
        if SONG_SPLIT[clip["song_id"]] != clip["split"]:
            raise ValueError("song split mismatch")
        audio = by_song[clip["song_id"]]["original_audio"]
        contexts.append({"context_id": clip["example_id"] + "-context", "example_id": clip["example_id"],
            "song_id": clip["song_id"], "split": clip["split"], "audio_path": audio["path"], "audio_sha256": audio["sha256"],
            "start_sec": max(0.0, clip["start_sec"] - 6.0), "end_sec": min(audio["duration_sec"], clip["end_sec"] + 6.0),
            "old_start_sec": clip["start_sec"], "old_end_sec": clip["end_sec"], "contains_labels_or_score_context": False})
    if len(contexts) != 18:
        raise ValueError("requires exactly eighteen train/development contexts")
    policy_hash = write_json_new(destination / "policy.json", policy)
    bindings_hash = write_json_new(destination / "preservation-before.json", {"sources": sources, "artifacts": baseline, "auxiliary_models": auxiliary, "test_prediction_content_read": False})
    context_hash = write_json_new(destination / "contexts.json", {"schema_version": "quality8h_contexts_v1", "policy_sha256": policy_hash, "reference_sha256": sha256_file(reference_path), "split": SPLITS, "contexts": contexts, "model_outputs_used_to_select": False})
    status = {"schema_version": "quality8h_execution_start_v1", "started_at": parsed.isoformat(), "initialized_at_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_execution_seconds": 28800, "policy_sha256": policy_hash, "contexts_sha256": context_hash, "preservation_sha256": bindings_hash,
        "source_policy_sha256": sha256_file(policy_path), "dataset_counts": counts, "training_gate": gate,
        "status": "started_not_completed", "test_predictions_opened": False, "default_changed": False}
    write_json_new(destination / "execution-start.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--started-at", required=True)
    args = parser.parse_args()
    print(initialize(args.run_root, args.started_at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

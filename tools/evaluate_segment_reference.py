"""Evaluate frozen baseline and rendered-domain adapters on confirmed v2 segments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256, ctc_collapse  # noqa: E402
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.phone_error_metrics import phone_metrics  # noqa: E402
from any2xsampa.training.phoneticxeus_adaptation import make_residual_adapter  # noqa: E402
from any2xsampa.training.segment_reference import validate_segment_reference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate confirmed segment-reference v2 rows without selecting a checkpoint.")
    parser.add_argument("--segment-reference", required=True, type=Path)
    parser.add_argument("--feature-index", required=True, type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    segment = read_json(args.segment_reference); validate_segment_reference(segment)
    confirmed = [row for row in segment["examples"] if row["status"] == "human_confirmed"]
    if not confirmed:
        raise ValueError("no confirmed rows are available for formal evaluation")
    if any(row["split"] == "test" for row in confirmed):
        raise ValueError("test predictions cannot enter evaluation")
    if any(row["split"] == "dev" and "edge_evidence" not in row for row in confirmed):
        raise ValueError("development edge answers must be assembled before formal evaluation")
    index = read_json(args.feature_index); verify_file(args.feature_index, sha256_file(args.feature_index))
    if index.get("purpose") != "original-reviewed-v2":
        raise ValueError("feature index is not an original-reviewed-v2 index")
    if index.get("test_songs_loaded") is True or index.get("labels_or_score_received") is True or index.get("no_labels_or_score_received") is not True:
        raise ValueError("original evaluation evidence must be audio-only and test-sealed")
    if index.get("source_dataset_hash") != segment["source_reference_manifest_sha256"]:
        raise ValueError("feature index is bound to a different reference manifest")
    model = index.get("model_metadata", {})
    if (model.get("model_id"), model.get("revision"), model.get("weights_sha256")) != (MODEL_ID, REVISION, WEIGHT_SHA256):
        raise ValueError("feature index uses an unauthorized PhoneticXeus snapshot")
    from safetensors.torch import load_file
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires explicit CUDA")
    vocabulary = read_json(args.vocabulary); labels = [""] * len(vocabulary)
    for label, value in vocabulary.items(): labels[value] = label
    head_path = Path(index["head"]["path"])
    if not head_path.resolve().is_relative_to(args.feature_index.parent.resolve()):
        raise ValueError("feature head must remain inside its evidence directory")
    verify_file(head_path, index["head"]["sha256"])
    head_state = load_file(str(head_path), device="cpu")
    head = torch.nn.Linear(1024, 428, device="cuda"); head.load_state_dict(head_state, strict=True); head.requires_grad_(False).eval()
    entries = {row["example_id"]: row for row in index["entries"]}
    pairs = {"baseline": []}; details = {"baseline": []}
    reports = {}
    for seed in (17, 29, 43):
        report_path = args.training_root / f"seed{seed}" / "training-report.json"
        report = read_json(report_path); reports[str(seed)] = {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)}
        checkpoint = report["checkpoints"]["best_admissible"]
        if checkpoint is None:
            raise ValueError(f"seed {seed} has no admissible checkpoint")
        path = Path(checkpoint["path"]); verify_file(path, checkpoint["sha256"])
        adapter = make_residual_adapter(torch); adapter.load_state_dict(load_file(str(path), device="cpu"), strict=True); adapter.eval().requires_grad_(False).to("cuda")
        pairs[f"seed{seed}"] = []; details[f"seed{seed}"] = []
    adapters = {}
    for seed in (17, 29, 43):
        report = read_json(args.training_root / f"seed{seed}" / "training-report.json")
        path = Path(report["checkpoints"]["best_admissible"]["path"]); adapter = make_residual_adapter(torch); adapter.load_state_dict(load_file(str(path), device="cpu"), strict=True); adapters[seed] = adapter.eval().requires_grad_(False).to("cuda")
    with torch.no_grad():
        for row in confirmed:
            entry = entries.get(row["example_id"])
            if entry is None or entry.get("test_outputs_sealed") is not False or entry.get("audio_role") not in {"original", "original_resampled_mix", "original_separated_vocal"} or entry["audio_sha256"] != row["audio_sha256"] or entry["start_sec"] != row["start_sec"] or entry["end_sec"] != row["end_sec"]:
                raise ValueError("feature evidence does not bind the confirmed segment")
            if entry.get("vocabulary_sha256") != sha256_file(args.vocabulary):
                raise ValueError("feature evidence vocabulary differs from evaluation vocabulary")
            evidence = Path(entry["evidence_path"])
            if not evidence.resolve().is_relative_to(args.feature_index.parent.resolve()):
                raise ValueError("feature evidence must remain inside its evidence directory")
            verify_file(evidence, entry["evidence_sha256"]); values = load_file(str(evidence), device="cpu")
            features, logits = values["encoder_features"], values["frame_logits"]
            reference = [phone["phone"] for phone in row["phones"]]
            baseline_prediction = [labels[index] for index in ctc_collapse(logits.argmax(dim=-1).tolist())]
            pairs["baseline"].append((reference, baseline_prediction)); details["baseline"].append({"example_id": row["example_id"], "prediction": baseline_prediction, "song_id": row["song_id"]})
            cuda_features = features.to("cuda")
            for seed, adapter in adapters.items():
                prediction = [labels[index] for index in ctc_collapse(head(adapter(cuda_features)).argmax(dim=-1).cpu().tolist())]
                pairs[f"seed{seed}"].append((reference, prediction)); details[f"seed{seed}"].append({"example_id": row["example_id"], "prediction": prediction, "song_id": row["song_id"]})
    metrics = {name: phone_metrics(value) for name, value in pairs.items()}
    result = {"schema_version": "original_segment_evaluation_v2", "mode": "confirmed_original_segments", "segment_reference_path": str(args.segment_reference.resolve()), "segment_reference_sha256": sha256_file(args.segment_reference), "feature_index_path": str(args.feature_index.resolve()), "feature_index_sha256": sha256_file(args.feature_index), "metrics": metrics, "details": details, "training_reports": reports, "confirmed_windows": len(confirmed), "test_predictions_opened": False, "checkpoint_selection_performed": False, "default_model_replaced": False, "original_accuracy_claim_allowed": True}
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    write_json_new(output, result)
    print("; ".join(f"{name}_per={value['per']:.6f}" for name, value in metrics.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

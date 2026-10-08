"""Replay one previously frozen adapter on cached Shalong evidence, offline."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.reviewed_audibility_reference import validate_reviewed_reference, build_reviewed_reference
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256
from any2xsampa.training.phoneticxeus_adaptation import make_residual_adapter

QUALITY = ROOT / ".state/runs/quality-first8h-20260907T010738"
RAW = ROOT / ".state/runs/limited-data-20260906T050251/original-baselines/index.json"


def validate_entry(row: dict, entry: dict, condition: str, pair: dict | None = None) -> None:
    if row["song_id"] != "shalong" or row["split"] != "dev" or entry["song_id"] != "shalong" or entry["split"] != "dev":
        raise ValueError("only the frozen Shalong dev subset may be replayed")
    if entry.get("test_outputs_sealed") is not False:
        raise ValueError("sealed evidence")
    if condition == "historical_raw":
        expected = (row["example_id"], "original", row["audio_sha256"], row["start_sec"], row["end_sec"])
    else:
        if condition not in {"mixed", "vocal"} or pair is None:
            raise ValueError("unknown paired condition")
        context = pair["context"]
        if (context["example_id"], context["audio_sha256"], context["old_start_sec"], context["old_end_sec"]) != (
            row["example_id"], row["audio_sha256"], row["start_sec"], row["end_sec"]):
            raise ValueError("paired source clock or audio mismatch")
        role = "original_resampled_mix" if condition == "mixed" else "original_separated_vocal"
        expected = (row["example_id"] + "-paired-" + condition, role, pair["audio"][condition]["sha256"], 0, row["end_sec"] - row["start_sec"])
    actual = tuple(entry[k] for k in ("example_id", "audio_role", "audio_sha256", "start_sec", "end_sec"))
    if actual != expected:
        raise ValueError("evidence role, audio hash or window differs")


def score(reference: dict, predictions: dict) -> dict:
    validate_reviewed_reference(reference)
    rows = [r for r in reference["examples"] if r["status"] == "human_confirmed"]
    if set(predictions) != {r["example_id"] for r in rows}:
        raise ValueError("prediction/reference window mismatch")
    metrics, details = {}, []
    for condition in ("historical_raw", "mixed", "vocal"):
        metrics[condition] = {}
        for system in ("baseline", "seed43"):
            pairs = []
            for row in rows:
                target = [p["phone"] for p in row["phones"]]
                prediction = predictions[row["example_id"]][condition][system]
                if not isinstance(prediction, list) or any(not isinstance(p, str) or not p for p in prediction):
                    raise ValueError("invalid decoded phones")
                pairs.append((target, prediction))
                details.append({"example_id": row["example_id"], "song_id": row["song_id"],
                    "condition": condition, "system": system, "metrics": phone_metrics([(target, prediction)])})
            metrics[condition][system] = phone_metrics(pairs)
    return {"schema_version": "reviewed_audibility_evaluation_v1",
            "reference_manifest_hash": reference["manifest_hash"], "coverage": reference["coverage"],
            "metrics": metrics, "details": details, "predictions": predictions,
            "reference_nature": "human_reviewed_edges_with_inherited_calibrated_source_ipa",
            "scope": "three_reviewed_original_dev_clips_not_whole_song_accuracy",
            "same_denominator_for_all_conditions": True, "checkpoint_selection_performed": False,
            "test_predictions_opened": False, "training_started": False, "default_model_replaced": False,
            "partial_phones_counted_once": True, "ctc_timing_is_not_duration_accuracy": True}


def replay(reference_path: Path, output: Path) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline, configure_cuda, check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    enforce_offline(output / "runtime-cache")
    from safetensors.torch import load_file
    from evaluate_quality8h_pairs import predictions as decode_cached

    started = time.monotonic()
    reference = read_json(reference_path); validate_reviewed_reference(reference)
    frozen_inputs = read_json(reference_path.parent / "INPUTS.json")
    verify_file(reference_path, frozen_inputs["reference_sha256"])
    for name in ("packet", "answers"):
        verify_file(Path(frozen_inputs[name]["path"]), frozen_inputs[name]["sha256"])
    from import_original_edge_review import load_verified_packet
    packet, _ = load_verified_packet(Path(frozen_inputs["packet"]["path"]))
    if build_reviewed_reference(packet, read_json(Path(frozen_inputs["answers"]["path"]))) != reference:
        raise ValueError("reviewed reference does not reproduce from original human answers")
    rows = [r for r in reference["examples"] if r["status"] == "human_confirmed"]
    if len(rows) != 3:
        raise ValueError("this bounded replay requires the three frozen dev clips")
    sources = {}
    def bound(name: str, path: Path, expected: str | None = None) -> dict:
        digest = verify_file(path, expected) if expected else sha256_file(path)
        sources[name] = {"path": str(path), "sha256": digest}
        return read_json(path)
    old_inputs = bound("previous_inputs", QUALITY / "paired-diagnostics-v1/inputs.json")
    historical = bound("historical_index", RAW, old_inputs["historical_index_sha256"])
    paired = bound("paired_index", QUALITY / "paired-phonetic-v1/index.json", old_inputs["paired_index_sha256"])
    freeze = bound("candidate_freeze", QUALITY / "training-summary-v1/candidate-freeze.json", old_inputs["candidate_freeze_sha256"])
    previous = bound("previous_predictions", QUALITY / "paired-diagnostics-v1/diagnostics.json")
    bound("new_reference", reference_path)
    if freeze["selected_seed"] != 43 or freeze["selection_source"] != "rendered_development_only" or freeze["original_evaluation_used_for_selection"]:
        raise ValueError("Seed43 must already be frozen independently of these references")
    if historical.get("no_labels_or_score_received") is not True or paired.get("labels_or_score_received") is not False or paired.get("test_songs_loaded") is not False:
        raise ValueError("cached evidence must be audio-only and test-sealed")
    for index in (historical, paired):
        model = index["model_metadata"]
        if (model["model_id"], model["revision"], model["weights_sha256"]) != (MODEL_ID, REVISION, WEIGHT_SHA256):
            raise ValueError("cached features belong to a different model")
    paired_sources = bound("paired_inputs", Path(paired["paired_inputs_path"]), paired["paired_inputs_sha256"])
    contexts = {p["context"]["example_id"]: p for p in paired_sources["pairs"]}
    raw_entries = {r["example_id"]: r for r in historical["entries"] if r["song_id"] == "shalong"}
    pair_entries = {(r["original_example_id"], r["condition"]): r for r in paired["entries"] if r["song_id"] == "shalong"}
    if historical["head"]["sha256"] != paired["head"]["sha256"]:
        raise ValueError("cached raw and paired heads differ")
    head_spec = historical["head"]; checkpoint = freeze["checkpoint"]
    verify_file(Path(head_spec["path"]), head_spec["sha256"])
    verify_file(Path(checkpoint["path"]), checkpoint["sha256"])
    sources["head"] = head_spec; sources["checkpoint"] = {k: checkpoint[k] for k in ("path", "sha256")}
    sources["packet"] = frozen_inputs["packet"]
    sources["answers"] = frozen_inputs["answers"]
    vocab_spec = packet["source_bindings"]["vocabulary"]
    vocab = bound("vocabulary", Path(vocab_spec["path"]), vocab_spec["sha256"])
    labels = [p for p, _ in sorted(vocab.items(), key=lambda x: x[1])]
    if len(labels) != 428 or sorted(vocab.values()) != list(range(428)):
        raise ValueError("full 428-label vocabulary required")
    with ProcessResourceMonitor(output / "resource", timeout_seconds=600):
        torch = configure_cuda()
        head = torch.nn.Linear(1024, 428).eval().requires_grad_(False)
        head.load_state_dict(load_file(head_spec["path"], device="cpu"), strict=True); head.to("cuda")
        adapter = make_residual_adapter(torch).eval().requires_grad_(False)
        adapter.load_state_dict(load_file(checkpoint["path"], device="cpu"), strict=True); adapter.to("cuda")
        predictions, checks = {}, []
        for row in rows:
            eid = row["example_id"]; predictions[eid] = {}
            verify_file(Path(row["audio_path"]), row["audio_sha256"])
            for condition in ("historical_raw", "mixed", "vocal"):
                entry = raw_entries[eid] if condition == "historical_raw" else pair_entries[(eid, condition)]
                validate_entry(row, entry, condition, contexts[eid])
                if entry["vocabulary_sha256"] != vocab_spec["sha256"]:
                    raise ValueError("entry vocabulary mismatch")
                if condition != "historical_raw":
                    spec = contexts[eid]["audio"][condition]
                    verify_file(Path(spec["path"]), spec["sha256"])
                decoded = decode_cached(torch, head, {43: adapter}, labels, entry)
                if any(decoded[s] != previous["predictions"][eid][condition][s] for s in ("baseline", "seed43")):
                    raise ValueError("fresh cached-feature replay differs from frozen prediction")
                predictions[eid][condition] = decoded
                checks.append({"example_id": eid, "condition": condition,
                    "evidence": {"path": entry["evidence_path"], "sha256": entry["evidence_sha256"]},
                    "prediction_identical_to_previous": True, "resources": check_gpu_budget(torch)})
                print(f"replayed={eid}/{condition}", flush=True)
        result = score(reference, predictions)
        result.update(sources=sources, checks=checks, elapsed_seconds=time.monotonic() - started,
                      backbone_rerun=False, fp32=True, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      new_training=False)
        for spec in sources.values():
            verify_file(Path(spec["path"]), spec["sha256"])
        write_json_new(output / "evaluation.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / ".state/runs"):
        raise ValueError("output must be a new run directory")
    replay(args.reference.resolve(), output)


if __name__ == "__main__":
    main()

"""Independently cold-replay all six saved development checkpoint roles."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer, ctc_collapse, emission_payload
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.limited_data_reference import load_vocabulary
from any2xsampa.training.phone_error_metrics import development_gate, phone_metrics
from any2xsampa.training.phoneticxeus_adaptation import _load_cache, make_residual_adapter
from any2xsampa.training.protected_adaptation import _validate_inputs


def replay_checkpoint(torch, head, rows, labels, vocabulary, vocabulary_path: Path, index, checkpoint, destination: Path, seed: int) -> dict:
    from safetensors.torch import load_file, save_file

    path = Path(checkpoint["path"])
    verify_file(path, checkpoint["sha256"])
    verify_file(Path(checkpoint["evaluation_path"]), checkpoint["evaluation_sha256"])
    expected = read_json(Path(checkpoint["evaluation_path"]))
    by_id = {r["example_id"]: r for r in expected["examples"]}
    entries = {e["example_id"]: e for e in index["entries"]}
    adapter = make_residual_adapter(torch)
    adapter.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    adapter.eval().requires_grad_(False).to("cuda")
    pairs, details = [], []
    for row in rows:
        entry = entries[row["example_id"]]
        verify_file(Path(entry["posterior_path"]), entry["posterior_sha256"])
        original = read_json(Path(entry["posterior_path"]))
        with torch.inference_mode():
            logits = head(adapter(row["features"].to("cuda"))).float().cpu().contiguous()
        if not torch.isfinite(logits).all():
            raise ValueError("replayed logits are non-finite")
        ids = ctc_collapse(logits.numpy().argmax(-1).tolist())
        prediction = [labels[i] for i in ids]
        reference = [labels[i] for i in row["model_target_ids"]]
        if prediction != by_id[row["example_id"]]["prediction"] or reference != by_id[row["example_id"]]["reference"]:
            raise ValueError("cold checkpoint replay differs from its recorded development sequence")
        pairs.append((reference, prediction))
        folder = destination / row["example_id"]
        folder.mkdir(parents=True)
        tensors_path = folder / "logits.safetensors"
        save_file({"frame_logits": logits}, str(tensors_path))
        vocab_path = folder / "vocabulary.json"
        # Copy the exact pinned vocabulary bytes; JSON rewriting would change
        # its identity even if the parsed labels happened to agree.
        # Earlier immutable evidence predates adjacent vocabulary copies.
        # The source dataset still binds the exact canonical vocabulary file.
        source_vocab = vocabulary_path
        verify_file(source_vocab, entry["vocabulary_sha256"])
        with vocab_path.open("xb") as stream:
            stream.write(source_vocab.read_bytes())
        verify_file(vocab_path, entry["vocabulary_sha256"])
        producer = original["producer"]
        payload = emission_payload(logits.numpy(), vocabulary, input_audio=producer["input_audio"], crop=producer["crop"],
            model_metadata=producer["model_metadata"], geometry=original["decoding"]["geometry"],
            runtime={"device": "cuda:0", "verification_only": True, "frozen_feature_replay": True, "backbone_loaded": False})
        payload["producer"]["adaptation"] = {"seed": seed, "checkpoint_sha256": checkpoint["sha256"],
            "feature_evidence_sha256": entry["evidence_sha256"], "qualified_recognizer": False, "automatic_model_promotion": False}
        payload["raw_evidence"] = {"file": tensors_path.name, "sha256": sha256_file(tensors_path),
            "logit_shape": list(logits.shape), "dtype": "float32", "vocabulary_file": vocab_path.name,
            "vocabulary_sha256": entry["vocabulary_sha256"]}
        payload["test_outputs_sealed"] = False
        posterior = folder / "posterior.json"
        digest = write_json_new(posterior, payload)
        actual = PhoneticXeusPosteriorRecognizer(posterior).recognize(Path(row["audio_path"]))
        fixture = PhoneticXeusPosteriorRecognizer(posterior, fixture=True).recognize(Path(row["audio_path"]))
        signature = lambda values: [(p.time_range, p.candidate_phones, p.confidence) for p in values]
        if signature(actual) != signature(fixture):
            raise ValueError("real/fixture checkpoint posterior replay mismatch")
        details.append({"example_id": row["example_id"], "posterior_sha256": digest, "raw_logits_sha256": payload["raw_evidence"]["sha256"],
                        "real_fixture_identical": True, "prediction": prediction})
    measured = phone_metrics(pairs)
    if measured != checkpoint["metrics"] or measured != expected["metrics"]:
        raise ValueError("independently recomputed checkpoint metrics differ")
    verify_file(path, checkpoint["sha256"])
    return {"checkpoint_sha256": checkpoint["sha256"], "metrics": measured, "examples": details,
            "cold_sequences_identical": True, "all_real_fixture_posteriors_identical": True}


def verify(run: Path, destination: Path) -> dict:
    from any2xsampa.adapters.phoneticxeus_runtime import configure_cuda, enforce_offline
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor

    policy, dataset_path, index_path = _validate_inputs(run, 17, ROOT)
    enforce_offline(destination / "runtime-cache")
    torch = configure_cuda()
    started = time.monotonic()
    with ProcessResourceMonitor(destination, timeout_seconds=1800):
        dataset = read_json(dataset_path)
        rows, head, index = _load_cache(index_path, dataset, torch)
        head = head.eval().requires_grad_(False).to("cuda")
        dev = [r for r in rows if r["split"] == "dev"]
        vocabulary_path = Path(dataset["bindings"]["vocabulary"]["path"])
        vocabulary = load_vocabulary(vocabulary_path)
        labels = [label for label, _ in sorted(vocabulary.items(), key=lambda pair: pair[1])]
        results, candidates, baseline = [], [], None
        for seed in (17, 29, 43):
            report_path = run / "training-v2" / f"seed{seed}/training-report.json"
            report = read_json(report_path)
            if report["seed"] != seed or report["policy_sha256"] != sha256_file(run / "policy.json"):
                raise ValueError("checkpoint report policy/seed mismatch")
            if baseline is not None and baseline != report["baseline_dev"]:
                raise ValueError("seeds have different baselines")
            baseline = report["baseline_dev"]
            for role, checkpoint in report["checkpoints"].items():
                if checkpoint is None:
                    results.append({"seed": seed, "role": role, "checkpoint": None})
                    continue
                result = replay_checkpoint(torch, head, dev, labels, vocabulary, vocabulary_path, index, checkpoint, destination / f"seed{seed}" / role, seed)
                results.append({"seed": seed, "role": role, **result, "training_report_sha256": sha256_file(report_path)})
                if role == "best_admissible":
                    candidates.append({"seed": seed, "metrics": result["metrics"]})
                print(f"verified_seed={seed}; role={role}; per={result['metrics']['per']:.6f}; windows={len(result['examples'])}", flush=True)
        gate = development_gate(baseline, candidates)
        output = {"schema_version": "quality8h_checkpoint_verification_v1", "results": results,
            "development_gate": gate, "baseline_dev": baseline, "policy_sha256": sha256_file(run / "policy.json"),
            "selection_domain": "rendered_development_only", "test_songs_loaded": False,
            "original_accuracy_claim_allowed": False, "default_changed": False, "elapsed_seconds": time.monotonic() - started}
        write_json_new(destination / "verification.json", output)
        return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, destination = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not destination.is_relative_to(run) or destination.exists():
        parser.error("verification requires a new directory inside this run")
    result = verify(run, destination)
    print(f"development_gate={result['development_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

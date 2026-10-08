from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget, configure_cuda, enforce_offline  # noqa: E402
from any2xsampa.adapters.phoneticxeus import verify_snapshot  # noqa: E402
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.limited_data_reference import load_vocabulary  # noqa: E402
from any2xsampa.training.phone_error_metrics import development_gate  # noqa: E402
from any2xsampa.training.phoneticxeus_adaptation import _load_cache, evaluate_rows, make_residual_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Cold-load every fixed-seed checkpoint; no optimizer or held-out test access.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ROOT / ".state/runs") or destination.exists():
        parser.error("use a new verification directory under .state/runs")
    destination.mkdir(parents=True)
    enforce_offline(destination / "runtime-cache")
    torch = configure_cuda()
    from safetensors.torch import load_file

    dataset = read_json(args.dataset)
    rows, head, index = _load_cache(args.features, dataset, torch)
    identity = verify_snapshot(args.snapshot)
    if identity != index["model_metadata"]:
        raise ValueError("cache metadata differs from the verified authorized snapshot")
    from safetensors import safe_open

    with safe_open(str(args.snapshot / "model.safetensors"), framework="pt", device="cpu") as tensors:
        for name, value in head.state_dict().items():
            if not torch.equal(value, tensors.get_tensor(f"model.ctc.ctc_lo.{name}")):
                raise ValueError("cached output head differs from the original pretrained weights")
    dev = [row for row in rows if row["split"] == "dev"]
    vocabulary = load_vocabulary(Path(dataset["bindings"]["vocabulary"]["path"]))
    labels = [""] * len(vocabulary)
    for label, value in vocabulary.items():
        labels[value] = label
    head = head.to("cuda").eval()
    baseline = evaluate_rows(dev, labels, torch)
    candidates = []
    proofs = []
    for seed in (17, 29, 43):
        report_path = args.training_root / f"seed{seed}" / "training-report.json"
        report = read_json(report_path)
        if report["seed"] != seed or report["mode"] != "rendered_proxy_only":
            raise ValueError("checkpoint seed/domain mismatch")
        if report["dataset_hash"] != dataset["dataset_hash"] or report["feature_index_sha256"] != sha256_file(args.features):
            raise ValueError("checkpoint dataset/cache mismatch")
        if report["model_metadata"] != index["model_metadata"] or report["head"] != index["head"]:
            raise ValueError("checkpoint base model/head mismatch")
        checkpoint_path = Path(report["adapter"]["path"])
        if not checkpoint_path.resolve().is_relative_to(report_path.parent.resolve()):
            raise ValueError("checkpoint path escapes its recorded training run")
        verify_file(checkpoint_path, report["adapter"]["sha256"])
        state = load_file(str(checkpoint_path), device="cpu")
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("invalid adapter tensor dtype/values")
        adapter = make_residual_adapter(torch)
        adapter.load_state_dict(state, strict=True)
        adapter.requires_grad_(False).eval().to("cuda")
        repeated = evaluate_rows(dev, labels, torch, adapter, head)
        prior_path = report_path.parent / "adapted-dev.json"
        verify_file(prior_path, report["adapted_dev_sha256"])
        if sha256_object(repeated) != sha256_object(read_json(prior_path)):
            raise ValueError("cold checkpoint replay differs from recorded predictions/metrics")
        if baseline["metrics"] != report["baseline_dev"]:
            raise ValueError("baseline differs across training runs")
        write_json_new(destination / f"seed{seed}-dev-replayed.json", repeated)
        candidates.append({"seed": seed, "metrics": repeated["metrics"]})
        proofs.append({
            "seed": seed, "checkpoint_sha256": report["adapter"]["sha256"],
            "training_report_sha256": sha256_file(report_path),
            "exact_prediction_and_metric_replay": True,
            "memory": check_gpu_budget(torch),
        })
    gate = development_gate(baseline["metrics"], candidates)
    result = {
        "schema_version": "phoneticxeus_checkpoint_verification_v1",
        "dataset_hash": dataset["dataset_hash"], "feature_index_sha256": sha256_file(args.features),
        "proofs": proofs, "development_gate": gate,
        "mode": "rendered_proxy_only", "test_songs_opened": False,
        "optimizer_steps_executed": 0, "original_accuracy_claim_allowed": False,
        "pretrained_head_verified_against_base_weights": True,
        "qualified_recognizer": False,
    }
    print(f"verification_sha256={write_json_new(destination / 'verification.json', result)}; development_gate={gate['passed']}; selected_seed={gate['selected_seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

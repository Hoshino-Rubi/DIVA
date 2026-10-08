"""One fixed CTC+KL experiment; no search and no implicit model promotion."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.limited_data_reference import load_vocabulary, training_gate
from any2xsampa.training.phone_error_metrics import development_gate
from any2xsampa.training.phoneticxeus_adaptation import _batch, _load_cache, evaluate_rows, make_residual_adapter


def baseline_kl(torch: Any, student: Any, teacher: Any, lengths: Any, *, temperature: float = 2.0) -> Any:
    """KL(teacher || student), averaged over actual frames, all labels retained."""
    if tuple(student.shape) != tuple(teacher.shape) or student.ndim != 3:
        raise ValueError("KL tensors must have equal [batch, frames, labels] shape")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0 or not len(lengths) or len(lengths) != student.shape[0]:
        raise ValueError("invalid KL temperature/lengths")
    sizes = lengths.to(student.device)
    if sizes.dtype not in {torch.int32, torch.int64} or not torch.all((sizes > 0) & (sizes <= student.shape[1])):
        raise ValueError("KL lengths must describe nonempty valid frames")
    mask = torch.arange(student.shape[1], device=student.device)[None, :] < sizes[:, None]
    # Select before softmax, so even poisoned padding cannot affect the loss
    # or propagate NaN gradients through an otherwise masked softmax.
    actual = student[mask]
    frozen = teacher.detach()[mask]
    if not torch.isfinite(actual).all() or not torch.isfinite(frozen).all():
        raise ValueError("non-finite valid KL evidence")
    log_q = (actual / temperature).log_softmax(-1)
    log_p = (frozen / temperature).log_softmax(-1)
    return torch.nn.functional.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1).mean() * temperature ** 2


def candidate_check(baseline: dict, evaluated: dict, seed: int) -> dict:
    return development_gate(baseline, [{"seed": seed, "metrics": evaluated}])["seeds"][0]


def _teacher_batch(torch: Any, rows: Sequence[Mapping[str, Any]], maximum: int) -> Any:
    values = torch.zeros(len(rows), maximum, 428, dtype=torch.float32)
    for i, row in enumerate(rows):
        frames = len(row["baseline_logits"])
        values[i, :frames] = row["baseline_logits"]
    return values.to("cuda")


def _validate_inputs(run: Path, seed: int, root: Path) -> tuple[dict, Path, Path]:
    if seed not in {17, 29, 43}:
        raise ValueError("only the three approved seeds are allowed")
    start = read_json(run / "execution-start.json")
    verify_file(run / "policy.json", start["policy_sha256"])
    policy = read_json(run / "policy.json")
    spec = policy["training"]
    expected = {"temperature": 2.0, "kl_coefficient": .05, "learning_rate": .001,
        "weight_decay": .0001, "batch_size": 8, "max_epochs": 20, "patience": 4,
        "max_seconds_per_seed": 1500, "gradient_clip_norm": 1.0}
    if any(spec.get(k) != v for k, v in expected.items()) or policy["reference_mode"] != "rendered_proxy_only":
        raise ValueError("training policy differs from the approved single experiment")
    before = read_json(run / "preservation-before.json")
    preserved = {name.replace("\\", "/"): digest for name, digest in before["artifacts"].items()}
    base = root / ".state/runs/limited-data-20260906T050251"
    dataset_path, index_path = base / "reference/rendered-proxy-dataset.json", base / "rendered-features/index.json"
    for path in (dataset_path, index_path):
        verify_file(path, preserved[path.relative_to(root).as_posix()])
    return policy, dataset_path, index_path


def _epoch(torch: Any, adapter: Any, head: Any, rows: Sequence[Mapping[str, Any]], optimizer: Any,
           loss_fn: Any, spec: dict, deadline: float) -> dict:
    ctc_values, kl_values, losses = [], [], []
    adapter.train()
    for offset in range(0, len(rows), spec["batch_size"]):
        if time.monotonic() >= deadline:
            return {"complete": False, "optimizer_steps": len(losses)}
        batch = rows[offset:offset + spec["batch_size"]]
        features, lengths, targets, target_lengths = _batch(batch, torch)
        teacher = _teacher_batch(torch, batch, features.shape[1])
        optimizer.zero_grad(set_to_none=True)
        logits = head(adapter(features))
        ctc = loss_fn(logits.log_softmax(-1).transpose(0, 1), targets, lengths, target_lengths)
        kl = baseline_kl(torch, logits, teacher, lengths, temperature=spec["temperature"])
        loss = ctc + spec["kl_coefficient"] * kl
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite protected loss; no zero-infinity masking")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), spec["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        ctc_values.append(float(ctc.item()))
        kl_values.append(float(kl.item()))
        losses.append(float(loss.item()))
    return {"complete": True, "optimizer_steps": len(losses), "mean_ctc_loss": sum(ctc_values) / len(losses),
            "mean_kl_loss": sum(kl_values) / len(losses), "mean_total_loss": sum(losses) / len(losses)}


def run_protected_adaptation(run: Path, seed: int, root: Path) -> dict:
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget, configure_cuda, enforce_offline
    from safetensors.torch import save_file

    policy, dataset_path, index_path = _validate_inputs(run, seed, root)
    destination = run / "training-v2" / f"seed{seed}"
    if destination.exists():
        raise FileExistsError("seed already attempted; refusing a fourth/repeated optimization trial")
    dataset = read_json(dataset_path)
    if not training_gate(dataset)["passed"]:
        raise ValueError("training reference gate fails")
    destination.mkdir(parents=True)
    enforce_offline(destination / "runtime-cache")
    torch = configure_cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    before, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    spec = policy["training"]
    rows, head, index = _load_cache(index_path, dataset, torch)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if (len(train), len(dev)) != (129, 25):
        raise ValueError("protected adaptation requires the unchanged 129/25 subset")
    vocab = load_vocabulary(Path(dataset["bindings"]["vocabulary"]["path"]))
    labels = [label for label, _ in sorted(vocab.items(), key=lambda pair: pair[1])]
    baseline = evaluate_rows(dev, labels, torch)
    write_json_new(destination / "baseline-dev.json", baseline)
    head = head.to("cuda")
    adapter = make_residual_adapter(torch).to("cuda")
    probe = train[0]["features"].to("cuda")
    if not torch.equal(adapter(probe), probe) or not torch.equal(head(adapter(probe)), head(probe)):
        raise RuntimeError("zero-initialized adapter changed the initial output")
    initial = evaluate_rows(dev, labels, torch, adapter, head)
    if initial != baseline:
        raise RuntimeError("zero-initialized development predictions differ from frozen baseline")
    frozen_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    write_json_new(destination / "optimization-start.json", {"seed": seed, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": sha256_file(run / "policy.json"), "dataset_sha256": sha256_file(dataset_path),
        "feature_index_sha256": sha256_file(index_path), "zero_initial_predictions_identical": True})
    best, admissible = None, None
    best_per, admissible_per, stale = float("inf"), float("inf"), 0
    history, steps = [], 0
    reason = "maximum_epochs"
    rng = random.Random(seed)
    for epoch in range(1, spec["max_epochs"] + 1):
        ordered = list(train)
        rng.shuffle(ordered)
        trained = _epoch(torch, adapter, head, ordered, optimizer, criterion, spec, before + spec["max_seconds_per_seed"])
        steps += trained["optimizer_steps"]
        if not trained["complete"]:
            reason = "wall_clock_limit_incomplete_epoch_not_selected"
            write_json_new(destination / f"epoch-{epoch:02d}-incomplete.json", {"epoch": epoch, **trained})
            break
        adapter.eval()
        evaluated = evaluate_rows(dev, labels, torch, adapter, head)
        score = evaluated["metrics"]["per"]
        gate = candidate_check(baseline["metrics"], evaluated["metrics"], seed)
        event = {"epoch": epoch, **trained, "dev_metrics": evaluated["metrics"], "candidate_check": gate,
                 "elapsed_seconds": time.monotonic() - before, "memory": check_gpu_budget(torch)}
        history.append(event)
        write_json_new(destination / f"epoch-{epoch:02d}.json", event)
        if score < best_per:
            best_per, stale = score, 0
            best = {"epoch": epoch, "evaluation": evaluated, "state": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}}
        else:
            stale += 1
        if gate["passed"] and score < admissible_per:
            admissible_per = score
            admissible = {"epoch": epoch, "evaluation": evaluated, "state": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}}
        print(f"seed={seed}; epoch={epoch}; per={score:.6f}; protected={gate['passed']}; ctc={trained['mean_ctc_loss']:.6f}; kl={trained['mean_kl_loss']:.6f}", flush=True)
        if stale >= spec["patience"]:
            reason = "early_stopping_unconstrained_dev_per"
            break
    if best is None:
        raise RuntimeError("no complete development checkpoint; trial remains consumed")
    if any(not torch.equal(v.detach().cpu(), frozen_head[k]) for k, v in head.state_dict().items()) or any(p.grad is not None or p.requires_grad for p in head.parameters()):
        raise RuntimeError("frozen pretrained head was modified")
    checkpoints = {}
    for kind, candidate in (("best_overall", best), ("best_admissible", admissible)):
        if candidate is None:
            checkpoints[kind] = None
            continue
        path = destination / f"{kind}.safetensors"
        save_file(candidate["state"], str(path))
        evaluation_path = destination / f"{kind}-dev.json"
        evaluation_hash = write_json_new(evaluation_path, candidate["evaluation"])
        checkpoints[kind] = {"path": str(path.resolve()), "sha256": sha256_file(path), "epoch": candidate["epoch"],
            "evaluation_path": str(evaluation_path.resolve()), "evaluation_sha256": evaluation_hash, "metrics": candidate["evaluation"]["metrics"]}
    report = {"schema_version": "phoneticxeus_protected_adaptation_v2", "seed": seed, "mode": "rendered_proxy_only",
        "policy_sha256": sha256_file(run / "policy.json"), "dataset_file_sha256": sha256_file(dataset_path),
        "feature_index_sha256": sha256_file(index_path), "model_metadata": index["model_metadata"], "head": index["head"],
        "architecture": {"width": 1024, "bottleneck": 64, "trainable_parameters": sum(p.numel() for p in adapter.parameters())},
        "hyperparameters": spec, "baseline_dev": baseline["metrics"], "checkpoints": checkpoints,
        "history": history, "optimizer_steps": steps, "stop_reason": reason, "head_unchanged": True,
        "zero_initial_predictions_identical": True, "backbone_loaded_during_training": False, "test_songs_loaded": False,
        "original_accuracy_claim_allowed": False, "automatic_model_promotion": False,
        "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - before}
    write_json_new(destination / "training-report.json", report)
    return report

"""Matched continuation training with whole-song exclusion and an explicit risk loss."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import random
import time

from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.singing_domain_training import load_rows, predict, evaluate
from any2xsampa.training.singing_domain import partition
from any2xsampa.training.singing_domain_losses import sequence_ctc
from any2xsampa.training.protected_adaptation import baseline_kl
from any2xsampa.training.sequence_risk import candidate_pool, ctc_candidate_risk


def epoch_training(torch: Any, tail: Any, train: list[dict], arm: str, spec: dict,
                   pools: dict, optimizer: Any, rng: Any, deadline: float) -> dict:
    ordered = []
    for row in train:
        if row["audio_role"] in {"original", "rendered"}:
            ordered.extend([row] * (spec["training"]["original_repeats"] if row["audio_role"] == "original" else 1))
    rng.shuffle(ordered)
    tail.train()
    parameters = [p for p in tail.parameters() if p.requires_grad]
    means = {"ctc": 0.0, "prior_kl": 0.0, "risk": 0.0}
    accumulation, steps = spec["training"]["gradient_accumulation"], 0
    for offset in range(0, len(ordered), accumulation):
        if time.monotonic() > deadline:
            raise TimeoutError("fixed continuation budget exceeded")
        optimizer.zero_grad(set_to_none=True)
        batch = ordered[offset:offset+accumulation]
        for row in batch:
            logits, _ = predict(torch, tail, row)
            loss_ctc = sequence_ctc(torch, logits, row["target_ids"])
            prior = row["frame_logits"].unsqueeze(0).cuda()
            loss_kl = baseline_kl(torch, logits, prior,
                torch.tensor([logits.shape[1]], device="cuda"), temperature=spec["training"]["temperature"])
            weight = spec["training"]["rendered_loss_weight"] if row["audio_role"] == "rendered" else 1.0
            loss = weight*loss_ctc + spec["training"]["pretrained_kl_weight"]*loss_kl
            risk_value = 0.0
            if arm == "sequence_risk":
                pool = pools[row["example_id"]]
                if pool["target_ids"] != row["target_ids"]:
                    raise ValueError("candidate pool reference binding changed")
                risk, _ = ctc_candidate_risk(torch, logits, pool, temperature=spec["risk"]["posterior_temperature"])
                loss = loss + weight*spec["risk"]["coefficient"]*risk
                risk_value = float(risk.detach())
            if not torch.isfinite(loss):
                raise ValueError("non-finite continuation objective")
            (loss/len(batch)).backward()
            means["ctc"] += float(loss_ctc.detach())
            means["prior_kl"] += float(loss_kl.detach())
            means["risk"] += risk_value
        torch.nn.utils.clip_grad_norm_(parameters, spec["training"]["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        steps += 1
        if torch.cuda.memory_reserved() > 7*2**30:
            raise RuntimeError("continuation exceeded own reserved GPU budget")
    return {"optimizer_steps": steps, "supervised_microsteps": len(ordered),
            "means": {k:v/len(ordered) for k,v in means.items()}}


def refresh_candidates(torch: Any, tail: Any, rows: list[dict], labels: list[str], spec: dict,
                       output: Path, epoch: int, deadline: float) -> dict:
    before = time.monotonic()
    result = {}
    tail.eval()
    with torch.no_grad():
        for row in rows:
            if row["audio_role"] not in {"original", "rendered"}:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError("candidate refresh exceeded the same training job budget")
            logits, _ = predict(torch, tail, row)
            result[row["example_id"]] = {**candidate_pool(logits[0].cpu().numpy(), row["target_ids"], labels, spec["risk"]),
                "song_id": row["song_id"], "cache_sha256": row["sha256"]}
    write_json_new(output / f"candidate-pool-epoch-{epoch:02d}.json", {"epoch": epoch,
        "rows": result, "elapsed_seconds": time.monotonic()-before, "only_training_rows": True,
        "references_entered_decoder": False})
    return result


def train_job(run: Path, arm: str, held_song: str | None, seed: int, output: Path) -> None:
    from safetensors.torch import load_file, save_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime, make_tail, load_trainable_state, trainable_state
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
    spec, freeze = read_json(run / "POLICY.json"), read_json(run / "FREEZE.json")
    verify_file(run / "POLICY.json", freeze["policy_sha256"])
    if arm not in {"control", "sequence_risk"}:
        raise ValueError("whole-sequence route requires a passed source/correspondence gate; no implicit fallback")
    if seed not in spec["training"]["final_seeds"] or (held_song is not None and (held_song not in spec["songs"] or seed != 17)):
        raise ValueError("unplanned continuation seed/fold")
    if held_song is None:
        selection = read_json(run / "ROUTE_SELECTION.json")
        if not selection["passed"] or selection["selected_arm"] != arm:
            raise ValueError("final jobs require a passing cross-song selection")
    parent_run = Path(freeze["parents"]["singing"])
    parent_job = parent_run / "architecture/full" / held_song if held_song else parent_run / "final-training" / ("seed"+str(seed))
    parent = read_json(parent_job / "RESULT.json")
    if (not parent["complete"] or parent["arm"] != "full" or parent["variant"] != "baseline"
            or parent["held_song"] != held_song or parent["seed"] != seed):
        raise ValueError("parent checkpoint belongs to another fold/recipe/seed")
    if held_song is not None and held_song in parent["training_songs"]:
        raise ValueError("parent checkpoint leaks the held song")
    for path, digest in parent["code_bindings"].items():
        verify_file(Path(path), digest)
    verify_file(Path(parent["checkpoint_path"]), parent["checkpoint_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with serial_candidate_job(), ProcessResourceMonitor(output / "monitor", timeout_seconds=spec["training"]["maximum_job_seconds"]):
        original_freeze = read_json(parent_run / "FREEZE.json")
        rt = SingingDomainRuntime(Path(original_freeze["model_snapshot"]), output / "runtime-cache")
        torch = rt.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        rows, index = load_rows(parent_run, torch)
        if rt.identity["sha256"] != index["architecture"]["sha256"] or rt.identity["sha256"] != parent["architecture"]["sha256"]:
            raise ValueError("parent/prefix architecture mismatch")
        train, validation = partition(rows, held_song)
        tail = make_tail(torch, rt.model, "full")
        load_trainable_state(tail, load_file(parent["checkpoint_path"], device="cpu"))
        rt.model.frontend.cpu()
        rt.model.preencoder.cpu()
        for block in list(rt.model.encoder.encoders)[:15]:
            block.cpu()
        rt.model.encoder.conditioning_layer.cpu()
        labels = [p for p,i in sorted(rt.vocabulary.items(), key=lambda x:x[1])]
        initial = evaluate(torch, tail, validation, labels) if validation else None
        if initial is not None:
            expected = parent["final_validation"]
            if initial["metrics"] != expected["metrics"] or [r["prediction"] for r in initial["examples"]] != [r["prediction"] for r in expected["examples"]]:
                raise ValueError("step-zero continuation differs from its parent checkpoint")
        frozen = [p for p in rt.model.parameters() if not p.requires_grad]
        versions = [p._version for p in frozen]
        parameters = [p for p in tail.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=spec["training"]["learning_rate"],
            weight_decay=spec["training"]["weight_decay"], foreach=False)
        record = {"arm": arm, "held_song": held_song, "seed": seed, "architecture": rt.identity,
            "parent_checkpoint_path": parent["checkpoint_path"], "parent_checkpoint_sha256": parent["checkpoint_sha256"],
            "parent_result_sha256": sha256_file(parent_job / "RESULT.json"), "initial_validation": initial,
            "initialization": "continued full tail; not zero initialization", "training_example_ids": [r["example_id"] for r in train],
            "training_songs": sorted({r["song_id"] for r in train}),
            "training_counts": {role:sum(r["audio_role"] == role for r in train) for role in ("original", "rendered")},
            "policy_sha256": sha256_file(run / "POLICY.json"), "cache_sha256": sha256_file(parent_run / "CACHE.json"),
            "trainable_parameters": sum(p.numel() for p in parameters), "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_bindings": {str(p):sha256_file(p) for p in [Path(__file__), Path(__file__).with_name("sequence_risk.py")]}}
        write_json_new(output / "START.json", record)
        history, pools, rng = [], {}, random.Random(seed)
        torch.cuda.reset_peak_memory_stats()
        deadline = start + spec["training"]["maximum_job_seconds"]
        for epoch in range(1, spec["training"]["epochs"]+1):
            if arm == "sequence_risk" and epoch in spec["risk"]["refresh_epochs"]:
                pools = refresh_candidates(torch, tail, train, labels, spec, output, epoch, deadline)
            trained = epoch_training(torch, tail, train, arm, spec, pools, optimizer, rng, deadline)
            evaluated = evaluate(torch, tail, validation, labels) if validation and epoch in {5,10,20} else None
            event = {"epoch": epoch, **trained, "validation": evaluated["metrics"] if evaluated else None,
                "elapsed_seconds": time.monotonic()-start, "memory": check_gpu_budget(torch)}
            history.append(event)
            write_json_new(output / f"epoch-{epoch:02d}.json", event)
            print(arm, held_song or "all", seed, epoch, "CTC", round(trained["means"]["ctc"],4),
                "risk", round(trained["means"]["risk"],4), "PER", round(evaluated["metrics"]["per"],5) if evaluated else "-", flush=True)
        if versions != [p._version for p in frozen]:
            raise RuntimeError("frozen model parameters changed")
        final = evaluate(torch, tail, validation, labels) if validation else None
        checkpoint = output / "final.safetensors"
        save_file(trainable_state(tail), str(checkpoint))
        write_json_new(output / "RESULT.json", {**record, "complete": True, "fixed_epoch": 20,
            "history": history, "final_validation": final, "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint), "frozen_parameters_unchanged": True,
            "optimizer_steps": sum(h["optimizer_steps"] for h in history),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "elapsed_seconds": time.monotonic()-start,
            "default_replacement": False, "test_predictions_opened": False})

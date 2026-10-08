"""Fixed-budget adaptation on explicitly confirmed original segments.

This module is intentionally opt-in.  It refuses to start until the v2
reference and feature cache satisfy the complete-song-group and coverage gates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.limited_data_reference import load_vocabulary
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.phoneticxeus_adaptation import _batch, make_residual_adapter
from any2xsampa.training.protected_adaptation import baseline_kl
from any2xsampa.training.reviewed_training_reference import validate_training_reference
from any2xsampa.training.phoneticxeus_adaptation import required_ctc_frames
from any2xsampa.training.limited_data_reference import SONG_SPLIT
from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256


SEEDS = (17, 29, 43)
MAX_SECONDS = 1500



def validate_original_index(index: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], vocabulary_hash: str) -> None:
    if index.get("purpose") != "original-reviewed-v2" or index.get("all_parameters_frozen") is not True or index.get("no_labels_or_score_received") is not True:
        raise ValueError("original cache must be frozen and audio only")
    model=index.get("model_metadata", {})
    if (model.get("model_id"), model.get("revision"), model.get("weights_sha256")) != (MODEL_ID, REVISION, WEIGHT_SHA256):
        raise ValueError("original model identity mismatch")
    entries=index["entries"]
    expected={r["example_id"].lower():r for r in rows}
    if len(entries)!=len(rows) or len(expected)!=len(rows) or {e["example_id"] for e in entries}!=set(expected):
        raise ValueError("original cache coverage mismatch")
    for entry in entries:
        row=expected[entry["example_id"]]
        if row["split"] not in {"train","dev"} or SONG_SPLIT.get(row["song_id"])!=row["split"] or row["audio_role"]!="original":
            raise ValueError("original song leakage or role mismatch")
        for key in ("song_id","split","audio_role","audio_sha256","start_sec","end_sec"):
            if entry[key]!=row[key]:
                raise ValueError("original cache binding mismatch: " + key)
        if entry["test_outputs_sealed"] or entry["vocabulary_sha256"]!=vocabulary_hash or entry["model_snapshot_lock_sha256"]!=model["snapshot_lock_sha256"]:
            raise ValueError("sealed or stale original cache")
        if entry["frames"] < required_ctc_frames(row["model_target_ids"]):
            raise ValueError("insufficient CTC frames")


def _labels(vocabulary: Mapping[str, int]) -> list[str]:
    result = [""] * len(vocabulary)
    for label, index in vocabulary.items():
        result[index] = label
    return result


def _load_evidence(index: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], torch: Any, *, root: Path) -> list[dict[str, Any]]:
    from safetensors.torch import load_file

    entries = {entry["example_id"]: entry for entry in index.get("entries", [])}
    if len(entries) != len(rows) or set(entries) != {row["example_id"].lower() for row in rows}:
        raise ValueError("original feature cache does not cover exactly the confirmed segments")
    loaded = []
    for row in rows:
        entry = entries[row["example_id"].lower()]
        if entry["audio_sha256"] != row["audio_sha256"] or entry["start_sec"] != row["start_sec"] or entry["end_sec"] != row["end_sec"]:
            raise ValueError("original feature cache source binding differs from segment reference")
        path = Path(entry["evidence_path"])
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("original feature evidence must remain in the run")
        verify_file(path, entry["evidence_sha256"])
        values = load_file(str(path), device="cpu")
        features, logits = values["encoder_features"], values["frame_logits"]
        if tuple(features.shape) != (entry["frames"], 1024) or tuple(logits.shape) != (entry["frames"], 428):
            raise ValueError("invalid original feature evidence shape")
        if features.dtype != torch.float32 or logits.dtype != torch.float32 or not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise ValueError("invalid original feature evidence")
        loaded.append({**row, "features": features, "baseline_logits": logits, "frames": entry["frames"]})
    return loaded


def _reference_ids(rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        phones = row.get("phones", [])
        if row.get("status") != "human_confirmed" or not phones:
            raise ValueError("training/evaluation rows must be human confirmed")
        symbols = [phone["phone"] for phone in phones]
        ids = [vocabulary.get(symbol) for symbol in symbols]
        if any(value is None or isinstance(value, bool) or value < 4 for value in ids):
            raise ValueError("confirmed phone is absent from the pinned non-special vocabulary")
        result.append({**row, "model_target_ipa": symbols, "model_target_ids": ids})
    return result


def _evaluate(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], torch: Any, adapter: Any, head: Any) -> dict[str, Any]:
    pairs = []
    details = []
    with torch.no_grad():
        for row in rows:
            logits = head(adapter(row["features"].to("cuda")))
            ids = ctc_collapse(logits.argmax(dim=-1).cpu().tolist())
            prediction = [labels[index] for index in ids]
            reference = list(row["model_target_ipa"])
            pairs.append((reference, prediction))
            details.append({"example_id": row["example_id"], "song_id": row["song_id"], "reference": reference, "prediction": prediction,
                            "metrics": phone_metrics([(reference, prediction)])})
    return {"metrics": phone_metrics(pairs), "examples": details}


def _evaluate_logits(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    pairs = []
    for row in rows:
        ids = ctc_collapse(row["baseline_logits"].argmax(dim=-1).tolist())
        pairs.append((list(row["model_target_ipa"]), [labels[index] for index in ids]))
    return {"metrics": phone_metrics(pairs)}


def _gate(candidate: Mapping[str, Any], baseline: Mapping[str, Any], teacher: Mapping[str, Any]) -> dict[str, Any]:
    per = candidate["metrics"]["per"]
    base = baseline["metrics"]["per"]
    old = teacher["metrics"]
    improved = base is not None and per is not None and old["per"] is not None and per <= old["per"] * 0.95 and per <= base
    regressions = []
    for name, old_group in old["groups"].items():
        new_group = candidate["metrics"]["groups"][name]
        if old_group["reference_count"] >= 20 and new_group["per"] is not None and new_group["per"] > old_group["per"] + 0.03:
            regressions.append(name)
    return {"passed": improved and not regressions, "relative_improvement_vs_seed43": (old["per"] - per) / old["per"] if old.get("per") and per is not None else None, "not_worse_than_pretrained": per <= base if base is not None and per is not None else False,
            "regressions_vs_seed43": regressions}


def _rendered_gate(candidate: Mapping[str, Any], teacher: Mapping[str, Any]) -> dict[str, Any]:
    base = teacher["metrics"]
    new = candidate["metrics"]
    regressions = []
    if new["per"] > base["per"] + 0.03:
        regressions.append("overall")
    for name, old_group in base["groups"].items():
        row = new["groups"][name]
        if old_group["reference_count"] >= 20 and row["per"] is not None and row["per"] > old_group["per"] + 0.03:
            regressions.append(name)
    return {"passed": not regressions, "regressions_vs_seed43": regressions}


def _train_epoch(torch: Any, adapter: Any, head: Any, rows: Sequence[Mapping[str, Any]], teacher_adapter: Any, optimizer: Any, deadline: float) -> dict[str, Any]:
    if time.monotonic() >= deadline:
        return {"complete": False, "steps": 0}
    adapter.train()
    total, ctc_total, kl_total, steps = 0.0, 0.0, 0.0, 0
    for offset in range(0, len(rows), 8):
        if time.monotonic() >= deadline:
            return {"complete": False, "steps": steps}
        batch = rows[offset:offset + 8]
        features, lengths, targets, target_lengths = _batch(batch, torch)
        optimizer.zero_grad(set_to_none=True)
        logits = head(adapter(features))
        with torch.no_grad():
            teacher_logits = head(teacher_adapter(features))
        ctc = torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0, 1), targets, lengths, target_lengths, blank=0, zero_infinity=False)
        kl = baseline_kl(torch, logits, teacher_logits, lengths, temperature=2.0)
        loss = ctc + 0.05 * kl
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite original adaptation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.item()); ctc_total += float(ctc.item()); kl_total += float(kl.item()); steps += 1
    return {"complete": True, "steps": steps, "mean_total_loss": total / steps, "mean_ctc_loss": ctc_total / steps, "mean_kl_loss": kl_total / steps}


def run_original_adaptation(run: Path, seed: int, root: Path) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("only seeds 17, 29, and 43 are authorized")
    manifest = read_json(run / "original-training-manifest.json")
    if manifest.get("schema_version") != "original_training_manifest_v1":
        raise ValueError("unsupported original training manifest")
    if manifest.get("manifest_hash") != sha256_object({k:v for k,v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("training manifest hash mismatch")
    for key in ("segment_reference", "original_feature_index", "rendered_dataset", "rendered_feature_index", "seed43_checkpoint", "vocabulary"):
        verify_file(Path(manifest[key]["path"]), manifest[key]["sha256"])
    if manifest.get("gate", {}).get("passed") is not True:
        raise ValueError("frozen training gate not passed")
    segment_path = Path(manifest["segment_reference"]["path"])
    segment = read_json(segment_path)
    validate_training_reference(segment)
    confirmed = [row for row in segment["examples"] if row["status"] == "human_confirmed"]
    train_refs = [row for row in confirmed if row["split"] == "train"]
    dev_refs = [row for row in confirmed if row["split"] == "dev"]
    if len(train_refs) < 20 or len({row["song_id"] for row in train_refs}) < 3 or len(dev_refs) < 3 or sum(len(row["phones"]) for row in dev_refs) < 100:
        raise ValueError("original training gate is not satisfied")
    train_ids = {row["example_id"] for row in train_refs}
    for left in train_refs:
        for right in train_refs:
            if left is right or left["song_id"] != right["song_id"] or left["audio_sha256"] != right["audio_sha256"]:
                continue
            if left["start_sec"] < right["end_sec"] and right["start_sec"] < left["end_sec"]:
                raise ValueError("overlapping original training windows")
    destination = run / "original-training" / f"seed{seed}"
    if destination.exists():
        raise FileExistsError("this original adaptation seed was already attempted")
    destination.mkdir(parents=True)
    from any2xsampa.adapters.phoneticxeus_runtime import configure_cuda, enforce_offline, check_gpu_budget
    from safetensors.torch import load_file, save_file
    torch = configure_cuda(); enforce_offline(destination / "runtime-cache")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    vocab_path = Path(manifest["vocabulary"]["path"]); verify_file(vocab_path, manifest["vocabulary"]["sha256"]); vocab = load_vocabulary(vocab_path); labels = _labels(vocab)
    train_refs, dev_refs = _reference_ids(train_refs, vocab), _reference_ids(dev_refs, vocab)
    index = read_json(Path(manifest["original_feature_index"]["path"])); verify_file(Path(manifest["original_feature_index"]["path"]), manifest["original_feature_index"]["sha256"])
    validate_original_index(index, train_refs + dev_refs, manifest["vocabulary"]["sha256"])
    all_rows = _load_evidence(index, train_refs + dev_refs, torch, root=run)
    train_rows = [row for row in all_rows if row["split"] == "train"]
    dev_rows = [row for row in all_rows if row["split"] == "dev"]
    head_path = Path(index["head"]["path"]); verify_file(head_path, index["head"]["sha256"]); head_state = load_file(str(head_path), device="cpu")
    head = torch.nn.Linear(1024, 428, device="cuda"); head.load_state_dict(head_state, strict=True); head.requires_grad_(False).eval()
    teacher_path = Path(manifest["seed43_checkpoint"]["path"]); verify_file(teacher_path, manifest["seed43_checkpoint"]["sha256"])
    teacher = make_residual_adapter(torch); teacher.load_state_dict(load_file(str(teacher_path), device="cpu"), strict=True); teacher.eval().requires_grad_(False).to("cuda")
    adapter = make_residual_adapter(torch); adapter.load_state_dict(load_file(str(teacher_path), device="cpu"), strict=True); adapter.to("cuda")
    frozen_head = {k:v.detach().clone() for k,v in head.state_dict().items()}
    frozen_teacher = {k:v.detach().clone() for k,v in teacher.state_dict().items()}
    before = time.monotonic(); started = datetime.now(timezone.utc).isoformat()
    with torch.no_grad():
        probe = train_rows[0]["features"].to("cuda")
        if not torch.equal(head(adapter(probe)), head(teacher(probe))):
            raise RuntimeError("seed43 initialization does not reproduce the teacher adapter")
    with torch.no_grad():
        for row in all_rows:
            actual = head(row["features"].to("cuda")).cpu()
            if not torch.allclose(actual, row["baseline_logits"], rtol=1e-4, atol=1e-4):
                raise ValueError("original logits differ from frozen head(features)")
    baseline_original = _evaluate_logits(dev_rows, labels); teacher_original = _evaluate(dev_rows, labels, torch, teacher, head)
    rendered_data = read_json(Path(manifest["rendered_dataset"]["path"])); rendered_index = Path(manifest["rendered_feature_index"]["path"])
    from any2xsampa.training.phoneticxeus_adaptation import _load_cache
    rendered_rows, rendered_head, _ = _load_cache(rendered_index, rendered_data, torch)
    rendered_head = rendered_head.to("cuda"); rendered_dev = [row for row in rendered_rows if row["split"] == "dev"]
    if len(rendered_dev) != 25 or any(row["song_id"] != "shalong" for row in rendered_dev):
        raise ValueError("rendered protection set changed")
    if not all(torch.equal(head.state_dict()[k], rendered_head.state_dict()[k]) for k in frozen_head):
        raise ValueError("original and rendered heads differ")
    teacher_rendered = _evaluate(rendered_dev, labels, torch, teacher, rendered_head)
    write_json_new(destination / "baseline-original-dev.json", baseline_original); write_json_new(destination / "seed43-original-dev.json", teacher_original); write_json_new(destination / "seed43-rendered-dev.json", teacher_rendered)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = random.Random(seed); best = None; best_pass = None; best_per = float("inf"); best_pass_per = float("inf"); stale = 0; history=[]; reason="maximum_epochs"
    for epoch in range(1, 21):
        order = list(train_rows); rng.shuffle(order)
        event = _train_epoch(torch, adapter, head, order, teacher, optimizer, before + MAX_SECONDS)
        if not event["complete"]:
            reason = "wall_clock_limit_incomplete_epoch_not_selected"; break
        adapter.eval(); original_eval = _evaluate(dev_rows, labels, torch, adapter, head); rendered_eval = _evaluate(rendered_dev, labels, torch, adapter, rendered_head)
        gate = _gate(original_eval, baseline_original, teacher_original); render_gate = _rendered_gate(rendered_eval, teacher_rendered); passed = gate["passed"] and render_gate["passed"]
        score = original_eval["metrics"]["per"]; event = {"epoch": epoch, **event, "original_dev": original_eval["metrics"], "rendered_dev": rendered_eval["metrics"], "original_gate": gate, "rendered_gate": render_gate, "passed": passed, "memory": check_gpu_budget(torch), "elapsed_seconds": time.monotonic() - before}; history.append(event); write_json_new(destination / f"epoch-{epoch:02d}.json", event)
        if score < best_per: best_per = score; best = {"epoch": epoch, "state": {k: v.detach().cpu().clone() for k,v in adapter.state_dict().items()}, "original": original_eval, "rendered": rendered_eval}; stale=0
        else: stale += 1
        if passed and score < best_pass_per: best_pass_per = score; best_pass = {"epoch": epoch, "state": {k: v.detach().cpu().clone() for k,v in adapter.state_dict().items()}, "original": original_eval, "rendered": rendered_eval}
        if stale >= 4: reason="early_stopping_original_dev_per"; break
    if best is None: raise RuntimeError("no complete original adaptation epoch")
    if not all(torch.equal(head.state_dict()[k],v) for k,v in frozen_head.items()):
        raise RuntimeError("frozen head parameters changed")
    if not all(torch.equal(teacher.state_dict()[k],v) for k,v in frozen_teacher.items()) or any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError("immutable teacher changed")
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("frozen pretrained output head became trainable")
    checkpoints={}
    for role,candidate in (("best_overall",best),("best_admissible",best_pass)):
        if candidate is None: checkpoints[role]=None; continue
        path=destination/f"{role}.safetensors"; save_file(candidate["state"],str(path)); checkpoints[role]={"path":str(path.resolve()),"sha256":sha256_file(path),"epoch":candidate["epoch"],"original_metrics":candidate["original"]["metrics"],"rendered_metrics":candidate["rendered"]["metrics"]}
    report={"schema_version":"original_residual_adaptation_v1","seed":seed,"mode":"confirmed_original_segments","manifest_sha256":sha256_file(run/"original-training-manifest.json"),"architecture":{"width":1024,"bottleneck":64,"initialized_from_seed43":True,"trainable_parameters":sum(p.numel() for p in adapter.parameters())},"hyperparameters":{"learning_rate":1e-4,"weight_decay":1e-4,"batch_size":8,"max_epochs":20,"patience":4,"gradient_clip_norm":1.0,"max_seconds":MAX_SECONDS,"kl_temperature":2.0,"kl_coefficient":0.05},"baseline_original":baseline_original["metrics"],"seed43_original":teacher_original["metrics"],"seed43_rendered":teacher_rendered["metrics"],"checkpoints":checkpoints,"history":history,"stop_reason":reason,"head_unchanged":True,"backbone_loaded_during_training":False,"test_songs_loaded":False,"automatic_model_promotion":False,"original_accuracy_claim_allowed":True,"started_at_utc":started,"ended_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-before}
    report.update(teacher_unchanged=True, initialization_output_equal=True, frozen_parameters_compared=True)
    for role,candidate in (("best_overall",best),("best_admissible",best_pass)):
        if candidate is not None:
            write_json_new(destination / (role + "-evaluation.json"), {"original":candidate["original"], "rendered":candidate["rendered"]})
    write_json_new(destination / "training-report.json", report); return report

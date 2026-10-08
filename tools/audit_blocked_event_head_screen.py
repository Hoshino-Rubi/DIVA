"""CPU-only evidence audit after a stopped H/E screen; never select a model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from any2xsampa.training.limited_data_artifacts import (
    read_json, sha256_file, sha256_object, verify_file, write_json_new,
)
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.event_head_evaluation import same_references


def job_state(folder: Path, exit_code: int | None) -> dict:
    """Never turn epoch logs into a recoverable optimizer checkpoint."""
    epochs = [read_json(p) for p in sorted(folder.glob("epoch-*.json"))]
    if [e["epoch"] for e in epochs] != list(range(1, len(epochs) + 1)):
        raise ValueError("nonconsecutive epoch records")
    result_path = folder / "RESULT.json"
    if result_path.exists():
        result = read_json(result_path)
        if not result["complete"] or result["selected_epoch"] != 20 or len(epochs) != 20:
            raise ValueError("incomplete result cannot count as completed")
        if exit_code != 0 and not (
            folder.parent.name == "E" and folder.name == "freedom-flower"
            and exit_code == 73 and result.get("quality_training_restarted") is False
            and result.get("quality_process_exit_code") == 73
        ):
            raise ValueError("unapproved recovery")
        state = "complete_with_readonly_recovery" if exit_code else "complete"
    elif not (folder / "START.json").exists():
        if epochs or exit_code is not None or (folder / "final.safetensors").exists():
            raise ValueError("artifacts without a launch")
        state = "not_started"
    else:
        if exit_code is None or exit_code == 0:
            raise ValueError("worker is still running or missing a successful result")
        state = "stopped_without_final_checkpoint" if not (folder / "final.safetensors").exists() else "stopped_with_checkpoint_unverified"
    return {
        "state": state, "exit_code": exit_code, "logged_epochs": len(epochs),
        "logged_optimizer_steps": sum(e["optimizer_steps"] for e in epochs),
        "unlogged_optimizer_steps_unknown": state.startswith("stopped_"),
        "final_checkpoint_exists": (folder / "final.safetensors").exists(),
        "completed": state in {"complete", "complete_with_readonly_recovery"},
    }


def compare_common(originals: dict[str, list[dict]]) -> dict:
    songs = sorted({r["song_id"] for r in originals["H"]} & {r["song_id"] for r in originals["E"]})
    rows = {a: [r for r in originals[a] if r["song_id"] in songs] for a in ("H", "E")}
    same_references(rows["H"], rows["E"])
    aggregate = lambda rr: phone_metrics([(r["reference"], r["prediction"]) for r in rr])
    return {
        "songs": songs, "metrics": {a: aggregate(rows[a]) for a in ("H", "E")},
        "per_song": {s: {a: aggregate([r for r in rows[a] if r["song_id"] == s]) for a in ("H", "E")} for s in songs},
        "partial_subset_only": True, "screen_gates_evaluated": False,
        "selection_allowed": False,
    }


def audit(run: Path, output: Path) -> None:
    if output.exists() or not output.is_relative_to(run):
        raise ValueError("new output directory inside source run required")
    blocked = read_json(run / "postprocess/BLOCKED.json")
    if blocked.get("blocked") is not True:
        raise ValueError("stopped workflow evidence required")
    spec = read_json(run / "POLICY.json")
    freeze = read_json(run / "FREEZE.json")
    bound = dict(freeze["bindings"])
    for phase in ("preflight", "training", "recovery", "training_v2", "replay", "postprocess"):
        path = run / f"CODE_REVIEW_{phase}.json"
        review = read_json(path)
        bound[str(path)] = sha256_file(path)
        for name, digest in review["code_bindings"].items():
            verify_file(Path(name), digest)
    for name, digest in bound.items():
        verify_file(Path(name), digest)
    rows = {r["example_id"]: r for r in read_json(run / "CACHE_ROWS.json")["rows"]}
    schedule = read_json(run / "SCHEDULES.json")
    vocab = read_json(Path(freeze["vocabulary"]["path"]))
    labels = {i: p for p, i in vocab.items()}
    from safetensors.numpy import load_file
    from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer, ctc_collapse
    from any2xsampa.adapters.segmental_event import SegmentalEventRecognizer, decode_events
    from verify_event_head_delivery import distance

    started = time.monotonic()
    write_json_new(output / "START.json", {"cpu_evidence_only": True, "neural_model_forwards": 0,
        "optimizer_steps": 0, "script_sha256": sha256_file(Path(__file__))})
    ledger = []
    checks = []
    originals = {"H": [], "E": []}
    for arm in spec["arms"]:
        for song in spec["songs"]:
            folder = run / "screen" / arm / song
            exit_path = run / "commands" / f"{arm}--{song}.exit.json"
            code = read_json(exit_path)["exit_code"] if exit_path.exists() else None
            status = job_state(folder, code)
            ledger.append({"arm": arm, "song": song, **status})
            for path in sorted(folder.glob("epoch-*.json")):
                epoch = read_json(path)
                if epoch["order_sha256"] != sha256_object(schedule[song]["epochs"][epoch["epoch"] - 1]) or epoch["held_out_metrics_used"]:
                    raise ValueError("sampling or held-out selection drift")
                bound[str(path)] = sha256_file(path)
            for path in [exit_path, folder / "START.json", folder / "INITIALIZATION.json", folder / "monitor/resource-failure.json"]:
                if path.exists():
                    bound[str(path)] = sha256_file(path)
            if not status["completed"]:
                continue
            path = folder / "RESULT.json"
            result = read_json(path)
            if result["freeze_sha256"] != sha256_file(run / "FREEZE.json") or result["optimizer_steps"] != schedule[song]["optimizer_steps"]:
                raise ValueError("data freeze or optimizer budget drift")
            if result["history"] != [read_json(p) for p in sorted(folder.glob("epoch-*.json"))]:
                raise ValueError("epoch record differs from saved result")
            bound[str(path)] = sha256_file(path)
            verify_file(Path(result["checkpoint_path"]), result["checkpoint_sha256"])
            bound[result["checkpoint_path"]] = result["checkpoint_sha256"]
            init = read_json(folder / "INITIALIZATION.json")
            expected = {k for k, r in rows.items() if r["song_id"] == song}
            if set(init["held_example_ids"]) != expected or set(init["train_example_ids"]) != set(rows) - expected:
                raise ValueError("song leakage")
            details = result["validation"]["details"]
            if len(details) != len(expected) or {r["example_id"] for r in details} != expected:
                raise ValueError("missing or duplicate held output")
            for detail in details:
                row = rows[detail["example_id"]]
                if detail["reference"] != row["target_ipa"] or detail["song_id"] != song or detail["audio_role"] != row["audio_role"]:
                    raise ValueError("reference or role drift")
                key = "posterior" if arm == "H" else "event"
                evidence = Path(detail[key + "_path"])
                verify_file(evidence, detail[key + "_sha256"])
                payload = read_json(evidence)
                raw_spec = payload["raw_evidence" if arm == "H" else "scores"]
                raw_path = (evidence.parent / raw_spec["file"]).resolve()
                if not raw_path.is_relative_to(evidence.parent.resolve()):
                    raise ValueError("unsafe evidence path")
                verify_file(raw_path, raw_spec["sha256"])
                values = load_file(str(raw_path))
                if arm == "H":
                    prediction = [labels[i] for i in ctc_collapse(values["frame_logits"].argmax(-1).tolist())]
                    cls = PhoneticXeusPosteriorRecognizer
                    view = lambda seq: [(asdict(s.time_range), [asdict(c) for c in s.candidate_phones], s.confidence) for s in seq]
                else:
                    prediction = [labels[i] for i in decode_events(values["frame_log_probs"], values["boundary_logits"])["event_ids"]]
                    cls = SegmentalEventRecognizer
                    view = lambda seq: [asdict(s) for s in seq]
                real = cls(evidence).recognize(Path(row["audio_path"]))
                fixture = cls(evidence, fixture=True).recognize(Path(row["audio_path"]))
                metrics = phone_metrics([(detail["reference"], prediction)])
                if view(real) != view(fixture) or prediction != detail["prediction"] or metrics != detail["metrics"]:
                    raise ValueError("codec, fixture, or metric replay differs")
                if metrics["edit_distance"] != distance(detail["reference"], prediction):
                    raise ValueError("independent edit distance differs")
                checks.append({"arm": arm, "example_id": detail["example_id"], "fixture_equal": True, "metric_equal": True})
                bound[str(evidence)] = sha256_file(evidence)
                bound[str(raw_path)] = sha256_file(raw_path)
                if row["audio_role"] == "original":
                    originals[arm].append(detail)
            write_json_new(output / f"{arm}--{song}.json", {"passed": True, "output_count": len(details), "result_sha256": sha256_file(path)})
            print("CPU EVIDENCE VERIFIED", arm, song, len(details), flush=True)
    protected = read_json(ROOT / ".state/runs/dual-input-ctc-20260920T073445032166Z/PRESERVATION_BEFORE.json")["files"]
    protected = {p: h for p, h in protected.items() if Path(p).suffix.lower() in {".vsqx", ".wav", ".json"} or p.endswith("any2xsampa\\cli.py")}
    for name, digest in protected.items():
        verify_file(Path(name), digest)
    write_json_new(output / "RESULT.json", {
        "passed_for_completed_saved_evidence_only": True, "screen_complete": False,
        "ledger": ledger, "checks": checks, "common_original_subset": compare_common(originals),
        "all_completed_originals": {a: phone_metrics([(r["reference"], r["prediction"]) for r in originals[a]]) for a in originals},
        "artifact_bindings": bound, "protected_files_unchanged": protected,
        "neural_model_forwards": 0, "full_audio_replay_executed": False,
        "checkpoint_neural_replay_executed": False, "new_quality_starts": 0,
        "screen_gates_evaluated": False, "candidate_selected": None,
        "elapsed_seconds": time.monotonic() - started,
    })
    print("PARTIAL EVIDENCE AUDIT COMPLETE", len(checks), "outputs; full screen remains blocked", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.output.resolve())

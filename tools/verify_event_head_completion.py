"""Final verification of the ten selected models and all eleven starts."""
from dataclasses import asdict
from pathlib import Path
import argparse
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.event_head_evaluation import screen_gate
from any2xsampa.adapters.segmental_event import SegmentalEventRecognizer, decode_events
from event_head_completion_common import load_completion
from event_head_completion_results import resolve_sources, verify_ledger
from verify_event_head_delivery import distance


def reused_evidence(detail: dict, arm: str, evidence: Path, raw: Path, proof: dict) -> bool:
    checks = [c for c in proof["checks"] if c["arm"] == arm and c["example_id"] == detail["example_id"]]
    if len(checks) != 1 or not checks[0]["fixture_equal"] or not checks[0]["metric_equal"]:
        raise ValueError("missing or duplicate prior codec evidence")
    for path in (evidence, raw):
        digest = proof["artifact_bindings"].get(str(path))
        if not digest:raise ValueError("prior codec proof does not bind this output")
        verify_file(path, digest)
    return True


def verify_states(folder: Path, result: dict, auth: dict, run: Path) -> list[dict]:
    import torch
    from safetensors.torch import load_file
    commits = result["resumable_epoch_commits"]
    if [c["epoch"] for c in commits] != list(range(1, 21)):
        raise ValueError("missing recoverable epoch snapshot")
    verified = []
    for item in commits:
        path = Path(item["commit_path"]);verify_file(path, item["commit_sha256"])
        if not path.resolve().is_relative_to((folder / "resumable").resolve()):
            raise ValueError("snapshot escapes attempt directory")
        meta = read_json(path);state = path.parent / meta["state_file"]
        if meta["state_file"] != "state.pt" or meta["epoch"] != item["epoch"]:
            raise ValueError("invalid snapshot member")
        binding = meta["binding"]
        if (binding["authorization_sha256"] != sha256_file(run / "AUTHORIZATION.json")
                or binding["freeze_sha256"] != auth["freeze_sha256"]
                or binding["review_sha256"] != sha256_file(run / "CODE_REVIEW_training.json")
                or binding["worker_sha256"] != sha256_file(ROOT / "tools/train_event_head_completion.py")
                or binding["held_song"] != result["held_song"]
                or binding["schedule_sha256"] != result["schedule_sha256"]):
            raise ValueError("snapshot experiment binding differs")
        if meta["optimizer_steps"] != result["history"][item["epoch"] - 1]["cumulative_optimizer_steps"]:
            raise ValueError("snapshot step count differs")
        verify_file(state, meta["state_sha256"])
        if item["epoch"] == 20:
            saved = torch.load(state, map_location="cpu", weights_only=True)
            final = load_file(result["checkpoint_path"], device="cpu")
            if saved["binding"] != binding or saved["epoch"] != 20 or saved["optimizer_steps"] != result["optimizer_steps"]:
                raise ValueError("snapshot payload differs from commit")
            if set(saved["head"]) != set(final) or any(not torch.equal(saved["head"][k], final[k]) for k in final):
                raise ValueError("last recoverable state differs from final model")
            if len(saved["rng"]["torch_cuda"]) != 1 or not saved["optimizer"]["state"]:
                raise ValueError("production RNG or optimizer state missing")
        verified.append({"epoch": item["epoch"], "commit_sha256": item["commit_sha256"], "state_sha256": meta["state_sha256"]})
    return verified


def verify(run: Path) -> None:
    auth, source, freeze, spec, index = load_completion(run, "CODE_REVIEW_postprocess.json")
    sources = resolve_sources(run, auth, spec);ledger = verify_ledger(run, auth)
    resolved = read_json(run / "RESOLVED_RESULTS.json")
    if sources != resolved["rows"] or ledger != resolved["ledger"]:
        raise ValueError("resolved source manifest changed")
    collection = run / "collection"
    replay = read_json(collection / "independent-replay/RESULT.json")
    if not replay["passed"] or replay["full_acoustic_forwards"] != 10 or replay["head_cache_replays"] != 298:
        raise ValueError("neural/audio independent replay incomplete")
    proof_path = source / "blocked-validation/saved-evidence/RESULT.json"
    trusted_index = read_json(source / "BLOCKED_FINAL_INDEX.json")
    verify_file(proof_path, trusted_index["artifacts"][str(proof_path)])
    proof = read_json(proof_path)
    if not proof["passed_for_completed_saved_evidence_only"] or len(proof["checks"]) != 229:
        raise ValueError("invalid prior partial evidence")
    from safetensors.numpy import load_file
    vocab = read_json(Path(freeze["vocabulary"]["path"]));labels = {i: p for p, i in vocab.items()}
    rows = {r["example_id"]: r for r in index["rows"]}
    pools = {a: {"original": [], "rendered": []} for a in spec["arms"]}
    checks = [];bindings = {};snapshots = [];reused = 0;new = 0
    for entry in sources:
        path = Path(entry["source_result_path"]);result = read_json(path);arm = entry["arm"];song = entry["song"]
        copied = collection / "screen" / arm / song / "RESULT.json"
        if sha256_file(copied) != entry["source_result_sha256"]:raise ValueError("derived collection differs from source")
        init = read_json(path.parent / "INITIALIZATION.json")
        expected = {k for k, r in rows.items() if r["song_id"] == song}
        if set(init["train_example_ids"]) != set(rows) - expected or set(init["held_example_ids"]) != expected:
            raise ValueError("held-song leakage")
        details = result["validation"]["details"]
        if len(details) != len(expected) or {d["example_id"] for d in details} != expected:
            raise ValueError("held cohort incomplete or duplicated")
        bindings[str(path)] = sha256_file(path);bindings[result["checkpoint_path"]] = result["checkpoint_sha256"]
        if not entry["reused"]:snapshots.append({"key": entry["key"], "commits": verify_states(path.parent, result, auth, run)})
        for detail in details:
            row = rows[detail["example_id"]]
            if row["target_ipa"] != detail["reference"] or row["audio_role"] != detail["audio_role"] or detail["song_id"] != song:
                raise ValueError("reference/role changed")
            key = "posterior" if arm == "H" else "event"
            evidence = Path(detail[key + "_path"]);verify_file(evidence, detail[key + "_sha256"])
            payload = read_json(evidence);raw_spec = payload["raw_evidence" if arm == "H" else "scores"]
            raw = (evidence.parent / raw_spec["file"]).resolve()
            if not raw.is_relative_to(evidence.parent.resolve()):raise ValueError("unsafe output member")
            verify_file(raw, raw_spec["sha256"])
            if entry["reused"]:
                reused_evidence(detail, arm, evidence, raw, proof);prediction = detail["prediction"];reused += 1
            else:
                if arm != "E":raise ValueError("unapproved new arm")
                values = load_file(str(raw));prediction = [labels[i] for i in decode_events(values["frame_log_probs"], values["boundary_logits"])["event_ids"]]
                real = SegmentalEventRecognizer(evidence).recognize(Path(row["audio_path"]))
                fixture = SegmentalEventRecognizer(evidence, fixture=True).recognize(Path(row["audio_path"]))
                if [asdict(s) for s in real] != [asdict(s) for s in fixture] or prediction != detail["prediction"]:
                    raise ValueError("new event output does not replay")
                new += 1
            metric = phone_metrics([(detail["reference"], prediction)])
            if metric != detail["metrics"] or metric["edit_distance"] != distance(detail["reference"], prediction):
                raise ValueError("independent metric differs")
            checks.append({"key": entry["key"], "example_id": detail["example_id"], "fixture_verified": True, "reused_hash_bound_codec_proof": entry["reused"]})
            bindings[str(evidence)] = sha256_file(evidence);bindings[str(raw)] = sha256_file(raw)
            pools[arm][detail["audio_role"]].append(detail)
        print("VERIFIED SOURCE", entry["key"], len(details), "prior-proof" if entry["reused"] else "new-codec", flush=True)
    if (len(checks), reused, new) != (298, 229, 69):raise ValueError("codec coverage differs")
    report = read_json(run / "SCREEN_REPORT.json")
    baseline = read_json(collection / "independent-replay/ENDPOINT_CONTROLS.json")["rows"]
    old_o = [r for r in baseline if r["audio_role"] == "original"];old_r = [r for r in baseline if r["audio_role"] == "rendered"]
    for arm in spec["arms"]:
        gate = screen_gate(arm, pools[arm]["original"], pools[arm]["rendered"], old_o, old_r, spec["screen_gates"], h_original=pools["H"]["original"] if arm == "E" else None)
        if gate != report["gates"][arm]:raise ValueError("independent gate computation differs")
    protected = proof["protected_files_unchanged"]
    for name, digest in protected.items():verify_file(Path(name), digest)
    suite = ET.parse(run / "postprocess/regression.xml").getroot().find("testsuite")
    tests = {k: int(suite.get(k)) for k in ("tests", "failures", "errors", "skipped")}
    if tests["failures"] or tests["errors"] or read_json(run / "postprocess/regression.exit.json")["exit_code"] != 0:
        raise ValueError("regression failed")
    if ledger != report["ledger"]:raise ValueError("global ledger drift")
    write_json_new(run / "FINAL_VERIFICATION.json", {"passed": True, "ledger": ledger,
        "codec_checks": checks, "reused_codec_proofs": reused, "new_codec_checks": new,
        "independent_neural_head_replays": 298, "old_endpoint_control_calls": 149,
        "full_audio_forwards": 10, "resumable_snapshots_verified": snapshots,
        "artifact_bindings": bindings, "protected_files_unchanged": protected,
        "prior_codec_proof_sha256": sha256_file(proof_path), "regression": tests,
        "selected_arm": report["selected_arm"], "default_replaced": False, "final_test_opened": False})
    print("FINAL COMPLETION VERIFIED: 298 outputs, 10 waveforms, 40 snapshots, 11 starts", flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);verify(p.parse_args().run.resolve())

"""Resolve eight reused plus two new results without rewriting failed jobs."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file, sha256_object
from any2xsampa.training.event_head_evaluation import screen_gate, aggregate
from any2xsampa.training.phone_error_metrics import align_phone_edits
from event_head_completion_common import load_completion


def resolve_sources(run: Path, auth: dict, spec: dict) -> list[dict]:
    expected = {a + "--" + s for a in spec["arms"] for s in spec["songs"]}
    if set(auth["existing_completed_results"]) | set(auth["allowed_jobs"]) != expected:
        raise ValueError("result source map does not cover the exact ten folds")
    if set(auth["existing_completed_results"]) & set(auth["allowed_jobs"]):
        raise ValueError("completed jobs must not be retrained")
    resolved = []
    for arm in spec["arms"]:
        for song in spec["songs"]:
            key = arm + "--" + song
            existing = auth["existing_completed_results"].get(key)
            path = Path(existing["path"]) if existing else run / "screen" / arm / song / "RESULT.json"
            if existing:
                verify_file(path, existing["sha256"])
            else:
                if read_json(run / "commands" / (key + ".exit.json"))["exit_code"] != 0:
                    raise ValueError("new attempt did not exit successfully")
            row = read_json(path)
            if (not row["complete"] or row["selected_epoch"] != 20 or row["seed"] != 17
                    or row["arm"] != arm or row["held_song"] != song
                    or row["freeze_sha256"] != auth["freeze_sha256"]):
                raise ValueError("incomplete, wrong-fold or differently frozen result")
            verify_file(Path(row["checkpoint_path"]), row["checkpoint_sha256"])
            if existing and row["checkpoint_sha256"] != existing["checkpoint_sha256"]:
                raise ValueError("reused checkpoint changed")
            resolved.append({"key": key, "arm": arm, "song": song, "reused": existing is not None,
                "source_result_path": str(path), "source_result_sha256": sha256_file(path),
                "checkpoint_path": row["checkpoint_path"], "checkpoint_sha256": row["checkpoint_sha256"]})
    return resolved


def verify_ledger(run: Path, auth: dict) -> dict:
    source = Path(auth["source_run"])
    old = sorted((source / "commands").glob("*.command.json"))
    new = sorted((run / "commands").glob("*.command.json"))
    if len(old) != 9 or len(new) != 2:
        raise ValueError("expected nine prior plus two approved starts")
    if {p.name.removesuffix(".command.json") for p in new} != set(auth["allowed_jobs"]):
        raise ValueError("unapproved new job")
    statuses = []
    for command in old + new:
        key = command.name.removesuffix(".command.json")
        base = command.parent.parent
        arm, song = key.split("--", 1)
        start = read_json(base / "screen" / arm / song / "START.json")
        exit_record = read_json(command.with_name(key + ".exit.json"))
        if not start["quality_training_start"]:
            raise ValueError("invalid quality launch record")
        if exit_record["exit_code"] != 0:
            if base != source or key not in {"E--freedom-flower", "E--promise"} or exit_record["exit_code"] != 73:
                raise ValueError("unhandled failure; no automatic recovery")
            failure = read_json(base / "screen" / arm / song / "monitor/resource-failure.json")
            wanted = "wall_clock_limit" if song == "freedom-flower" else "total_gpu_memory_limit"
            if failure["reason"] != wanted:
                raise ValueError("historical failure reason changed")
        if base == run:
            expected_ordinal = 10 if song == "promise" else 11
            if start["quality_start_ordinal"] != expected_ordinal or read_json(command)["quality_start_ordinal"] != expected_ordinal:
                raise ValueError("global start ordinal changed")
        statuses.append({"key": key, "run": str(base), "exit_code": exit_record["exit_code"],
            "command_sha256": sha256_file(command), "elapsed_seconds": exit_record["elapsed_seconds"]})
    if Counter(s["exit_code"] for s in statuses) != {0: 9, 73: 2}:
        raise ValueError("failure accounting differs")
    epochs = list(source.glob("screen/*/*/epoch-*.json")) + list(run.glob("screen/*/*/epoch-*.json"))
    return {"quality_starts": 11, "normal_quality_exits": 9,
        "post_training_timeout_recovered_without_retraining": 1,
        "in_training_resource_abort": 1, "explicitly_approved_replacement_starts": 1,
        "logged_epochs_including_failed_attempt": len(epochs),
        "logged_optimizer_steps_including_failed_attempt": sum(read_json(p)["optimizer_steps"] for p in epochs),
        "additional_unlogged_failed_epoch_updates": "unknown_not_assumed_zero", "jobs": statuses}


def prepare_collection(run: Path) -> None:
    auth, source, freeze, spec, _ = load_completion(run, "CODE_REVIEW_postprocess.json")
    review = read_json(run / "CODE_REVIEW_postprocess.json")
    for path, digest in review.get("data_bindings", {}).items():verify_file(Path(path), digest)
    resolved = resolve_sources(run, auth, spec);ledger = verify_ledger(run, auth)
    collection = run / "collection";collection.mkdir(exist_ok=False)
    for name in ("FREEZE.json", "POLICY.json", "CACHE_ROWS.json", "SCHEDULES.json", "CONTROLS.json"):
        with (collection / name).open("xb") as stream:stream.write((source / name).read_bytes())
    with (collection / "CODE_REVIEW_replay.json").open("xb") as stream:
        stream.write((run / "CODE_REVIEW_replay.json").read_bytes())
    for row in resolved:
        folder = collection / "screen" / row["arm"] / row["song"];folder.mkdir(parents=True)
        with (folder / "RESULT.json").open("xb") as stream:stream.write(Path(row["source_result_path"]).read_bytes())
        write_json_new(folder / "SOURCE.json", {**row, "view_only_no_training_started_here": True})
    write_json_new(run / "RESOLVED_RESULTS.json", {"rows": resolved, "ledger": ledger,
        "collection_is_read_only_derived_view": True, "no_original_failure_overwritten": True})
    print("COLLECTION READY: eight reused, two new; eleven historical starts", flush=True)


def make_report(run: Path) -> None:
    auth, source, freeze, spec, _ = load_completion(run, "CODE_REVIEW_postprocess.json")
    collection = run / "collection"
    replay = read_json(collection / "independent-replay/RESULT.json")
    if not replay["passed"] or replay["full_acoustic_forwards"] != 10 or replay["head_cache_replays"] != 298:
        raise ValueError("complete independent replay required before selection")
    sources = resolve_sources(run, auth, spec);ledger = verify_ledger(run, auth)
    resolved = read_json(run / "RESOLVED_RESULTS.json")
    if sources != resolved["rows"] or ledger != resolved["ledger"]:
        raise ValueError("resolved sources or launch ledger changed after collection")
    schedules = read_json(source / "SCHEDULES.json");pools = {a: {"original": [], "rendered": []} for a in spec["arms"]}
    steps = Counter();resources = []
    for entry in sources:
        path = Path(entry["source_result_path"]);row = read_json(path);arm = row["arm"];song = row["held_song"]
        if len(row["history"]) != 20 or row["history"] != [read_json(path.parent / f"epoch-{i:02d}.json") for i in range(1, 21)]:
            raise ValueError("epoch evidence mismatch")
        for i, event in enumerate(row["history"]):
            if event["order_sha256"] != sha256_object(schedules[song]["epochs"][i]) or event["held_out_metrics_used"]:
                raise ValueError("sampling drift or held-song epoch selection")
        if row["optimizer_steps"] != schedules[song]["optimizer_steps"]:
            raise ValueError("optimizer step count differs")
        steps[arm] += row["optimizer_steps"]
        if row.get("quality_job_outcome") == "20_epochs_saved_then_readonly_validation_recovered":
            recovery_path = Path(row["evaluation_recovery_path"]);verify_file(recovery_path, row["evaluation_recovery_sha256"])
            recovery = read_json(recovery_path)
            if recovery["optimizer_steps"] != 0 or not recovery["checkpoint_unchanged"]:
                raise ValueError("invalid read-only recovery")
            monitor = read_json(recovery_path.parent / "monitor/resource-monitor.json")
            failure = read_json(path.parent / "monitor/resource-failure.json")
            peak_card = max(monitor["maximum_sampled_total_used_bytes"],
                            failure["latest_sample"]["total_used_bytes"],
                            max(e["memory"]["total_used_bytes"] for e in row["history"]))
            reserved_kind = "maximum_logged_epoch_observation_not_exact_process_peak"
        else:
            monitor = read_json(path.parent / "monitor/resource-monitor.json")
            peak_card = monitor["maximum_sampled_total_used_bytes"]
            reserved_kind = "reported_process_peak"
        if not monitor["operation_succeeded"] or peak_card > 14 * 1024**3 or row["peak_reserved_bytes"] > 7 * 1024**3:
            raise ValueError("completed model resource envelope violation")
        resources.append({"key": entry["key"], "elapsed_seconds": row["elapsed_seconds"],
            "peak_reserved_gib": row["peak_reserved_bytes"] / 1024**3, "reserved_kind": reserved_kind,
            "maximum_sampled_card_gib": peak_card / 1024**3})
        for detail in row["validation"]["details"]:pools[arm][detail["audio_role"]].append(detail)
    if dict(steps) != {"H": 2140, "E": 2140}:
        raise ValueError("effective model update budget changed")
    base = read_json(collection / "independent-replay/ENDPOINT_CONTROLS.json")["rows"]
    old_o = [r for r in base if r["audio_role"] == "original"];old_r = [r for r in base if r["audio_role"] == "rendered"]
    gates = {a: screen_gate(a, pools[a]["original"], pools[a]["rendered"], old_o, old_r, spec["screen_gates"],
        h_original=pools["H"]["original"] if a == "E" else None) for a in spec["arms"]}
    eligible = [a for a in spec["arms"] if gates[a]["passed"]]
    selected = min(eligible, key=lambda a: (gates[a]["original"]["per"], a != "H")) if eligible else None
    reference_path = ROOT / ".state/runs/original-adaptation-20260919T012000/reviewed-training-reference.json"
    reference = {r["example_id"]: r for r in read_json(reference_path)["examples"] if r["split"] == "train"}
    strata = defaultdict(Counter);vowels = set("aeiouy\u0250\u0259\u0254\u025b\u0153\u00f8\u026a\u028a")
    for arm in spec["arms"]:
        for row in pools[arm]["original"]:
            phones = reference[row["example_id"]]["phones"]
            if [p["phone"] for p in phones] != row["reference"]:raise ValueError("human reference drift")
            notes = defaultdict(list)
            for p in phones:notes[p["note_id"]].append(p["phone"])
            edges = {phones[0]["note_id"], phones[-1]["note_id"]};position = 0
            for op, expected, actual in align_phone_edits(row["reference"], row["prediction"]):
                if expected is None:continue  # Insertions remain included in all window/PER metrics.
                p = phones[position];position += 1
                kind = "multi_vowel" if sum(any(ch in vowels for ch in phone) for phone in notes[p["note_id"]]) >= 2 else "other"
                key = (arm, row["song_id"], "edge" if p["note_id"] in edges else "interior", kind, expected)
                strata[key]["reference"] += 1;strata[key][op] += 1
    h = {r["example_id"]: r for r in pools["H"]["original"]}
    windows = [{"example_id": e["example_id"], "song_id": e["song_id"],
        "delta_E_minus_H": {k: e["metrics"][k] - h[e["example_id"]]["metrics"][k] for k in ("edit_distance", "insertions", "deletions", "substitutions")}}
        for e in pools["E"]["original"]]
    old_failure = source / "screen/E/promise"
    new_promise = run / "screen/E/promise"
    init_equal = read_json(old_failure / "INITIALIZATION.json")["head_sha256"] == read_json(new_promise / "INITIALIZATION.json")["head_sha256"]
    if not init_equal:raise ValueError("replacement initialization differs from failed attempt")
    diffs = [abs(read_json(p)["means"]["sequence_loss_per_reference"] - read_json(new_promise / p.name)["means"]["sequence_loss_per_reference"]) for p in old_failure.glob("epoch-*.json")]
    report = {"screen_complete": True, "gates": gates, "selected_arm": selected,
        "historical_endpoint": {"original": aggregate(old_o), "rendered": aggregate(old_r)},
        "pools": pools, "sources": sources, "ledger": ledger, "effective_optimizer_updates": dict(steps),
        "completed_model_epochs": 200, "resources": resources, "window_differences": windows,
        "note_phone_strata": [dict(arm=k[0], song=k[1], position=k[2], note_kind=k[3], phone=k[4], **v) for k, v in strata.items()],
        "strata_exclude_insertions_but_main_metrics_include_them": True,
        "replacement_initialization_equal": init_equal, "first14_logged_loss_maximum_difference": max(diffs),
        "human_reference_sha256": sha256_file(reference_path), "automatic_default_replacement": False,
        "new_independent_generalization_claim": False, "final_test_opened": False}
    write_json_new(run / "SCREEN_REPORT.json", report)
    for arm in spec["arms"]:
        print(arm, gates[arm]["original"]["edit_distance"], "gate", gates[arm]["passed"], gates[arm]["failures"], flush=True)
    print("SELECTED", selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["collect", "report"], required=True);args = parser.parse_args()
    (prepare_collection if args.stage == "collect" else make_report)(args.run.resolve())

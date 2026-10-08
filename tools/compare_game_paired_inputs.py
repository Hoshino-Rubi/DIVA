"""Frozen, audio-only GAME paired-input experiment. Never retries a model job."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.local.common import read, sha, write
from any2xsampa.music.game_notes import decode_regions, load_snapshot

OLD = ROOT / ".state/runs/note-game-qualification-20261002T113704Z"
COUNT = ROOT / ".state/runs/note-correspondence-20261002T125911Z/COUNT_CEILING.json"
PAIRS = ROOT / ".state/runs/dual-input-ctc-20260920T073445032166Z/audio"
IDS = ("batch-r05", "batch-n01", "batch-n05", "batch-r11", "batch-c05")
LOCK_SHA = "efd9ea98c3b2345958bde4bae1e943776921515a1d5abfe3dedd8c3051aec2c0"
COUNT_SHA = "6cfff173d284c60253fa538ab6c21ad1b15314c9c230b0e352b3b62f8b74bdf7"
SELECTED_SHA = "2f9d7d761775ab8aeffe4d5fce48cc9f29b53fc9c35689a5a6a123cee015b2e0"
CODE = (
    "src/any2xsampa/adapters/game_onnx.py",
    "src/any2xsampa/music/game_notes.py",
    "src/any2xsampa/config/game_note_v1.json",
    "src/any2xsampa/local/common.py",
    "tools/run_game_note_job.py",
    "tools/compare_game_paired_inputs.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def count_gate(reference: dict[str, int], predicted: dict[str, int]) -> dict:
    require(set(reference) == set(predicted), "Song set mismatch")
    total_ref = sum(reference.values())
    total_pred = sum(predicted.values())
    maximum = sum(min(reference[song], predicted[song]) for song in reference)
    needed = math.ceil(.95 * total_ref)
    return {
        "reference_notes": total_ref, "predicted_notes": total_pred,
        "maximum_one_to_one_matches": maximum, "matches_needed": needed,
        "recall_upper_bound": maximum / total_ref,
        "precision_upper_bound": maximum / total_pred if total_pred else 0,
        "recall_necessary_pass": maximum >= needed,
        "precision_necessary_pass": total_pred > 0 and 20 * maximum >= 19 * total_pred,
        "not_measured_accuracy": True,
    }


def request_for(audio: dict, seconds: int) -> dict:
    require(seconds in (8, 10), "Only frozen 8/10-second requests are permitted")
    require(audio["samples"] == 441000 and audio["sample_rate"] == 44100,
            "Paired input must be the frozen 10-second WAV")
    return {"audio_path": audio["path"], "audio_sha256": audio["sha256"],
            "start_sec": 0, "end_sec": seconds}


def diagnose_old(rows: list[dict]) -> list[dict]:
    import numpy as np

    diagnosis = []
    for row in rows:
        song = row["song_id"]
        job = OLD / "jobs" / ("original-" + song)
        result_path = job / "prediction.json"
        require(read(job / "EXIT.json")["status"] == "completed", "Old job incomplete: " + song)
        require(sha(result_path) == row["prediction_sha256"], "Old prediction hash mismatch: " + song)
        prediction = read(result_path)
        rawfile = job / prediction["raw_evidence"]["file"]
        require(sha(rawfile) == prediction["raw_evidence"]["sha256"], "Old raw hash mismatch: " + song)
        with np.load(rawfile, allow_pickle=False) as archive:
            raw = dict(archive)
        decoded = decode_regions(raw, start_sec=prediction["source_start_sec"],
                                 duration_sec=prediction["input"]["model_duration_sec"],
                                 timestep=prediction["timestep_sec"])
        require(decoded["notes"] == prediction["notes"] and decoded["regions"] == prediction["regions"],
                "Old raw replay mismatch: " + song)
        iterations = raw["boundary_iterations"]
        require(iterations.shape == (8, *raw["maskT"].shape) and
                np.array_equal(iterations[-1], raw["boundaries"]), "Sampling history mismatch: " + song)
        active = int(raw["maskT"].sum())
        boundaries = [int(step[0, :active].sum()) for step in iterations]
        rests = [r for r in decoded["regions"] if not r["presence"]]
        diagnosis.append({
            "song_id": song, "example_id": row["example_id"],
            "reference_count_for_reporting_only": row["reference_notes"],
            "predicted_note_count": len(decoded["notes"]), "region_count": len(decoded["regions"]),
            "rest_count": len(rests), "rest_seconds": sum(r["duration_sec"] for r in rests),
            "rest_intervals_in_original_audio": [[r["start_sec"], r["end_sec"]] for r in rests],
            "boundary_counts_after_each_of_8_samples": boundaries,
            "feature_shapes": {name: list(raw[name].shape) for name in ("x_seg", "x_est")},
            "at_least_one_rest_region": bool(rests),
            "region_count_deficit_relative_to_reference": max(0, row["reference_notes"] - len(decoded["regions"])),
            "region_count_is_not_note_alignment": True,
            "rest_pitch_scores_are_not_evidence_of_singing": True,
            "prediction_sha256": sha(result_path), "raw_sha256": sha(rawfile),
        })
    return diagnosis


def check_pair(pair_file: Path, row: dict) -> dict:
    import numpy as np
    import soundfile as sf

    pair = read(pair_file)
    song, example = row["song_id"], row["example_id"]
    require(pair["schema_version"] == "dual_input_pair_v1" and pair["example_id"] == example and
            pair["song_id"] == song and pair["split"] == "train", "Pair song/split mismatch")
    require(not pair["contains_reference_context"], "Pair includes reference conditions")
    old_request = read(OLD / "requests" / ("original-" + song + ".json"))
    require(pair["source_audio_path"] == old_request["audio_path"] and
            pair["source_audio_sha256"] == old_request["audio_sha256"], "Source audio differs")
    source = Path(pair["source_audio_path"])
    require(sha(source) == pair["source_audio_sha256"], "Original source hash mismatch")
    with sf.SoundFile(source) as f:
        rate = f.samplerate
        require(pair["source_rate"] == rate and
                pair["source_start_sample"] == round(old_request["start_sec"] * rate) and
                pair["source_end_sample_exclusive"] == round(old_request["end_sec"] * rate),
                "Original source sample interval differs")
    require(abs(pair["effective_original_start_sec"] - pair["source_start_sample"] / rate) <= 1 / rate and
            abs(pair["effective_original_end_sec"] - pair["source_end_sample_exclusive"] / rate) <= 1 / rate,
            "Paired clock differs from original source")
    sep = Path(pair["separation_record_path"])
    require(sha(sep) == pair["separation_record_sha256"], "Separation record hash mismatch")
    record = read(sep)
    clock = record["clock"]
    if record["schema_version"] == "vocal_separation_evidence_v1":
        context = record["context"]
        require(context["song_id"] == song and context["split"] == "train" and
                context["audio_sha256"] == pair["source_audio_sha256"] and
                not context["contains_labels_or_score_context"] and not record["labels_or_score_used"],
                "Separation context has labels or differs")
    elif record["schema_version"] == "dual_input_separation_v1":
        require(record["labels_received"] is False and pair["cache_mode"] == "new_context",
                "Separation has labels or changed context type")
    else:
        raise ValueError("Unreviewed separation record schema")
    require(clock["source"]["sha256"] == pair["source_audio_sha256"] and
            clock["source"]["sample_rate"] == rate and
            clock["source_start_sample"] <= pair["source_start_sample"] <
            pair["source_end_sample_exclusive"] <= clock["source_end_sample_exclusive"],
            "Separation source interval differs")
    first, last = pair["transform"]["source_start_sample"], pair["transform"]["source_end_sample_exclusive"]
    require(last - first == 441000 and pair["source_end_sample_exclusive"] - pair["source_start_sample"] == 10 * rate and
            abs(pair["effective_original_start_sec"] - (clock["effective_original_start_sec"] + first / 44100)) < 1e-5 and
            abs(pair["effective_original_end_sec"] - (clock["effective_original_start_sec"] + last / 44100)) < 1e-5 and
            clock["no_time_stretch_or_compensation"], "Pair/separation time axis mismatch")
    transform = pair["transform"]
    gain = transform["shared_gain"]
    require(0 < gain <= 1 and transform["same_gain_for_both_conditions"] and
            transform["both_full_crop_samples_preserved"] and not transform["clipping_performed"],
            "Pair transform is not matched or is clipped")
    hashes = {"pair_json": sha(pair_file), "original_source": sha(source), "separation_record": sha(sep)}
    for role in ("mixed", "vocal"):
        audio = pair["audio"][role]
        path = Path(audio["path"])
        require(path.resolve().is_relative_to(PAIRS.resolve()) and sha(path) == audio["sha256"],
                "Paired WAV hash/path mismatch: " + role)
        with sf.SoundFile(path) as f:
            require((f.samplerate, f.frames, f.channels, f.subtype) == (44100, 441000, 2, "PCM_32"),
                    "Paired WAV geometry differs")
            values = f.read(dtype="float32", always_2d=True)
        master = record[role]
        master_path = Path(master["path"])
        require(sha(master_path) == master["sha256"], "Separation master hash mismatch")
        with sf.SoundFile(master_path) as f:
            require(f.samplerate == 44100 and f.channels == 2 and f.frames >= last,
                    "Separation master geometry differs")
            f.seek(first)
            original = f.read(last - first, dtype="float32", always_2d=True)
        require(values.shape == original.shape and np.isfinite(values).all() and
                np.isfinite(original).all() and np.max(np.abs(values - original * gain)) < 2e-7,
                "Paired WAV samples differ from shared-gain master")
        hashes[role + "_wav"] = sha(path)
        hashes[role + "_master"] = sha(master_path)
    return {"song_id": song, "example_id": example, "source_interval_sec":
            [pair["effective_original_start_sec"], pair["effective_original_end_sec"]],
            "shared_gain": gain, "hashes": hashes}


def preflight(run: Path) -> None:
    require(run.resolve().is_relative_to((ROOT / ".state/runs").resolve()) and not run.exists(),
            "Use a new exclusive run under .state/runs")
    require(sha(COUNT) == COUNT_SHA and sha(OLD / "SELECTED_INPUTS.json") == SELECTED_SHA and
            sha(OLD / "model/snapshot-lock.json") == LOCK_SHA, "Frozen manifest hash mismatch")
    load_snapshot(OLD / "model")
    selected = read(OLD / "SELECTED_INPUTS.json")
    rows = read(COUNT)["rows"]
    require(len(rows) == len(IDS) == 5 and len(selected["jobs"]) == 7 and
            all(read(OLD / "jobs" / job["id"] / "EXIT.json")["status"] == "completed"
                for job in selected["jobs"]), "Old seven jobs are not all completed")
    require([row["example_id"] for row in rows] == list(IDS), "Frozen song order differs")
    diagnosis = diagnose_old(rows)
    pairs = [check_pair(PAIRS / example / "pair.json", row) for example, row in zip(IDS, rows)]
    run.mkdir(parents=True, exist_ok=False)
    write(run / "DIAGNOSIS.json", {"rows": diagnosis, "model_forwards": 0,
                                    "reference_used_to_select_or_decode": False})
    write(run / "PRECHECK.json", {"schema_version": "game_paired_precheck_v1", "pairs": pairs,
                                   "frozen_count_sha256": COUNT_SHA, "selected_sha256": SELECTED_SHA,
                                   "snapshot_lock_sha256": LOCK_SHA, "old_jobs_completed": 7,
                                   "model_forwards": 0, "optimizer_updates": 0,
                                   "pair_labels_used_to_select": False})
    (run / "requests").mkdir()
    jobs = []
    for example, pair in zip(IDS, pairs):
        data = read(PAIRS / example / "pair.json")
        for role in ("mixed", "vocal"):
            name = example + "-" + role
            request_path = run / "requests" / (name + ".json")
            write(request_path, request_for(data["audio"][role], 10))
            jobs.append({"id": name, "request_path": str(request_path),
                         "request_sha256": sha(request_path), "output": str(run / "jobs" / name),
                         "song_id": pair["song_id"], "example_id": example, "role": role})
    smoke_audio = read(PAIRS / "batch-n01/pair.json")["audio"]["vocal"]
    smoke_request = run / "requests/smoke-8.json"
    write(smoke_request, request_for(smoke_audio, 8))
    jobs.insert(0, {"id": "smoke-8", "request_path": str(smoke_request),
                    "request_sha256": sha(smoke_request), "output": str(run / "jobs/smoke-8"),
                    "song_id": "queen-road-east", "example_id": "batch-n01", "role": "vocal-smoke-only"})
    write(run / "JOBS.json", {"jobs": jobs, "smoke_excluded_from_comparison": True,
                              "maximum_encoder_calls": 11, "maximum_graph_calls": 121,
                              "no_training_download_retry_or_parameter_search": True})


def freeze(run: Path, test_xml: Path) -> None:
    require(not (run / "LAUNCH_REVIEW.json").exists(), "Review already frozen")
    precheck = read(run / "PRECHECK.json")
    suites = ET.parse(test_xml).getroot().findall("testsuite") if test_xml.is_file() else []
    require(len(precheck["pairs"]) == 5 and (run / "DIAGNOSIS.json").is_file() and
            suites and sum(int(s.attrib["tests"]) for s in suites) > 0 and
            all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == 0 for s in suites),
            "Preflight/tests have not passed")
    jobs = read(run / "JOBS.json")["jobs"]
    require(len(jobs) == 11 and len({job["output"] for job in jobs}) == 11 and
            all(sha(Path(job["request_path"])) == job["request_sha256"] for job in jobs),
            "Request budget/paths differ")
    review = {"schema_version": "game_note_launch_review_v1", "device": "cpu", "provider": "CPUExecutionProvider",
              "cpu_threads": 4, "optimizer_updates": 0, "maximum_jobs": 11,
              "maximum_encoder_calls": 11, "maximum_graph_calls_per_job": 11,
              "maximum_total_graph_calls": 121, "timeout_seconds_per_job": 180,
              "snapshot_lock_sha256": LOCK_SHA,
              "code_sha256": {name: sha(ROOT / name) for name in CODE},
              "validation": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                              "sha256": sha(path)} for path in
                             (run / "PRECHECK.json", run / "DIAGNOSIS.json", run / "JOBS.json", test_xml)],
              "jobs": jobs, "no_automatic_retry": True,
              "reviewed": {"sampling": "fixed seed17, 8 segmenter steps, yue, unchanged 0.2/2/0.2 policy",
                           "inputs": "five source-hash-checked paired PCM_32 mix/vocal WAVs; no reference conditions",
                           "loss_updates": "zero; inference only", "failure": "exclusive output, serial jobs, stop on first failure"}}
    write(run / "LAUNCH_REVIEW.json", review)


def launch(run: Path) -> None:
    review = read(run / "LAUNCH_REVIEW.json")
    from tools.run_game_note_job import verify_review

    jobs = review["jobs"]
    require(len(jobs) == 11 and review["maximum_total_graph_calls"] == 121 and
            not any(Path(job["output"]).exists() for job in jobs),
            "Job already started: never silently resume or retry")
    for pair in read(run / "PRECHECK.json")["pairs"]:
        current = check_pair(PAIRS / pair["example_id"] / "pair.json", pair)
        require(current == pair, "Pair provenance drifted since preflight")
    for job in jobs:
        verify_review(run / "LAUNCH_REVIEW.json", Path(job["request_path"]), OLD / "model", Path(job["output"]))
    (run / "jobs").mkdir(exist_ok=False)
    for index, job in enumerate(jobs):
        command = [sys.executable, "-X", "utf8", str(ROOT / "tools/run_game_note_job.py"),
                   "--review", str(run / "LAUNCH_REVIEW.json"), "--request", job["request_path"],
                   "--snapshot", str(OLD / "model"), "--output", job["output"]]
        started = time.monotonic()
        try:
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=185,
                                  env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                                       "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"})
            record = {"job_id": job["id"], "ordinal": index + 1, "command": command,
                      "exit_code": done.returncode, "elapsed_seconds": time.monotonic() - started,
                      "stdout": done.stdout, "stderr": done.stderr, "timed_out": False}
        except subprocess.TimeoutExpired as exc:
            record = {"job_id": job["id"], "ordinal": index + 1, "command": command,
                      "exit_code": None, "elapsed_seconds": time.monotonic() - started,
                      "timed_out": True, "error": str(exc)}
        write(run / (f"COMMAND_{index + 1:02d}.json"), record)
        output = Path(job["output"])
        if record["exit_code"] != 0 or not (output / "EXIT.json").is_file() or \
                read(output / "EXIT.json").get("status") != "completed" or \
                read(output / "EXIT.json").get("graph_calls") != 11:
            write(run / "STOPPED.json", {"failed_job": job["id"], "ordinal": index + 1,
                                         "failed_or_timed_out": True, "automatic_retry": False})
            raise RuntimeError("GAME paired job failed; stopped without retry: " + job["id"])
        print(f"completed {index + 1}/11 {job['id']}", flush=True)
    write(run / "LAUNCH_COMPLETED.json", {"completed_jobs": 11, "encoder_calls": 11,
                                          "graph_calls": 121, "training_updates": 0,
                                          "smoke_is_not_a_formal_pair": True})


def evaluate(run: Path) -> dict:
    from any2xsampa.adapters.game_onnx import replay_result

    require(read(run / "LAUNCH_COMPLETED.json")["graph_calls"] == 121, "Incomplete paired experiment")
    jobs = read(run / "JOBS.json")["jobs"]
    frozen = read(COUNT)["rows"]
    reference = {row["song_id"]: row["reference_notes"] for row in frozen}
    counts: dict[str, dict[str, int]] = {role: {} for role in ("mixed", "vocal")}
    rows = []
    replay_dir = run / "replay"
    replay_dir.mkdir(exist_ok=False)
    for job in jobs:
        output = Path(job["output"])
        status = read(output / "EXIT.json")
        prediction_path = output / "prediction.json"
        require(status["status"] == "completed" and status["graph_calls"] == 11 and
                sha(prediction_path) == status["prediction_sha256"], "Saved job invalid: " + job["id"])
        replay = replay_result(prediction_path, replay_dir / (job["id"] + ".json"))
        pred = read(prediction_path)
        require(replay["status"] == "passed" and replay["model_forwards"] == 0 and
                pred["model_id"] == "openvpi/GAME-1.0.3-medium-onnx" and pred["language"] == "yue",
                "Independent raw replay failed")
        if job["id"] == "smoke-8":
            continue
        pair = read(PAIRS / job["example_id"] / "pair.json")
        require(pred["input"]["audio_sha256"] == pair["audio"][job["role"]]["sha256"] and
                pred["input"]["model_samples"] == 441000 and pred["source_start_sec"] == 0 and
                abs(pred["source_end_sec"] - 10) < 1e-6,
                "Pair result time/audio binding differs")
        note_count = len(pred["notes"])
        counts[job["role"]][job["song_id"]] = note_count
        rows.append({"song_id": job["song_id"], "example_id": job["example_id"],
                     "role": job["role"], "notes": note_count, "regions": len(pred["regions"]),
                     "rests": len(pred["regions"]) - note_count,
                     "original_audio_start_sec": pair["effective_original_start_sec"],
                     "prediction_sha256": sha(prediction_path), "raw_sha256": pred["raw_evidence"]["sha256"],
                     "replay_sha256": sha(replay_dir / (job["id"] + ".json"))})
    gates = {role: count_gate(reference, counts[role]) for role in counts}
    vocal_pass = gates["vocal"]["recall_necessary_pass"] and gates["vocal"]["precision_necessary_pass"]
    result = {"schema_version": "game_pair_count_diagnostic_v1", "rows": rows, "gates": gates,
              "vocal_pair_necessary_conditions_passed": vocal_pass,
              "decision": "prepare_independent_note_truth_only" if vocal_pass else "close_fixed_game_vocal_recipe",
              "joint_pitch_onset_offset_accuracy_measured": False,
              "original_note_clock_confirmed": False, "smoke_excluded_from_metrics": True,
              "model_forwards_during_replay": 0, "no_threshold_scan_or_ref_conditioning": True}
    write(run / "RESULTS.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "freeze", "launch", "evaluate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--test-xml", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.phase == "preflight":
        preflight(run)
    elif args.phase == "freeze":
        require(args.test_xml is not None and args.test_xml.resolve().is_relative_to(run),
                "Frozen tests must be in this run")
        freeze(run, args.test_xml.resolve())
    elif args.phase == "launch":
        launch(run)
    else:
        print(json.dumps(evaluate(run)["gates"], ensure_ascii=False))


if __name__ == "__main__":
    main()

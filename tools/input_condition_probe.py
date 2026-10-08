"""Fixed x/v/h input probe for the singing-domain experiment.

This tool deliberately has no training path.  It builds a new, hash-bound run
from the already verified original/separated pairs and evaluates all three
input conditions with the existing held-song full-tail checkpoints.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.adapters.candidate_inference import serial_candidate_job
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
from any2xsampa.adapters.singing_domain_inference import load_domain_model
from any2xsampa.training.dual_input import validate_pair
from any2xsampa.training.limited_data_artifacts import (
    read_json,
    sha256_file,
    verify_file,
    write_json_new,
)
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.singing_domain import validate_audio_row


HISTORICAL_RUN = ROOT / ".state/runs/singing-domain-20260920T095918143867Z"
PAIR_RUN = ROOT / ".state/runs/dual-input-ctc-20260920T073445032166Z"
TRAIN_SONGS = (
    "freedom-flower",
    "half-moon-night-song",
    "like-you",
    "promise",
    "queen-road-east",
)
CONDITIONS = ("x", "v", "h")


def _require_new_run(run: Path) -> None:
    run = run.resolve()
    if not run.is_relative_to(ROOT / ".state/runs"):
        raise ValueError("input probe output must be under .state/runs")
    if run.exists():
        raise FileExistsError(f"refusing to overwrite input probe run: {run}")


def _write_wav(path: Path, samples: Any) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("generated audio must be finite stereo float32")
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak >= 1.0:
        raise ValueError(f"fixed blend would clip: peak={peak}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        sf.write(stream, values, 44100, subtype="PCM_32", format="WAV")
    with sf.SoundFile(path) as stream:
        if stream.samplerate != 44100 or stream.channels != 2 or stream.frames != values.shape[0]:
            raise ValueError("generated WAV geometry changed after write")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sample_rate": 44100,
        "channels": 2,
        "samples": int(values.shape[0]),
        "subtype": "PCM_32",
        "peak_abs": peak,
    }


def _load_pairs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load the 20 original windows and verify their existing pair bindings."""
    references = read_json(HISTORICAL_RUN / "REFERENCES.json")["rows"]
    originals = [row for row in references if row["split"] == "train" and row["audio_role"] == "original"]
    if len(originals) != 20 or {row["song_id"] for row in originals} != set(TRAIN_SONGS):
        raise ValueError("historical run does not contain the fixed 20 original training windows")
    pair_payload = read_json(PAIR_RUN / "PAIRS.json")
    pair_index = {row["example_id"]: row for row in pair_payload["entries"]}
    old_requests_payload = read_json(PAIR_RUN / "AUDIO_REQUESTS.json")
    old_requests = {row["example_id"]: row for row in old_requests_payload["requests"]}
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for reference in sorted(originals, key=lambda row: row["example_id"]):
        example_id = reference["example_id"]
        if example_id not in pair_index or example_id not in old_requests:
            raise ValueError(f"missing verified pair for {example_id}")
        index_row = pair_index[example_id]
        pair_path = Path(index_row["path"])
        verify_file(pair_path, index_row["sha256"])
        pair = read_json(pair_path)
        validate_pair(old_requests[example_id], pair)
        if pair["source_audio_sha256"] != reference["audio_sha256"]:
            raise ValueError(f"pair/reference source hash mismatch: {example_id}")
        if abs(pair["effective_original_start_sec"] - reference["start_sec"]) > 1 / 44100:
            raise ValueError(f"pair/reference start clock mismatch: {example_id}")
        if abs(pair["effective_original_end_sec"] - reference["end_sec"]) > 1 / 44100:
            raise ValueError(f"pair/reference end clock mismatch: {example_id}")
        mixed = pair["audio"]["mixed"]
        vocal = pair["audio"]["vocal"]
        for audio in (mixed, vocal):
            verify_file(Path(audio["path"]), audio["sha256"])
            source_hashes[str(Path(audio["path"]).resolve())] = audio["sha256"]
        rows.append(
            {
                "window_id": example_id,
                "song_id": reference["song_id"],
                "reference_phones": reference["target_ipa"],
                "reference_sha256": sha256_file(HISTORICAL_RUN / "REFERENCES.json"),
                "source_audio_sha256": reference["audio_sha256"],
                "source_start_sec": reference["start_sec"],
                "source_end_sec": reference["end_sec"],
                "pair_path": str(pair_path.resolve()),
                "pair_sha256": index_row["sha256"],
                "mixed": mixed,
                "vocal": vocal,
                "transform": pair["transform"],
            }
        )
    return rows, source_hashes


def prepare(run: Path) -> None:
    """Create the immutable x/v/h audio condition manifest."""
    _require_new_run(run)
    run.mkdir(parents=True)
    rows, source_hashes = _load_pairs()
    import numpy as np
    import soundfile as sf

    requests: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for row in rows:
        mixed, mixed_rate = sf.read(row["mixed"]["path"], dtype="float32", always_2d=True)
        vocal, vocal_rate = sf.read(row["vocal"]["path"], dtype="float32", always_2d=True)
        if mixed_rate != 44100 or vocal_rate != 44100 or mixed.shape != vocal.shape:
            raise ValueError(f"pair geometry mismatch: {row['window_id']}")
        if not np.isfinite(mixed).all() or not np.isfinite(vocal).all():
            raise ValueError(f"non-finite pair audio: {row['window_id']}")
        blend = ((mixed.astype(np.float64) + vocal.astype(np.float64)) * 0.5).astype(np.float32)
        h_audio = _write_wav(run / "audio" / row["window_id"] / "h-pcm32.wav", blend)
        generated.append({"window_id": row["window_id"], "audio": h_audio})
        condition_audio = {"x": row["mixed"], "v": row["vocal"], "h": h_audio}
        for condition in CONDITIONS:
            audio = condition_audio[condition]
            condition_id = f"{row['window_id']}--{condition}"
            role = "separated_vocal" if condition == "v" else "original"
            request = {
                "example_id": condition_id,
                "song_id": row["song_id"],
                "split": "train",
                "audio_role": role,
                "audio_path": audio["path"],
                "audio_sha256": audio["sha256"],
                "start_sec": 0.0,
                "end_sec": 10.0,
            }
            validate_audio_row(request)
            requests.append(request)
            conditions.append(
                {
                    "condition_id": condition_id,
                    "window_id": row["window_id"],
                    "song_id": row["song_id"],
                    "condition": condition,
                    "audio_role": role,
                    "audio_sha256": audio["sha256"],
                    "audio_path": audio["path"],
                    "pair_path": row["pair_path"],
                    "pair_sha256": row["pair_sha256"],
                    "formula": "x" if condition == "x" else "v" if condition == "v" else "0.5*x+0.5*v",
                    "fixed_input_probe": True,
                }
            )
        references.append(
            {
                "window_id": row["window_id"],
                "song_id": row["song_id"],
                "target_ipa": row["reference_phones"],
                "target_sha256": sha256_file(HISTORICAL_RUN / "REFERENCES.json"),
                "source_audio_sha256": row["source_audio_sha256"],
                "source_start_sec": row["source_start_sec"],
                "source_end_sec": row["source_end_sec"],
            }
        )
    snapshot = read_json(HISTORICAL_RUN / "FREEZE.json")["model_snapshot"]
    freeze = {
        "schema_version": "input_condition_probe_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_run": str(HISTORICAL_RUN.resolve()),
        "historical_policy_sha256": sha256_file(HISTORICAL_RUN / "POLICY.json"),
        "historical_cache_sha256": sha256_file(HISTORICAL_RUN / "CACHE.json"),
        "historical_references_sha256": sha256_file(HISTORICAL_RUN / "REFERENCES.json"),
        "pair_run": str(PAIR_RUN.resolve()),
        "pair_policy_sha256": sha256_file(PAIR_RUN / "POLICY.json"),
        "pair_manifest_sha256": sha256_file(PAIR_RUN / "PAIRS.json"),
        "model_snapshot": snapshot,
        "conditions": list(CONDITIONS),
        "blend_formula": "h=0.5*x+0.5*v",
        "blend_weight": 0.5,
        "training_started": False,
        "development_or_test_audio_opened": False,
        "final_test_predictions_allowed": False,
    }
    write_json_new(run / "FREEZE.json", freeze)
    write_json_new(run / "INPUT_REQUESTS.json", {"schema_version": "input_probe_audio_requests_v1", "rows": requests, "labels_received": False})
    write_json_new(run / "CONDITIONS.json", {"schema_version": "input_probe_conditions_v1", "rows": conditions, "generated": generated})
    write_json_new(run / "EVALUATION_REFERENCES.json", {"schema_version": "input_probe_references_v1", "rows": references})
    write_json_new(
        run / "SOURCE_BINDINGS.json",
        {
            "schema_version": "input_probe_source_bindings_v1",
            "sources": {**source_hashes, str((HISTORICAL_RUN / "REFERENCES.json").resolve()): sha256_file(HISTORICAL_RUN / "REFERENCES.json")},
            "old_checkpoint_dirs": {
                song: {
                    "result": str((HISTORICAL_RUN / "architecture" / "full" / song / "RESULT.json").resolve()),
                    "result_sha256": sha256_file(HISTORICAL_RUN / "architecture" / "full" / song / "RESULT.json"),
                }
                for song in TRAIN_SONGS
            },
        },
    )
    print(json.dumps({"run": str(run.resolve()), "windows": len(rows), "requests": len(requests), "conditions": list(CONDITIONS)}, ensure_ascii=False))


def cache(run: Path) -> None:
    """Run the frozen acoustic encoder once per condition, without labels."""
    run = run.resolve()
    freeze = read_json(run / "FREEZE.json")
    requests = read_json(run / "INPUT_REQUESTS.json")["rows"]
    if len(requests) != 60:
        raise ValueError("input probe requires exactly 60 x/v/h requests")
    from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime

    started = time.monotonic()
    entries: list[dict[str, Any]] = []
    runtime_python = Path(read_json(HISTORICAL_RUN / "FREEZE.json")["runtime_python"])
    if Path(sys.executable).resolve() != runtime_python.resolve():
        raise RuntimeError(f"run cache with the isolated Python 3.12 runtime: {runtime_python}")
    with serial_candidate_job(), ProcessResourceMonitor(run / "cache-monitor", timeout_seconds=1800), LockedAudioCache(max_bytes=16 * 1024**2, max_sources=12) as audio_cache:
        runtime = SingingDomainRuntime(Path(freeze["model_snapshot"]), run / "runtime-cache", audio_cache=audio_cache)
        write_json_new(run / "ARCHITECTURE.json", runtime.identity)
        for index, request in enumerate(requests, 1):
            entry = runtime.save_prefix(request, run / "cache" / request["example_id"])
            entry["condition_id"] = request["example_id"]
            entries.append(entry)
            if index % 10 == 0:
                print(f"cached {index}/{len(requests)}", flush=True)
        cache_index = {
            "schema_version": "input_probe_cache_v1",
            "architecture": runtime.identity,
            "input_requests_sha256": sha256_file(run / "INPUT_REQUESTS.json"),
            "entries": entries,
            "labels_received": False,
            "all_parameters_frozen": True,
            "peak_reserved_bytes": runtime.torch.cuda.max_memory_reserved(),
            "elapsed_seconds": time.monotonic() - started,
            "audio_cache": audio_cache.stats(),
        }
        write_json_new(run / "CACHE.json", cache_index)
    print(json.dumps({"run": str(run), "entries": len(entries), "elapsed_seconds": cache_index["elapsed_seconds"]}, ensure_ascii=False))


def _edit_metrics(reference: list[str], prediction: list[str]) -> dict[str, Any]:
    return phone_metrics([(reference, prediction)])


def _condition_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    return phone_metrics([(row["reference"], row["prediction"]) for row in details])


def evaluate(run: Path) -> None:
    """Evaluate every held-song checkpoint on x, v and h."""
    run = run.resolve()
    if (run / 'RESULT.json').exists():
        raise FileExistsError('completed input probe is immutable')
    attempt = run / 'evaluation-attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt.mkdir(parents=True, exist_ok=False)
    freeze = read_json(run / "FREEZE.json")
    if freeze["development_or_test_audio_opened"] or freeze["final_test_predictions_allowed"]:
        raise ValueError("input probe is restricted to training-song held windows")
    references = {row["window_id"]: row for row in read_json(run / "EVALUATION_REFERENCES.json")["rows"]}
    conditions = {row["condition_id"]: row for row in read_json(run / "CONDITIONS.json")["rows"]}
    cache = read_json(run / "CACHE.json")
    verify_file(run / "INPUT_REQUESTS.json", cache["input_requests_sha256"])
    entries = {row["condition_id"]: row for row in cache["entries"]}
    if set(entries) != set(conditions) or len(entries) != 60:
        raise ValueError("input probe cache coverage changed")
    from safetensors.torch import load_file, save_file

    summaries: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    by_condition_song: dict[str, dict[str, list[dict[str, Any]]]] = {condition: defaultdict(list) for condition in CONDITIONS}
    x_identity: list[dict[str, Any]] = []
    label_map: list[str] | None = None
    started = time.monotonic()
    process_peak_reserved = 0
    with serial_candidate_job(), ProcessResourceMonitor(attempt / "monitor", timeout_seconds=1800):
        for held_song in TRAIN_SONGS:
            job = HISTORICAL_RUN / "architecture" / "full" / held_song
            if not (job / "RESULT.json").exists():
                raise ValueError(f"missing held-song checkpoint: {held_song}")
            fold_windows = [row for row in references.values() if row["song_id"] == held_song]
            if not fold_windows:
                raise ValueError(f"held fold has no windows: {held_song}")
            rt, tail, checkpoint = load_domain_model(HISTORICAL_RUN, job, attempt / "model-runtime" / held_song)
            torch = rt.torch
            labels = [phone for phone, index in sorted(rt.vocabulary.items(), key=lambda item: item[1])]
            if label_map is None:
                label_map = labels
            elif label_map != labels:
                raise ValueError("vocabulary changed between held-song checkpoints")
            tail.eval()
            try:
                for reference in sorted(fold_windows, key=lambda row: row["window_id"]):
                    window_id = reference["window_id"]
                    predictions_for_x: list[str] | None = None
                    for condition in CONDITIONS:
                        condition_id = f"{window_id}--{condition}"
                        entry = entries[condition_id]
                        verify_file(Path(entry["path"]), entry["sha256"])
                        tensors = load_file(entry["path"], device="cpu")
                        prefix = tensors["prefix_features"].unsqueeze(0).to("cuda")
                        lengths = torch.tensor([prefix.shape[1]], device="cuda", dtype=torch.long)
                        before = time.monotonic()
                        with torch.no_grad():
                            logits, _ = tail(prefix, lengths)
                        values = logits[0].float().cpu().contiguous()
                        raw_ids = values.argmax(dim=-1).tolist()
                        collapsed_ids = ctc_collapse(raw_ids)
                        prediction = [labels[index] for index in collapsed_ids]
                        detail = {
                            "condition_id": condition_id,
                            "window_id": window_id,
                            "song_id": held_song,
                            "held_song": held_song,
                            "condition": condition,
                            "reference": reference["target_ipa"],
                            "prediction": prediction,
                            "prediction_ids": collapsed_ids,
                            "metrics": _edit_metrics(reference["target_ipa"], prediction),
                            "cache_path": entry["path"],
                            "cache_sha256": entry["sha256"],
                            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                            "elapsed_seconds": time.monotonic() - before,
                        }
                        summaries[condition].append(detail)
                        by_condition_song[condition][held_song].append(detail)
                        if condition == "x":
                            predictions_for_x = prediction
                        logits_dir = attempt / "logits" / held_song
                        logits_dir.mkdir(parents=True, exist_ok=True)
                        logits_path = logits_dir / f"{condition_id}.safetensors"
                        save_file({"frame_logits": values}, str(logits_path))
                        detail["logits_path"] = str(logits_path.resolve())
                        detail["logits_sha256"] = sha256_file(logits_path)
                    old_result = read_json(job / "RESULT.json")
                    old_example = next((row for row in old_result["final_validation"]["examples"] if row["example_id"] == window_id), None)
                    if old_example is not None:
                        x_identity.append(
                            {
                                "window_id": window_id,
                                "held_song": held_song,
                                "historical_prediction": old_example["prediction"],
                                "matched_x_prediction": predictions_for_x,
                                "sequence_equal": old_example["prediction"] == predictions_for_x,
                                "historical_metrics": old_example["metrics"],
                                "matched_x_metrics": next(row["metrics"] for row in summaries["x"] if row["window_id"] == window_id),
                            }
                        )
            finally:
                process_peak_reserved = max(process_peak_reserved, torch.cuda.max_memory_reserved())
                del tail, rt
                gc.collect()
                torch.cuda.empty_cache()
    report: dict[str, Any] = {
        "schema_version": "input_condition_probe_result_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run),
        "conditions": list(CONDITIONS),
        "windows": len(references),
        "reference_phones": sum(len(row["target_ipa"]) for row in references.values()),
        "condition_metrics": {condition: _condition_metrics(rows) for condition, rows in summaries.items()},
        "per_song": {
            condition: {song: _condition_metrics(rows) for song, rows in sorted(song_rows.items())}
            for condition, song_rows in by_condition_song.items()
        },
        "details": {condition: rows for condition, rows in summaries.items()},
        "matched_x_identity": {
            "windows": len(x_identity),
            "sequence_equal": sum(row["sequence_equal"] for row in x_identity),
            "all_equal": bool(x_identity) and all(row["sequence_equal"] for row in x_identity),
            "rows": x_identity,
        },
        "input_policy": {
            "h_formula": "0.5*x+0.5*v",
            "coefficient_scan": False,
            "per_window_input_selection": False,
            "reference_selection": False,
            "new_training_started": False,
            "shalong_opened": False,
            "final_test_opened": False,
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_reserved_bytes": process_peak_reserved,
        "resource_monitor_path": str((attempt / 'monitor/resource-monitor.json').resolve()),
        "resource_monitor_sha256": sha256_file(attempt / 'monitor/resource-monitor.json'),
    }
    write_json_new(run / "RESULT.json", report)
    lines = [
        "# Fixed Input Condition Probe",
        "",
        "本报告只使用五首训练歌的 20 个留歌窗口；未打开沙龙或最终测试。",
        "",
        "| condition | PER | S | D | I | refs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metric = report["condition_metrics"][condition]
        lines.append(f"| {condition} | {metric['per']:.6f} | {metric['substitutions']} | {metric['deletions']} | {metric['insertions']} | {metric['reference_phones']} |")
    lines.extend(
        [
            "",
            f"匹配 x 与历史 x 的整段预测完全相同窗口：{report['matched_x_identity']['sequence_equal']}/{report['matched_x_identity']['windows']}。",
            "",
            "h 固定为 `0.5*x+0.5*v`，没有扫描混合比例，也没有按参考逐窗口挑输入。",
        ]
    )
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "metrics": report["condition_metrics"], "matched_x": report["matched_x_identity"]["sequence_equal"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "cache", "evaluate"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run.resolve())
    elif args.command == "cache":
        cache(args.run.resolve())
    else:
        evaluate(args.run.resolve())


if __name__ == "__main__":
    main()

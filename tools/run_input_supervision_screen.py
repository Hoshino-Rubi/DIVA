"""Serially run the predeclared V/MV five-fold screen."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.input_supervision_evaluation import compare_screen
from any2xsampa.training.input_supervision import load_manifest
from any2xsampa.training.input_supervision_training import require_screen_budget, require_code_review

SONGS = ("freedom-flower", "half-moon-night-song", "like-you", "promise", "queen-road-east")

def run_job(run: Path, variant: str, song: str, output: Path) -> dict:
    freeze, _, _ = load_manifest(run)
    if f'{variant}-{song}' in freeze.get('unavailable_screen_jobs', []):
        raise ValueError('previous failed job is unavailable; no retry authorized')
    if output.exists():
        result = output / "RESULT.json"
        if result.exists():
            value = read_json(result)
            if (value.get('schema_version') == 'input_supervision_job_v2' and value.get("complete") is True
                    and value.get("variant") == variant and value.get("held_song") == song
                    and value.get('freeze_sha256') == sha256_file(run/'FREEZE.json')
                    and value.get('policy_sha256') == sha256_file(run/'POLICY.json')):
                verify_file(Path(value["checkpoint_path"]), value["checkpoint_sha256"]); return value
        raise RuntimeError(f"incomplete job requires inspection: {output}")
    require_screen_budget(run)
    require_code_review(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    python = Path(read_json(run / "FREEZE.json")["runtime_python"])
    command = [str(python), "-X", "utf8", "-B", str(ROOT / "tools/train_input_supervision.py"), "--run", str(run), "--variant", variant, "--held-song", song, "--seed", "17", "--output", str(output)]
    logdir = run / "commands"; logdir.mkdir(exist_ok=True)
    name = f"{variant}-{song}"
    write_json_new(logdir / f"{name}.command.json", {"argv": command, "started_at_utc": datetime.now(timezone.utc).isoformat(), "policy_sha256": sha256_file(run / "POLICY.json")})
    started = time.monotonic()
    log_path = logdir / f"{name}.log"
    print('START', name, flush=True)
    exit_code = None
    try:
        with log_path.open("x", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=1560, shell=False, env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            exit_code = process.returncode
    finally:
        write_json_new(logdir / f"{name}.exit.json", {"exit_code": exit_code, "elapsed_seconds": time.monotonic() - started})
    if exit_code:
        raise RuntimeError(f"{name} failed; see {log_path}")
    print('DONE', name, flush=True)
    return read_json(output / "RESULT.json")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True); parser.add_argument("--variant", choices=["V", "MV"], required=True); parser.add_argument("--song", choices=SONGS); args = parser.parse_args(); run = args.run.resolve()
    freeze, _, _ = load_manifest(run)
    unavailable = set(freeze.get('unavailable_screen_jobs', []))
    selected = [args.song] if args.song else [s for s in SONGS if f'{args.variant}-{s}' not in unavailable]
    results = [run_job(run, args.variant, song, run / "screen" / args.variant / song) for song in selected]
    destination = run / (f'{args.variant}-{args.song}-FOLD.json' if args.song else f'{args.variant}-SCREEN.json')
    value = {"variant": args.variant, "folds": [{"held_song": r["held_song"], "checkpoint_sha256": r["checkpoint_sha256"], "x": r["validation_x"]["metrics"], "v": r["validation_v"]["metrics"]} for r in results], "all_folds_reported": len(results) == 5, "unavailable_jobs": sorted(unavailable), "new_training_only": True, "final_test_opened": False}
    if destination.exists():
        if read_json(destination) != value: raise ValueError('completed summary changed')
    else: write_json_new(destination, value)

if __name__ == "__main__": main()

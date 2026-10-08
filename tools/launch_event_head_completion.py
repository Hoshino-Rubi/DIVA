"""Run only the two approved jobs serially; eleven total starts, no retries."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from event_head_completion_common import load_completion, check_attempt
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, sha256_file


def launch(run: Path) -> None:
    from any2xsampa.adapters.phoneticxeus_monitor import query_total_gpu_bytes
    for held in ("promise", "queen-road-east"):
        auth, source, freeze, spec, _ = load_completion(run, "CODE_REVIEW_training.json")
        ordinal = check_attempt(run, held, auth)
        probes = []
        for i in range(3):
            value = query_total_gpu_bytes();probes.append(value)
            if value > 12 * 1024**3:
                raise RuntimeError("insufficient stable launch headroom; no quality job started")
            if i < 2:time.sleep(15)
        key = "E--" + held
        command = [freeze["python"], "-X", "utf8", str(ROOT / "tools/train_event_head_completion.py"), "--run", str(run), "--held", held]
        write_json_new(run / "commands" / (key + ".command.json"), {
            "command": command, "quality_start_ordinal": ordinal, "gpu_preflight_bytes": probes,
            "preflight_span_seconds_at_least": 30, "preflight_admission_gib": 12,
            "runtime_card_cap_gib": 14, "review_sha256": sha256_file(run / "CODE_REVIEW_training.json"),
            "authorization_sha256": sha256_file(run / "AUTHORIZATION.json"),
            "counts_even_on_failure": True, "automatic_retry": False,
        })
        started = time.monotonic();code = None;error = None
        try:
            with (run / "commands" / (key + ".log")).open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1550,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
            code = result.returncode
        except Exception as exc:
            error = repr(exc)
        finally:
            write_json_new(run / "commands" / (key + ".exit.json"), {"exit_code": code, "error": error, "elapsed_seconds": time.monotonic() - started})
        if error or code != 0:
            raise RuntimeError(key + " failed; no automatic retry")
        print("APPROVED JOB COMPLETE", key, "global start", ordinal, flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);launch(p.parse_args().run.resolve())

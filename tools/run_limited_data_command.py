from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import canonical_json, sha256_file, sha256_object, write_json_new  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one bounded command with immutable logs.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT / ".state/runs"):
        parser.error("logs must remain under .state/runs")
    if Path(args.name).name != args.name or not args.name or args.timeout <= 0:
        parser.error("invalid command name or timeout")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    logs = root / "commands"
    logs.mkdir(parents=True, exist_ok=True)
    metadata_path = logs / f"{args.name}.command.json"
    paths = [logs / f"{args.name}.{suffix}" for suffix in ("stdout.log", "stderr.log", "started.json", "command.json")]
    if any(path.exists() for path in paths):
        parser.error("refusing to overwrite existing command evidence")
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "NUMBA_NUM_THREADS": "4", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NO_PROXY": "", "no_proxy": "", "PIP_CONFIG_FILE": os.devnull,
        "NUMBA_CACHE_DIR": str(root / "runtime-cache/numba"),
        "LIBROSA_CACHE_DIR": str(root / "runtime-cache/librosa"),
        "JOBLIB_TEMP_FOLDER": str(root / "runtime-cache/joblib"),
    })
    environment.pop("PIP_EXTRA_INDEX_URL", None)
    started = datetime.now(timezone.utc).isoformat()
    record = {"argv": command, "cwd": str(ROOT), "started_at_utc": started, "timeout_seconds": args.timeout, "cpu_threads": 4}
    files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "src").rglob("*.json")),
        *sorted((ROOT / "tools").glob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
        *sorted((ROOT / "tests/fixtures").rglob("*.json")),
        *sorted((ROOT / "tools/requirements").glob("*.txt")),
        ROOT / "pyproject.toml",
    ]
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}
    source_path = logs / f"{args.name}.source-files.json"
    write_json_new(source_path, {
        "schema_version": "command_source_fingerprints_v1",
        "recorded_before_launch": True, "files": source_hashes,
        "source_tree_sha256": sha256_object(source_hashes),
    })
    record["source_fingerprints_path"] = str(source_path)
    record["source_fingerprints_sha256"] = sha256_file(source_path)
    write_json_new(paths[2], record)
    before = time.monotonic()
    timed_out = False
    process = None
    startup_error = None
    with paths[0].open("x", encoding="utf-8") as stdout, paths[1].open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, shell=False)
        except OSError as exc:
            startup_error = f"{type(exc).__name__}: {exc}"
            stderr.write(startup_error + "\n")
            return_code = 127
        try:
            if process is not None:
                return_code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False, shell=False)
            else:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return_code = 124
    record.update({
        "pid": process.pid if process is not None else None,
        "startup_error": startup_error,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - before, 6),
        "return_code": return_code,
        "timed_out": timed_out,
        "stdout_path": str(paths[0]),
        "stderr_path": str(paths[1]),
    })
    write_json_new(metadata_path, record)
    print(canonical_json(record))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

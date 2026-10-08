"""One-shot serial stage launch, with accounting before model startup."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import subprocess
import time
from input_native_probe_common import ROOT, verify_review, check_stage, LIMITS
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new


def launch(run: Path, stage: str) -> None:
    freeze, _ = verify_review(run)
    tasks = read_json(run / 'tasks' / (stage + '.json'))['tasks']
    check_stage(run, stage, tasks)
    from any2xsampa.adapters.phoneticxeus_monitor import query_total_gpu_bytes
    probes = []
    try:
        for i in range(3):
            used = query_total_gpu_bytes(); probes.append(used)
            if used > 14 * 1024**3:
                raise RuntimeError('card memory budget exceeded')
    except Exception as exc:
        write_json_new(run / 'commands' / (stage + '.preflight-failed.json'),
            {'probes': probes, 'error': type(exc).__name__ + ': ' + str(exc), 'model_launched': False})
        raise
    write_json_new(run / 'commands' / (stage + '.gpu-preflight.json'), {'three_queries_passed': True, 'used_bytes': probes})
    command = [freeze['python']['path'], '-X', 'utf8', str(ROOT / 'tools/run_input_native_probe.py'), '--run', str(run), '--stage', stage]
    write_json_new(run / 'commands' / (stage + '.command.json'), {'command': command,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'planned_forwards': len(tasks), 'failed_launch_counts': True})
    before = time.monotonic(); code = None; error = None
    try:
        with (run / 'commands' / (stage + '.log')).open('x', encoding='utf-8') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1900,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                     'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
            code = process.returncode
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        write_json_new(run / 'commands' / (stage + '.exit.json'),
            {'exit_code': code, 'error': error, 'elapsed_seconds': time.monotonic() - before})
    if code:
        raise RuntimeError(f'{stage} failed ({code}); no automatic retry')
    print(stage, 'completed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', choices=list(LIMITS), required=True)
    args = p.parse_args(); launch(args.run.resolve(), args.stage)

"""Exactly thirteen fixed serial model slots; failures stop the sequence."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import subprocess
import time
from speed_invariance_common import ROOT, protocol, slots, read_json, write_json_new, sha256_file, stable_memory


def check_slot(run: Path, slot: str) -> None:
    sequence = slots()
    if slot not in sequence: raise ValueError('unplanned launch')
    if (run / 'commands' / (slot + '.json')).exists() or (run / 'jobs' / slot / 'START.json').exists():
        raise ValueError('slot already consumed; no retry')
    for previous in sequence[:sequence.index(slot)]:
        exit_path = run / 'commands' / (previous + '.exit.json')
        if not exit_path.exists() or read_json(exit_path)['exit_code'] != 0:
            raise ValueError('prior job incomplete/failed; cannot advance')
        if read_json(run / 'jobs' / previous / 'RESULT.json')['complete'] is not True:
            raise ValueError('prior artifacts incomplete')
    if len(list((run / 'commands').glob('*.json'))) and len(list((run / 'commands').glob('*.pid.json'))) >= 13:
        raise ValueError('model launch cap exhausted')


def launch(run: Path, slot: str) -> int:
    freeze = protocol(run, reviewed=True); check_slot(run, slot)
    probe = stable_memory(slot.startswith('encode'))
    worker = ROOT / 'tools/run_speed_worker.py'
    command = [freeze['python'], '-X', 'utf8', str(worker), '--run', str(run.resolve()), '--slot', slot]
    path = run / 'commands' / (slot + '.json')
    write_json_new(path, {'command': command, 'slot': slot, 'worker_sha256': sha256_file(worker),
        'review_sha256': sha256_file(run / 'CODE_REVIEW.json'), 'gpu_preflight': probe, 'retry': False})
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    start = time.monotonic(); code = -1; reason = None
    try:
        with (run / 'commands' / (slot + '.log')).open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            write_json_new(run / 'commands' / (slot + '.pid.json'), {'pid': process.pid})
            try: code = process.wait(timeout=1530)
            except subprocess.TimeoutExpired:
                process.kill(); code = process.wait(); reason = 'only_own_child_killed_after_timeout'
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
    write_json_new(run / 'commands' / (slot + '.exit.json'), {'exit_code': code, 'reason': reason,
        'elapsed_seconds': time.monotonic() - start, 'automatic_retry': False, 'command_sha256': sha256_file(path)})
    return code


def execute(run: Path) -> None:
    from validate_speed_cache import accept
    from finish_speed_invariance import finish
    protocol(run, reviewed=True)
    write_json_new(run / 'EXECUTION_START.json', {'pid': os.getpid(), 'slots': slots(), 'no_auto_retry': True})
    for slot in slots():
        print('Launching', slot, flush=True)
        if launch(run, slot) != 0: raise RuntimeError('job failed; sequence stopped: ' + slot)
        if slot == 'encode-replay': accept(run)
    finish(run)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    try: execute(a.run.resolve())
    except Exception as exc:
        target = a.run / 'EXECUTION_STOPPED.json'
        if not target.exists(): write_json_new(target, {'error_type': type(exc).__name__, 'error': str(exc), 'no_auto_retry': True})
        raise

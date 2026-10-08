"""Single accounted model-worker entry. Serial, offline, no retry."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
from speed_invariance_common import protocol, slots, policy, read_json, write_json_new, sha256_file, Ledger


def caps_for(run: Path, slot: str) -> dict:
    if slot in ('encode', 'encode-replay'):
        return {'encoder': 45 if slot == 'encode' else 10}
    if slot == 'head-replay': return {'head': 298}
    arm, song = slot.split('--')
    if arm not in ('C', 'A') or song not in policy()['per_fold']: raise ValueError('unknown training slot')
    queries = read_json(run / 'folds' / song / 'QUERIES.json')
    f = policy()['per_fold'][song]
    return {'head': f['training_requests_per_arm'] + 1 + len(queries['held']) + len(queries['seen_original']),
        'optimizer': f['optimizer_steps_per_arm']}


def worker(run: Path, slot: str) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline, configure_cuda
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    import importlib.metadata as metadata
    if slot not in slots(): raise ValueError('unplanned GPU process')
    out = run / 'jobs' / slot
    command_path = run / 'commands' / (slot + '.json'); command = read_json(command_path)
    if command['slot'] != slot or command['worker_sha256'] != sha256_file(Path(__file__)):
        raise ValueError('unaccounted direct worker launch')
    write_json_new(out / 'START.json', {'slot': slot, 'pid': os.getpid(), 'command_sha256': sha256_file(command_path)})
    before = time.monotonic(); enforce_offline(out / 'offline')
    freeze = protocol(run, reviewed=True)
    versions = {name: metadata.version(name) for name in ('torch', 'torchaudio', 'numpy', 'scipy', 'safetensors')}
    if versions['scipy'] != '1.15.3' or versions['numpy'] != '2.2.6': raise ValueError('fixed signal runtime changed')
    ledger = Ledger(out / 'attempts.jsonl', caps_for(run, slot))
    try:
        with serial_candidate_job(), ProcessResourceMonitor(out / 'monitor', timeout_seconds=1500):
            if slot in ('encode', 'encode-replay'):
                from run_speed_encoder import encode
                result = encode(run, slot, freeze, ledger)
            else:
                from run_speed_head import train, replay
                torch = configure_cuda()
                result = replay(run, freeze, ledger, torch) if slot == 'head-replay' else train(run, slot, freeze, ledger, torch)
            if ledger.counts != ledger.caps: raise ValueError('incomplete fixed request/update ledger')
            result.update(attempt_counts=ledger.counts, elapsed_seconds=time.monotonic() - before,
                runtime_versions=versions, python=sys.version, slot=slot, review_sha256=sha256_file(run / 'CODE_REVIEW.json'))
            write_json_new(out / 'RESULT.json', result)
    finally:
        ledger.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--slot', choices=slots(), required=True)
    a = p.parse_args()
    try: worker(a.run, a.slot)
    except Exception as exc:
        target = a.run / 'jobs' / a.slot / 'FAILURE.json'
        if not target.exists(): write_json_new(target, {'error_type': type(exc).__name__, 'error': str(exc), 'automatic_retry': False})
        raise

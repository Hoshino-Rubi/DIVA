"""Local immutable protocol for the single authorized acoustic-memory screen."""
from __future__ import annotations
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, sha256_file, sha256_object, verify_file

RESEARCH = ROOT / '.state/runs/post-event-research-20260925T052926Z'
OLD = ROOT / '.state/runs/event-head-comparison-20260924T101019Z'
COMPLETED = ROOT / '.state/runs/event-head-completion-20260924T162858Z'
REFERENCE = ROOT / '.state/runs/original-adaptation-20260919T012000/reviewed-training-reference.json'
POLICY = {
    'schema': 'original_soft_acoustic_memory_v1', 'k': 16, 'fusion_weight': .25,
    'temperature': 'median_nearest_other_training_song_squared_distance_floor_1e-6',
    'shuffle_seed': 17, 'classes': 428, 'blank_id': 0, 'feature': 'encoder_features',
    'cache_workers': 2, 'head_requests_per_worker': 229, 'total_head_requests': 458,
    'head_timeout_seconds': 1500, 'cpu_stage_timeout_seconds': 1500,
    'cpu_threads': 4, 'gpu_reserved_limit_gib': 7, 'card_limit_gib': 14,
    'new_training': 0, 'optimizer_steps': 0, 'encoder_forward': 0,
    'downloads': 0, 'automatic_retry': False, 'dev_test_opened': False,
    'maximum_errors': 564, 'maximum_insertions': 32, 'maximum_deletions': 248,
    'minimum_improved_songs': 3, 'group_minimum': 20, 'group_tolerance': .03,
    'source_sort': 'song_id,example_id,frame', 'nn_blocks': [128, 2048],
    'distance_precision': 'cpu_float64', 'head_precision': 'cuda_float32',
}


def bound(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


def load_bound(spec: dict) -> dict:
    path = Path(spec['path']); verify_file(path, spec['sha256']); return read_json(path)


def protocol(run: Path, *, review: bool = False) -> dict:
    freeze = read_json(run / 'FREEZE.json')
    if freeze['policy'] != POLICY:
        raise ValueError('fixed policy changed')
    for path, digest in freeze['manifest_bindings'].items():
        verify_file(Path(path), digest)
    if review:
        r = read_json(run / 'CODE_REVIEW.json')
        if r['freeze_sha256'] != sha256_file(run / 'FREEZE.json') or r['approved_for_launch'] is not True:
            raise ValueError('missing/invalid executed-code review')
        for path, digest in r['code_bindings'].items():
            verify_file(Path(path), digest)
        evidence = load_bound(r['verification'])
        if evidence['passed'] is not True:
            raise ValueError('prelaunch tests failed')
        for spec in evidence['artifacts']:
            verify_file(Path(spec['path']), spec['sha256'])
    return freeze


def deadline(seconds: float):
    until = time.monotonic() + seconds
    def check() -> None:
        if time.monotonic() >= until:
            raise TimeoutError('fixed CPU stage timeout; no automatic retry')
    return check


def save_arrays(path: Path, arrays: dict) -> str:
    from safetensors.numpy import save_file
    import numpy as np
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, str(path))
    return sha256_file(path)


def load_arrays(spec: dict) -> dict:
    from safetensors.numpy import load_file
    verify_file(Path(spec['path']), spec['sha256'])
    return load_file(spec['path'])

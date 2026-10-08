"""Immutable artifacts and launch ledger for speed-invariance v2."""
from __future__ import annotations
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file, sha256_object
from any2xsampa.training.acoustic_memory import SONGS, QUERY_FIELDS, validate_query

RESEARCH = ROOT / '.state/runs/post-memory-research-20260925T115419Z'
OLD = ROOT / '.state/runs/event-head-comparison-20260924T101019Z'
MEMORY = ROOT / '.state/runs/acoustic-memory-20260925T073038Z'
POLICY_HASH = 'de07c4c65f8a812069fd3971439f12d5c7e3c77795ce40ae0792ba8e731b0ff9'


def bound(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


def load_bound(spec: dict) -> dict:
    verify_file(Path(spec['path']), spec['sha256'])
    return read_json(Path(spec['path']))


def policy() -> dict:
    path = RESEARCH / 'PLAN_V2_POLICY.json'; verify_file(path, POLICY_HASH)
    return read_json(path)


def slots() -> list[str]:
    return ['encode', 'encode-replay'] + [arm + '--' + song for song in SONGS for arm in ('C', 'A')] + ['head-replay']


def protocol(run: Path, *, reviewed: bool = False) -> dict:
    f = read_json(run / 'FREEZE.json')
    if f['policy_sha256'] != POLICY_HASH or f['budget'] != policy()['budget']:
        raise ValueError('fixed policy or budget changed')
    for path, digest in f['manifests'].items(): verify_file(Path(path), digest)
    if reviewed:
        r = read_json(run / 'CODE_REVIEW.json')
        if r['freeze_sha256'] != sha256_file(run / 'FREEZE.json') or not r['approved_for_launch']:
            raise ValueError('missing/stale launch review')
        for path, digest in r['code_bindings'].items(): verify_file(Path(path), digest)
        tests = load_bound(r['verification'])
        if tests['passed'] is not True: raise ValueError('verification not passed')
        for spec in tests['artifacts']: verify_file(Path(spec['path']), spec['sha256'])
    return f


class Ledger:
    def __init__(self, path: Path, caps: dict[str, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open('x', encoding='utf-8', buffering=1)
        self.caps = dict(caps); self.counts = dict.fromkeys(caps, 0)

    def attempt(self, kind: str, **detail) -> None:
        if kind not in self.caps or self.counts[kind] >= self.caps[kind]:
            raise RuntimeError('fixed request budget exhausted: ' + kind)
        self.counts[kind] += 1
        self.stream.write(json.dumps({'event': 'attempt', 'kind': kind, 'number': self.counts[kind], **detail}) + '\n')
        self.stream.flush()

    def completed(self, kind: str, **detail) -> None:
        self.stream.write(json.dumps({'event': 'completed', 'kind': kind, 'number': self.counts[kind], **detail}) + '\n')
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def stable_memory(encoder: bool) -> dict:
    from any2xsampa.adapters.phoneticxeus_monitor import query_total_gpu_bytes
    samples = []
    for _ in range(3): samples.append(query_total_gpu_bytes()); time.sleep(.25)
    reserve = (3 * 1024**3 if encoder else 256 * 1024**2)
    if max(samples) + reserve > 14 * 1024**3:
        raise RuntimeError('insufficient stable GPU headroom; no launch or fallback')
    return {'samples_bytes': samples, 'required_headroom_bytes': reserve}


def load_cache(torch, q: dict) -> dict:
    from safetensors.torch import load_file
    validate_query(q); verify_file(Path(q['path']), q['sha256'])
    data = load_file(q['path'], device='cpu')
    for key, width in (('encoder_features', 1024), ('frame_logits', 428)):
        t = data[key]
        if tuple(t.shape) != (q['frames'], width) or t.dtype != torch.float32 or not torch.isfinite(t).all():
            raise ValueError('invalid finite final feature cache')
    return {key: data[key] for key in ('encoder_features', 'frame_logits')}

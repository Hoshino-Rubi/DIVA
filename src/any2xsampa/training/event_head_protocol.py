"""Immutable cache identity, old-order sampling, and budget gates for H/E."""
from __future__ import annotations
from pathlib import Path
import random
from typing import Any
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,sha256_object

CONFIG=Path(__file__).parents[1]/'config/event_head_comparison_v1.json'


def load_protocol(run: Path,review_name: str) -> tuple[dict,dict,dict]:
    freeze=read_json(run/'FREEZE.json');spec=read_json(run/'POLICY.json')
    verify_file(run/'POLICY.json',freeze['policy_sha256'])
    if spec!=read_json(CONFIG):raise ValueError('runtime policy changed after freeze')
    review=read_json(run/review_name)
    if not review['approved_for_launch'] or review['freeze_sha256']!=sha256_file(run/'FREEZE.json'):raise ValueError('invalid launch review')
    for p,h in review['code_bindings'].items():verify_file(Path(p),h)
    v=review['verification'];verify_file(Path(v['path']),v['sha256'])
    if not read_json(Path(v['path']))['passed']:raise ValueError('preflight verification failed')
    for p,h in freeze['bindings'].items():verify_file(Path(p),h)
    rows=read_json(run/'CACHE_ROWS.json')
    return freeze,spec,rows


def epoch_schedule(rows: list[dict],unlabeled: list[dict],held: str|None,seed: int,epochs: int=20) -> list[list[str]]:
    if any(r['split']!='train' or r['audio_role'] not in ('original','rendered') for r in rows):raise ValueError('nontraining labelled row')
    selected=[r for r in rows if r['song_id']!=held]
    if not selected:raise ValueError('empty training split')
    rng=random.Random(seed);schedule=[]
    for _ in range(epochs):
        ids=[]
        for r in selected:ids.extend([r['example_id']]*(4 if r['audio_role']=='original' else 1))
        rng.shuffle(ids)
        unused=[r['example_id'] for r in unlabeled if r['song_id']!=held]
        rng.shuffle(unused)  # preserve old baseline RNG consumption, no loss.
        schedule.append(ids)
    return schedule


def validate_cohort(rows: list[dict],spec: dict) -> None:
    if len(rows)!=149 or len({r['example_id'] for r in rows})!=149:raise ValueError('149 unique labelled rows required')
    original=[r for r in rows if r['audio_role']=='original'];render=[r for r in rows if r['audio_role']=='rendered']
    if len(original)!=20 or len(render)!=129 or sum(len(r['target_ids']) for r in original)!=1235:raise ValueError('cohort denominator changed')
    for r in rows:
        if r['split']!='train' or r['song_id'] not in spec['songs']:raise ValueError('song/split leakage')
        if 'vocabulary' in r and r['target_ids']!=[r['vocabulary'][p] for p in r['target_ipa']]:raise ValueError('target identity mismatch')
        ids=r['target_ids']
        if not ids or any(isinstance(i,bool) or not isinstance(i,int) or not 0<i<428 for i in ids):raise ValueError('invalid reference target')
        if r['frames']<len(ids)+sum(a==b for a,b in zip(ids,ids[1:])):raise ValueError('insufficient CTC frames')


def load_cache(torch: Any,rows: list[dict]) -> list[dict]:
    from safetensors.torch import load_file
    output=[]
    for row in rows:
        verify_file(Path(row['path']),row['sha256']);d=load_file(row['path'],device='cpu')
        for name,width in [('encoder_features',1024),('frame_logits',428)]:
            if d[name].shape!=(row['frames'],width) or d[name].dtype!=torch.float32 or not torch.isfinite(d[name]).all():raise ValueError('invalid frozen final cache')
        output.append({**row,'features':d['encoder_features'],'prior':d['frame_logits']})
    return output


def fold_steps(schedule: list[list[str]],micro: int=8) -> int:
    return sum((len(epoch)+micro-1)//micro for epoch in schedule)


def check_job(run: Path,arm: str,held: str,spec: dict,*,worker: bool=False) -> None:
    if arm not in spec['arms'] or held not in spec['songs']:raise ValueError('unplanned screen job')
    key=arm+'--'+held
    if (run/'screen'/arm/held/'START.json').exists():raise ValueError('job already started; no retry')
    command=run/'commands'/(key+'.command.json')
    if worker and not command.exists():raise ValueError('unaccounted direct worker launch')
    if not worker and command.exists():raise ValueError('job command already exists; no retry')
    if len(list((run/'commands').glob('*.command.json')))>=(11 if worker else 10):raise ValueError('screen launch budget exhausted')
    preflight=read_json(run/'preflight/RESULT.json')
    if not preflight['passed'] or not preflight['cost_gate'][arm]['passed']:raise ValueError('cost/correctness preflight did not permit arm')

"""Freeze the authorized C/A comparison without acoustic forwards or updates."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from speed_invariance_common import ROOT, OLD, MEMORY, RESEARCH, bound, policy, read_json, write_json_new, verify_file, sha256_file, sha256_object
from speed_invariance_common import SONGS, QUERY_FIELDS, validate_query, load_cache
from any2xsampa.training.speed_invariance import validate_fold, schedule_views, make_head, adapter_state
from any2xsampa.training.event_head_protocol import validate_cohort


def prepare(run: Path) -> None:
    import torch
    from safetensors.torch import load_file, save_file
    if run.exists() or not run.resolve().is_relative_to(ROOT / '.state/runs'):
        raise ValueError('new independent run required; never restart existing experiments')
    spec = policy(); old = read_json(OLD / 'FREEZE.json')
    preserved = {}
    for records in (read_json(RESEARCH / 'RESEARCH_COMPLETE_V2.json')['artifacts'], spec['source_bindings'], old['bindings']):
        for p, h in records.items(): verify_file(Path(p), h); preserved[p] = h
    memory = read_json(MEMORY / 'FREEZE.json')
    for p, h in memory['manifest_bindings'].items(): verify_file(Path(p), h); preserved[p] = h
    protected = read_json(MEMORY / 'PROTECTED.json')
    for records in protected.values():
        for p, h in records.items(): verify_file(Path(p), h); preserved[p] = h
    for p, h in read_json(MEMORY / 'CODE_REVIEW.json')['code_bindings'].items():
        verify_file(Path(p), h); preserved[p] = h
    data = read_json(OLD / 'CACHE_ROWS.json'); rows = data['rows']
    validate_cohort(rows, {'songs': SONGS})
    queries = [{k: row[k] for k in QUERY_FIELDS} for row in rows]
    for q in queries: validate_query(q); load_cache(torch, q)
    if queries != read_json(MEMORY / 'QUERIES.json')['rows']:
        raise ValueError('frozen cohort identity differs from completed memory experiment')
    scoring = read_json(MEMORY / 'SCORING.json')
    baseline = [{k: r[k] for k in ('example_id', 'song_id', 'audio_role', 'reference')} |
                {'prediction': r['baseline_prediction']} for r in scoring['rows']]
    vocabulary = read_json(Path(old['vocabulary']['path']))
    for row in rows:
        if row['target_ids'] != [vocabulary[p] for p in row['target_ipa']]:
            raise ValueError('target vocabulary mismatch')
        prior = next(r for r in scoring['rows'] if r['example_id'] == row['example_id'])
        if prior['reference'] != row['target_ipa']:
            raise ValueError('human complete sequence reference changed')
    run.mkdir(parents=True)
    write_json_new(run / 'AUTHORIZATION.json', {'user_message': '现在开始启动新实验',
        'time_utc': datetime.now(timezone.utc).isoformat(), 'plan': bound(RESEARCH / 'NEXT_PLAN_V2_2026-09-26.md'),
        'budget': spec['budget'], 'no_automatic_retries': True})
    write_json_new(run / 'QUERIES.json', {'rows': queries})
    write_json_new(run / 'SCORING.json', scoring)
    write_json_new(run / 'HISTORICAL.json', {'rows': baseline})
    write_json_new(run / 'PROTECTED.json', {'files': preserved})
    schedules = read_json(OLD / 'SCHEDULES.json'); initial = {}; fold_audit = {}
    torch.set_num_threads(4)
    head = load_file(old['pretrained_head']['path'], device='cpu')
    for song in SONGS:
        selected = [r for r in rows if r['song_id'] != song]
        train = [{'query': {k: r[k] for k in QUERY_FIELDS}, 'target_ids': r['target_ids'],
            'reference_sha256': sha256_object(next(s for s in scoring['rows'] if s['example_id'] == r['example_id'])),
            'reference_nature': r['reference_nature']} for r in selected]
        held = [q for q in queries if q['song_id'] == song]
        validate_fold(train, held, song)
        roles = {r['example_id']: r['audio_role'] for r in selected}
        epochs = schedules[song]['epochs']; counts = Counter(); steps = 0
        if len(epochs) != 20: raise ValueError('epoch budget changed')
        for i, epoch in enumerate(epochs):
            for key, view in schedule_views(epoch, roles, 'A', i):
                counts['rendered_raw' if roles[key] == 'rendered' else view] += 1
            steps += (len(epoch) + 7) // 8
        fixed = spec['per_fold'][song]
        if dict(counts) != fixed['augmented_arm_views'] or steps != fixed['optimizer_steps_per_arm']:
            raise ValueError('saved schedule no longer matches authorized budget')
        write_json_new(run / 'folds' / song / 'TRAINING.json', {'rows': train, 'epochs': epochs})
        write_json_new(run / 'folds' / song / 'QUERIES.json', {'held': held,
            'seen_original': [r['query'] for r in train if r['query']['audio_role'] == 'original']})
        # Explicit CPU initialization: no real-data neural call, no optimizer.
        torch.manual_seed(17); model = make_head(torch, head, device='cpu')
        path = run / 'folds' / song / 'initial.safetensors'; save_file(adapter_state(model), str(path))
        initial[song] = bound(path)
        fold_audit[song] = {'training_rows': len(train), 'held_rows': len(held), 'steps': steps, 'views': dict(counts)}
    originals = sorted([q for q in queries if q['audio_role'] == 'original'],
        key=lambda q: (SONGS.index(q['song_id']), q['start_sec'], q['example_id']))
    first = {song: next(q['example_id'] for q in originals if q['song_id'] == song) for song in SONGS}
    requests = []
    for q in originals:
        views = ['raw', 'slow_0p9', 'fast_1p1'] if q['example_id'] in first.values() else ['slow_0p9', 'fast_1p1']
        requests.extend({'query': q, 'view': v} for v in views)
    replay = [r for r in requests if r['query']['example_id'] in first.values() and r['view'] != 'raw']
    if len(requests) != 45 or len(replay) != 10: raise ValueError('encoder request cap changed')
    write_json_new(run / 'ENCODER_REQUESTS.json', {'encode': requests, 'encode-replay': replay, 'fixed_first': first})
    manifests = {str(p.resolve()): sha256_file(p) for p in run.rglob('*') if p.is_file()}
    write_json_new(run / 'FREEZE.json', {'schema': 'speed_invariance_execution_v2',
        'policy_sha256': bound(RESEARCH / 'PLAN_V2_POLICY.json')['sha256'], 'budget': spec['budget'],
        'manifests': manifests, 'architecture': data['architecture'], 'head': old['pretrained_head'],
        'vocabulary': old['vocabulary'], 'python': old['python'], 'snapshot': old['snapshot'], 'initial': initial,
        'plan': bound(RESEARCH / 'NEXT_PLAN_V2_2026-09-26.md')})
    write_json_new(run / 'PREPARED.json', {'passed': True, 'folds': fold_audit, 'protected_files': len(preserved),
        'new_acoustic_forwards': 0, 'optimizer_updates': 0, 'unique_original_seconds': 200,
        'original_windows': 20, 'derived_views': 40, 'original_references': 1235, 'rendered_references': 4814})
    print(run, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path)
    a = p.parse_args(); prepare((a.run or ROOT / '.state/runs' / ('speed-invariance-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve())

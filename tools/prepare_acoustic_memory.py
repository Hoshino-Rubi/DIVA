"""Read-only source preflight and separated manifests; no neural calls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from acoustic_memory_common import ROOT, RESEARCH, OLD, COMPLETED, REFERENCE, POLICY, bound
from acoustic_memory_common import read_json, write_json_new, verify_file, sha256_file, sha256_object
import numpy as np
from safetensors import safe_open
from any2xsampa.training.acoustic_memory import SONGS, QUERY_FIELDS, validate_query, validate_sources
from any2xsampa.training.event_head_protocol import validate_cohort
from any2xsampa.training.phone_error_metrics import phone_metrics


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError('run already exists; inspect rather than restart')
    compaction = read_json(RESEARCH / 'COMPACTION_INDEX_2026-09-25.json')
    history = read_json(RESEARCH / 'RESEARCH_COMPLETE.json')
    verified = {}
    for records in (compaction['artifacts'], history['research_artifacts'], history['source_archive']):
        for p, h in records.items():
            verify_file(Path(p), h); verified[p] = h
    old_freeze = read_json(OLD / 'FREEZE.json')
    for p, h in old_freeze['bindings'].items():
        verify_file(Path(p), h)
    index = read_json(OLD / 'CACHE_ROWS.json'); rows = index['rows']
    validate_cohort(rows, {'songs': SONGS})
    controls = read_json(OLD / 'CONTROLS.json')
    old = read_json(COMPLETED / 'collection/independent-replay/ENDPOINT_CONTROLS.json')['rows']
    if len(old) != 149 or {r['example_id'] for r in old} != {r['example_id'] for r in rows}:
        raise ValueError('old baseline cohort mismatch')
    vocab = read_json(Path(old_freeze['vocabulary']['path']))
    if len(vocab) != 428 or vocab.get('<blank>') != 0 or sorted(vocab.values()) != list(range(428)):
        raise ValueError('vocabulary changed')
    references = read_json(REFERENCE)
    originals = {r['example_id']: r for r in references['examples'] if r['split'] == 'train'}
    if len(originals) != 20:
        raise ValueError('original supervision changed')
    source_rows = []; queries = []; score = []; features = []
    baseline = {r['example_id']: r for r in old}
    for row in rows:
        q = {k: row[k] for k in QUERY_FIELDS}; validate_query(q)
        if row['architecture_sha256'] != index['architecture']['sha256'] or row['target_ids'] != [vocab[p] for p in row['target_ipa']]:
            raise ValueError('cache/label identity mismatch')
        if row['audio_role'] == 'original' and row['target_ipa'] != [p['phone'] for p in originals[row['example_id']]['phones']]:
            raise ValueError('human reference drift')
        if row['target_ipa'] != baseline[row['example_id']]['reference']:
            raise ValueError('baseline denominator drift')
        with safe_open(row['path'], framework='numpy') as tensors:
            f = tensors.get_tensor('encoder_features')
            if f.shape != (row['frames'], 1024) or f.dtype != np.float32 or not np.isfinite(f).all():
                raise ValueError('invalid final cache')
            features.append({'example_id': row['example_id'], 'keys': list(tensors.keys()), 'final_shape': list(f.shape)})
        queries.append(q)
        if row['audio_role'] == 'original':
            source_rows.append({'query': q, 'example_id': q['example_id'], 'song_id': q['song_id'],
                'target_ids': row['target_ids'], 'reference_sha256': sha256_object(originals[q['example_id']]),
                'reference_nature': row['reference_nature']})
        score.append({'example_id': q['example_id'], 'song_id': q['song_id'], 'audio_role': q['audio_role'],
            'reference': row['target_ipa'], 'baseline_prediction': baseline[q['example_id']]['prediction'],
            'phones': originals[q['example_id']]['phones'] if q['audio_role'] == 'original' else None})
    metrics = {role: phone_metrics([(r['reference'], r['baseline_prediction']) for r in score if r['audio_role'] == role]) for role in ('original', 'rendered')}
    if [metrics['original'][k] for k in ('reference_phones','edit_distance','substitutions','deletions','insertions')] != [1235,595,315,248,32] or [metrics['rendered'][k] for k in ('reference_phones','edit_distance')] != [4814,1551]:
        raise ValueError('baseline saved metrics differ')
    folds = {}; requests = []; checkpoints = {}
    for song in SONGS:
        selected = sorted([r for r in source_rows if r['song_id'] != song], key=lambda r:(r['song_id'],r['example_id']))
        validate_sources(selected, song); folds[song] = selected
        checkpoint = controls[song]
        verify_file(Path(checkpoint['checkpoint_path']), checkpoint['checkpoint_sha256'])
        checkpoints[song] = {'path': checkpoint['checkpoint_path'], 'sha256': checkpoint['checkpoint_sha256']}
        for purpose, subset in [('source', [s['query'] for s in selected]), ('query', [q for q in queries if q['song_id'] == song])]:
            requests.extend({'fold': song, 'purpose': purpose, 'query': q} for q in subset)
    if len(requests) != 229 or sum(r['purpose'] == 'source' for r in requests) != 80:
        raise ValueError('head budget mismatch')
    run.mkdir(parents=True)
    write_json_new(run / 'SOURCES.json', {'folds': folds})
    write_json_new(run / 'QUERIES.json', {'rows': queries})
    write_json_new(run / 'SCORING.json', {'rows': score})
    write_json_new(run / 'HEAD_REQUESTS.json', {'requests': requests})
    protected = read_json(COMPLETED / 'FINAL_VERIFICATION.json')['protected_files_unchanged']
    for p, h in protected.items():verify_file(Path(p), h)
    write_json_new(run / 'PROTECTED.json', {'files': protected, 'historical_bindings': old_freeze['bindings'], 'research_bindings': verified})
    manifests = {str(p.resolve()): sha256_file(p) for p in run.glob('*.json')}
    freeze = {'schema': 'acoustic_memory_freeze_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'policy': POLICY, 'manifest_bindings': manifests, 'checkpoints': checkpoints,
        'head': old_freeze['pretrained_head'], 'vocabulary': old_freeze['vocabulary'], 'python': old_freeze['python'],
        'architecture': index['architecture'], 'plan': bound(RESEARCH / 'COMPACTION_PLAN_2026-09-25.md'),
        'source_ledger': bound(OLD / 'CACHE_ROWS.json'), 'historical_endpoint': bound(COMPLETED / 'collection/independent-replay/ENDPOINT_CONTROLS.json')}
    write_json_new(run / 'FREEZE.json', freeze)
    write_json_new(run / 'PREFLIGHT.json', {'passed': True, 'research_hash_count': len(verified),
        'historical_hash_count': len(old_freeze['bindings']), 'protected_count': len(protected),
        'saved_baseline_metrics': metrics, 'cache_checks': features, 'requests': 229, 'new_head_calls': 0,
        'query_reference_fields': False, 'actual_worker_processes_observed_before_prepare': 0})
    print(str(run.resolve()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    prepare(args.run or ROOT / '.state/runs' / ('acoustic-memory-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))

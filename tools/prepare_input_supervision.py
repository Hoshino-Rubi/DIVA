"""Freeze direct-supervision inputs without changing the historical slot order."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from any2xsampa.training.input_supervision import policy, validate_rows, CONFIG
from any2xsampa.training.singing_domain import load_manifest, validate_cache_entry
from any2xsampa.training.dual_input import validate_pair
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def binding(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


def prepare(run: Path, probe: Path, incident: Path, authorized_total_starts: int = 10) -> None:
    run, probe = run.resolve(), probe.resolve()
    if not run.is_relative_to(ROOT / '.state/runs') or run.exists():
        raise ValueError('choose a new output under .state/runs')
    prior = read_json(incident)
    unavailable = prior.get('unavailable_screen_jobs', [])
    planned = {f'{variant}-{song}' for variant in policy()['variants'] for song in policy()['songs']}
    if not isinstance(unavailable, list) or len(set(unavailable)) != len(unavailable) or not set(unavailable) <= planned:
        raise ValueError('invalid prior failed-job exclusion list')
    if isinstance(authorized_total_starts, bool) or not isinstance(authorized_total_starts, int) or authorized_total_starts < 0:
        raise ValueError('authorized start budget must be a nonnegative integer')
    probe_freeze = read_json(probe / 'FREEZE.json')
    historical = Path(probe_freeze['historical_run'])
    spec, refs = load_manifest(historical)
    source_freeze = read_json(historical / 'FREEZE.json')
    vocabulary_path = Path(source_freeze['vocabulary_path'])
    vocabulary = read_json(vocabulary_path)
    index = read_json(historical / 'CACHE.json')
    verify_file(historical / 'CACHE.json', probe_freeze['historical_cache_sha256'])
    verify_file(historical / 'REFERENCES.json', probe_freeze['historical_references_sha256'])
    verify_file(historical / 'AUDIO_REQUESTS.json', index['audio_requests_sha256'])
    requests = {r['example_id']: r for r in read_json(historical / 'AUDIO_REQUESTS.json')['rows']}
    if index['reference_received'] is not False or not index['all_parameters_frozen']:
        raise ValueError('historical cache has reference contamination')
    cached = {r['example_id']: r for r in index['entries']}
    if len(cached) != len(index['entries']) or set(cached) != set(requests):
        raise ValueError('historical cache coverage changed')
    references = {r['example_id']: r for r in refs['rows']}
    pairs_root = Path(probe_freeze['pair_run'])
    pairs = {r['example_id']: r for r in read_json(pairs_root / 'PAIRS.json')['entries']}
    pair_requests = {r['example_id']: r for r in read_json(pairs_root / 'AUDIO_REQUESTS.json')['requests']}
    bindings = {name: binding(path) for name, path in {
        'old_freeze': historical / 'FREEZE.json', 'old_cache': historical / 'CACHE.json',
        'old_references': historical / 'REFERENCES.json', 'old_policy': historical / 'POLICY.json',
        'probe_result': probe / 'RESULT.json', 'probe_cache': probe / 'CACHE.json',
        'vocabulary': vocabulary_path, 'incident': incident,
    }.items()}
    rows, entries, unused = [], [], []
    for order, entry in enumerate(index['entries']):
        validate_cache_entry(entry, requests[entry['example_id']], index['architecture']['sha256'])
        role, source_id = entry['audio_role'], entry['example_id']
        if role == 'unlabeled_original':
            unused.append({'example_id': source_id, 'song_id': entry['song_id']})
        if role not in {'original', 'rendered'}:
            continue
        reference = references[source_id]
        if any(reference[k] != entry[k] for k in ('song_id', 'split', 'audio_sha256', 'start_sec', 'end_sec')):
            raise ValueError('historical reference is not bound to cached audio')
        bindings.setdefault('audio_' + entry['audio_sha256'], {'path': entry['audio_path'], 'sha256': entry['audio_sha256']})
        if role == 'original':
            mate = cached[reference['paired_example_id']]
            pair_binding = pairs[source_id]
            verify_file(Path(pair_binding['path']), pair_binding['sha256'])
            pair = read_json(Path(pair_binding['path']))
            validate_pair(pair_requests[source_id], pair)
            old_request = pair_requests[source_id]
            if any(old_request[k] != entry[k] for k in ('audio_sha256', 'start_sec', 'end_sec')):
                raise ValueError('pair source interval differs from confirmed original')
            if mate['song_id'] != entry['song_id'] or mate['audio_role'] != 'separated_vocal' or mate['audio_sha256'] != pair['audio']['vocal']['sha256'] or mate['start_sec'] != 0. or abs(mate['end_sec']-(entry['end_sec']-entry['start_sec'])) > 1e-6:
                raise ValueError('separated view is not bound to this original window')
            bindings['pair_' + source_id] = dict(pair_binding)
            bindings['vocal_' + source_id] = {'path': mate['audio_path'], 'sha256': mate['audio_sha256']}
            views = [('x', entry), ('v', mate)]
        else:
            views = [('rendered', entry)]
        for condition, acoustic in views:
            verify_file(Path(acoustic['path']), acoustic['sha256'])
            row_id = source_id + '--' + condition
            rows.append({
                'row_id': row_id, 'cache_id': row_id, 'window_id': source_id,
                'song_id': entry['song_id'], 'split': 'train', 'audio_kind': role,
                'condition': condition, 'source_order': order, 'frames': acoustic['frames'],
                'target_ipa': reference['target_ipa'], 'target_ids': reference['target_ids'],
                'reference_nature': 'derived_from_confirmed_original_not_separately_audited' if condition == 'v' else 'confirmed_original' if condition == 'x' else 'rendered_proxy',
                'source_audio_sha256': entry['audio_sha256'], 'source_start_sec': entry['start_sec'], 'source_end_sec': entry['end_sec'],
                'source_cache_example_id': acoustic['example_id'],
            })
            entries.append({**acoustic, 'cache_id': row_id, 'condition': condition})
    validate_rows(rows, vocabulary)
    for value in bindings.values():
        verify_file(Path(value['path']), value['sha256'])
    if not read_json(probe / 'RESULT.json')['matched_x_identity']['all_equal']:
        raise ValueError('matched PCM x predictions differ from historical x control')
    run.mkdir(parents=True)
    write_json_new(run / 'POLICY.json', policy())
    write_json_new(run / 'ROWS.json', {'schema_version': 'input_supervision_rows_v2', 'rows': rows, 'unused_unlabeled_rng_rows': unused})
    write_json_new(run / 'CACHE_BINDINGS.json', {'architecture': index['architecture'], 'entries': entries, 'labels_received': False, 'all_parameters_frozen': True})
    for name in ('POLICY.json', 'ROWS.json', 'CACHE_BINDINGS.json'):
        bindings[name] = binding(run / name)
    code_paths = [CONFIG, Path(__file__), ROOT/'tools/train_input_supervision.py', ROOT/'tools/run_input_supervision_screen.py',
                  ROOT/'src/any2xsampa/training/input_supervision.py', ROOT/'src/any2xsampa/training/input_supervision_training.py',
                  ROOT/'src/any2xsampa/training/input_supervision_evaluation.py', ROOT/'src/any2xsampa/training/singing_domain_losses.py',
                  ROOT/'src/any2xsampa/training/singing_domain_training.py', ROOT/'src/any2xsampa/training/singing_domain.py',
                  ROOT/'src/any2xsampa/training/protected_adaptation.py', ROOT/'src/any2xsampa/training/phone_error_metrics.py',
                  ROOT/'src/any2xsampa/training/limited_data_artifacts.py', ROOT/'src/any2xsampa/config/singing_domain_v1.json',
                  ROOT/'src/any2xsampa/adapters/singing_domain_runtime.py', ROOT/'src/any2xsampa/adapters/phoneticxeus_runtime.py',
                  ROOT/'src/any2xsampa/adapters/phoneticxeus_monitor.py', ROOT/'src/any2xsampa/adapters/phoneticxeus.py',
                  ROOT/'src/any2xsampa/adapters/candidate_inference.py']
    write_json_new(run / 'FREEZE.json', {
        'schema_version': 'input_supervision_freeze_v2', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'historical_run': str(historical), 'probe_run': str(probe), 'model_snapshot': source_freeze['model_snapshot'],
        'runtime_python': source_freeze['runtime_python'], 'vocabulary_path': str(vocabulary_path), 'bindings': bindings,
        'code_bindings': {str(p.resolve()): sha256_file(p) for p in code_paths},
        'budget': {'authorized_screen_starts': authorized_total_starts, 'prior_screen_starts': prior['starts_so_far'],
                   'remaining_screen_starts': max(0, authorized_total_starts-prior['starts_so_far']), 'required_for_valid_complete_screen': 10},
        'unavailable_screen_jobs': unavailable,
        'development_or_final_test_opened': False, 'default_replacement': False,
    })
    print(run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--incident', type=Path, required=True)
    parser.add_argument('--authorized-total-starts', type=int, default=10,
                        help='Use a larger value only after explicit user authorization; previous attempts remain counted.')
    args = parser.parse_args()
    prepare(args.run, args.probe, args.incident, args.authorized_total_starts)

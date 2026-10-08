"""Frozen cohort, source-clock and acceptance checks for dual-input inference."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from any2xsampa.training.limited_data_artifacts import read_json, sha256_object, verify_file
from any2xsampa.training.decoder_experiment import TRAIN_SONGS


def load_policy(run: Path) -> dict:
    policy = read_json(run/'POLICY.json')
    config = read_json(Path(__file__).parents[1]/'config/dual_input_ctc_v1.json')
    if any(policy.get(k) != v for k, v in config.items()):
        raise ValueError('dual-input rules differ from frozen config')
    if policy['policy_sha256'] != sha256_object({k: v for k, v in policy.items() if k != 'policy_sha256'}):
        raise ValueError('dual-input policy hash mismatch')
    for spec in policy['bindings'].values(): verify_file(Path(spec['path']), spec['sha256'])
    return policy


def audio_requests(payload: dict) -> list[dict]:
    if set(payload) != {'schema_version', 'requests', 'contains_reference_context', 'test_allowed'}:
        raise ValueError('unexpected audio request fields')
    if payload['schema_version'] != 'dual_input_audio_requests_v1' or payload['contains_reference_context'] is not False or payload['test_allowed'] is not False:
        raise ValueError('audio-only, non-test requests required')
    allowed = {'example_id', 'song_id', 'split', 'audio_path', 'audio_sha256', 'start_sec', 'end_sec',
               'source_rate', 'source_start_sample', 'source_end_sample_exclusive',
               'context_start_sec', 'context_end_sec', 'reuse_separation', 'reuse_exact_pair', 'contains_reference_context'}
    seen = set()
    for row in payload['requests']:
        if set(row) != allowed or row['contains_reference_context'] is not False:
            raise ValueError('unexpected reference/score data in audio request')
        if row['example_id'] in seen: raise ValueError('duplicate window')
        seen.add(row['example_id'])
        if not ((row['split'] == 'train' and row['song_id'] in TRAIN_SONGS) or (row['split'] == 'dev' and row['song_id'] == 'shalong')):
            raise ValueError('song split leakage')
        if not 8 <= row['end_sec']-row['start_sec'] <= 20:
            raise ValueError('recognizer crop outside bounds')
        if not 0 <= row['context_start_sec'] <= row['start_sec'] < row['end_sec'] <= row['context_end_sec']:
            raise ValueError('context does not contain crop')
        if row['context_end_sec']-row['context_start_sec'] > 30.000001:
            raise ValueError('context exceeds bound')
        if (row['source_start_sample'] != round(row['start_sec']*row['source_rate']) or
                row['source_end_sample_exclusive'] != round(row['end_sec']*row['source_rate'])):
            raise ValueError('source sample interval mismatch')
    if Counter(r['split'] for r in payload['requests']) != {'train': 20, 'dev': 3}:
        raise ValueError('fixed window count changed')
    if len({r['song_id'] for r in payload['requests'] if r['split'] == 'train'}) != 5:
        raise ValueError('five training songs required')
    return payload['requests']


def validate_pair(request: dict, pair: dict) -> None:
    if pair['example_id'] != request['example_id'] or pair['source_audio_sha256'] != request['audio_sha256']:
        raise ValueError('paired audio source mismatch')
    if pair['source_start_sample'] != request['source_start_sample'] or pair['source_end_sample_exclusive'] != request['source_end_sample_exclusive']:
        raise ValueError('pair rebound to another original interval')
    if set(pair['audio']) != {'mixed', 'vocal'} or pair['contains_reference_context'] is not False:
        raise ValueError('pair roles/reference contamination')
    a, b = pair['audio']['mixed'], pair['audio']['vocal']
    for row in (a, b):
        verify_file(Path(row['path']), row['sha256'])
        if row['sample_rate'] != 44100 or row['channels'] != 2 or row['subtype'] != 'PCM_32':
            raise ValueError('pair PCM geometry mismatch')
    if a['samples'] != b['samples'] or a['samples'] != round((request['end_sec']-request['start_sec'])*44100):
        raise ValueError('paired timeline duration mismatch')
    for edge in ('start', 'end'):
        expected = request['source_'+edge+'_sample' if edge == 'start' else 'source_end_sample_exclusive']/request['source_rate']
        if abs(pair['effective_original_'+edge+'_sec']-expected) > 1/44100:
            raise ValueError('paired source clock shifted')
    if pair['transform']['same_gain_for_both_conditions'] is not True or pair['transform']['clipping_performed'] is not False:
        raise ValueError('unmatched gain or clipped pair')


def acceptance(metrics: dict, baseline: dict, rules: dict) -> dict:
    if metrics['reference_phones'] != 214 or baseline['reference_phones'] != 214 or metrics['sequences'] != 3:
        raise ValueError('development denominator changed')
    failures = []
    for key, limit in [('edit_distance', 'maximum_dev_errors'), ('insertions', 'maximum_dev_insertions'), ('deletions', 'maximum_dev_deletions')]:
        if metrics[key] > rules[limit]: failures.append(key+'_over_limit')
    for group, limit in [('aspirated', 'maximum_dev_aspirated_errors'), ('glottal_h', 'maximum_dev_h_errors')]:
        row = metrics['groups'][group]
        if sum(row[k] for k in ('substitutions', 'deletions', 'insertions')) > rules[limit]:
            failures.append(group+'_error_increase')
    for name, old in baseline['groups'].items():
        new = metrics['groups'][name]
        if old['reference_count'] != new['reference_count']: raise ValueError('phone group denominator changed')
        if old['reference_count'] >= rules['minimum_group_reference_count'] and new['per'] > old['per']+rules['maximum_group_per_increase']:
            failures.append(name+'_regression')
    return {'passed': not failures, 'failures': failures, 'thresholds': rules,
            'small_sample_groups': [k for k,v in metrics['groups'].items() if v['reference_count'] < rules['minimum_group_reference_count']],
            'development_already_observed': True, 'automatic_deployment': False}

"""Frozen train/dev-only CTC decoder experiment and error attribution."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any

from any2xsampa.training.limited_data_artifacts import read_json, sha256_object, verify_file
from any2xsampa.training.phone_error_metrics import align_phone_edits

TRAIN_SONGS = {'freedom-flower', 'queen-road-east', 'half-moon-night-song', 'promise', 'like-you'}


def validate_policy(policy: dict) -> None:
    if policy.get('schema_version') != 'ctc_prefix_experiment_v1':
        raise ValueError('unexpected decoder experiment policy')
    expected = read_json(Path(__file__).parents[1] / 'config/ctc_prefix_experiment_v1.json')
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError('unfrozen decoder rule: ' + key)
    for spec in policy['bindings'].values():
        verify_file(Path(spec['path']), spec['sha256'])


def validate_cohort(rows: list[dict]) -> None:
    counts: Counter = Counter()
    ids: set[str] = set()
    for row in rows:
        cohort, sid = row['cohort'], row['song_id']
        if row['cache_id'] in ids or row['split'] not in {'train', 'dev'}:
            raise ValueError('duplicate or test example')
        ids.add(row['cache_id'])
        if (row['split'] == 'train' and sid not in TRAIN_SONGS) or (row['split'] == 'dev' and sid != 'shalong'):
            raise ValueError('song leakage in decoder experiment')
        if cohort != row['audio_role'] + '_' + row['split']:
            raise ValueError('original/rendered role mismatch')
        counts[cohort] += 1
    if counts != {'original_train': 20, 'original_dev': 3, 'rendered_train': 129, 'rendered_dev': 25}:
        raise ValueError('frozen decoder cohort differs')


def group_errors(group: dict) -> int:
    return sum(group[k] for k in ('substitutions', 'deletions', 'insertions'))


def promotion_gate(base: dict, trial: dict, policy: dict) -> dict:
    rules = policy['selection']
    original, rendered = trial['original_dev'], trial['rendered_dev']
    old, protect = base['original_dev'], base['rendered_dev']
    for cohort in ('original_dev', 'rendered_dev'):
        if any(trial[cohort][k] != base[cohort][k] for k in ('reference_phones', 'sequences')):
            raise ValueError('decoder reference denominator changed')
        if any(trial[cohort]['groups'][name]['reference_count'] != g['reference_count'] for name, g in base[cohort]['groups'].items()):
            raise ValueError('decoder class denominator changed')
    failures: list[str] = []
    if original['per'] > old['per'] * (1 - rules['minimum_relative_per_reduction']):
        failures.append('original_relative_improvement_below_5_percent')
    for kind in ('deletions', 'insertions'):
        if original[kind] > old[kind]:
            failures.append('original_' + kind + '_increased')
    if rendered['per'] > protect['per'] + rules['maximum_rendered_overall_per_increase']:
        failures.append('rendered_overall_regression')
    insufficient = []
    for cohort in ('original_dev', 'rendered_dev'):
        for name, prior in base[cohort]['groups'].items():
            new = trial[cohort]['groups'][name]
            if prior['reference_count'] >= rules['minimum_group_references']:
                if new['per'] > prior['per'] + rules['maximum_group_per_increase']:
                    failures.append(cohort + ':' + name + ':regression')
            else:
                insufficient.append(cohort + ':' + name)
                if name in {'aspirated', 'glottal_h'} and group_errors(new) > group_errors(prior):
                    failures.append(cohort + ':' + name + ':small_sample_error_increase')
    return {'passed': not failures, 'failures': failures, 'insufficient_groups': insufficient,
            'relative_original_per_reduction': (old['per'] - original['per']) / old['per']}


def reference_edits(reference: list[str], prediction: list[str]) -> list[dict]:
    result: list[dict] = []
    position, predicted = 0, 0
    for operation, expected, actual in align_phone_edits(reference, prediction):
        result.append({'operation': operation, 'reference': expected, 'prediction': actual,
                       'reference_index': position if expected is not None else None,
                       'prediction_index': predicted if actual is not None else None})
        position += expected is not None
        predicted += actual is not None
    return result


def sequence_digest(rows: list[dict]) -> str:
    return sha256_object([{k: row[k] for k in ('cache_id', 'song_id', 'split', 'audio_role', 'evidence_sha256')} for row in rows])

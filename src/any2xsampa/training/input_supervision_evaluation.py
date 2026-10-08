"""Reference-bound gates for the two predeclared input-supervision pipelines."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Any
from any2xsampa.training.input_supervision import policy
from any2xsampa.training.phone_error_metrics import phone_metrics


def aggregate(details: list[Mapping[str, Any]]) -> dict:
    return phone_metrics([(r['reference'], r['prediction']) for r in details])


def by_song(details: list[Mapping[str, Any]]) -> dict:
    groups = defaultdict(list)
    for row in details:
        groups[row['song_id']].append(row)
    return {song: aggregate(rows) for song, rows in sorted(groups.items())}


def compare_screen(variant: str, results: list[Mapping[str, Any]], baseline: Mapping[str, Any],
                   baseline_by_condition_song: Mapping[str, Any], baseline_details: Mapping[str, list]) -> dict:
    spec = policy()
    if variant not in spec['variants']:
        raise ValueError('unplanned pipeline')
    indexed = {r['held_song']: r for r in results}
    if len(indexed) != len(results):
        raise ValueError('duplicate held-song result')
    if set(indexed) != set(spec['songs']) or not all(r.get('complete') is True for r in results):
        return {'variant': variant, 'complete': False, 'passed': False, 'failures': ['incomplete_folds']}
    for row in results:
        if row.get('schema_version') != 'input_supervision_job_v2' or row['variant'] != variant or row['seed'] != 17 or row['fixed_epoch_count'] != 20:
            raise ValueError('legacy schedule or job identity cannot enter selection')
        if row['held_song'] in row['training_songs'] or set(row['training_songs']) != set(spec['songs']) - {row['held_song']}:
            raise ValueError('training/held song leakage')
    condition = spec['primary_inputs'][variant]
    for row in results:
        if any(d['song_id'] != row['held_song'] or d.get('condition') != condition
               for d in row['validation_' + condition]['examples']):
            raise ValueError('evaluation uses wrong held song or input condition')
    details = [d for row in results for d in row['validation_' + condition]['examples']]
    expected = {r['window_id']: r for r in baseline_details[condition]}
    current = {r['window_id']: r for r in details}
    if len(current) != len(details) or set(current) != set(expected) or len(current) != 20:
        raise ValueError('window coverage differs from frozen baseline')
    for name, row in current.items():
        if row['song_id'] != expected[name]['song_id'] or row['reference'] != expected[name]['reference']:
            raise ValueError('reference identity differs from frozen baseline')
    for c in ('x', 'v'):
        if aggregate(baseline_details[c]) != baseline[c] or by_song(baseline_details[c]) != baseline_by_condition_song[c]:
            raise ValueError('baseline metrics cannot be reproduced')
    metric, songs = aggregate(details), by_song(details)
    if metric['reference_phones'] != 1235:
        raise ValueError('fixed denominator changed')
    for row in results:
        validation = row['validation_' + condition]
        if aggregate(validation['examples']) != validation['metrics']:
            raise ValueError('fold metric differs from saved prediction')
    failures = []
    controls = list(dict.fromkeys([condition, 'x']))
    relative, macro = {}, {'new': sum(m['per'] for m in songs.values()) / 5}
    group_regressions = {}
    for control in controls:
        old = baseline[control]
        relative[control] = (old['per']-metric['per']) / old['per'] if old['per'] else None
        if relative[control] is None or relative[control] < spec['minimum_relative_improvement'] - 1e-12:
            failures.append(control + '_relative_improvement_below_5pct')
        macro[control] = sum(m['per'] for m in baseline_by_condition_song[control].values()) / 5
        if macro['new'] > macro[control] + 1e-12:
            failures.append(control + '_macro_regression')
        group_regressions[control] = [name for name, group in old['groups'].items()
                                     if group['reference_count'] >= spec['minimum_group_references'] and
                                     metric['groups'][name]['per'] > group['per'] + spec['maximum_group_per_increase'] + 1e-12]
        if group_regressions[control]:
            failures.append(control + '_protected_group_regression')
    improved = [song for song in spec['songs'] if songs[song]['per'] < baseline_by_condition_song[condition][song]['per']]
    if len(improved) < 3:
        failures.append('fewer_than_three_songs_improved')
    sensitivity = {}
    for omit in spec['songs']:
        new = aggregate([r for r in details if r['song_id'] != omit])
        old = aggregate([r for r in baseline_details[condition] if r['song_id'] != omit])
        sensitivity[omit] = {'new_errors': new['edit_distance'], 'old_errors': old['edit_distance'], 'references': new['reference_phones']}
    return {'variant': variant, 'complete': True, 'passed': not failures, 'failures': failures,
            'primary_condition': condition, 'metric': metric, 'song_metrics': songs, 'song_macro': macro,
            'relative_improvement_by_control': relative, 'strict_improved_songs': improved,
            'protected_group_regressions': group_regressions, 'leave_one_song_out_sensitivity': sensitivity,
            'insufficient_group_evidence': [name for name, g in metric['groups'].items() if g['reference_count'] < 20],
            'automatic_promotion': False}


def choose_pipeline(gates: list[dict]) -> str | None:
    if {r['variant'] for r in gates} != {'V', 'MV'} or len(gates) != 2:
        raise ValueError('both predeclared pipelines must be reported')
    if not all(r['complete'] for r in gates):
        return None
    eligible = [r for r in gates if r['passed']]
    return min(eligible, key=lambda r: (r['metric']['per'], r['variant'] != 'MV'))['variant'] if eligible else None

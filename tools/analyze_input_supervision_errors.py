"""Reference-position error changes on identical available held-song windows."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new
from any2xsampa.training.phone_error_metrics import align_phone_edits, phone_metrics


def indexed_edits(reference: list[str], prediction: list[str]) -> tuple[dict, Counter]:
    """Positions index references, not uncertain CTC emission timestamps."""
    position, states, insertions = 0, {}, Counter()
    for operation, expected, actual in align_phone_edits(reference, prediction):
        if operation == 'insertion':
            insertions[(position, actual)] += 1
        else:
            states[position] = (operation, expected, actual)
            position += 1
    if position != len(reference):
        raise ValueError('reference alignment coverage differs')
    return states, insertions


def compare_rows(old_rows: list[dict], new_rows: list[dict]) -> dict:
    originals = {r['window_id']: r for r in old_rows}
    if len({r['window_id'] for r in new_rows}) != len(new_rows):
        raise ValueError('duplicate new windows')
    changes, window_changes, controls = [], [], []
    counts = Counter()
    for new in new_rows:
        old = originals[new['window_id']]
        if new['reference'] != old['reference'] or new['song_id'] != old['song_id']:
            raise ValueError('error analysis requires identical reference and song')
        controls.append(old)
        a, ai = indexed_edits(old['reference'], old['prediction'])
        b, bi = indexed_edits(new['reference'], new['prediction'])
        old_metric = phone_metrics([(old['reference'], old['prediction'])])
        new_metric = phone_metrics([(new['reference'], new['prediction'])])
        window_changes.append({'window_id': new['window_id'], 'song_id': new['song_id'],
            'reference_phones': len(new['reference']),
            'old': {k: old_metric[k] for k in ('edit_distance', 'substitutions', 'deletions', 'insertions')},
            'new': {k: new_metric[k] for k in ('edit_distance', 'substitutions', 'deletions', 'insertions')}})
        for position in a:
            if a[position] == b[position]:
                continue
            if a[position][0] == 'match':
                kind = 'new_error'
            elif b[position][0] == 'match':
                kind = 'fixed_error'
            else:
                kind = 'changed_error_type_or_prediction'
            counts[kind] += 1
            changes.append({'window_id': new['window_id'], 'song_id': new['song_id'],
                            'reference_index': position, 'reference_phone': new['reference'][position],
                            'kind': kind, 'old': list(a[position]), 'new': list(b[position])})
        for key in sorted(set(ai) | set(bi)):
            if ai[key] != bi[key]:
                counts['insertion_count_delta'] += bi[key]-ai[key]
                changes.append({'window_id': new['window_id'], 'song_id': new['song_id'],
                                'reference_boundary_index': key[0], 'inserted_phone': key[1],
                                'kind': 'insertion_count_changed', 'old_count': ai[key], 'new_count': bi[key]})
    return {'covered_windows': len(new_rows), 'reference_phones': sum(len(r['reference']) for r in new_rows),
            'old_metrics_same_windows': phone_metrics([(r['reference'], r['prediction']) for r in controls]),
            'new_metrics': phone_metrics([(r['reference'], r['prediction']) for r in new_rows]),
            'counts': dict(counts), 'windows': window_changes, 'changes': changes,
            'alignment_caution': 'Repeated phones can admit multiple optimal edit alignments; positions are diagnostic, not acoustic boundary ground truth.'}


def analyze(run: Path, replay_dir: Path) -> None:
    freeze = read_json(run/'FREEZE.json')
    baseline = read_json(Path(freeze['probe_run'])/'RESULT.json')
    replay = read_json(replay_dir/'VERIFICATION.json')
    if not replay['all_checkpoints_verified'] or replay['freeze_sha256'] != sha256_file(run/'FREEZE.json'):
        raise ValueError('unverified replay')
    comparisons = {}
    for variant, condition in [('V', 'v'), ('MV', 'x')]:
        rows = [r for job in replay['checkpoints'] if job['variant'] == variant
                for r in job['details'] if r['condition'] == condition]
        for control in dict.fromkeys([condition, 'x']):
            comparisons[f'{variant}({condition}) vs M({control})'] = compare_rows(baseline['details'][control], rows)
    write_json_new(run/'ERROR_ANALYSIS.json', {
        'schema_version': 'input_supervision_error_analysis_v1', 'comparisons': comparisons,
        'source_replay_sha256': sha256_file(replay_dir/'VERIFICATION.json'),
        'no_new_human_reference_claimed': True, 'no_new_manual_listening_required': True,
        'error_positions_are_not_audio_timestamps': True,
    })
    print(run/'ERROR_ANALYSIS.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.replay.resolve())

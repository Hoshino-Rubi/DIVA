"""Score already-frozen audio hypotheses; references cannot reach decoding."""
from __future__ import annotations
import argparse, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.dual_input import load_policy, acceptance
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.decoder_experiment import reference_edits


def evaluate(run: Path) -> None:
    policy = load_policy(run); decoded = read_json(run/'DECODED.json')
    verify_file(run/'DECODER_CODE_LOCK.json', decoded['code_lock_sha256'])
    lock = read_json(run/'DECODER_CODE_LOCK.json')
    for record in lock['files']: verify_file(Path(record['path']), record['sha256'])
    if decoded['policy_sha256'] != sha256_file(run/'POLICY.json') or decoded['reference_received'] is not False:
        raise ValueError('unfrozen or reference-conditioned decoding')
    references = read_json(run/'SCORING_REFERENCES.json')['rows']
    if {e['example_id'] for e in decoded['entries']} != {r['example_id'] for r in references}:
        raise ValueError('hypothesis/reference coverage mismatch')
    source = read_json(Path(policy['bindings']['source_policy']['path']))
    vocabulary = read_json(Path(source['bindings']['vocabulary']['path']))
    labels = [p for p, i in sorted(vocabulary.items(), key=lambda x: x[1])]
    aggregate, per_song, windows = defaultdict(list), defaultdict(list), []
    for reference in references:
        spec = next(e for e in decoded['entries'] if e['example_id'] == reference['example_id'])
        verify_file(Path(spec['path']), spec['sha256']); hypotheses = read_json(Path(spec['path']))
        old = read_json(Path(policy['bindings']['source_policy']['path']).parent/'examples'/(reference['cache_id']+'.json'))
        if old['reference'] != reference['reference'] or old['song_id'] != reference['song_id'] or old['split'] != reference['split']:
            raise ValueError('historical reference denominator changed')
        predictions = {'historical_raw': old['predictions']['candidate'],
                       'mixed_greedy': [labels[i] for i in hypotheses['greedy']['mixed']],
                       'vocal_greedy': [labels[i] for i in hypotheses['greedy']['vocal']],
                       'dual_equal_ctc': [labels[i] for i in hypotheses['selected_ids']]}
        metrics = {}
        for name, prediction in predictions.items():
            pair = (reference['reference'], prediction)
            metrics[name] = phone_metrics([pair]); aggregate[(reference['split'], name)].append(pair)
            per_song[(reference['song_id'], name)].append(pair)
        row = {**reference, 'predictions': predictions, 'metrics': metrics,
               'edit_operations': {k: reference_edits(reference['reference'], v) for k,v in predictions.items()},
               'decode_path': spec['path'], 'decode_sha256': spec['sha256']}
        path = run/'evaluation'/f"{reference['example_id']}.json"; write_json_new(path, row)
        windows.append({'example_id': reference['example_id'], 'song_id': reference['song_id'], 'split': reference['split'],
                        'path': str(path), 'sha256': sha256_file(path), 'metrics': metrics})
    metrics = {split: {name: phone_metrics(aggregate[(split, name)]) for name in predictions} for split in ('train','dev')}
    baseline = metrics['dev']['historical_raw']
    if (baseline['edit_distance'], baseline['insertions'], baseline['deletions']) != (92,11,25):
        raise ValueError('current frozen baseline did not replay')
    outcome = acceptance(metrics['dev']['dual_equal_ctc'], baseline, policy['acceptance'])
    report = {'schema_version': 'dual_input_evaluation_v1', 'policy_sha256': sha256_file(run/'POLICY.json'),
        'decoded_index_sha256': sha256_file(run/'DECODED.json'), 'reference_sha256': sha256_file(run/'SCORING_REFERENCES.json'),
        'aggregate': metrics, 'by_song': {song: {name: phone_metrics(per_song[(song,name)]) for name in predictions}
                                       for song in sorted({r['song_id'] for r in references})},
        'windows': windows, 'acceptance': outcome, 'new_training_started': False, 'default_changed': False,
        'test_predictions_opened': False, 'reference_counts_unchanged': True,
        'sample_selection_after_score': False, 'ctc_time_is_phone_duration_accuracy': False}
    write_json_new(run/'EVALUATION.json', report)
    for split in ('train','dev'):
        for name, m in metrics[split].items():
            print(split, name, m['edit_distance'], m['reference_phones'], round(m['per'],6), 'S/D/I', m['substitutions'], m['deletions'], m['insertions'])
    print(outcome)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    if not a.run.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('output outside runs')
    evaluate(a.run.resolve())

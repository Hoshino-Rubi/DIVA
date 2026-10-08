"""Fresh-process replay, with independent Torch CTC verification of every score."""
from __future__ import annotations
import argparse, math, os, sys, time
from dataclasses import asdict
from collections import Counter, defaultdict
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.dual_input import load_policy, audio_requests, validate_pair, acceptance
from any2xsampa.adapters.dual_ctc import dual_decode
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
from any2xsampa.training.phone_error_metrics import phone_metrics


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1): current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def verify(run: Path) -> None:
    import numpy as np
    import torch
    from safetensors.numpy import load_file
    torch.set_num_threads(4)
    started = time.monotonic(); policy = load_policy(run); source = read_json(Path(policy['bindings']['source_policy']['path']))
    vocabulary_path = Path(source['bindings']['vocabulary']['path']); vocabulary = read_json(vocabulary_path)
    labels = [p for p, i in sorted(vocabulary.items(), key=lambda x: x[1])]
    decoded = read_json(run/'DECODED.json'); verify_file(run/'DECODER_CODE_LOCK.json', decoded['code_lock_sha256'])
    lock = read_json(run/'DECODER_CODE_LOCK.json'); verify_file(run/'EVIDENCE.json', lock['evidence_sha256'])
    for spec in lock['files']: verify_file(Path(spec['path']), spec['sha256'])
    evidence = read_json(run/'EVIDENCE.json'); verify_file(run/'PAIRS.json', evidence['pairs_sha256'])
    pairs = read_json(run/'PAIRS.json'); requests = audio_requests(read_json(run/'AUDIO_REQUESTS.json'))
    scoring = {r['example_id']: r for r in read_json(run/'SCORING_REFERENCES.json')['rows']}
    evaluation = read_json(run/'EVALUATION.json')
    for name, path in [('decoded_index_sha256', run/'DECODED.json'), ('reference_sha256', run/'SCORING_REFERENCES.json')]: verify_file(path, evaluation[name])
    checked_source = set(); score_count = fixture_count = 0; max_score_error = 0.; aggregates = defaultdict(list)
    insertions = defaultdict(Counter); unique_refs = {}; window_records = []
    for request in requests:
        if request['audio_sha256'] not in checked_source:
            verify_file(Path(request['audio_path']), request['audio_sha256']); checked_source.add(request['audio_sha256'])
        pair_spec = next(e for e in pairs['entries'] if e['example_id'] == request['example_id'])
        verify_file(Path(pair_spec['path']), pair_spec['sha256']); pair = read_json(Path(pair_spec['path'])); validate_pair(request, pair)
        spec = next(e for e in decoded['entries'] if e['example_id'] == request['example_id'])
        verify_file(Path(spec['path']), spec['sha256']); prior = read_json(Path(spec['path']))
        arrays, tensor_probs = {}, {}
        for role in ('mixed', 'vocal'):
            row = next(e for e in evidence['entries'] if e['example_id'] == request['example_id'] and e['condition'] == role)
            artifact = row['artifact']; verify_file(Path(artifact['evidence_path']), artifact['evidence_sha256'])
            verify_file(Path(artifact['posterior_path']), artifact['posterior_sha256'])
            arrays[role] = load_file(artifact['evidence_path'])['frame_logits']
            tensor_probs[role] = torch.from_numpy(arrays[role]).double().log_softmax(-1)[:, None, :]
            real = PhoneticXeusPosteriorRecognizer(Path(artifact['posterior_path']), vocabulary_path=vocabulary_path).recognize(Path(pair['audio'][role]['path']))
            fixture = PhoneticXeusPosteriorRecognizer(Path(artifact['posterior_path']), vocabulary_path=vocabulary_path, fixture=True).recognize(Path(pair['audio'][role]['path']))
            view = lambda rows: [{'time_range': asdict(r.time_range), 'candidate_phones': [asdict(p) for p in r.candidate_phones], 'confidence': r.confidence} for r in rows]
            if view(real) != view(fixture): raise ValueError('independent posterior fixture differs')
            fixture_count += 1
        actual = dual_decode(arrays['mixed'], arrays['vocal'])
        if any(prior[k] != value for k, value in actual.items()): raise ValueError('independent decoder replay differs')
        for candidate in prior['candidates']:
            target = torch.tensor(candidate['ids'], dtype=torch.long)
            for role, logp in tensor_probs.items():
                measured = -float(torch.nn.functional.ctc_loss(logp, target, torch.tensor([len(logp)]), torch.tensor([len(target)]), blank=0, reduction='sum', zero_infinity=False))
                expected = candidate['exact_log_probabilities'][role]
                if expected is None:
                    if measured != -math.inf: raise ValueError('impossible CTC path differs')
                else:
                    error = abs(measured-expected); max_score_error = max(max_score_error, error)
                    if not math.isfinite(error) or error > 1e-7: raise ValueError('independent exact CTC score differs')
                score_count += 1
        ref = scoring[request['example_id']]['reference']; split = request['split']
        row_spec = next(w for w in evaluation['windows'] if w['example_id'] == request['example_id'])
        verify_file(Path(row_spec['path']), row_spec['sha256']); scored = read_json(Path(row_spec['path']))
        predictions = {'mixed_greedy': [labels[i] for i in actual['greedy']['mixed']], 'vocal_greedy': [labels[i] for i in actual['greedy']['vocal']],
                       'dual_equal_ctc': [labels[i] for i in actual['selected_ids']], 'historical_raw': scored['predictions']['historical_raw']}
        for name, sequence in predictions.items():
            metric = phone_metrics([(ref,sequence)])
            if metric != scored['metrics'][name] or edit_distance(ref,sequence) != metric['edit_distance']:
                raise ValueError('independent edit/metric replay differs')
            aggregates[(split,name)].append((ref,sequence))
            for edit in scored['edit_operations'][name]:
                if edit['operation'] == 'insertion': insertions[(split,name)][edit['prediction']] += 1
        if split == 'train':
            # Candidate-pool diagnostic only. Never emitted or used to choose a system.
            pool_distance = min(edit_distance(ref,[labels[i] for i in c['ids']]) for c in actual['candidates'])
            unique_refs[request['example_id']] = {'reference_phones': len(ref), 'best_pool_edit_distance': pool_distance,
                'actual_edit_distance': scored['metrics']['dual_equal_ctc']['edit_distance'], 'scope': 'training_only_oracle_lower_bound_not_a_recognizer'}
        window_records.append({'example_id': request['example_id'], 'decoder_exact': True, 'candidate_count': len(actual['candidates'])})
    for (split,name), sequence_pairs in aggregates.items():
        if phone_metrics(sequence_pairs) != evaluation['aggregate'][split][name]: raise ValueError('aggregate metric differs')
    gate = acceptance(evaluation['aggregate']['dev']['dual_equal_ctc'], evaluation['aggregate']['dev']['historical_raw'], policy['acceptance'])
    if gate != evaluation['acceptance']: raise ValueError('acceptance changed during replay')
    for path, digest in read_json(run/'PRESERVATION_BEFORE.json')['files'].items(): verify_file(Path(path), digest)
    report = {'schema_version': 'dual_input_independent_verification_v1', 'windows': window_records,
        'independent_torch_ctc_scores_checked': score_count, 'maximum_ctc_score_absolute_error': max_score_error,
        'real_fixture_posteriors': fixture_count, 'all_metrics_reproduced': True, 'acceptance': gate,
        'source_files_and_historical_code_unchanged': True, 'gpu_model_loaded': False, 'cpu_math_verification_not_model_fallback': True,
        'elapsed_seconds': time.monotonic()-started, 'test_predictions_loaded': False}
    write_json_new(run/'INDEPENDENT_VERIFICATION.json', report)
    write_json_new(run/'ERROR_ANALYSIS.json', {'schema_version': 'dual_input_error_analysis_v1',
        'insertions_by_phone': {split: {name: dict(insertions[(split,name)]) for name in predictions} for split in ('train','dev')},
        'training_candidate_pool_diagnostic': unique_refs,
        'actual_train_errors': sum(v['actual_edit_distance'] for v in unique_refs.values()),
        'oracle_train_lower_bound_errors': sum(v['best_pool_edit_distance'] for v in unique_refs.values()),
        'oracle_predictions_used': False, 'no_reference_changes': True, 'new_experiment_or_tuning_started': False})
    print('Verified', len(window_records), 'windows;', score_count, 'Torch CTC scores;', fixture_count, 'real/fixture pairs.')
    print('Maximum independent CTC score error:', max_score_error)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    if not a.run.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('output outside runs')
    verify(a.run.resolve())

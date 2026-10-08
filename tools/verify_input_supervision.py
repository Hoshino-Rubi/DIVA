"""Independent evidence replay and bounded, unscored runtime verification."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new, sha256_object
from any2xsampa.training.phone_error_metrics import phone_metrics, align_phone_edits
from any2xsampa.adapters.phoneticxeus import emission_payload, PhoneticXeusPosteriorRecognizer, ctc_collapse


def distance(a: list, b: list) -> int:
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def export_replay(values, vocabulary: dict, entry: dict, metadata: dict, output: Path, vocabulary_path: Path) -> dict:
    from safetensors.numpy import save_file
    output.mkdir(parents=True, exist_ok=False)
    raw = output/'logits.safetensors'
    save_file({'frame_logits': values}, str(raw))
    vocab = output/'vocabulary.json'
    with vocab.open('xb') as stream:
        stream.write(vocabulary_path.read_bytes())
    payload = emission_payload(values, vocabulary, input_audio=entry['audio_metadata'], crop=entry['crop'],
                               model_metadata=metadata, geometry=entry['geometry'],
                               runtime={**entry['runtime'], 'verification_kind': 'saved_logit_replay'})
    payload['raw_evidence'] = {'file': raw.name, 'sha256': sha256_file(raw), 'vocabulary_file': vocab.name,
                               'vocabulary_sha256': sha256_file(vocab), 'logit_shape': list(values.shape), 'dtype': 'float32'}
    posterior = output/'posterior.json'
    write_json_new(posterior, payload)
    real = PhoneticXeusPosteriorRecognizer(posterior).recognize(Path(entry['audio_path']))
    fixture = PhoneticXeusPosteriorRecognizer(posterior, fixture=True).recognize(Path(entry['audio_path']))
    def view(segments):
        return [{'time_range': asdict(s.time_range), 'phones': [asdict(p) for p in s.candidate_phones], 'confidence': s.confidence} for s in segments]
    if view(real) != view(fixture):
        raise ValueError('real/fixture PhoneSegment fields differ')
    return {'posterior_path': str(posterior.resolve()), 'posterior_sha256': sha256_file(posterior),
            'prediction': payload['decoding']['collapsed_raw_labels'], 'real_fixture_equal': True,
            'logits_sha256': sha256_file(raw)}


def evidence(probe: Path, output: Path) -> None:
    from safetensors.numpy import load_file
    import numpy as np
    import soundfile as sf
    output.mkdir(parents=True, exist_ok=False)
    freeze = read_json(probe/'FREEZE.json')
    result = read_json(probe/'RESULT.json')
    cache = read_json(probe/'CACHE.json')
    index = {r['condition_id']: r for r in cache['entries']}
    source_refs = read_json(Path(freeze['historical_run'])/'FREEZE.json')
    vp = Path(source_refs['vocabulary_path'])
    vocabulary = read_json(vp)
    human = read_json(ROOT/'.state/runs/original-adaptation-20260919T012000/reviewed-training-reference.json')
    original = {r['example_id']: r for r in human['examples'] if r['split'] == 'train'}
    vowels = {'a','e','i','o','u','y','ɐ','ə','ɔ','ɛ','œ','ø','ɪ','ʊ'}
    checks, waveform_checks = [], []
    grouped = defaultdict(Counter)
    for condition in ('x', 'v', 'h'):
        for row in result['details'][condition]:
            entry = index[row['condition_id']]
            verify_file(Path(row['logits_path']), row['logits_sha256'])
            verify_file(Path(entry['path']), entry['sha256'])
            values = load_file(row['logits_path'])['frame_logits']
            artifact = export_replay(values, vocabulary, entry, {**cache['architecture']['snapshot'],
                'singing_domain_architecture': cache['architecture'],
                'experimental_checkpoint': {'sha256': row['checkpoint_sha256'], 'arm': 'full', 'variant': 'M'}},
                output/'posteriors'/row['condition_id'], vp)
            if artifact['prediction'] != row['prediction'] or distance(row['reference'], row['prediction']) != row['metrics']['edit_distance']:
                raise ValueError('independent saved-logit or distance verification failed')
            source = original[row['window_id']]
            if row['reference'] != [p['phone'] for p in source['phones']]:
                raise ValueError('reference changed from confirmed source')
            note_phones = defaultdict(list)
            for p in source['phones']:
                note_phones[p['note_id']].append(p['phone'])
            position = 0
            for operation, expected, actual in align_phone_edits(row['reference'], row['prediction']):
                if expected is None:
                    continue
                phone = source['phones'][position]
                category = 'multi_vowel_note' if sum(v in vowels for v in note_phones[phone['note_id']]) >= 2 else 'other_note'
                key = (condition, row['song_id'], expected, category)
                grouped[key]['reference_count'] += 1
                grouped[key][operation] += 1
                position += 1
            checks.append({'condition_id': row['condition_id'], 'distance': row['metrics']['edit_distance'], **artifact})
    for row in read_json(probe/'CONDITIONS.json')['generated']:
        wid = row['window_id']
        arrays = {}
        for c in ('x', 'v', 'h'):
            entry = index[wid+'--'+c]
            verify_file(Path(entry['audio_path']), entry['audio_sha256'])
            arrays[c], rate = sf.read(entry['audio_path'], dtype='float32', always_2d=True)
            if rate != 44100 or arrays[c].shape != (441000, 2) or not np.isfinite(arrays[c]).all():
                raise ValueError('paired waveform geometry/finite check failed')
        expected = ((arrays['x'].astype('float64')+arrays['v'].astype('float64'))*.5).astype('float32')
        error = float(np.max(np.abs(arrays['h']-expected)))
        if error > 2**-30:
            raise ValueError('midpoint waveform differs from fixed formula')
        waveform_checks.append({'window_id': wid, 'maximum_pcm_error': error, 'peak_abs': float(np.max(np.abs(arrays['h'])))})
    write_json_new(output/'VERIFICATION.json', {'saved_logit_checks': len(checks), 'independent_distance_checks': len(checks),
                   'fixture_checks': checks, 'waveform_checks': waveform_checks,
                   'source_reference_sha256': sha256_file(ROOT/'.state/runs/original-adaptation-20260919T012000/reviewed-training-reference.json'),
                   'probe_result_sha256': sha256_file(probe/'RESULT.json'),
                   'stratified_note_diagnostics': [dict(condition=k[0], song_id=k[1], phone=k[2], note_kind=k[3], **v) for k,v in sorted(grouped.items())],
                   'process_reserved_peak_recoverable': False,
                   'historical_peak_field_caveat': 'v1 RESULT recorded zero via nonexistent monitor keys; zero is not a measured peak',
                   'new_gpu_inference': False, 'final_test_opened': False})
    print('verified', len(checks), 'saved posteriors;', len(waveform_checks), 'fixed midpoint waveforms', flush=True)


def runtime_checks(run: Path, output: Path) -> None:
    from any2xsampa.training.input_supervision import load_manifest
    from any2xsampa.training.input_supervision_training import load_rows
    from any2xsampa.training.singing_domain_training import supervised_loss, predict
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime, make_tail
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    freeze, _, _ = load_manifest(run)
    historical = Path(freeze['historical_run'])
    source = next(r for r in read_json(historical/'AUDIO_REQUESTS.json')['rows'] if r['audio_role'] == 'original')
    output.mkdir(parents=True, exist_ok=False)
    with serial_candidate_job(), ProcessResourceMonitor(output/'monitor', timeout_seconds=900):
        rt = SingingDomainRuntime(Path(freeze['model_snapshot']), output/'runtime-cache')
        torch = rt.torch
        smoke = []
        for seconds in (8., 20.):
            request = {**source, 'example_id': 'smoke'+str(int(seconds)), 'end_sec': source['start_sec']+seconds}
            tensors, payload = rt.encode_prefix(request)
            entry = {**request, 'audio_metadata': payload['producer']['input_audio'], 'crop': payload['producer']['crop'],
                     'geometry': payload['decoding']['geometry'], 'runtime': payload['producer']['runtime']}
            artifact = export_replay(tensors['frame_logits'].numpy(), rt.vocabulary, entry,
                                    payload['producer']['model_metadata'], output/request['example_id'], Path(freeze['vocabulary_path']))
            smoke.append({'seconds': seconds, 'frames': len(tensors['frame_logits']), 'reference_used': False, **artifact})
        rows, _ = load_rows(run, torch)
        tail = make_tail(torch, rt.model, 'full')
        rt.model.frontend.cpu(); rt.model.preencoder.cpu()
        for block in list(rt.model.encoder.encoders)[:15]: block.cpu()
        rt.model.encoder.conditioning_layer.cpu()
        frozen = [p for p in rt.model.parameters() if not p.requires_grad]
        versions = [p._version for p in frozen]
        same_window = next(r['window_id'] for r in rows if r['condition'] == 'x')
        probes = [r for r in rows if r['window_id'] == same_window]
        probes.append(next(r for r in rows if r['condition'] == 'rendered'))
        spec = read_json(historical/'POLICY.json')
        initial_errors, losses = {}, {}
        tail.eval()
        for row in probes:
            with torch.no_grad(): logits, _ = predict(torch, tail, row)
            error = float((logits[0].cpu()-row['frame_logits']).abs().max())
            if error > 2e-5: raise ValueError('initial frozen-prefix logits differ')
            initial_errors[row['condition']] = error
        tail.train()
        for row in probes:
            loss, _ = supervised_loss(torch, tail, row, {}, 'baseline', spec, {})
            (loss/len(probes)).backward()
            losses[row['condition']] = float(loss.detach())
        parameters = [p for p in tail.parameters() if p.requires_grad]
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('non-finite/missing direct-supervision gradients')
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4, foreach=False)
        before = parameters[0].detach().clone()
        optimizer.step()
        if torch.equal(before, parameters[0]) or versions != [p._version for p in frozen]:
            raise ValueError('trainable/frozen parameter verification failed')
        write_json_new(output/'VERIFICATION.json', {'smoke': smoke, 'initial_logit_errors': initial_errors,
                       'one_unscored_optimizer_step': True, 'losses': losses, 'frozen_parameters_unchanged': True,
                       'finite_gradients': True, 'checkpoint_saved': False, 'quality_jobs_started': 0,
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'final_test_opened': False})
        print('8s/20s real+fixture and unscored gradient/optimizer verification passed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['evidence', 'runtime'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('output outside runs')
    (evidence if a.mode == 'evidence' else runtime_checks)(a.run.resolve(), a.output.resolve())

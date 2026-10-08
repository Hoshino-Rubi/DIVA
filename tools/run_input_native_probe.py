"""Bounded audio-only worker for fixed-G input and passive native-CTC probes."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from input_native_probe_common import (ROOT, verify_review, native_observer, native_outputs,
    compare_logits, parameter_digest, check_stage, LIMITS)
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file
from verify_input_supervision import export_replay


def worker(run: Path, stage: str) -> None:
    freeze, review = verify_review(run)
    task_path = run / 'tasks' / (stage + '.json')
    task_hash = freeze['bindings'].get(str(task_path))
    if task_hash is None:
        raise ValueError('unbound stage task list')
    verify_file(task_path, task_hash)
    tasks = read_json(task_path)['tasks']
    check_stage(run, stage, tasks, worker=True)
    if not tasks or len(tasks) > LIMITS[stage] or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('invalid fixed stage budget')
    from any2xsampa.training.singing_domain import validate_audio_row
    for task in tasks:
        validate_audio_row(task['request'], allow_dev=stage in {'A2', 'ADR'})
    dest = run / stage
    write_json_new(dest / 'START.json', {'stage': stage, 'task_sha256': task_hash,
        'review_sha256': sha256_file(run / 'CODE_REVIEW.json'), 'planned_forwards': len(tasks),
        'training_starts': 0, 'failed_launches_count': True})
    from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime, enforce_offline
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.guarded_phoneticxeus import GuardedPhoneHead
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from safetensors.numpy import load_file
    enforce_offline(dest / 'runtime-cache')
    start = time.monotonic(); records = []
    modes = {t['mode'] for t in tasks}
    if len(modes) != 1:
        raise ValueError('one architecture per stage process required')
    mode = modes.pop()
    with serial_candidate_job(), ProcessResourceMonitor(dest / 'monitor', timeout_seconds=1800):
        if mode == 'G':
            base = PhoneticXeusRuntime(Path(freeze['model_snapshot']), cache_dir=dest / 'runtime-cache')
            spec = freeze['candidate']
            engine = GuardedPhoneHead(base.torch, base.model.model.ctc.ctc_lo, base.vocabulary, Path(spec['path']), spec['sha256'])
            architecture = {'kind': 'legacy_G', 'interctc_layer_idx': list(base.model.model.encoder.interctc_layer_idx)}
            if architecture['interctc_layer_idx']:
                raise ValueError('legacy G effective architecture changed')
        else:
            rt = SingingDomainRuntime(Path(freeze['model_snapshot']), dest / 'runtime-cache')
            base = rt.base
            architecture = rt.identity
        model = base.model.model
        before = parameter_digest(model)
        head_before = {k: parameter_digest(getattr(engine, k)) for k in ('teacher', 'student')} if mode == 'G' else {}
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('probe parameters must remain frozen')
        initial = {'model_parameters_sha256': before, 'residual_parameters': head_before, 'architecture': architecture,
                   'torch': base.torch.__version__, 'runtime': base.runtime}
        write_json_new(dest / 'INITIALIZATION.json', initial)
        for task in tasks:
            ident = task['id']; directory = dest / 'forwards' / ident
            if len(list(run.glob('*/forwards/*/START.json'))) >= 111:
                raise ValueError('complete forward budget exhausted')
            write_json_new(directory / 'START.json', {'request': task['request'], 'mode': mode,
                'task_id': ident, 'stage': stage, 'complete_forward_attempt': True})
            t0 = time.monotonic(); obs = None
            if mode == 'P':
                with native_observer(model, capture=task.get('capture', False)) as obs:
                    features, logits, payload = base.encode(task['request'])
                if obs['condition_calls'] != 3:
                    raise ValueError('conditioning call count changed')
                values = logits.numpy()
                outputs = native_outputs(obs, values, base.vocabulary) if task.get('capture') else {'19': values}
                metadata = {**base.identity, 'singing_domain_architecture': architecture}
                if task.get('expected_architecture') not in (None, architecture['sha256']):
                    raise ValueError('P cache architecture differs')
            else:
                features, _, payload = base.encode(task['request'])
                outputs = {'G': engine.predict(features)['frame_logits']}
                metadata = {**base.identity, 'guarded_candidate_sha256': freeze['candidate']['sha256'],
                            'effective_architecture': architecture}
                values = outputs['G']
            p = payload['producer']
            if task.get('expected_crop') is not None and p['crop'] != task['expected_crop']:
                raise ValueError('crop sample/preprocessing identity changed')
            checks = {}
            if task.get('compare'):
                spec = task['compare']; verify_file(Path(spec['path']), spec['sha256'])
                expected = load_file(spec['path'])[spec['key']]
                checks['historical_logits'] = compare_logits(values, expected, base.vocabulary)
            if task.get('compare_smoke'):
                prior = dest / 'forwards' / task['compare_smoke'] / '19/logits.safetensors'
                checks['observation_equivalence'] = compare_logits(values, load_file(str(prior))['frame_logits'], base.vocabulary)
            artifacts = {}
            entry = {'audio_path': task['request']['audio_path'], 'audio_metadata': p['input_audio'],
                     'crop': p['crop'], 'geometry': payload['decoding']['geometry'], 'runtime': p['runtime']}
            for layer, array in outputs.items():
                artifact = export_replay(array, base.vocabulary, entry, {**metadata, 'native_outlet': layer},
                                         directory / layer, Path(freeze['vocabulary']['path']))
                artifacts[layer] = artifact
                if task.get('replay_of'):
                    prior = run / task['replay_of'] / 'forwards' / ident / layer / 'logits.safetensors'
                    checks['independent_' + layer] = compare_logits(array, load_file(str(prior))['frame_logits'], base.vocabulary)
                    oldp = read_json(prior.parent / 'posterior.json')
                    newp = read_json(Path(artifact['posterior_path']))
                    if oldp['decoding'] != newp['decoding'] or oldp['segments'] != newp['segments']:
                        raise ValueError('independent posterior replay mismatch')
            result = {'task_id': ident, 'request': task['request'], 'checks': checks, 'outputs': artifacts,
                'conditioning_calls': obs['condition_calls'] if obs else None,
                'crop': p['crop'], 'runtime': p['runtime'], 'elapsed_seconds': time.monotonic() - t0}
            write_json_new(directory / 'RESULT.json', result); records.append(result)
            print(stage, ident, 'verified', round(result['elapsed_seconds'], 3), flush=True)
        after = parameter_digest(model)
        if after != before or any(p.requires_grad for p in model.parameters()):
            raise ValueError('frozen model parameters changed')
        head_after = {k: parameter_digest(getattr(engine, k)) for k in ('teacher', 'student')} if mode == 'G' else {}
        if head_before != head_after:
            raise ValueError('frozen guarded residual parameters changed')
        report = {'passed': True, 'stage': stage, 'mode': mode, 'records': records,
            'complete_forwards': len(records), 'model_parameters_before': before, 'model_parameters_after': after,
            'frozen_parameters_unchanged': True, 'residual_parameters': head_after, 'architecture': architecture, 'labels_received': False,
            'peak_reserved_bytes': base.torch.cuda.max_memory_reserved(), 'elapsed_seconds': time.monotonic() - start,
            'training_starts': 0, 'default_replaced': False, 'final_test_opened': False}
    write_json_new(dest / 'RESULT.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', choices=list(LIMITS), required=True)
    args = p.parse_args(); worker(args.run.resolve(), args.stage)

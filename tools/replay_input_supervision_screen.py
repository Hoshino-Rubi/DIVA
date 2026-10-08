"""Independent x/v/h replay of completed, reviewed V/MV screen checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from any2xsampa.training.input_supervision import load_manifest, policy
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from verify_input_supervision import export_replay, distance


def replay(run: Path, output: Path) -> None:
    from safetensors.torch import load_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime, make_tail, load_trainable_state
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    freeze, manifest, cache = load_manifest(run)
    if output.exists() or not output.is_relative_to(run):
        raise ValueError('choose a new replay directory inside this run')
    probe = Path(freeze['probe_run'])
    probe_cache = read_json(probe / 'CACHE.json')
    refs = {r['window_id']: r for r in manifest['rows'] if r['condition'] == 'x'}
    entries = {(r['cache_id'].rsplit('--', 1)[0], r['condition']): r
               for r in cache['entries'] if r['condition'] in ('x', 'v')}
    for row in probe_cache['entries']:
        if row['condition_id'].endswith('--h'):
            entries[(row['condition_id'].rsplit('--', 1)[0], 'h')] = row
    jobs = []
    unavailable = set(freeze.get('unavailable_screen_jobs', []))
    for variant in policy()['variants']:
        for held in policy()['songs']:
            if f'{variant}-{held}' in unavailable:
                continue
            path = run / 'screen' / variant / held / 'RESULT.json'
            result = read_json(path)
            if (result.get('schema_version') != 'input_supervision_job_v2' or not result.get('complete')
                    or result['variant'] != variant or result['held_song'] != held
                    or result['freeze_sha256'] != sha256_file(run / 'FREEZE.json')
                    or result['code_bindings'] != freeze['code_bindings']
                    or result['frozen_parameters_unchanged'] is not True
                    or [e['epoch'] for e in result['history']] != list(range(1, 21))):
                raise ValueError('invalid or incomplete screen job')
            verify_file(Path(result['checkpoint_path']), result['checkpoint_sha256'])
            jobs.append((path, result))
    output.mkdir(parents=True)
    reports = []
    with serial_candidate_job(), ProcessResourceMonitor(output / 'monitor', timeout_seconds=1200):
        rt = SingingDomainRuntime(Path(freeze['model_snapshot']), output / 'runtime-cache')
        torch = rt.torch
        tail = make_tail(torch, rt.model, 'full')
        labels = [p for p, _ in sorted(rt.vocabulary.items(), key=lambda item: item[1])]
        for path, result in jobs:
            load_trainable_state(tail, load_file(result['checkpoint_path'], device='cpu'))
            tail.eval()
            details = []
            with torch.no_grad():
                for (wid, condition), entry in entries.items():
                    if entry['song_id'] != result['held_song']:
                        continue
                    verify_file(Path(entry['path']), entry['sha256'])
                    prefix = load_file(entry['path'], device='cpu')['prefix_features'].unsqueeze(0).cuda()
                    logits, _ = tail(prefix, torch.tensor([prefix.shape[1]], device='cuda'))
                    values = logits[0].cpu().contiguous()
                    if not torch.isfinite(values).all():
                        raise ValueError('non-finite replay logits')
                    prediction = [labels[i] for i in ctc_collapse(values.argmax(-1).tolist())]
                    logit_error = None
                    if condition in ('x', 'v'):
                        old = next(r for r in result['validation_' + condition]['examples'] if r['window_id'] == wid)
                        verify_file(Path(old['logits_path']), old['logits_sha256'])
                        saved = load_file(old['logits_path'], device='cpu')['frame_logits']
                        if saved.shape != values.shape:
                            raise ValueError('replay logit geometry differs')
                        logit_error = float((values - saved).abs().max())
                        if (old['reference'] != refs[wid]['target_ipa'] or old['prediction'] != prediction
                                or logit_error > 2e-5):
                            raise ValueError('checkpoint replay differs from saved validation')
                    artifact = export_replay(values.numpy(), rt.vocabulary, entry,
                        {**rt.base.identity, 'experimental_checkpoint': {
                            'sha256': result['checkpoint_sha256'], 'variant': result['variant'],
                            'held_song': result['held_song'], 'h_is_diagnostic_only': True}},
                        output / result['variant'] / (wid + '--' + condition), Path(freeze['vocabulary_path']))
                    if artifact['prediction'] != prediction:
                        raise ValueError('posterior replay changed the phone sequence')
                    metrics = phone_metrics([(refs[wid]['target_ipa'], prediction)])
                    if distance(refs[wid]['target_ipa'], prediction) != metrics['edit_distance']:
                        raise ValueError('independent edit distance mismatch')
                    details.append({'window_id': wid, 'song_id': result['held_song'], 'condition': condition,
                                    'reference': refs[wid]['target_ipa'], 'prediction': prediction, 'metrics': metrics,
                                    'saved_logit_maximum_error': logit_error, **artifact})
            # A fresh full waveform checks the frozen-prefix path for each checkpoint.
            entry = next(e for (wid, c), e in entries.items()
                         if c == 'v' and e['song_id'] == result['held_song'])
            request = {k: entry[k] for k in ('example_id', 'song_id', 'split', 'audio_role',
                                            'audio_path', 'audio_sha256', 'start_sec', 'end_sec')}
            live, _ = rt.encode_prefix(request)
            prefix = load_file(entry['path'], device='cpu')['prefix_features'].unsqueeze(0).cuda()
            with torch.no_grad():
                expected, _ = tail(prefix, torch.tensor([prefix.shape[1]], device='cuda'))
            error = float((live['frame_logits'] - expected[0].cpu()).abs().max())
            if error > 2e-5:
                raise ValueError('fresh audio and frozen-prefix replay differ')
            reports.append({'variant': result['variant'], 'held_song': result['held_song'],
                            'source_result_path': str(path), 'source_result_sha256': sha256_file(path),
                            'checkpoint_sha256': result['checkpoint_sha256'], 'details': details,
                            'full_audio_maximum_logit_error': error,
                            'metrics_by_condition': {c: phone_metrics([(r['reference'], r['prediction'])
                                for r in details if r['condition'] == c]) for c in ('x', 'v', 'h')}})
            print('REPLAYED', result['variant'], result['held_song'], flush=True)
        write_json_new(output / 'VERIFICATION.json', {
            'schema_version': 'input_supervision_screen_replay_v1', 'checkpoints': reports,
            'freeze_sha256': sha256_file(run / 'FREEZE.json'), 'all_checkpoints_verified': True,
            'unavailable_jobs': sorted(unavailable), 'complete_ten_fold_screen': len(reports) == 10,
            'quality_training_started': False, 'development_opened': False, 'final_test_opened': False,
            'h_is_diagnostic_only': True, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.run.resolve(), args.output.resolve())

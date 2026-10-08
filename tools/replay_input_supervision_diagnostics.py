"""Replay superseded V checkpoints without treating them as valid quality trials."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from verify_input_supervision import export_replay, distance


def replay(incident: Path, probe: Path, output: Path) -> None:
    from safetensors.torch import load_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime, make_tail, load_trainable_state
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    jobs = [a for a in read_json(incident)['attempts'] if a['complete']]
    source = read_json(probe/'FREEZE.json')
    vocab_path = Path(source['model_snapshot'])/'src/model/xeusphoneme/resources/ipa_vocab.json'
    cache = read_json(probe/'CACHE.json')
    refs = {r['window_id']: r for r in read_json(probe/'EVALUATION_REFERENCES.json')['rows']}
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with serial_candidate_job(), ProcessResourceMonitor(output/'monitor', timeout_seconds=900):
        rt = SingingDomainRuntime(Path(source['model_snapshot']), output/'runtime-cache')
        torch = rt.torch
        tail = make_tail(torch, rt.model, 'full')
        labels = [p for p, _ in sorted(rt.vocabulary.items(), key=lambda item: item[1])]
        for number, job in enumerate(jobs):
            path = Path(job['path'])
            verify_file(path/'RESULT.json', job['result_sha256'])
            result = read_json(path/'RESULT.json')
            verify_file(Path(result['checkpoint_path']), result['checkpoint_sha256'])
            load_trainable_state(tail, load_file(result['checkpoint_path'], device='cpu'))
            tail.eval()
            entries = [r for r in cache['entries'] if r['song_id'] == result['held_song']]
            details = []
            with torch.no_grad():
                for entry in entries:
                    cid = entry['condition_id']
                    wid, condition = cid.rsplit('--', 1)
                    verify_file(Path(entry['path']), entry['sha256'])
                    tensors = load_file(entry['path'], device='cpu')
                    prefix = tensors['prefix_features'].unsqueeze(0).cuda()
                    logits, _ = tail(prefix, torch.tensor([prefix.shape[1]], device='cuda'))
                    values = logits[0].cpu().contiguous()
                    prediction = [labels[i] for i in ctc_collapse(values.argmax(-1).tolist())]
                    if condition in ('x', 'v'):
                        old = next(r for r in result['validation_'+condition]['examples'] if r['window_id'] == wid)
                        if old['reference'] != refs[wid]['target_ipa'] or old['prediction'] != prediction:
                            raise ValueError('checkpoint replay differs from saved prediction')
                    artifact = export_replay(values.numpy(), rt.vocabulary, entry,
                        {**rt.base.identity, 'experimental_checkpoint': {'sha256': result['checkpoint_sha256'], 'variant': 'V',
                         'status': 'superseded_schedule_mismatch_not_for_selection'}},
                        output/f'job-{number+1}'/cid, vocab_path)
                    details.append({'window_id': wid, 'condition': condition, 'reference': refs[wid]['target_ipa'],
                                    'prediction': prediction, 'distance': distance(refs[wid]['target_ipa'], prediction), **artifact})
            # One fresh waveform per checkpoint confirms prefix replay equality.
            entry = next(r for r in entries if r['condition_id'].endswith('--v'))
            audio_request = {k: entry[k] for k in ('example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec')}
            live, _ = rt.encode_prefix(audio_request)
            prefix = load_file(entry['path'], device='cpu')['prefix_features'].unsqueeze(0).cuda()
            with torch.no_grad(): expected, _ = tail(prefix, torch.tensor([prefix.shape[1]], device='cuda'))
            error = float((live['frame_logits']-expected[0].cpu()).abs().max())
            if error > 2e-5: raise ValueError('fresh waveform/tail replay differs')
            results.append({'job_path': str(path), 'checkpoint_sha256': result['checkpoint_sha256'], 'details': details,
                            'source_result_sha256': job['result_sha256'], 'full_audio_maximum_logit_error': error,
                            'metrics_by_condition': {c: phone_metrics([(r['reference'], r['prediction']) for r in details if r['condition']==c]) for c in ('x','v','h')},
                            'valid_for_selection': False})
            print('replayed', number+1, '/', len(jobs), flush=True)
        write_json_new(output/'VERIFICATION.json', {'checkpoints': results, 'all_valid_for_selection': False,
                       'quality_training_started': False, 'h_is_diagnostic_only': True,
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'final_test_opened': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('incident', 'probe', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('output outside runs')
    replay(a.incident.resolve(), a.probe.resolve(), a.output.resolve())

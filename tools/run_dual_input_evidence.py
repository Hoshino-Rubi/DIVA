"""Extract matched train evidence; reuse exact frozen development evidence."""
from __future__ import annotations
import argparse, sys, time
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, sha256_object, write_json_new
from any2xsampa.training.dual_input import load_policy, audio_requests, validate_pair


def run_worker(run: Path) -> None:
    policy = load_policy(run); requests = audio_requests(read_json(run/'AUDIO_REQUESTS.json'))
    pairs = read_json(run/'PAIRS.json')
    if pairs['policy_sha256'] != sha256_file(run/'POLICY.json'): raise ValueError('pair policy mismatch')
    if {e['example_id'] for e in pairs['entries']} != {r['example_id'] for r in requests}: raise ValueError('incomplete paired cohort')
    transfer = read_json(Path(policy['bindings']['guarded_transfer']['path']))
    if transfer['candidate_sha256'] != policy['bindings']['candidate']['sha256']: raise ValueError('cached development candidate differs')
    source = read_json(Path(policy['bindings']['source_policy']['path']))
    vp = Path(source['bindings']['vocabulary']['path'])
    from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime, enforce_offline, gpu_total_used_bytes
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.guarded_phoneticxeus import GuardedPhoneHead
    from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
    from safetensors.numpy import load_file
    output = run/'evidence'; output.mkdir(exist_ok=False); enforce_offline(output/'runtime-cache')
    if gpu_total_used_bytes() > 14*1024**3: raise RuntimeError('card budget exceeded')
    started = time.monotonic(); entries = []; runtime = engine = None
    with ProcessResourceMonitor(output/'monitor', timeout_seconds=1800):
        for request in requests:
            spec = next(e for e in pairs['entries'] if e['example_id'] == request['example_id'])
            verify_file(Path(spec['path']), spec['sha256']); pair = read_json(Path(spec['path'])); validate_pair(request, pair)
            for role in ('mixed', 'vocal'):
                audio = pair['audio'][role]; before = time.monotonic()
                if request['split'] == 'dev' and pair['cache_mode'] == 'exact_pair_reused':
                    cached = next(e for e in transfer['posteriors'] if e['example_id'] == request['example_id'] and e['condition'] == role)
                    artifact = {k: cached[k] for k in ('posterior_path', 'posterior_sha256', 'evidence_path', 'evidence_sha256', 'frames', 'guarded_frames')}
                    verify_file(Path(artifact['posterior_path']), artifact['posterior_sha256'])
                    verify_file(Path(artifact['evidence_path']), artifact['evidence_sha256'])
                    payload = read_json(Path(artifact['posterior_path']))
                    if payload['producer']['input_audio']['sha256'] != audio['sha256']:
                        raise ValueError('cached development waveform does not match exact pair')
                    cache_mode = 'exact_frozen_posterior_reused'
                else:
                    if runtime is None:
                        runtime = PhoneticXeusRuntime(Path(policy['phonetic_snapshot']), cache_dir=output/'runtime-cache')
                        candidate = policy['bindings']['candidate']
                        engine = GuardedPhoneHead(runtime.torch, runtime.model.model.ctc.ctc_lo, runtime.vocabulary, Path(candidate['path']), candidate['sha256'])
                    row = {'example_id': request['example_id']+'-'+role, 'song_id': request['song_id'], 'split': request['split'],
                           'audio_role': 'original_resampled_mix' if role == 'mixed' else 'original_separated_vocal',
                           'audio_path': audio['path'], 'audio_sha256': audio['sha256'], 'start_sec': 0., 'end_sec': audio['samples']/44100}
                    features, _, payload = runtime.encode(row)
                    artifact = engine.save(features, payload, output/request['example_id']/role)
                    cache_mode = 'new_frozen_model_inference'
                pp = Path(artifact['posterior_path']); source_audio = Path(audio['path'])
                a = PhoneticXeusPosteriorRecognizer(pp, vocabulary_path=vp).recognize(source_audio)
                b = PhoneticXeusPosteriorRecognizer(pp, fixture=True, vocabulary_path=vp).recognize(source_audio)
                view = lambda rows: [{'time_range': asdict(s.time_range), 'candidate_phones': [asdict(p) for p in s.candidate_phones], 'confidence': s.confidence} for s in rows]
                if view(a) != view(b): raise ValueError('real/fixture posterior differs')
                logits = load_file(artifact['evidence_path'])['frame_logits']
                if logits.shape != (artifact['frames'], 428): raise ValueError('full inventory/frame geometry changed')
                entries.append({'example_id': request['example_id'], 'song_id': request['song_id'], 'split': request['split'],
                    'condition': role, 'paired_record': spec, 'input_audio': audio, 'artifact': artifact, 'cache_mode': cache_mode,
                    'elapsed_seconds': time.monotonic()-before, 'real_fixture_equal': True, 'contract_sha256': sha256_object(view(a))})
                print(request['example_id'], role, cache_mode, round(time.monotonic()-before, 3), flush=True)
        write_json_new(run/'EVIDENCE.json', {'schema_version': 'dual_input_evidence_v1', 'policy_sha256': sha256_file(run/'POLICY.json'),
            'pairs_sha256': sha256_file(run/'PAIRS.json'), 'candidate_sha256': policy['bindings']['candidate']['sha256'],
            'entries': entries, 'elapsed_seconds': time.monotonic()-started,
            'peak_reserved_bytes': runtime.torch.cuda.max_memory_reserved() if runtime else 0, 'reference_received': False, 'test_predictions_loaded': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    if not a.run.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('output outside runs')
    run_worker(a.run.resolve())

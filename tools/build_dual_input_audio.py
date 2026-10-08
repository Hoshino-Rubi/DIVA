"""Reuse verified separation masters and create only missing frozen contexts."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.dual_input import load_policy, audio_requests, validate_pair


def build(run: Path) -> None:
    policy = load_policy(run); requests = audio_requests(read_json(run/'AUDIO_REQUESTS.json'))
    output = run/'audio'; output.mkdir(exist_ok=False)
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline, gpu_total_used_bytes
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.auxiliary_audio import read_stereo_context, write_stereo_float
    from any2xsampa.adapters.melband_runtime import MelBandRuntime
    from build_quality8h_pairs import paired_pcm
    import soundfile as sf
    enforce_offline(output/'runtime-cache')
    if gpu_total_used_bytes() > 14*1024**3: raise RuntimeError('card budget exceeded')
    entries, runtime = [], None
    started = time.monotonic()
    with ProcessResourceMonitor(output/'monitor', timeout_seconds=1800):
        for request in requests:
            folder = output/request['example_id']; before = time.monotonic()
            verify_file(Path(request['audio_path']), request['audio_sha256'])
            if request['reuse_exact_pair'] is not None:
                old = request['reuse_exact_pair']
                pair = {k: old[k] for k in ('audio', 'transform', 'effective_original_start_sec', 'effective_original_end_sec', 'separation_record_path', 'separation_record_sha256')}
                mode = 'exact_pair_reused'; measured = None
            else:
                if request['reuse_separation'] is not None:
                    spec = request['reuse_separation']; verify_file(Path(spec['path']), spec['sha256'])
                    record = read_json(Path(spec['path'])); separation_path = Path(spec['path']); mode = 'context_reused'; measured = None
                else:
                    if runtime is None: runtime = MelBandRuntime(Path(policy['quality_asset_run']), output/'runtime-cache')
                    mixed, clock = read_stereo_context(Path(request['audio_path']), request['audio_sha256'], request['context_start_sec'], request['context_end_sec'])
                    vocal, measured = runtime.separate(mixed)
                    record = {'schema_version': 'dual_input_separation_v1', 'clock': clock,
                              'mixed': write_stereo_float(folder/'mixed-master.wav', mixed, 44100),
                              'vocal': write_stereo_float(folder/'vocal-master.wav', vocal, 44100),
                              'model': runtime.identity, 'runtime': measured, 'labels_received': False}
                    separation_path = folder/'separation.json'; write_json_new(separation_path, record); mode = 'new_context'
                if record['clock']['source']['sha256'] != request['audio_sha256']: raise ValueError('cached context belongs to another source')
                arrays = {}
                for role in ('mixed', 'vocal'):
                    verify_file(Path(record[role]['path']), record[role]['sha256'])
                    arrays[role], rate = sf.read(record[role]['path'], dtype='float32', always_2d=True)
                    if rate != 44100 or len(arrays[role]) != record[role]['samples']: raise ValueError('cached master geometry changed')
                origin = record['clock']['effective_original_start_sec']
                first = round((request['start_sec']-origin)*44100); last = round((request['end_sec']-origin)*44100)
                audio, transform = paired_pcm(arrays['mixed'], arrays['vocal'], first, last, folder/'pair')
                pair = {'audio': audio, 'transform': transform, 'effective_original_start_sec': origin+first/44100,
                        'effective_original_end_sec': origin+last/44100, 'separation_record_path': str(separation_path), 'separation_record_sha256': sha256_file(separation_path)}
            pair.update(schema_version='dual_input_pair_v1', example_id=request['example_id'], song_id=request['song_id'], split=request['split'],
                        source_audio_path=request['audio_path'], source_audio_sha256=request['audio_sha256'],
                        source_start_sample=request['source_start_sample'], source_end_sample_exclusive=request['source_end_sample_exclusive'],
                        source_rate=request['source_rate'], contains_reference_context=False, cache_mode=mode,
                        stage_elapsed_seconds=time.monotonic()-before, separation_runtime=measured)
            validate_pair(request, pair)
            path = folder/'pair.json'; write_json_new(path, pair)
            entries.append({'example_id': request['example_id'], 'path': str(path), 'sha256': sha256_file(path), 'cache_mode': mode})
            print(request['example_id'], mode, round(time.monotonic()-before, 3), flush=True)
        write_json_new(run/'PAIRS.json', {'schema_version': 'dual_input_pairs_v1', 'policy_sha256': sha256_file(run/'POLICY.json'),
            'entries': entries, 'elapsed_seconds': time.monotonic()-started, 'peak_reserved_bytes': runtime.torch.cuda.max_memory_reserved() if runtime else 0,
            'labels_received': False, 'test_audio_opened': False, 'new_contexts': sum(x['cache_mode'] == 'new_context' for x in entries)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    if not a.run.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('output outside runs')
    build(a.run.resolve())

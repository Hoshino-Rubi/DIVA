"""Only cached FP32 CUDA endpoint heads, never an acoustic encoder or labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
from acoustic_memory_common import protocol, read_json, write_json_new, verify_file, sha256_file, bound, load_bound
from any2xsampa.training.acoustic_memory import SONGS, validate_query


def run_head(run: Path, phase: str) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline, configure_cuda, check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor, query_total_gpu_bytes
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from safetensors.torch import load_file, save_file
    from any2xsampa.adapters.phoneticxeus import ctc_collapse
    if phase not in ('execute', 'replay'):
        raise ValueError('only two fixed worker slots')
    out = run / phase / 'head'
    command = read_json(run / 'commands' / (phase + '.json'))
    if command['worker'] != str(Path(__file__).resolve()) or command['phase'] != phase or (out / 'START.json').exists():
        raise ValueError('unaccounted or duplicate launch')
    write_json_new(out / 'START.json', {'phase': phase, 'pid': __import__('os').getpid(),
        'command_sha256': sha256_file(run / 'commands' / (phase + '.json')), 'request_cap': 229})
    before = time.monotonic()
    enforce_offline(out / 'offline')
    freeze = protocol(run, review=True)
    if phase == 'replay':
        if not read_json(run / 'execute/score/RESULT.json')['complete']:
            raise ValueError('execution incomplete; replay not permitted')
    requests = read_json(run / 'HEAD_REQUESTS.json')['requests']
    if len(requests) != 229 or sum(r['purpose'] == 'source' for r in requests) != 80:
        raise ValueError('fixed request count changed')
    for r in requests:
        if set(r) != {'fold','purpose','query'} or r['fold'] not in SONGS:
            raise ValueError('invalid head request schema')
        q = r['query']; validate_query(q)
        if r['purpose'] == 'source':
            if q['song_id'] == r['fold'] or q['audio_role'] != 'original':raise ValueError('source leakage')
        elif r['purpose'] != 'query' or q['song_id'] != r['fold']:
            raise ValueError('query fold mismatch')
    probes = []
    for _ in range(3):
        probes.append(query_total_gpu_bytes()); time.sleep(.25)
    # Reserve 256 MiB headroom for the tiny frozen heads, without killing others.
    if max(probes) + 256 * 1024**2 > 14 * 1024**3:
        raise RuntimeError('insufficient stable card headroom')
    write_json_new(out / 'PREFLIGHT.json', {'gpu_samples_bytes': probes, 'headroom_bytes': 256 * 1024**2})
    with serial_candidate_job(), ProcessResourceMonitor(out / 'monitor', timeout_seconds=1500):
        torch = configure_cuda(); torch.set_grad_enabled(False)
        head_spec = freeze['head']; verify_file(Path(head_spec['path']), head_spec['sha256'])
        head = load_file(head_spec['path'], device='cuda')
        if set(head) != {'weight','bias'} or head['weight'].shape != (428,1024) or head['bias'].shape != (428,):
            raise ValueError('pretrained head shape changed')
        output = []; attempted = 0
        for song in SONGS:
            spec = freeze['checkpoints'][song]; verify_file(Path(spec['path']), spec['sha256'])
            state = load_file(spec['path'], device='cuda')
            shapes = {'residual.0.weight':(64,1024),'residual.0.bias':(64,),
                'residual.2.weight':(1024,64),'residual.2.bias':(1024,)}
            if set(state) != set(shapes) or any(tuple(state[k].shape) != v for k,v in shapes.items()):
                raise ValueError('endpoint shape changed')
            for value in [*state.values(), *head.values()]:
                if value.dtype != torch.float32 or not torch.isfinite(value).all():raise ValueError('invalid frozen weights')
            snapshots = {k:v.clone() for k,v in {**state, **head}.items()}
            for request in [r for r in requests if r['fold'] == song]:
                q = request['query']; verify_file(Path(q['path']), q['sha256'])
                cache = load_file(q['path'], device='cpu'); f = cache['encoder_features']
                if f.shape != (q['frames'],1024) or f.dtype != torch.float32 or not torch.isfinite(f).all():
                    raise ValueError('invalid final feature cache; no reencoding authorized')
                attempted += 1
                if attempted > 229:raise RuntimeError('request budget exhausted')
                write_json_new(out / 'requests' / (str(attempted).zfill(3) + '.json'),
                    {'attempt': attempted, 'fold': song, 'purpose': request['purpose'], 'example_id': q['example_id']})
                with torch.no_grad():
                    f = f.cuda()
                    h = torch.nn.functional.linear(f, state['residual.0.weight'], state['residual.0.bias'])
                    residual = torch.nn.functional.linear(torch.nn.functional.gelu(h), state['residual.2.weight'], state['residual.2.bias'])
                    z = f + residual
                    logits = torch.nn.functional.linear(z, head['weight'], head['bias'])
                    if not torch.isfinite(z).all() or not torch.isfinite(logits).all():raise ValueError('nonfinite head output')
                    raw = {'z':z.detach().cpu().contiguous(), 'logits':logits.detach().cpu().contiguous()}
                path = out / song / (q['example_id'] + '.safetensors'); path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():raise FileExistsError('duplicate fold/request')
                save_file(raw, str(path))
                ids = ctc_collapse(raw['logits'].argmax(-1).tolist())
                output.append({'fold': song, 'purpose': request['purpose'], 'example_id':q['example_id'],
                    'raw':bound(path), 'baseline_ids':ids, 'cache_sha256':q['sha256'], 'checkpoint_sha256':spec['sha256']})
                if attempted % 20 == 0:
                    check_gpu_budget(torch); print('head requests', attempted, flush=True)
                del cache, f, h, residual, z, logits, raw
            if any(not torch.equal(v, snapshots[k]) for k,v in {**state, **head}.items()):
                raise ValueError('frozen parameter changed')
            verify_file(Path(spec['path']), spec['sha256'])
        if attempted != 229 or len(output) != 229:raise ValueError('incomplete requests')
        memory = check_gpu_budget(torch)
        verify_file(Path(head_spec['path']), head_spec['sha256'])
        result = {'complete':True, 'phase':phase, 'rows':output, 'head_requests':attempted,
            'optimizer_steps':0, 'acoustic_encoder_forwards':0, 'labels_received':False,
            'runtime':{'python':sys.version, 'torch':torch.__version__, 'cuda':torch.version.cuda,
                'device':'cuda:0', 'cpu_threads':torch.get_num_threads(), 'offline':True},
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(), 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'memory_after':memory, 'elapsed_seconds':time.monotonic()-before}
        write_json_new(out / 'RESULT.json', result)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=('execute','replay'),required=True)
    a=p.parse_args()
    try:run_head(a.run,a.phase)
    except Exception as exc:
        target=a.run/a.phase/'head/FAILURE.json'
        if not target.exists():write_json_new(target,{'error_type':type(exc).__name__,'error':str(exc),'no_automatic_retry':True})
        raise

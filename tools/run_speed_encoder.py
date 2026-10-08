"""Two bounded, audio-only native-P encoding slots (45 + 10)."""
from __future__ import annotations
import time
from pathlib import Path
from speed_invariance_common import bound, read_json, write_json_new, verify_file, load_cache, sha256_file
from any2xsampa.training.speed_invariance import speed_view, pcm_hash
from any2xsampa.training.singing_domain import validate_audio_row


def encode(run: Path, slot: str, freeze: dict, ledger) -> dict:
    import numpy as np
    from safetensors.torch import load_file, save_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.phoneticxeus_runtime import read_audio_crop, check_gpu_budget
    from any2xsampa.adapters.phoneticxeus import convolution_geometry, ctc_collapse
    out = run / 'jobs' / slot
    rt = SingingDomainRuntime(Path(freeze['snapshot']), out / 'offline'); torch = rt.torch
    if rt.identity != freeze['architecture']:
        raise ValueError('native P architecture identity differs')
    verify_file(Path(freeze['head']['path']), freeze['head']['sha256'])
    head = load_file(freeze['head']['path'], device='cpu')
    if not torch.equal(rt.model.ctc.ctc_lo.weight.cpu(), head['weight']) or not torch.equal(rt.model.ctc.ctc_lo.bias.cpu(), head['bias']):
        raise ValueError('cached output head differs from live frozen model')
    if any(p.requires_grad for p in rt.model.parameters()): raise ValueError('backbone is not frozen')
    versions = [p._version for p in rt.model.parameters()]
    requests = read_json(run / 'ENCODER_REQUESTS.json')[slot]
    if len(requests) != (45 if slot == 'encode' else 10): raise ValueError('encoder quota changed')
    output = []; last_key = None; source = None
    for request in requests:
        q, view = request['query'], request['view']
        row = {k: q[k] for k in ('example_id', 'song_id', 'split', 'audio_role', 'audio_path', 'audio_sha256', 'start_sec', 'end_sec')}
        validate_audio_row(row)
        key = (q['audio_path'], q['audio_sha256'], q['start_sec'], q['end_sec'])
        if key != last_key:
            source = read_audio_crop(Path(key[0]), *key[1:]); last_key = key
        raw, audio, crop = source
        if audio != q['audio_metadata'] or crop != q['crop']:
            raise ValueError('historical raw preprocessing/crop mismatch')
        waveform, transform = speed_view(raw, view)
        transform.update(source_audio_sha256=q['audio_sha256'], source_crop=crop,
            source_example_id=q['example_id'], source_song_id=q['song_id'])
        destination = out / 'cache' / (q['example_id'] + '--' + view)
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / 'pcm.npy').open('xb') as stream:
            np.save(stream, waveform, allow_pickle=False)
        speech = torch.from_numpy(waveform).unsqueeze(0).to('cuda')
        lengths = torch.tensor([len(waveform)], dtype=torch.long, device='cuda')
        calls = []
        hook = rt.model.encoder.conditioning_layer.register_forward_hook(lambda m, a, o: calls.append(1))
        check_gpu_budget(torch); start = time.monotonic()
        ledger.attempt('encoder', example_id=q['example_id'], view=view, samples=len(waveform))
        try:
            with torch.inference_mode():
                features, sizes = rt.model.encode(speech, lengths)
                if isinstance(features, tuple): features = features[0]
                features = features[0, :int(sizes[0])]
                logits = rt.model.ctc.ctc_lo(features)
                tensors = {'encoder_features': features.cpu().contiguous(), 'frame_logits': logits.cpu().contiguous()}
        finally:
            hook.remove()
        torch.cuda.synchronize()
        ledger.completed('encoder', example_id=q['example_id'], view=view)
        if len(calls) != 3: raise ValueError('native interCTC conditioning calls differ')
        geometry = convolution_geometry(rt.base.shapes, len(waveform))
        for name, width in (('encoder_features', 1024), ('frame_logits', 428)):
            if tuple(tensors[name].shape) != (geometry['frame_count'], width) or tensors[name].dtype != torch.float32 or not torch.isfinite(tensors[name]).all():
                raise ValueError('invalid encoded shape/numeric values')
        ids = ctc_collapse(tensors['frame_logits'].argmax(-1).tolist())
        check = None
        if view == 'raw':
            old = load_cache(torch, q)
            errors = {name: float((old[name] - tensors[name]).abs().max()) for name in tensors}
            bitwise = {name: torch.equal(old[name], tensors[name]) for name in tensors}
            if max(errors.values()) > 2e-5 or ids != ctc_collapse(old['frame_logits'].argmax(-1).tolist()):
                raise ValueError('unchanged control does not reproduce old cache')
            check = {'max_abs': errors, 'bitwise': bitwise, 'greedy_equal': True}
        path = destination / 'features.safetensors'; save_file(tensors, str(path))
        record = {'source_query': q, 'view': view, 'transform': transform, 'pcm': bound(destination / 'pcm.npy'),
            'raw': bound(path), 'frames': geometry['frame_count'], 'geometry': geometry, 'ids': ids,
            'architecture_sha256': rt.identity['sha256'], 'old_raw_cache_check': check,
            'elapsed_seconds': time.monotonic() - start, 'memory': check_gpu_budget(torch),
            'no_reference_received': True, 'training_view_not_original_timeline_posterior': True}
        write_json_new(destination / 'ENTRY.json', record); output.append(record)
        print(slot, len(output), '/', len(requests), q['example_id'], view, flush=True)
        del speech, lengths, features, logits, tensors
    if versions != [p._version for p in rt.model.parameters()]: raise ValueError('frozen encoder changed')
    return {'complete': True, 'rows': output, 'encoder_forwards': ledger.counts['encoder'],
        'parameters_frozen': True, 'architecture': rt.identity, 'runtime': rt.base.runtime,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}

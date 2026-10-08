"""Cached-only C/A training and held-audio prediction, never held references."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import time
from speed_invariance_common import bound, read_json, write_json_new, verify_file, load_cache, load_bound, sha256_file, policy
from any2xsampa.training.speed_invariance import make_head, adapter_state, load_adapter, objective, validate_fold, schedule_views
from any2xsampa.training.singing_domain_losses import required_frames
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def model_for(torch, freeze: dict, song: str, checkpoint: dict):
    from safetensors.torch import load_file
    verify_file(Path(freeze['head']['path']), freeze['head']['sha256'])
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    model = make_head(torch, load_file(freeze['head']['path'], device='cpu'), device='cuda')
    verify_file(Path(checkpoint['path']), checkpoint['sha256'])
    load_adapter(torch, model, load_file(checkpoint['path'], device='cpu'))
    return model


def load_views(torch, run: Path, train: list[dict], held: str, arm: str) -> dict:
    from safetensors.torch import load_file
    acceptance = read_json(run / 'CACHE_ACCEPTED.json')
    if acceptance['passed'] is not True or acceptance['code_review_sha256'] != sha256_file(run / 'CODE_REVIEW.json'):
        raise ValueError('stale/unaccepted augmented caches')
    for spec in acceptance['bindings']: verify_file(Path(spec['path']), spec['sha256'])
    entries = {(r['source_query']['example_id'], r['view']): r for r in read_json(run / 'jobs/encode/RESULT.json')['rows']}
    out = {}
    for row in train:
        q = row['query']; key = q['example_id']
        raw = load_cache(torch, q); out[(key, 'raw')] = raw
        if arm != 'A' or q['audio_role'] != 'original': continue
        for view in ('slow_0p9', 'fast_1p1'):
            e = entries[(key, view)]
            if e['source_query'] != q or q['song_id'] == held or e['architecture_sha256'] != q['architecture_sha256'] or not e['no_reference_received']:
                raise ValueError('derived source/song/model identity mismatch')
            if required_frames(row['target_ids']) > e['frames']: raise ValueError('derived CTC infeasible')
            verify_file(Path(e['raw']['path']), e['raw']['sha256'])
            tensors = load_file(e['raw']['path'], device='cpu')
            for name, width in (('encoder_features', 1024), ('frame_logits', 428)):
                if tuple(tensors[name].shape) != (e['frames'], width) or tensors[name].dtype != torch.float32 or not torch.isfinite(tensors[name]).all():
                    raise ValueError('invalid same-view feature/teacher cache')
            out[(key, view)] = tensors
    return out


def predict(torch, model, q: dict, cached: dict, destination: Path, ledger, purpose: str) -> dict:
    from safetensors.torch import save_file
    ledger.attempt('head', purpose=purpose, example_id=q['example_id'])
    model.eval()
    with torch.no_grad():
        logits = model(cached['encoder_features'][None].cuda())[0].cpu().contiguous()
    if not torch.isfinite(logits).all(): raise ValueError('nonfinite prediction')
    ledger.completed('head', purpose=purpose, example_id=q['example_id'])
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'evidence.safetensors'; save_file({'frame_logits': logits}, str(path))
    record = {'example_id': q['example_id'], 'song_id': q['song_id'], 'audio_role': q['audio_role'],
        'raw': bound(path), 'ids': ctc_collapse(logits.argmax(-1).tolist()), 'cache_sha256': q['sha256'],
        'purpose': purpose, 'no_reference_received_by_predict': True}
    write_json_new(destination / 'ENTRY.json', record)
    return record


def train(run: Path, slot: str, freeze: dict, ledger, torch) -> dict:
    from safetensors.torch import save_file
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
    arm, song = slot.split('--'); out = run / 'jobs' / slot
    source = read_json(run / 'folds' / song / 'TRAINING.json'); train_rows = source['rows']
    queries = read_json(run / 'folds' / song / 'QUERIES.json')
    validate_fold(train_rows, queries['held'], song)
    by_id = {r['query']['example_id']: r for r in train_rows}
    roles = {key: r['query']['audio_role'] for key, r in by_id.items()}
    views = load_views(torch, run, train_rows, song, arm)
    model = model_for(torch, freeze, song, freeze['initial'][song])
    weight, bias = model.weight.clone(), model.bias.clone()
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    probe = next(r['query'] for r in train_rows if r['query']['audio_role'] == 'original')
    data = views[(probe['example_id'], 'raw')]
    initial = predict(torch, model, probe, data, out / 'initial-probe', ledger, 'initial')
    from safetensors.torch import load_file
    initial_logits = load_file(initial['raw']['path'])['frame_logits']
    error = float((initial_logits - data['frame_logits']).abs().max())
    if error > 2e-5 or initial['ids'] != ctc_collapse(data['frame_logits'].argmax(-1).tolist()):
        raise ValueError('step zero differs from frozen P')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, foreach=False)
    history = []; exposure = Counter(); total_frames = 0; completed_updates = 0
    for epoch_index, epoch in enumerate(source['epochs']):
        ordered = schedule_views(epoch, roles, arm, epoch_index)
        model.train(); sums = Counter()
        for offset in range(0, len(ordered), 8):
            batch = ordered[offset:offset + 8]; optimizer.zero_grad(set_to_none=True)
            for key, view in batch:
                cached = views[(key, view)]; row = by_id[key]
                features = cached['encoder_features'][None].cuda(); teacher = cached['frame_logits'][None].cuda()
                ledger.attempt('head', purpose='training', epoch=epoch_index + 1, example_id=key, view=view)
                logits = model(features)
                loss, diag = objective(torch, logits, teacher, row['target_ids'], roles[key])
                (loss / len(batch)).backward()
                ledger.completed('head', purpose='training')
                sums.update(diag); exposure['rendered_raw' if roles[key] == 'rendered' else view] += 1
                total_frames += logits.shape[1]
                del features, teacher, logits, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            ledger.attempt('optimizer', epoch=epoch_index + 1, accumulation_size=len(batch))
            optimizer.step(); completed_updates += 1
            ledger.completed('optimizer', epoch=epoch_index + 1)
            if torch.cuda.memory_reserved() > 7 * 1024**3: raise RuntimeError('reserved memory limit')
        event = {'epoch': epoch_index + 1, 'completed_updates': completed_updates,
            'training_requests': len(ordered), 'means': {k: v / len(ordered) for k, v in sums.items()},
            'memory': check_gpu_budget(torch), 'no_validation_used': True}
        history.append(event); write_json_new(out / f'epoch-{epoch_index+1:02d}.json', event)
        print(slot, 'epoch', epoch_index + 1, 'updates', completed_updates, flush=True)
    if not torch.equal(weight, model.weight) or not torch.equal(bias, model.bias):
        raise ValueError('frozen output parameters changed')
    path = out / 'final.safetensors'; save_file(adapter_state(model), str(path))
    final = []; seen = []
    for purpose, qq, target in [('held', queries['held'], final), ('seen_original', queries['seen_original'], seen)]:
        for q in qq:
            data = views[(q['example_id'], 'raw')] if purpose == 'seen_original' else load_cache(torch, q)
            target.append(predict(torch, model, q, data, out / purpose / q['example_id'], ledger, purpose))
    fixed = policy()['per_fold'][song]
    if completed_updates != fixed['optimizer_steps_per_arm'] or sum(exposure.values()) != fixed['training_requests_per_arm']:
        raise ValueError('effective update/request counts differ')
    if arm == 'A' and dict(exposure) != fixed['augmented_arm_views']: raise ValueError('view exposure mismatch')
    return {'complete': True, 'arm': arm, 'fold': song, 'seed': 17, 'fixed_epoch': 20,
        'checkpoint': bound(path), 'initial_state': freeze['initial'][song], 'initial_max_error': error,
        'initial_prediction': initial, 'held': final, 'seen_original': seen, 'history': history,
        'exposure_counts': dict(exposure), 'training_frames': total_frames, 'completed_updates': completed_updates,
        'trainable_parameters': 132160, 'frozen_parameters_unchanged': True, 'encoder_forwards': 0,
        'held_references_received': False, 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def replay(run: Path, freeze: dict, ledger, torch) -> dict:
    from safetensors.torch import load_file
    from speed_invariance_common import SONGS
    rows = []
    for song in SONGS:
        queries = read_json(run / 'folds' / song / 'QUERIES.json')['held']
        for arm in ('C', 'A'):
            slot = arm + '--' + song; prior = read_json(run / 'jobs' / slot / 'RESULT.json')
            model = model_for(torch, freeze, song, prior['checkpoint']); checks = []
            for q, expected in zip(queries, prior['held'], strict=True):
                actual = predict(torch, model, q, load_cache(torch, q),
                    run / 'jobs/head-replay' / slot / q['example_id'], ledger, 'held_replay')
                verify_file(Path(expected['raw']['path']), expected['raw']['sha256'])
                a = load_file(actual['raw']['path'])['frame_logits']; b = load_file(expected['raw']['path'])['frame_logits']
                if not torch.equal(a, b) or actual['ids'] != expected['ids'] or actual['example_id'] != expected['example_id']:
                    raise ValueError('independent cached head replay mismatch')
                checks.append(actual)
            rows.append({'slot': slot, 'checkpoint': prior['checkpoint'], 'rows': checks, 'bitwise': True})
    if ledger.counts['head'] != 298: raise ValueError('incomplete independent replay')
    return {'complete': True, 'rows': rows, 'encoder_forwards': 0, 'optimizer_updates': 0,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'no_reference_received': True}

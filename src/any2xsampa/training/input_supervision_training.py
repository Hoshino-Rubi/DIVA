"""Direct vocal supervision with the historical control's exact source schedule."""
from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.input_supervision import load_manifest, policy, supervision_slots, shuffle_epoch
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.singing_domain_training import supervised_loss


def load_rows(run: Path, torch: Any) -> tuple[list[dict], dict]:
    from safetensors.torch import load_file
    _, manifest, index = load_manifest(run)
    entries = {r['cache_id']: r for r in index['entries']}
    if set(entries) != {r['cache_id'] for r in manifest['rows']}:
        raise ValueError('row/cache identity mismatch')
    rows = []
    for ref in manifest['rows']:
        entry = entries[ref['cache_id']]
        for key in ('condition', 'song_id', 'frames'):
            if entry[key] != ref[key]:
                raise ValueError('cache/reference mismatch: ' + key)
        verify_file(Path(entry['path']), entry['sha256'])
        tensors = load_file(entry['path'], device='cpu')
        for name, width in (('prefix_features', 1024), ('encoder_features', 1024), ('frame_logits', 428)):
            if tensors[name].shape != (ref['frames'], width) or tensors[name].dtype != torch.float32 or not torch.isfinite(tensors[name]).all():
                raise ValueError('invalid cached tensor: ' + name)
        rows.append({**entry, **ref, **tensors})
    return rows, index


def _supervision_rows(rows: list[dict], held_song: str | None, variant: str) -> list[dict]:
    return supervision_slots(rows, held_song, variant)


def _evaluate(torch: Any, tail: Any, rows: list[dict], labels: list[str], output: Path) -> dict:
    from any2xsampa.training.singing_domain_training import predict
    from safetensors.torch import save_file
    output.mkdir(parents=True, exist_ok=False)
    details = []
    tail.eval()
    with torch.no_grad():
        for row in rows:
            logits, _ = predict(torch, tail, row)
            values = logits[0].float().cpu().contiguous()
            if not torch.isfinite(values).all():
                raise ValueError('non-finite validation logits')
            prediction = [labels[i] for i in ctc_collapse(values.argmax(-1).tolist())]
            raw = output / (row['row_id'] + '.safetensors')
            save_file({'frame_logits': values}, str(raw))
            details.append({'row_id': row['row_id'], 'window_id': row['window_id'], 'song_id': row['song_id'],
                            'condition': row['condition'], 'reference': row['target_ipa'], 'prediction': prediction,
                            'metrics': phone_metrics([(row['target_ipa'], prediction)]),
                            'logits_path': str(raw.resolve()), 'logits_sha256': sha256_file(raw)})
    return {'examples': details, 'metrics': phone_metrics([(r['reference'], r['prediction']) for r in details])}


def train_epoch(torch: Any, tail: Any, slots: list[dict], ignored: list[str], optimizer: Any,
                rng: Any, old_spec: dict, deadline: float) -> dict:
    spec = policy()
    tail.train()
    ordered = shuffle_epoch(slots, ignored, rng)
    params = [p for p in tail.parameters() if p.requires_grad]
    micro = spec['gradient_accumulation']
    ctc = kl = 0.
    steps = 0
    for offset in range(0, len(ordered), micro):
        if time.monotonic() > deadline:
            raise TimeoutError('fixed job deadline; partial epochs cannot enter selection')
        batch = ordered[offset:offset+micro]
        optimizer.zero_grad(set_to_none=True)
        for row in batch:
            # Reuse the control's exact arithmetic, including rendered weighting.
            loss, diagnostic = supervised_loss(torch, tail, row, {}, 'baseline', old_spec, {})
            (loss / len(batch)).backward()
            ctc += diagnostic['ctc']
            kl += diagnostic['prior_kl']
        torch.nn.utils.clip_grad_norm_(params, spec['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        steps += 1
        if torch.cuda.memory_reserved() > spec['process_reserved_gib'] * 1024**3:
            raise RuntimeError('own reserved GPU memory exceeds policy')
    return {'optimizer_steps': steps, 'supervision_slots': len(ordered),
            'source_order_sha256': sha256_object([r['window_id'] for r in ordered]),
            'slot_order_sha256': sha256_object([r['slot_id'] for r in ordered]),
            'view_order_sha256': sha256_object([r['condition'] for r in ordered]),
            'rng_after_sha256': sha256_object(rng.getstate()),
            'mean_ctc': ctc/len(ordered), 'mean_prior_kl': kl/len(ordered)}


def require_code_review(run: Path) -> None:
    """A reviewed hash set is required before any quality-training launch."""
    freeze, _, _ = load_manifest(run)
    review = read_json(run / 'CODE_REVIEW.json')
    if (review.get('approved_for_training') is not True
            or review.get('freeze_sha256') != sha256_file(run / 'FREEZE.json')
            or review.get('code_bindings') != freeze['code_bindings']):
        raise ValueError('training requires an approved review of these exact code hashes')
    checks = review.get('verification_bindings', [])
    if not checks:
        raise ValueError('code review lacks verification evidence')
    for check in checks:
        verify_file(Path(check['path']), check['sha256'])


def require_screen_budget(run: Path, *, current_launch: str | None = None) -> None:
    freeze, _, _ = load_manifest(run)
    completed = list(run.glob('screen/*/*/RESULT.json'))
    started = {f'{p.parent.parent.name}-{p.parent.name}' for p in run.glob('screen/*/*/START.json')}
    commands = {p.name.removesuffix('.command.json') for p in (run / 'commands').glob('*.command.json')}
    # The parent records dispatch before Python/model initialization. Count it
    # even if the worker never reaches START; do not count the same job twice.
    if current_launch not in started:
        commands.discard(current_launch)
    starts = started | commands
    budget = freeze['budget']
    unavailable = set(freeze.get('unavailable_screen_jobs', []))
    planned = {f'{variant}-{song}' for variant in policy()['variants'] for song in policy()['songs']}
    if not unavailable <= planned or unavailable & starts:
        raise ValueError('invalid unavailable jobs or attempted retry')
    if budget['remaining_screen_starts'] - len(starts) < 10 - len(unavailable) - len(completed):
        raise RuntimeError('insufficient authorized starts for complete V/MV screen; requires explicit additional budget')


def train_job(run: Path, variant: str, held_song: str, seed: int, output: Path) -> None:
    spec = policy()
    if variant not in spec['variants'] or held_song not in spec['songs'] or seed != spec['screen_seed']:
        raise ValueError('unplanned quality job')
    freeze, manifest, _ = load_manifest(run)
    if f'{variant}-{held_song}' in freeze.get('unavailable_screen_jobs', []):
        raise ValueError('previously failed job is excluded; no retry authorized')
    require_code_review(run)
    require_screen_budget(run, current_launch=f'{variant}-{held_song}')
    if output.resolve() != (run / 'screen' / variant / held_song).resolve():
        raise ValueError('output must use the unique planned screen job path')
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / 'START.json', {
        'schema_version': 'input_supervision_launch_v2', 'variant': variant,
        'held_song': held_song, 'seed': seed, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'freeze_sha256': sha256_file(run / 'FREEZE.json'),
        'code_review_sha256': sha256_file(run / 'CODE_REVIEW.json'),
        'budget_consumed_before_model_load': True,
    })
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime, make_tail, trainable_state
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.training.singing_domain_training import predict
    from safetensors.torch import save_file
    started = time.monotonic()
    with serial_candidate_job(), ProcessResourceMonitor(output/'monitor', timeout_seconds=spec['maximum_job_seconds']):
        rt = SingingDomainRuntime(Path(freeze['model_snapshot']), output/'runtime-cache')
        torch = rt.torch
        rows, index = load_rows(run, torch)
        if index['architecture']['sha256'] != rt.identity['sha256']:
            raise ValueError('live/cache architecture mismatch')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        tail = make_tail(torch, rt.model, 'full')
        rt.model.frontend.cpu()
        rt.model.preencoder.cpu()
        for block in list(rt.model.encoder.encoders)[:15]:
            block.cpu()
        rt.model.encoder.conditioning_layer.cpu()
        slots = supervision_slots(rows, held_song, variant)
        ignored = [r['example_id'] for r in manifest['unused_unlabeled_rng_rows'] if r['song_id'] != held_song]
        initial = {}
        tail.eval()
        for condition in ('x', 'v', 'rendered'):
            probe = next(r for r in rows if r['song_id'] != held_song and r['condition'] == condition)
            with torch.no_grad():
                logits, _ = predict(torch, tail, probe)
            error = float((logits[0].cpu() - probe['frame_logits']).abs().max())
            if error > 2e-5:
                raise ValueError('initial logits differ from frozen model: ' + condition)
            initial[condition] = error
        frozen = [p for p in rt.model.parameters() if not p.requires_grad]
        versions = [p._version for p in frozen]
        trainable = [p for p in tail.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=spec['learning_rate'], weight_decay=spec['weight_decay'], foreach=False)
        old_spec = read_json(Path(freeze['historical_run'])/'POLICY.json')
        old_result = read_json(Path(freeze['historical_run'])/'architecture/full'/held_song/'RESULT.json')
        if len(slots) != old_result['history'][0]['labelled_microsteps']:
            raise ValueError('control microstep budget changed')
        record = {'schema_version': 'input_supervision_job_v2', 'variant': variant, 'held_song': held_song, 'seed': seed,
                  'arm': 'full', 'architecture': rt.identity, 'policy_sha256': sha256_file(run/'POLICY.json'),
                  'freeze_sha256': sha256_file(run/'FREEZE.json'), 'code_bindings': freeze['code_bindings'],
                  'training_row_ids': [r['row_id'] for r in slots], 'initial_logit_errors': initial,
                  'trainable_parameters': sum(p.numel() for p in trainable), 'fixed_epoch_count': 20,
                  'training_songs': sorted({r['song_id'] for r in slots}), 'default_replacement': False}
        write_json_new(output/'INITIALIZATION.json', record)
        history = []
        rng = random.Random(seed)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, 21):
            event = train_epoch(torch, tail, slots, ignored, optimizer, rng, old_spec, started+spec['maximum_job_seconds'])
            if event['optimizer_steps'] != old_result['history'][epoch-1]['optimizer_steps']:
                raise ValueError('optimizer step budget differs from control')
            event.update(epoch=epoch, elapsed_seconds=time.monotonic()-started, memory=check_gpu_budget(torch))
            write_json_new(output/f'epoch-{epoch:02d}.json', event)
            history.append(event)
            print(variant, held_song, 'epoch', epoch, flush=True)
        if versions != [p._version for p in frozen]:
            raise RuntimeError('frozen weights changed')
        labels = [p for p, _ in sorted(rt.vocabulary.items(), key=lambda item: item[1])]
        evaluations = {c: _evaluate(torch, tail, [r for r in rows if r['song_id'] == held_song and r['condition'] == c], labels, output/f'validation-{c}') for c in ('x', 'v')}
        checkpoint = output/'final.safetensors'
        save_file(trainable_state(tail), str(checkpoint))
        write_json_new(output/'RESULT.json', {**record, 'complete': True, 'history': history, 'validation_x': evaluations['x'],
                       'validation_v': evaluations['v'], 'checkpoint_path': str(checkpoint.resolve()), 'checkpoint_sha256': sha256_file(checkpoint),
                       'elapsed_seconds': time.monotonic()-started, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'frozen_parameters_unchanged': True, 'final_test_predictions_opened': False})

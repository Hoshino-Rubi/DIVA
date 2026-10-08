"""Frozen-feature original adaptation with bounded teacher replay experiments."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.original_adaptation import (
    _labels, _reference_ids, _load_evidence, _evaluate, _evaluate_logits,
    _gate, _rendered_gate, validate_original_index,
)
from any2xsampa.training.phoneticxeus_adaptation import _batch, _load_cache, make_residual_adapter
from any2xsampa.training.protected_adaptation import baseline_kl
from any2xsampa.training.reviewed_training_reference import validate_training_reference
from any2xsampa.training.limited_data_reference import SONG_SPLIT


def validate_policy(policy: dict) -> None:
    unsigned = {k: v for k, v in policy.items() if k != 'policy_sha256'}
    if policy.get('schema_version') not in {'retention_iteration_policy_v1', 'retention_iteration_policy_v2'} or policy.get('policy_sha256') != sha256_object(unsigned):
        raise ValueError('retention policy hash or schema mismatch')
    expected = [
        {'id': 'replay_005', 'rendered_kl_coefficient': .05, 'frame_selection': 'all', 'radius': 0},
        {'id': 'replay_100', 'rendered_kl_coefficient': 1., 'frame_selection': 'all', 'radius': 0},
        {'id': 'symmetric_100', 'rendered_kl_coefficient': 1., 'frame_selection': 'symmetric', 'radius': 2},
    ]
    supervised = policy['schema_version'] == 'retention_iteration_policy_v2'
    if supervised:
        expected = [{'id': 'rehearsal_100', 'rendered_kl_coefficient': .05, 'frame_selection': 'all', 'radius': 0, 'rendered_ctc_coefficient': 1.}]
    if policy['variants'] != expected or policy['seeds'] != [17, 29, 43] or policy['max_training_attempts'] != (3 if supervised else 9):
        raise ValueError('unfrozen experiment variant or seed')
    fixed = dict(original_kl_coefficient=.05, temperature=2., batch_size=8, learning_rate=1e-4,
                 weight_decay=1e-4, gradient_clip=1., max_epochs=20, patience=4, timeout_seconds=1500)
    if any(policy[k] != v for k, v in fixed.items()):
        raise ValueError('retention budget or optimization changed')
    if policy['replay']['synthetic_reference_targets_used'] is not supervised or policy['sealed_test_opening'] is not False:
        raise ValueError('replay labels or test data forbidden')


def teacher_frame_mask(torch: Any, teacher: Any, lengths: Any, *, mode: str, radius: int) -> Any:
    if teacher.ndim != 3 or teacher.shape[-1] != 428 or len(lengths) != teacher.shape[0]:
        raise ValueError('replay requires batch, time and all 428 labels')
    sizes = lengths.to(teacher.device)
    if sizes.dtype not in (torch.int32, torch.int64) or not torch.all((sizes > 0) & (sizes <= teacher.shape[1])):
        raise ValueError('invalid replay lengths')
    valid = torch.arange(teacher.shape[1], device=teacher.device)[None, :] < sizes[:, None]
    if not torch.isfinite(teacher[valid]).all():
        raise ValueError('non-finite valid teacher frames')
    if mode == 'all' and radius == 0:
        return valid
    if mode != 'symmetric' or radius != 2:
        raise ValueError('unfrozen replay frame selection')
    # Padding cannot create a spike or extend a real spike beyond valid audio.
    clean = teacher.detach().masked_fill(~valid[..., None], 0)
    nonblank = clean.argmax(-1).ne(0) & valid
    around = torch.nn.functional.max_pool1d(nonblank[:, None, :].float(), 2 * radius + 1, 1, radius)[:, 0].bool() & valid
    empty = ~around.any(dim=1)
    return torch.where(empty[:, None], valid, around)


def replay_kl(torch: Any, student: Any, teacher: Any, lengths: Any, variant: Mapping[str, Any]) -> tuple[Any, dict]:
    if student.shape != teacher.shape:
        raise ValueError('student/teacher replay shape mismatch')
    mask = teacher_frame_mask(torch, teacher, lengths, mode=variant['frame_selection'], radius=variant['radius'])
    actual, frozen = student[mask], teacher.detach()[mask]
    if not torch.isfinite(actual).all():
        raise ValueError('non-finite student replay frames')
    log_q, log_p = (actual / 2.).log_softmax(-1), (frozen / 2.).log_softmax(-1)
    loss = torch.nn.functional.kl_div(log_q, log_p, log_target=True, reduction='none').sum(-1).mean() * 4.
    return loss, {'selected_frames': int(mask.sum()), 'valid_frames': int(lengths.sum())}


class ReplayCycle:
    """Deterministic training-only replay; a wrap never changes source membership."""
    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int) -> None:
        if not rows or len({r['example_id'] for r in rows}) != len(rows):
            raise ValueError('empty or duplicate replay pool')
        for row in rows:
            if row['split'] != 'train' or row['audio_role'] != 'rendered' or SONG_SPLIT.get(row['song_id']) != 'train':
                raise ValueError('replay pool contains dev/test or non-rendered audio')
            if any(k in row for k in ('model_target_ids', 'model_target_ipa', 'phones')):
                raise ValueError('replay loss must not receive reference labels')
        self.rows = list(rows)
        self.random = random.Random(seed)
        self.order: list[int] = []
        self.position = 0
        self.seen: set[str] = set()

    def take(self, count: int) -> list[Mapping[str, Any]]:
        if count < 1:
            raise ValueError('empty replay batch')
        result = []
        while len(result) < count:
            if self.position == len(self.order):
                self.order = list(range(len(self.rows)))
                self.random.shuffle(self.order)
                self.position = 0
            row = self.rows[self.order[self.position]]
            self.position += 1
            result.append(row)
            self.seen.add(row['example_id'])
        return result


def feature_batch(torch: Any, rows: Sequence[Mapping[str, Any]], *, device: str = 'cuda') -> tuple[Any, Any, Any]:
    maximum = max(r['features'].shape[0] for r in rows)
    features = torch.zeros(len(rows), maximum, 1024, dtype=torch.float32)
    teachers = torch.zeros(len(rows), maximum, 428, dtype=torch.float32)
    lengths = []
    for i, row in enumerate(rows):
        n = row['features'].shape[0]
        features[i, :n] = row['features']
        teachers[i, :n] = row['teacher_logits']
        lengths.append(n)
    return features.to(device), torch.tensor(lengths, dtype=torch.long), teachers.to(device)


def load_inputs(source: Path, expected_hash: str, torch: Any) -> dict:
    path = source / 'original-training-manifest.json'
    verify_file(path, expected_hash)
    manifest = read_json(path)
    if manifest['manifest_hash'] != sha256_object({k: v for k, v in manifest.items() if k != 'manifest_hash'}) or manifest['gate']['passed'] is not True:
        raise ValueError('invalid source training manifest')
    for name in ('segment_reference', 'original_feature_index', 'rendered_dataset', 'rendered_feature_index', 'seed43_checkpoint', 'vocabulary'):
        spec = manifest[name]
        verify_file(Path(spec['path']), spec['sha256'])
    reference = read_json(Path(manifest['segment_reference']['path']))
    validate_training_reference(reference)
    vocabulary = read_json(Path(manifest['vocabulary']['path']))
    refs = _reference_ids(reference['examples'], vocabulary)
    index = read_json(Path(manifest['original_feature_index']['path']))
    if index['source_dataset_hash'] != reference['manifest_hash']:
        raise ValueError('stale original reference cache')
    validate_original_index(index, refs, manifest['vocabulary']['sha256'])
    originals = _load_evidence(index, refs, torch, root=source)
    train = [r for r in originals if r['split'] == 'train']
    dev = [r for r in originals if r['split'] == 'dev']
    if len(train) != 20 or len({r['song_id'] for r in train}) != 5 or len(dev) != 3 or sum(len(r['phones']) for r in dev) != 214:
        raise ValueError('fixed original cohort changed')
    rendered, head, rendered_index = _load_cache(Path(manifest['rendered_feature_index']['path']), read_json(Path(manifest['rendered_dataset']['path'])), torch)
    verify_file(Path(index['head']['path']), index['head']['sha256'])
    if index['head']['sha256'] != rendered_index['head']['sha256']:
        raise ValueError('original and rendered frozen heads differ')
    from safetensors.torch import load_file
    teacher = make_residual_adapter(torch).eval().requires_grad_(False).to('cuda')
    teacher.load_state_dict(load_file(manifest['seed43_checkpoint']['path']), strict=True)
    head.to('cuda')
    with torch.no_grad():
        for row in originals:
            if not torch.allclose(head(row['features'].to('cuda')).cpu(), row['baseline_logits'], rtol=1e-4, atol=1e-4):
                raise ValueError('original feature/head projection mismatch')
    pool = [{k: r[k] for k in ('example_id', 'song_id', 'split', 'audio_role', 'features')}
            for r in rendered if r['split'] == 'train']
    protection = [r for r in rendered if r['split'] == 'dev']
    if len(pool) != 129 or len(protection) != 25:
        raise ValueError('fixed replay/protection cohort changed')
    ReplayCycle(pool, 17)  # Check every replay member before computing teachers.
    with torch.no_grad():
        for row in train + pool:
            row['teacher_logits'] = head(teacher(row['features'].to('cuda'))).cpu()
    return dict(manifest=manifest, reference=reference, vocabulary=vocabulary, labels=_labels(vocabulary),
                train=train, dev=dev, replay=pool, protection=protection, head=head, teacher=teacher,
                original_index=index, rendered_targets=validated_replay_targets(rendered))



def validated_replay_targets(rendered: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    result = {}
    for row in rendered:
        if row['split'] == 'dev':
            continue
        if row['split'] != 'train' or row['audio_role'] != 'rendered' or SONG_SPLIT.get(row['song_id']) != 'train':
            raise ValueError('supervised replay source leakage')
        ids = row['model_target_ids']
        if not ids or any(isinstance(i, bool) or not isinstance(i, int) or not 4 <= i < 428 for i in ids):
            raise ValueError('invalid supervised replay target')
        if row['example_id'] in result:
            raise ValueError('duplicate supervised replay target')
        result[row['example_id']] = {'model_target_ids': list(ids)}
    return result


def epoch(torch: Any, adapter: Any, data: dict, cycle: ReplayCycle, order: Sequence[Mapping[str, Any]],
          optimizer: Any, variant: Mapping[str, Any], deadline: float) -> dict:
    adapter.train()
    head = data['head']
    sums = dict(ctc=0., rendered_ctc=0., original_kl=0., replay_kl=0., loss=0.)
    steps = selected = valid = 0
    for offset in range(0, len(order), 8):
        if time.monotonic() >= deadline:
            return dict(complete=False, steps=steps)
        rows = order[offset:offset + 8]
        features, lengths, targets, target_lengths = _batch(rows, torch)
        _, _, teacher = feature_batch(torch, rows)
        replay_rows = cycle.take(8)
        rf, rl, rt = feature_batch(torch, replay_rows)
        optimizer.zero_grad(set_to_none=True)
        logits = head(adapter(features))
        ctc = torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0, 1), targets, lengths, target_lengths, blank=0, zero_infinity=False)
        original_loss = baseline_kl(torch, logits, teacher, lengths, temperature=2.)
        replay_logits = head(adapter(rf))
        retention, count = replay_kl(torch, replay_logits, rt, rl, variant)
        rendered_ctc = logits.new_zeros(())
        if variant.get('rendered_ctc_coefficient', 0):
            targets = [data['rendered_targets'][row['example_id']]['model_target_ids'] for row in replay_rows]
            target_ids = torch.tensor([i for seq in targets for i in seq], dtype=torch.long)
            target_sizes = torch.tensor([len(seq) for seq in targets], dtype=torch.long)
            rendered_ctc = torch.nn.functional.ctc_loss(replay_logits.log_softmax(-1).transpose(0, 1), target_ids, rl, target_sizes, blank=0, zero_infinity=False)
        loss = ctc + .05 * original_loss + variant['rendered_kl_coefficient'] * retention + variant.get('rendered_ctc_coefficient', 0) * rendered_ctc
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite retention loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        for key, value in dict(ctc=ctc, rendered_ctc=rendered_ctc, original_kl=original_loss, replay_kl=retention, loss=loss).items():
            sums[key] += float(value.detach())
        selected += count['selected_frames']; valid += count['valid_frames']; steps += 1
    return dict(complete=True, steps=steps, mean_losses={k: v / steps for k, v in sums.items()},
                replay_selected_frames=selected, replay_valid_frames=valid, replay_unique_seen=len(cycle.seen))


def run_retention(suite: Path, variant_id: str, seed: int) -> dict:
    policy = read_json(suite / 'POLICY.json'); validate_policy(policy)
    if seed not in policy['seeds'] or variant_id not in {v['id'] for v in policy['variants']}:
        raise ValueError('unauthorized experiment')
    variant = next(v for v in policy['variants'] if v['id'] == variant_id)
    destination = suite / variant_id / f'seed{seed}'
    if destination.exists():
        raise FileExistsError('refusing to repeat a completed or attempted training job')
    destination.mkdir(parents=True)
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline, configure_cuda, check_gpu_budget
    from safetensors.torch import save_file
    enforce_offline(destination / 'runtime-cache'); torch = configure_cuda()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    data = load_inputs(Path(policy['source_run']), policy['source_manifest_sha256'], torch)
    teacher, head = data['teacher'], data['head']
    adapter = make_residual_adapter(torch).to('cuda'); adapter.load_state_dict(teacher.state_dict(), strict=True)
    frozen_teacher = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    frozen_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
    with torch.no_grad():
        for row in data['train'] + data['replay']:
            if not torch.equal(head(adapter(row['features'].to('cuda'))).cpu(), row['teacher_logits']):
                raise RuntimeError('step-zero output differs from the immutable teacher')
    baseline = _evaluate_logits(data['dev'], data['labels'])
    original_teacher = _evaluate(data['dev'], data['labels'], torch, teacher, head)
    rendered_teacher = _evaluate(data['protection'], data['labels'], torch, teacher, head)
    for name, value in [('baseline-original', baseline), ('teacher-original', original_teacher), ('teacher-rendered', rendered_teacher)]:
        write_json_new(destination / (name + '.json'), value)
    started = datetime.now(timezone.utc).isoformat(); before = time.monotonic()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = random.Random(seed); cycle = ReplayCycle(data['replay'], seed)
    best = best_pass = None; best_per = best_pass_per = float('inf'); stale = 0
    history = []; reason = 'maximum_epochs'
    for number in range(1, 21):
        order = list(data['train']); rng.shuffle(order)
        event = epoch(torch, adapter, data, cycle, order, optimizer, variant, before + 1500)
        if not event['complete']:
            reason = 'wall_clock_limit_incomplete_epoch_not_selected'; break
        adapter.eval()
        original = _evaluate(data['dev'], data['labels'], torch, adapter, head)
        rendered = _evaluate(data['protection'], data['labels'], torch, adapter, head)
        og = _gate(original, baseline, original_teacher); rg = _rendered_gate(rendered, rendered_teacher)
        passed = og['passed'] and rg['passed']
        event.update(epoch=number, original_dev=original['metrics'], rendered_dev=rendered['metrics'],
                     original_gate=og, rendered_gate=rg, passed=passed, resources=check_gpu_budget(torch), elapsed_seconds=time.monotonic()-before)
        history.append(event); write_json_new(destination / f'epoch-{number:02d}.json', event)
        score = original['metrics']['per']
        def capture() -> dict:
            return dict(epoch=number, state={k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}, original=original, rendered=rendered)
        if score < best_per:
            best_per = score; best = capture(); stale = 0
        else:
            stale += 1
        if passed and score < best_pass_per:
            best_pass_per = score; best_pass = capture()
        if stale >= 4:
            reason = 'early_stopping_original_dev_per'; break
    if best is None:
        raise RuntimeError('no completed retention epoch')
    for name, module, state in [('teacher', teacher, frozen_teacher), ('head', head, frozen_head)]:
        if any(p.requires_grad for p in module.parameters()) or not all(torch.equal(module.state_dict()[k], v) for k, v in state.items()):
            raise RuntimeError(name + ' changed during retention training')
    checkpoints = {}
    for role, value in [('best_overall', best), ('best_admissible', best_pass)]:
        checkpoints[role] = None
        if value is None:
            continue
        path = destination / (role + '.safetensors'); save_file(value['state'], str(path))
        evaluation = dict(original=value['original'], rendered=value['rendered'])
        evaluation_path = destination / (role + '-evaluation.json'); write_json_new(evaluation_path, evaluation)
        checkpoints[role] = dict(path=str(path.resolve()), sha256=sha256_file(path), epoch=value['epoch'],
                                 evaluation_path=str(evaluation_path.resolve()), evaluation_sha256=sha256_file(evaluation_path),
                                 original_metrics=value['original']['metrics'], rendered_metrics=value['rendered']['metrics'])
    report = dict(schema_version='retention_adaptation_result_v1', variant=variant, seed=seed,
                  policy_sha256=sha256_file(suite/'POLICY.json'), source_manifest_sha256=policy['source_manifest_sha256'],
                  baseline_original=baseline['metrics'], teacher_original=original_teacher['metrics'], teacher_rendered=rendered_teacher['metrics'],
                  checkpoints=checkpoints, history=history, stop_reason=reason, initialization_equal_all_training_rows=True,
                  teacher_unchanged=True, head_unchanged=True, backbone_loaded=False, test_opened=False, synthetic_reference_labels_in_gradient=bool(variant.get('rendered_ctc_coefficient', 0)),
                  trainable_parameters=sum(p.numel() for p in adapter.parameters()), replay_unique_seen=len(cycle.seen),
                  started_at_utc=started, ended_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-before,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), model_promoted=False)
    write_json_new(destination / 'training-report.json', report)
    return report

"""Equal-stratum teacher KL on full-vocabulary distributions, training only."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from any2xsampa.training.limited_data_artifacts import read_json, sha256_object, verify_file


def validate_policy(policy: dict) -> None:
    config = read_json(Path(__file__).parents[1]/'config/stratified_projected_replay_v1.json')
    if any(policy.get(k) != v for k, v in config.items()):
        raise ValueError('unfrozen stratified replay recipe')
    if policy.get('policy_sha256') != sha256_object({k: v for k, v in policy.items() if k != 'policy_sha256'}):
        raise ValueError('stratified policy hash mismatch')
    for binding in policy['bindings'].values():
        verify_file(Path(binding['path']), binding['sha256'])


def teacher_strata(torch: Any, teacher: Any, lengths: Any, labels: list[str]) -> dict:
    if teacher.ndim != 3 or teacher.shape[-1] != 428 or len(labels) != 428 or len(set(labels)) != 428:
        raise ValueError('expected full 428-label vocabulary')
    if lengths.ndim != 1 or len(lengths) != teacher.shape[0] or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError('invalid sequence lengths')
    sizes = lengths.to(teacher.device)
    if not torch.all((sizes > 0) & (sizes <= teacher.shape[1])):
        raise ValueError('length outside padded batch')
    valid = torch.arange(teacher.shape[1], device=teacher.device)[None, :] < sizes[:, None]
    if not torch.isfinite(teacher[valid]).all():
        raise ValueError('nonfinite valid teacher logits')
    ids = teacher.detach().masked_fill(~valid[..., None], 0).argmax(-1)
    masks = {}
    for name, indices in [('aspirated', [i for i, p in enumerate(labels) if '\u02b0' in p]),
                          ('glottal_h', [i for i, p in enumerate(labels) if p == 'h'])]:
        if not indices:
            raise ValueError('protected labels absent from vocabulary')
        hits = torch.isin(ids, torch.tensor(indices, device=ids.device)) & valid
        masks[name] = torch.nn.functional.max_pool1d(hits[:, None].float(), 5, 1, 2)[:, 0].bool() & valid
    masks['glottal_h'] &= ~masks['aspirated']
    masks['other'] = valid & ~(masks['aspirated'] | masks['glottal_h'])
    return masks


def stratified_kl(torch: Any, student: Any, teacher: Any, lengths: Any, labels: list[str]) -> tuple[Any, dict]:
    if student.shape != teacher.shape:
        raise ValueError('student/teacher shape mismatch')
    masks = teacher_strata(torch, teacher, lengths, labels)
    losses, counts, values = [], {}, {}
    for name, mask in masks.items():
        count = int(mask.sum()); counts[name] = count
        if not count:
            values[name] = None
            continue
        actual, frozen = student[mask], teacher.detach()[mask]
        if not torch.isfinite(actual).all():
            raise ValueError('nonfinite valid student logits')
        log_q, log_p = (actual/2.).log_softmax(-1), (frozen/2.).log_softmax(-1)
        value = torch.nn.functional.kl_div(log_q, log_p, log_target=True, reduction='none').sum(-1).mean()*4.
        if not torch.isfinite(value):
            raise ValueError('nonfinite stratum KL')
        losses.append(value); values[name] = float(value.detach())
    loss = torch.stack(losses).mean()
    return loss, {'frames_by_stratum': counts, 'kl_by_stratum': values,
                  'nonempty_strata': len(losses), 'valid_frames': int(lengths.sum()),
                  'teacher_is_a_model_not_acoustic_truth': True}

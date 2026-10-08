"""One fixed speed-replacement experiment; no acoustic model imports."""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence
import hashlib
import numpy as np

from any2xsampa.training.acoustic_memory import SONGS, validate_query
from any2xsampa.training.singing_domain_losses import required_frames, sequence_ctc
from any2xsampa.training.protected_adaptation import baseline_kl
from any2xsampa.training.event_head_evaluation import aggregate, same_references, compare_songs, group_regressions

VIEWS = ('raw', 'slow_0p9', 'fast_1p1')
RATIOS = {'raw': (1, 1), 'slow_0p9': (10, 9), 'fast_1p1': (10, 11)}
SLOTS = ('raw', 'raw', 'slow_0p9', 'fast_1p1')


def pcm_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes(order='C')).hexdigest()


def speed_view(waveform: np.ndarray, view: str) -> tuple[np.ndarray, dict]:
    from scipy.signal import resample_poly
    x = np.asarray(waveform)
    if view not in RATIOS or x.ndim != 1 or x.dtype != np.float32 or not len(x) or not np.isfinite(x).all():
        raise ValueError('invalid waveform/view')
    up, down = RATIOS[view]
    y = x if view == 'raw' else resample_poly(x, up, down, window=('kaiser', 5.0), padtype='constant', cval=0).astype(np.float32)
    expected = (len(x) * up + down - 1) // down
    if len(y) != expected or len(y) > 320000 or not np.isfinite(y).all():
        raise ValueError('transformed length/nonfinite/budget mismatch')
    return y, {'view': view, 'up': up, 'down': down, 'speed': down / up,
        'original_samples': len(x), 'derived_samples': len(y), 'sample_rate': 16000,
        'source_pcm_sha256': pcm_hash(x), 'derived_pcm_sha256': pcm_hash(y),
        'peak_absolute': float(np.max(np.abs(y))), 'identity_bypass': view == 'raw',
        'derived_to_source_seconds_scale': down / up,
        'duration_residual_source_samples': len(y) * down / up - len(x),
        'resample_window': ['kaiser', 5.0], 'padtype': 'constant', 'cval': 0,
        'full_source_interval_retained': True, 'new_human_confirmation': False}


def validate_fold(train: Sequence[dict], queries: Sequence[dict], held: str) -> None:
    if held not in SONGS or not train or not queries:
        raise ValueError('invalid fold')
    held_hashes = {q['audio_sha256'] for q in queries}
    if len({r['query']['example_id'] for r in train}) != len(train):
        raise ValueError('duplicate training source')
    for q in queries:
        validate_query(q)
        if q['song_id'] != held:
            raise ValueError('held query song mismatch')
    for r in train:
        if set(r) != {'query', 'target_ids', 'reference_sha256', 'reference_nature'}:
            raise ValueError('training source fields changed')
        q = r['query']; validate_query(q)
        if q['song_id'] == held or q['audio_sha256'] in held_hashes:
            raise ValueError('song/audio lineage leakage')
        ids = r['target_ids']
        if not ids or any(type(i) is not int or not 0 < i < 428 for i in ids) or required_frames(ids) > q['frames']:
            raise ValueError('invalid complete CTC target')


def schedule_views(epoch: Sequence[str], roles: dict[str, str], arm: str, epoch_index: int) -> list[tuple[str, str]]:
    if arm not in ('C', 'A') or not 0 <= epoch_index < 20 or set(epoch) != set(roles):
        raise ValueError('fixed schedule/cohort mismatch')
    counts = Counter(epoch)
    for key, role in roles.items():
        if role not in ('original', 'rendered') or counts[key] != (4 if role == 'original' else 1):
            raise ValueError('training slot count mismatch')
    seen = Counter(); output = []
    for key in epoch:
        view = SLOTS[(seen[key] + epoch_index) % 4] if arm == 'A' and roles[key] == 'original' else 'raw'
        output.append((key, view)); seen[key] += 1
    return output


def make_head(torch: Any, state: dict, *, device: str) -> Any:
    """Same endpoint numerical path, without loading any acoustic blocks."""
    if set(state) != {'weight', 'bias'} or tuple(state['weight'].shape) != (428, 1024) or tuple(state['bias'].shape) != (428,):
        raise ValueError('immutable head shape mismatch')
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('immutable head requires finite FP32 weights')
    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residual = torch.nn.Sequential(torch.nn.Linear(1024, 64), torch.nn.GELU(), torch.nn.Linear(64, 1024))
            torch.nn.init.zeros_(self.residual[2].weight); torch.nn.init.zeros_(self.residual[2].bias)
            self.register_buffer('weight', state['weight'].detach().clone())
            self.register_buffer('bias', state['bias'].detach().clone())

        def forward(self, features: Any) -> Any:
            if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] != 1024 or not torch.isfinite(features).all():
                raise ValueError('finite microbatch1 final features required')
            return torch.nn.functional.linear(features + self.residual(features), self.weight, self.bias)
    model = Head().to(device)
    if sum(p.numel() for p in model.parameters()) != 132160:
        raise ValueError('residual capacity changed')
    return model


def adapter_state(model: Any) -> dict:
    return {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items() if k.startswith('residual.')}


def load_adapter(torch: Any, model: Any, state: dict) -> None:
    current = adapter_state(model)
    if set(state) != set(current) or any(state[k].shape != current[k].shape or state[k].dtype != torch.float32 or not torch.isfinite(state[k]).all() for k in state):
        raise ValueError('invalid residual checkpoint')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {'weight', 'bias'} or unexpected:
        raise ValueError('unexpected checkpoint keys')


def objective(torch: Any, logits: Any, teacher: Any, ids: list[int], role: str) -> tuple[Any, dict]:
    if role not in ('original', 'rendered'):
        raise ValueError('unplanned supervision role')
    ctc = sequence_ctc(torch, logits, ids)
    sizes = torch.tensor([logits.shape[1]], device=logits.device)
    kl = baseline_kl(torch, logits, teacher, sizes, temperature=2.)
    loss = (1. if role == 'original' else .5) * ctc + .01 * kl
    if not torch.isfinite(loss):
        raise ValueError('invalid objective')
    return loss, {'ctc': float(ctc.detach()), 'kl': float(kl.detach())}


def compare_gates(augmented: list[dict], control: list[dict], historical: list[dict]) -> dict:
    same_references(augmented, control); same_references(augmented, historical)
    pools = {name: {role: [r for r in rows if r['audio_role'] == role] for role in ('original', 'rendered')}
             for name, rows in [('A', augmented), ('C', control), ('historical', historical)]}
    metrics = {name: {role: aggregate(rows) for role, rows in pp.items()} for name, pp in pools.items()}
    for pp in pools.values():
        if len(pp['original']) != 20 or len(pp['rendered']) != 129:
            raise ValueError('fixed windows changed')
    for mm in metrics.values():
        if mm['original']['reference_phones'] != 1235 or mm['rendered']['reference_phones'] != 4814:
            raise ValueError('fixed denominator changed')
    a, c = metrics['A']['original'], metrics['C']['original']
    if metrics['historical']['original']['edit_distance'] != 595:
        raise ValueError('historical anchor changed')
    failures = []; songs = {}
    if a['edit_distance'] > 564: failures.append('historical_error_cap')
    if 100 * a['edit_distance'] > 95 * c['edit_distance']: failures.append('matched_control_gain_below_5pct')
    if a['insertions'] > min(32, c['insertions']): failures.append('insertion_cap')
    if a['deletions'] > min(248, c['deletions']): failures.append('deletion_cap')
    for name in ('C', 'historical'):
        songs[name] = compare_songs(pools['A']['original'], pools[name]['original'])
        if len(songs[name]['improved']) < 3: failures.append(name + '_fewer_than_three_improved')
        if not songs[name]['macro_nonregressing']: failures.append(name + '_macro_regression')
        for role in ('original', 'rendered'):
            failures.extend(name + '_' + role + '_' + group for group in group_regressions(metrics['A'][role], metrics[name][role]))
        if metrics['A']['rendered']['per'] > metrics[name]['rendered']['per'] + .03:
            failures.append(name + '_rendered_overall_regression')
    return {'passed': not failures, 'failures': failures, 'metrics': metrics, 'song_comparisons': songs,
        'independent_generalization_claim': False, 'automatic_deployment': False}

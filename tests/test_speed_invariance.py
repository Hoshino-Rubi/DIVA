"""Bounded offline fixtures for the fixed speed-replacement screen."""
from __future__ import annotations
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from any2xsampa.training.speed_invariance import speed_view, pcm_hash, schedule_views, validate_fold, make_head, adapter_state, load_adapter, objective
from any2xsampa.training.protected_adaptation import baseline_kl
from any2xsampa.training.singing_domain_losses import required_frames, sequence_ctc
from speed_invariance_common import Ledger, slots, read_json, write_json_new, verify_file, stable_memory
from launch_speed_invariance import check_slot


def query(song='freedom-flower', key='one', role='original'):
    rate = 16000
    return {'example_id': key, 'song_id': song, 'split': 'train', 'audio_role': role,
        'audio_path': 'fixture.wav', 'audio_sha256': song.ljust(64, '0'), 'start_sec': 1., 'end_sec': 2.,
        'architecture_sha256': 'a' * 64, 'frames': 49, 'path': 'fixture.safetensors', 'sha256': 'b' * 64,
        'audio_metadata': {'sha256': song.ljust(64, '0'), 'sample_rate': rate, 'channels': 1,
            'sample_width_bytes': 4, 'frame_count': 64000, 'duration_sec': 4.},
        'crop': {'requested_start_sec': 1., 'requested_end_sec': 2., 'source_start_sample': rate,
            'source_end_sample_exclusive': 2 * rate, 'effective_start_sec': 1., 'effective_end_sec': 2.,
            'model_sample_rate': rate, 'model_samples': rate, 'channel_policy': 'arithmetic_mean',
            'resample_policy': 'scipy_resample_poly_v1', 'waveform_normalization': 'none_before_model_frontend'},
        'geometry': {'frame_count': 49, 'stride_samples': 320, 'receptive_field_samples': 400}}


def training(q):
    return {'query': q, 'target_ids': [1, 1, 427], 'reference_sha256': 'c' * 64, 'reference_nature': 'human_confirmed'}


@pytest.mark.parametrize('samples', [1, 63, 160000])
@pytest.mark.parametrize('view,up,down', [('raw', 1, 1), ('slow_0p9', 10, 9), ('fast_1p1', 10, 11)])
def test_complete_waveform_transform(samples, view, up, down):
    x = np.zeros(samples, np.float32); x[-1] = .75
    y, meta = speed_view(x, view)
    assert len(y) == (samples * up + down - 1) // down
    assert np.any(y[-min(32, len(y)):] != 0)
    assert meta['source_pcm_sha256'] == pcm_hash(x)
    assert 0 <= meta['duration_residual_source_samples'] < down / up + 1e-10
    assert meta['new_human_confirmation'] is False
    if view == 'raw': assert y is x and np.array_equal(y, x)


@pytest.mark.parametrize('x,view', [(np.array([np.nan], np.float32), 'raw'),
    (np.array([np.inf], np.float32), 'fast_1p1'), (np.zeros(2, np.float64), 'raw'),
    (np.zeros((2, 1), np.float32), 'raw'), (np.zeros(0, np.float32), 'raw'),
    (np.zeros(160000, np.float32), 'unplanned'), (np.zeros(320000, np.float32), 'slow_0p9')])
def test_invalid_waveforms_fail(x, view):
    with pytest.raises(ValueError): speed_view(x, view)


def test_no_integer_clipping_or_gain_normalization():
    x = np.full(100, 1.8, np.float32)
    y, meta = speed_view(x, 'slow_0p9')
    assert y.dtype == np.float32 and y.max() > 1 and meta['peak_absolute'] == float(abs(y).max())


@pytest.mark.parametrize('epoch', range(4))
def test_schedule_replaces_not_expands(epoch):
    ordered = ['o', 'r', 'o', 'o', 'o']; roles = {'o': 'original', 'r': 'rendered'}
    a = schedule_views(ordered, roles, 'A', epoch); c = schedule_views(ordered, roles, 'C', epoch)
    assert [k for k, _ in a] == ordered and [k for k, _ in c] == ordered
    assert Counter(v for k, v in a if k == 'o') == {'raw': 2, 'slow_0p9': 1, 'fast_1p1': 1}
    assert all(v == 'raw' for _, v in c) and a[1] == ('r', 'raw')


@pytest.mark.parametrize('ordered,roles,arm,epoch', [(['o'], {'o': 'original'}, 'A', 0),
    (['o'] * 4, {'o': 'vocal'}, 'A', 0), (['o'] * 4, {'o': 'original'}, 'B', 0),
    (['o'] * 4, {'o': 'original'}, 'A', 20), (['o'] * 4, {'z': 'original'}, 'A', 0)])
def test_schedule_corruption_fail(ordered, roles, arm, epoch):
    with pytest.raises(ValueError): schedule_views(ordered, roles, arm, epoch)


def test_fold_valid_and_repeated_ctc_frames():
    assert required_frames([1, 1, 1, 2]) == 6
    validate_fold([training(query())], [query('promise', 'held')], 'promise')


@pytest.mark.parametrize('bad', ['held_song', 'held_hash', 'test', 'vocal', 'label_in_query', 'nested_lyrics', 'sample_offset', 'blank', 'impossible'])
def test_fold_rejects_leakage_and_bad_reference(bad):
    train = training(query()); held = query('promise', 'held')
    if bad == 'held_song': train['query']['song_id'] = 'promise'
    elif bad == 'held_hash':
        train['query']['audio_sha256'] = held['audio_sha256']; train['query']['audio_metadata']['sha256'] = held['audio_sha256']
    elif bad == 'test': train['query']['split'] = 'test'
    elif bad == 'vocal': train['query']['audio_role'] = 'separated_vocal'
    elif bad == 'label_in_query': train['query']['target_ids'] = [1]
    elif bad == 'nested_lyrics': train['query']['crop']['lyrics'] = 'not_allowed'
    elif bad == 'sample_offset': train['query']['crop']['source_start_sample'] += 1
    elif bad == 'blank': train['target_ids'] = [0]
    elif bad == 'impossible': train['target_ids'] = [1] * 49
    with pytest.raises(ValueError): validate_fold([train], [held], 'promise')


def test_budget_attempts_before_completion_and_no_retry(tmp_path):
    path = tmp_path / 'attempts.jsonl'; ledger = Ledger(path, {'head': 1, 'optimizer': 1})
    ledger.attempt('head', purpose='training')
    assert json.loads(path.read_text().splitlines()[0])['event'] == 'attempt'
    ledger.completed('head'); ledger.attempt('optimizer')
    with pytest.raises(RuntimeError): ledger.attempt('head')
    ledger.close()
    assert 'optimizer' in path.read_text()
    with pytest.raises(FileExistsError): Ledger(path, {'head': 1})


def test_slot_prerequisites_and_duplicate_fail(tmp_path):
    assert len(slots()) == 13 and slots()[:2] == ['encode', 'encode-replay']
    check_slot(tmp_path, 'encode')
    with pytest.raises(ValueError): check_slot(tmp_path, 'encode-replay')
    write_json_new(tmp_path / 'commands/encode.json', {'attempted': True})
    with pytest.raises(ValueError): check_slot(tmp_path, 'encode')
    write_json_new(tmp_path / 'commands/encode.exit.json', {'exit_code': 73})
    with pytest.raises(ValueError): check_slot(tmp_path, 'encode-replay')


def test_headroom_fail_closed(monkeypatch):
    import any2xsampa.adapters.phoneticxeus_monitor as monitor
    import speed_invariance_common as common
    monkeypatch.setattr(common.time, 'sleep', lambda _: None)
    monkeypatch.setattr(monitor, 'query_total_gpu_bytes', lambda: 12 * 1024**3)
    with pytest.raises(RuntimeError): stable_memory(True)
    assert stable_memory(False)['required_headroom_bytes'] == 256 * 1024**2
    def fail(): raise OSError('nvidia-smi unavailable')
    monkeypatch.setattr(monitor, 'query_total_gpu_bytes', fail)
    with pytest.raises(OSError): stable_memory(False)


def test_hash_mismatch_rejected(tmp_path):
    path = tmp_path / 'pcm'; path.write_bytes(b'old')
    with pytest.raises(ValueError): verify_file(path, 'a' * 64)


def test_zero_init_checkpoint_equivalence_frozen_head_and_same_view_loss():
    torch = pytest.importorskip('torch'); torch.set_num_threads(4)
    torch.manual_seed(17)
    frozen = {'weight': torch.randn(428, 1024) * .01, 'bias': torch.zeros(428)}
    torch.manual_seed(17); model = make_head(torch, frozen, device='cpu')
    state = adapter_state(model)
    torch.manual_seed(29); other = make_head(torch, frozen, device='cpu'); load_adapter(torch, other, state)
    x = torch.randn(1, 8, 1024)
    expected = torch.nn.functional.linear(x, frozen['weight'], frozen['bias'])
    assert torch.equal(model(x), expected) and torch.equal(other(x), expected)
    assert sum(p.numel() for p in model.parameters()) == 132160
    loss, _ = objective(torch, model(x), expected.detach(), [1, 1, 427], 'original')
    loss.backward()
    assert model.residual[2].weight.grad is not None and model.weight.grad is None
    assert torch.equal(model.weight, frozen['weight']) and torch.equal(model.bias, frozen['bias'])
    with pytest.raises(ValueError): objective(torch, model(x), expected[:, :-1], [1], 'original')
    with pytest.raises(ValueError): objective(torch, model(x), expected, [0], 'original')
    with pytest.raises(ValueError): objective(torch, model(x), expected, [1] * 8, 'original')
    with pytest.raises(ValueError): objective(torch, model(x), expected, [1], 'separated_vocal')
    wrong = dict(state); wrong['residual.2.bias'] = torch.full((1024,), float('nan'))
    with pytest.raises(ValueError): load_adapter(torch, other, wrong)


def test_padding_teacher_detach_and_invalid_valid_values():
    torch = pytest.importorskip('torch'); torch.set_num_threads(4)
    student = torch.randn(1, 5, 428, requires_grad=True)
    teacher = torch.randn(1, 5, 428, requires_grad=True)
    sizes = torch.tensor([3]); expected = baseline_kl(torch, student[:, :3], teacher[:, :3], sizes)
    a = student.detach().clone(); b = teacher.detach().clone(); a[:, 3:] = float('nan'); b[:, 3:] = float('inf')
    a.requires_grad_(True); b.requires_grad_(True)
    actual = baseline_kl(torch, a, b, sizes)
    assert torch.equal(expected, actual); actual.backward()
    assert b.grad is None and torch.isfinite(a.grad).all() and torch.count_nonzero(a.grad[:, 3:]) == 0
    a = a.detach(); a[:, 0] = float('nan')
    with pytest.raises(ValueError): baseline_kl(torch, a, b, sizes)


def test_ctc_blank_repeat_and_empty():
    from any2xsampa.adapters.phoneticxeus import ctc_collapse
    assert ctc_collapse([0, 1, 1, 0, 1, 0, 427, 427]) == [1, 1, 427]
    assert ctc_collapse([0] * 8) == []
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError): sequence_ctc(torch, torch.zeros(1, 2, 428), [])


def test_exact_historical_endpoint_path_fixture():
    from types import SimpleNamespace
    from any2xsampa.adapters.singing_domain_runtime import make_tail, trainable_state
    torch = pytest.importorskip('torch'); torch.set_num_threads(4)
    frozen = torch.nn.Linear(1024, 428)
    parent = SimpleNamespace(encoder=SimpleNamespace(encoders=torch.nn.ModuleList([torch.nn.Identity() for _ in range(19)]),
        after_norm=torch.nn.Identity()), ctc=SimpleNamespace(ctc_lo=frozen))
    torch.manual_seed(17); old = make_tail(torch, parent, 'endpoint')
    torch.manual_seed(17); new = make_head(torch, {'weight': frozen.weight, 'bias': frozen.bias}, device='cpu')
    assert all(torch.equal(value, adapter_state(new)[name]) for name, value in trainable_state(old).items())
    x = torch.randn(1, 9, 1024); lengths = torch.tensor([9])
    y, _ = old(x, lengths, final_features=x)
    assert torch.equal(y, new(x))
    ref = torch.nn.functional.linear(x, frozen.weight, frozen.bias)
    a, _ = objective(torch, y, ref, [1, 2, 2], 'original'); b, _ = objective(torch, new(x), ref, [1, 2, 2], 'original')
    a.backward(); b.backward()
    for (_, left), (_, right) in zip(old.residual.named_parameters(), new.residual.named_parameters()):
        assert torch.equal(left.grad, right.grad)


def synthetic_cohort(errors):
    from any2xsampa.training.acoustic_memory import SONGS
    rows = []
    for i in range(20):
        n = 61 if i < 19 else 76; changes = errors // 20 + (i < errors % 20)
        rows.append({'example_id': 'o' + str(i), 'song_id': SONGS[i % 5], 'audio_role': 'original',
            'reference': ['a'] * n, 'prediction': ['b'] * changes + ['a'] * (n - changes)})
    for i in range(129):
        n = 37 if i < 128 else 78
        rows.append({'example_id': 'r' + str(i), 'song_id': SONGS[i % 5], 'audio_role': 'rendered',
            'reference': ['a'] * n, 'prediction': ['a'] * n})
    return rows


def test_fixed_gate_and_5pct_integer_boundary():
    from any2xsampa.training.speed_invariance import compare_gates
    history = synthetic_cohort(595)
    assert compare_gates(synthetic_cohort(564), history, history)['passed']
    failures = compare_gates(synthetic_cohort(565), history, history)['failures']
    assert 'historical_error_cap' in failures
    failures = compare_gates(synthetic_cohort(564), synthetic_cohort(593), history)['failures']
    assert 'matched_control_gain_below_5pct' in failures
    wrong = synthetic_cohort(564); wrong[0]['reference'].pop()
    with pytest.raises(ValueError): compare_gates(wrong, history, history)


def test_training_worker_control_flow_fixture(tmp_path, monkeypatch):
    """Run actual worker loop on synthetic CPU tensors, with a NO-OP optimizer."""
    import run_speed_head as worker
    import any2xsampa.adapters.phoneticxeus_runtime as runtime
    torch = pytest.importorskip('torch'); torch.set_num_threads(4)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    monkeypatch.setattr(runtime, 'check_gpu_budget', lambda _: {'reserved_bytes': 0})
    o = query(key='o'); r = query('like-you', 'r', 'rendered'); held = query('promise', 'held')
    source = [training(o), training(r)]; ordered = ['o', 'r', 'o', 'o', 'o']
    write_json_new(tmp_path / 'folds/promise/TRAINING.json', {'rows': source, 'epochs': [ordered] * 20})
    write_json_new(tmp_path / 'folds/promise/QUERIES.json', {'held': [held], 'seen_original': [o]})
    torch.manual_seed(17)
    frozen = {'weight': torch.randn(428, 1024) / 100, 'bias': torch.zeros(428)}
    model = make_head(torch, frozen, device='cpu'); features = torch.randn(49, 1024)
    prior = torch.nn.functional.linear(features[None], frozen['weight'], frozen['bias'])[0]
    data = {'encoder_features': features, 'frame_logits': prior}
    monkeypatch.setattr(worker, 'load_views', lambda *a: {('o', 'raw'): data, ('r', 'raw'): data})
    monkeypatch.setattr(worker, 'load_cache', lambda *a: data)
    monkeypatch.setattr(worker, 'model_for', lambda *a: model)
    monkeypatch.setattr(worker, 'policy', lambda: {'per_fold': {'promise': {'optimizer_steps_per_arm': 20, 'training_requests_per_arm': 100}}})
    attempts = []
    class NoOpOptimizer:
        def __init__(self, parameters, **kwargs): self.parameters = list(parameters); assert kwargs == {'lr': .001, 'weight_decay': .0001, 'foreach': False}
        def zero_grad(self, **kwargs):
            for p in self.parameters: p.grad = None
        def step(self): attempts.append(1)  # deliberately no parameter update
    monkeypatch.setattr(torch.optim, 'AdamW', NoOpOptimizer)
    ledger = Ledger(tmp_path / 'ledger.jsonl', {'head': 103, 'optimizer': 20})
    result = worker.train(tmp_path, 'C--promise', {'initial': {'promise': {'path': 'fixture', 'sha256': 'f' * 64}}}, ledger, torch)
    ledger.close()
    assert result['complete'] and len(result['history']) == 20 and len(attempts) == 20
    assert ledger.counts == {'head': 103, 'optimizer': 20} and result['exposure_counts'] == {'raw': 80, 'rendered_raw': 20}
    assert torch.count_nonzero(model.residual[2].weight) == 0

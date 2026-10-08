import json
from pathlib import Path
import numpy as np
import pytest
import torch
from any2xsampa.training.phone_diagnostic_analysis import frame_strata
from any2xsampa.training.stratified_replay import teacher_strata, stratified_kl, validate_policy
from any2xsampa.training.limited_data_artifacts import sha256_object

torch.set_num_threads(4)


def batch():
    labels = [str(i) for i in range(428)]; labels[1] = 'kʰ'; labels[2] = 'h'
    t = torch.zeros(2, 12, 428); t[..., 0] = 1
    t[0, 1, 1] = 3; t[0, 7, 2] = 3; t[1, 5, 1] = 3
    return t, torch.tensor([12, 8]), labels


def test_numpy_and_torch_masks_agree_padding_never_selected():
    t, lengths, labels = batch(); masks = teacher_strata(torch, t, lengths, labels)
    for i, n in enumerate(lengths.tolist()):
        expected = frame_strata(np, t[i, :n].numpy(), labels)
        for name in expected:
            assert np.array_equal(masks[name][i, :n].numpy(), expected[name])
            assert not masks[name][i, n:].any()
    assert sum(int(m.sum()) for m in masks.values()) == int(lengths.sum())


def test_same_logits_zero_loss_and_no_teacher_gradient():
    t, lengths, labels = batch(); t.requires_grad_(True); s = t.detach().clone().requires_grad_()
    loss, detail = stratified_kl(torch, s, t, lengths, labels); loss.backward()
    assert abs(float(loss.detach())) < 1e-7 and t.grad is None
    assert torch.isfinite(s.grad).all() and detail['valid_frames'] == 20


def test_invalid_padding_cannot_change_loss_or_gradients():
    t, lengths, labels = batch(); s = t.clone(); s[..., 5] += .5
    clean = stratified_kl(torch, s, t, lengths, labels)[0]
    t[1, 8:] = float('nan'); s[1, 8:] = float('nan'); s.requires_grad_()
    loss, _ = stratified_kl(torch, s, t, lengths, labels); loss.backward()
    assert torch.equal(clean, loss.detach()) and not s.grad[1, 8:].any()
    assert torch.isfinite(s.grad).all()


def test_equal_stratum_mean_not_frame_weighted_mean():
    t, lengths, labels = batch(); s = t.clone(); s[0, :4, 427] += 5
    loss, detail = stratified_kl(torch, s, t, lengths, labels)
    values = [v for v in detail['kl_by_stratum'].values() if v is not None]
    assert float(loss) == pytest.approx(sum(values)/len(values), abs=1e-7)
    weighted = sum(detail['kl_by_stratum'][k]*n for k, n in detail['frames_by_stratum'].items() if n)/20
    assert float(loss) != pytest.approx(weighted, abs=1e-5)


def test_blank_only_batch_uses_other_stratum_not_fake_protected_frames():
    t, lengths, labels = batch(); t.zero_(); t[..., 0] = 1
    s = t.clone(); s[..., 427] = .5
    loss, detail = stratified_kl(torch, s, t, lengths, labels)
    assert float(loss) > 0 and detail['nonempty_strata'] == 1
    assert detail['frames_by_stratum'] == {'aspirated': 0, 'glottal_h': 0, 'other': 20}


@pytest.mark.parametrize('bad', ['length_zero', 'length_float', 'nan_teacher', 'nan_student', 'narrow_logits'])
def test_bad_valid_data_fails(bad):
    t, lengths, labels = batch(); s = t.clone()
    if bad == 'length_zero': lengths[0] = 0
    if bad == 'length_float': lengths = lengths.float()
    if bad == 'nan_teacher': t[0, 0, 0] = float('nan')
    if bad == 'nan_student': s[0, 0, 0] = float('nan')
    if bad == 'narrow_logits': t = t[..., :427]; s = s[..., :427]
    with pytest.raises(ValueError): stratified_kl(torch, s, t, lengths, labels)


@pytest.mark.parametrize('field,value', [('seeds', [17, 29, 43, 99]), ('test_allowed', True),
                                      ('human_preferences_used_as_labels', True), ('max_epochs', 21)])
def test_recipe_cannot_expand_budget_or_promote_diagnostic_answers(field, value):
    path = Path(__file__).parents[1]/'src/any2xsampa/config/stratified_projected_replay_v1.json'
    p = json.loads(path.read_text()); p['bindings'] = {}; p[field] = value
    p['policy_sha256'] = sha256_object(p)
    with pytest.raises(ValueError, match='unfrozen'): validate_policy(p)

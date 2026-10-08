from copy import deepcopy
import random
import pytest

torch=pytest.importorskip('torch')
from any2xsampa.training.retention_adaptation import teacher_frame_mask,replay_kl,ReplayCycle,feature_batch
from any2xsampa.training.protected_adaptation import baseline_kl

ALL={'frame_selection':'all','radius':0}
SYM={'frame_selection':'symmetric','radius':2}


def teacher_batch():
    t=torch.full((2,9,428),-3.)
    t[:,:,0]=3.;t[0,3,5]=5.
    t[1,8,5]=30. # Outside the second utterance's length: never a spike.
    return t,torch.tensor([9,5])


def test_symmetric_edges_padding_and_blank_only_fallback():
    t,l=teacher_batch();mask=teacher_frame_mask(torch,t,l,mode='symmetric',radius=2)
    assert mask[0].tolist()==[False,True,True,True,True,True,False,False,False]
    assert mask[1].tolist()==[True,True,True,True,True,False,False,False,False]


def test_all_frame_replay_equals_existing_full_vocabulary_kl():
    torch.manual_seed(4);t,l=teacher_batch();s=t+.1*torch.randn_like(t)
    value,count=replay_kl(torch,s,t,l,ALL)
    assert torch.equal(value,baseline_kl(torch,s,t,l))
    assert count=={'selected_frames':14,'valid_frames':14}


def test_poisoned_padding_cannot_affect_loss_or_gradient():
    t,l=teacher_batch();s=t.clone();s[0,3,5]-=1.
    s[1,5:]=float('nan');t[1,5:]=float('nan');s.requires_grad_();t.requires_grad_()
    loss,count=replay_kl(torch,s,t,l,SYM);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad).all()
    assert torch.count_nonzero(s.grad[1,5:])==0 and t.grad is None
    assert count['selected_frames']==10


def test_zero_step_equality_and_unselected_gradient():
    t,l=teacher_batch();s=t.clone().requires_grad_();loss,_=replay_kl(torch,s,t,l,SYM)
    assert abs(float(loss.detach()))<1e-6
    changed=t.clone();changed[0,3,5]-=1.;changed.requires_grad_()
    v,_=replay_kl(torch,changed,t,l,SYM);v.backward()
    assert torch.count_nonzero(changed.grad[0,0])==0 and torch.count_nonzero(changed.grad[0,3])>0
    assert changed.grad[0,3,427]!=0 # Complete 428-label distribution, not a small phone projection.


@pytest.mark.parametrize('lengths',[[0,5],[10,5]])
def test_invalid_audio_lengths_fail(lengths):
    t,_=teacher_batch()
    with pytest.raises(ValueError):teacher_frame_mask(torch,t,torch.tensor(lengths),mode='all',radius=0)


def test_nonfinite_selected_frames_fail():
    t,l=teacher_batch();s=t.clone();s[0,3,4]=float('inf')
    with pytest.raises(ValueError,match='non-finite'):replay_kl(torch,s,t,l,SYM)


def pool():
    return [dict(example_id=f'r{i}',song_id='promise',split='train',audio_role='rendered') for i in range(11)]


def test_cycle_exhausts_pool_before_wrap_and_replays_deterministically():
    a,b=ReplayCycle(pool(),17),ReplayCycle(pool(),17)
    xa=[r['example_id'] for r in a.take(8)+a.take(8)]
    xb=[r['example_id'] for r in b.take(16)]
    assert xa==xb and len(set(xa[:11]))==11 and len(a.seen)==11


@pytest.mark.parametrize('change',[{'split':'dev','song_id':'shalong'},{'song_id':'king-of-karaoke'},{'audio_role':'original'},{'model_target_ids':[4]}])
def test_replay_rejects_leakage_and_reference_targets(change):
    rows=pool();rows[0].update(change)
    with pytest.raises(ValueError):ReplayCycle(rows,17)


def test_teacher_feature_batch_preserves_variable_lengths():
    rows=[{'features':torch.ones(n,1024),'teacher_logits':torch.full((n,428),float(n))} for n in (2,4)]
    f,l,t=feature_batch(torch,rows,device='cpu')
    assert l.tolist()==[2,4] and torch.count_nonzero(f[0,2:])==0 and torch.count_nonzero(t[0,2:])==0
    assert t[1,3,427]==4


def test_supervised_replay_only_projects_training_targets():
    from any2xsampa.training.retention_adaptation import validated_replay_targets
    rows=[dict(example_id='train',song_id='promise',split='train',audio_role='rendered',model_target_ids=[4,5]),
          dict(example_id='dev',song_id='shalong',split='dev',audio_role='rendered',model_target_ids=[6])]
    assert validated_replay_targets(rows)=={'train':{'model_target_ids':[4,5]}}
    rows[0]['song_id']='shalong'
    with pytest.raises(ValueError,match='leakage'):validated_replay_targets(rows)


def test_supervised_replay_rejects_test_and_blank_targets():
    from any2xsampa.training.retention_adaptation import validated_replay_targets
    row=dict(example_id='bad',song_id='promise',split='train',audio_role='rendered',model_target_ids=[0])
    with pytest.raises(ValueError,match='target'):validated_replay_targets([row])
    row.update(split='test',song_id='king-of-karaoke',model_target_ids=[4])
    with pytest.raises(ValueError,match='leakage'):validated_replay_targets([row])

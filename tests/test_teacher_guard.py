import numpy as np
import pytest
from any2xsampa.adapters.teacher_guard import guard_logits
from any2xsampa.training.limited_data_artifacts import sha256_object


def case(n=9):
    vocab={('<blank>' if i==0 else 't\u02b0' if i==4 else f'label{i}'):i for i in range(428)}
    p=dict(schema_version='teacher_aspiration_guard_policy_v1',radius_frames=2,guard_symbols=['t\u02b0'],acoustic_only=True,reference_conditioning=False)
    p['policy_sha256']=sha256_object(p)
    t=np.zeros((n,428),np.float32);t[:,0]=2;s=np.full_like(t,-1);s[:,6]=3
    return s,t,vocab,p


def test_guard_preserves_full_teacher_distribution_only_at_frozen_neighbourhood():
    s,t,v,p=case();t[3,4]=4
    output,mask=guard_logits(s,t,v,p)
    assert np.flatnonzero(mask).tolist()==[1,2,3,4,5]
    assert np.array_equal(output[mask],t[mask]) and np.array_equal(output[~mask],s[~mask])
    assert np.array_equal(s[:,6],np.full(9,3))


def test_guard_no_anchors_and_single_frame_are_defined():
    s,t,v,p=case(1);out,mask=guard_logits(s,t,v,p);assert np.array_equal(out,s) and not mask.any()
    t[0,4]=5;out,mask=guard_logits(s,t,v,p);assert np.array_equal(out,t) and mask.all()


def test_equal_models_remain_equal_with_guard():
    s,t,v,p=case();t[0,4]=5;t[-1,4]=5
    out,_=guard_logits(t,t,v,p);assert np.array_equal(out,t)


@pytest.mark.parametrize('fault',['nan','shape','policy','classes'])
def test_guard_rejects_invalid_evidence(fault):
    s,t,v,p=case()
    if fault=='nan':s[0,0]=np.nan
    if fault=='shape':t=t[:-1]
    if fault=='policy':p['radius_frames']=3
    if fault=='classes':p['guard_symbols']=[];p['policy_sha256']=sha256_object({k:x for k,x in p.items() if k!='policy_sha256'})
    with pytest.raises(ValueError):guard_logits(s,t,v,p)

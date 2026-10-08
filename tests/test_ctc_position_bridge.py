from __future__ import annotations
from itertools import product
import numpy as np
import pytest
from any2xsampa.training.ctc_position_bridge import position_occupancy,phone_key,exact_sequence_matches
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def exhaustive_states(logits,target):
    logp=logits-np.logaddexp.reduce(logits,axis=1,keepdims=True);labels=[0]
    for k in target:labels.extend([k,0])
    paths=[]
    def walk(frame,state,score,visited):
        score+=logp[frame,labels[state]];visited=visited+[state]
        if frame==len(logits)-1:
            if state in [len(labels)-1,len(labels)-2]:paths.append((score,visited))
            return
        for nxt in (state,state+1,state+2):
            if nxt>=len(labels):continue
            if nxt==state+2 and (labels[nxt]==0 or labels[nxt]==labels[state]):continue
            walk(frame+1,nxt,score,visited)
    walk(0,0,0.,[]);walk(0,1,0.,[])
    z=np.logaddexp.reduce([p[0] for p in paths]);gamma=np.zeros((len(logits),len(labels)))
    for v,states in paths:
        for t,s in enumerate(states):gamma[t,s]+=np.exp(v-z)
    return z,gamma


@pytest.mark.parametrize('target',[[1],[1,1],[1,2],[2,1,2]])
def test_soft_positions_against_exhaustive(target):
    x=np.random.default_rng(17).normal(size=(5,3));a=position_occupancy(x,target);z,g=exhaustive_states(x,target)
    assert abs(a['log_target_probability']-z)<1e-12
    np.testing.assert_allclose(a['state_posteriors'],g,atol=1e-12)
    np.testing.assert_allclose(a['position_weights'].sum(0),1,atol=1e-12)


def test_repeated_same_phone_uses_distinct_positions():
    x=np.array([[-9.,9.],[-9.,9.],[9.,-9.],[-9.,9.],[-9.,9.]])
    d=position_occupancy(x,[1,1])
    assert d['position_weights'][:,0].argmax()<d['position_weights'][:,1].argmax()
    assert d['positions'][0]['central_90_end_frame_exclusive']<=3


@pytest.mark.parametrize('target',[[0],[1,1,1,1],[],[True]])
def test_invalid_targets(target):
    with pytest.raises(ValueError):position_occupancy(np.zeros((4,3)),target)


def test_nan_rejected_and_all_blank_finite_logits_not_truth():
    with pytest.raises(ValueError):position_occupancy(np.full((4,3),np.nan),[1])
    x=np.full((4,3),-100.);x[:,0]=100.;r=position_occupancy(x,[1])
    assert r['reference_conditioned'] and r['acoustic_boundary_truth'] is False
    assert r['log_target_probability'] < -100


def test_key_requires_same_note_position_and_source_not_just_symbol():
    a=phone_key('h1',{'note_id':'song-n1','phone_index':0,'phone':'i'})
    b=phone_key('h1',{'note_id':'song-n2','phone_index':0,'phone':'i'})
    assert a!=b
    original={'song_id':'s','source_vsqx_sha256':'h','target_ipa':['i']}
    candidates=[dict(original,example_id='yes'),dict(original,example_id='wrong_source',source_vsqx_sha256='x'),
                dict(original,example_id='other_song',song_id='q')]
    assert exact_sequence_matches(original,candidates)==['yes']

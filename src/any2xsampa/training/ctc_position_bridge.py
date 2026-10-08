"""Reference-conditioned soft CTC positions, for training-data audit only.

This module is not a recognizer. Reference-conditioned occupancies are not
independently observed phone boundaries or proof of correct pronunciation.
"""
from __future__ import annotations
from typing import Any,Sequence


def position_occupancy(logits: Any,target: Sequence[int]) -> dict:
    import numpy as np
    z=np.asarray(logits,dtype=np.float64)
    if z.ndim!=2 or len(z)<1 or z.shape[1]<2 or not np.isfinite(z).all():raise ValueError('invalid CTC evidence')
    if not target or any(isinstance(i,bool) or not isinstance(i,int) or not 0<i<z.shape[1] for i in target):
        raise ValueError('expected nonempty nonblank target sequence')
    if len(target)+sum(a==b for a,b in zip(target,target[1:]))>len(z):raise ValueError('insufficient CTC frames')
    logp=z-np.logaddexp.reduce(z,axis=1,keepdims=True)
    labels=np.zeros(2*len(target)+1,dtype=np.int64);labels[1::2]=target
    emissions=logp[:,labels];t,s=emissions.shape
    alpha=np.full((t,s),-np.inf);alpha[0,:2]=emissions[0,:2]
    skip=np.zeros(s,dtype=bool)
    skip[2:]=(labels[2:]!=0)&(labels[2:]!=labels[:-2])
    for frame in range(1,t):
        previous=alpha[frame-1]
        one=np.r_[-np.inf,previous[:-1]]
        two=np.r_[[-np.inf,-np.inf],previous[:-2]]
        two=np.where(skip,two,-np.inf)
        alpha[frame]=emissions[frame]+np.logaddexp(np.logaddexp(previous,one),two)
    logz=float(np.logaddexp(alpha[-1,-1],alpha[-1,-2]))
    if not np.isfinite(logz):raise ValueError('target has no finite CTC path')
    beta=np.full((t,s),-np.inf);beta[-1,-2:]=0.
    for frame in range(t-2,-1,-1):
        next_value=emissions[frame+1]+beta[frame+1]
        one=np.r_[next_value[1:],-np.inf]
        two=np.r_[next_value[2:],[-np.inf,-np.inf]]
        permitted=np.r_[skip[2:],[False,False]]
        beta[frame]=np.logaddexp(np.logaddexp(next_value,one),np.where(permitted,two,-np.inf))
    reverse=float(np.logaddexp(emissions[0,0]+beta[0,0],emissions[0,1]+beta[0,1]))
    if abs(reverse-logz)>1e-7:raise ValueError('CTC forward/backward disagreement')
    states=np.exp(alpha+beta-logz)
    if not np.allclose(states.sum(1),1.,atol=1e-7,rtol=0):raise ValueError('CTC state posterior not normalized')
    phones=states[:,1::2];mass=phones.sum(0)
    if not np.isfinite(mass).all() or (mass<1-1e-6).any():raise ValueError('invalid reference-position occupancy')
    weights=phones/mass[None,:];summary=[]
    for i in range(len(target)):
        w=weights[:,i];cdf=w.cumsum();lo=int(np.searchsorted(cdf,.05));hi=min(t-1,int(np.searchsorted(cdf,.95)))
        summary.append({'position':i,'label_id':target[i],'expected_occupied_frames':float(mass[i]),
            'normalized_time_entropy':float(-(w[w>0]*np.log(w[w>0])).sum()/np.log(t)) if t>1 else 0.,
            'central_90_start_frame':lo,'central_90_end_frame_exclusive':hi+1,
            'central_90_width_frames':hi-lo+1,'maximum_position_weight':float(w.max())})
    return {'log_target_probability':logz,'position_weights':weights,'state_posteriors':states,
            'positions':summary,'reference_conditioned':True,'acoustic_boundary_truth':False}


def phone_key(source_vsqx: str,phone: dict) -> tuple:
    if not source_vsqx or not phone.get('note_id') or not isinstance(phone.get('phone_index'),int):
        raise ValueError('missing phone provenance')
    return source_vsqx,phone['note_id'],phone['phone_index'],phone['phone']


def exact_sequence_matches(original: dict,rendered: Sequence[dict]) -> list[str]:
    """No same-song shortcut; sequence equality is necessary but not sufficient."""
    return [r['example_id'] for r in rendered if r['song_id']==original['song_id']
            and r['source_vsqx_sha256']==original['source_vsqx_sha256']
            and r['target_ipa']==original['target_ipa']]

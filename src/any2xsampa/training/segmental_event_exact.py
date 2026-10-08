"""Exact tiled event loss with analytic arc-marginal backward.

Same mean-logp/start/hold score as segmental_event.py. Neural and segment score
arithmetic stays FP32 on CUDA; small dynamic-programming tables and marginal
accumulation use CPU FP64 for stability. No durations or labels are pruned.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Any,Sequence
import numpy as np
from any2xsampa.training.segmental_event import _validate


@lru_cache(maxsize=32)
def arc_indices(frames: int) -> tuple:
    if frames<1:raise ValueError('positive frames required')
    ends=np.repeat(np.arange(1,frames+1),np.arange(1,frames+1))
    starts=np.concatenate([np.arange(end) for end in range(1,frames+1)])
    return starts,ends,ends-starts


def _lse(x,axis=None):
    # logaddexp avoids undefined all-minus-infinity intermediate arithmetic.
    return np.logaddexp.reduce(x,axis=axis)


def partitions_and_marginals(blank,arcs,unique_scores,target_indices):
    """CPU FP64 exact forward/backward, returning gradients wrt arc scores."""
    blank=np.asarray(blank,dtype=np.float64);arcs=np.asarray(arcs,dtype=np.float64)
    selected=np.asarray(unique_scores,dtype=np.float64);target=np.asarray(target_indices,dtype=np.int64)
    t=len(blank);u=len(target);s,e,_=arc_indices(t);n=len(s)
    if arcs.shape!=(n,) or selected.ndim!=2 or selected.shape[0]!=n:raise ValueError('packed arc shape mismatch')
    if not np.isfinite(blank).all() or not np.isfinite(arcs).all() or not np.isfinite(selected).all():raise ValueError('nonfinite scores')
    den=np.full(t+1,-np.inf);den[0]=0.
    num=np.full((t+1,u+1),-np.inf);num[0,0]=0.
    for end in range(1,t+1):
        lo=end*(end-1)//2;hi=lo+end
        den[end]=np.logaddexp(den[end-1]+blank[end-1],_lse(den[:end]+arcs[lo:hi]))
        num[end]=num[end-1]+blank[end-1]
        if u:num[end,1:]=np.logaddexp(num[end,1:],_lse(num[:end,:-1]+selected[lo:hi][:,target],axis=0))
    bd=np.full(t+1,-np.inf);bd[t]=0.
    bn=np.full((t+1,u+1),-np.inf);bn[t,u]=0.
    for start in range(t-1,-1,-1):
        end=np.arange(start+1,t+1);idx=end*(end-1)//2+start
        bd[start]=np.logaddexp(blank[start]+bd[start+1],_lse(arcs[idx]+bd[end]))
        bn[start]=blank[start]+bn[start+1]
        if u:bn[start,:-1]=np.logaddexp(bn[start,:-1],_lse(selected[idx][:,target]+bn[end,1:],axis=0))
    z,y=den[-1],num[-1,-1]
    if not np.isfinite(y) or abs(z-bd[0])>1e-7 or abs(y-bn[0,0])>1e-7:raise ValueError('event forward/backward mismatch')
    gd=np.exp(den[s]+arcs+bd[e]-z)
    gn=np.zeros_like(selected)
    for position,label in enumerate(target):
        gn[:,label]+=np.exp(num[s,position]+selected[:,label]+bn[e,position+1]-y)
    blank_den=np.exp(den[:-1]+blank+bd[1:]-z)
    blank_num=np.exp(num[:-1]+blank[:,None]+bn[1:]-y).sum(axis=1)
    return {'loss':float(z-y),'denominator':float(z),'numerator':float(y),
            'den_arc':gd,'num_unique_arc':gn,'blank_gradient':blank_den-blank_num}


def _scores(torch,q,b,starts,ends):
    prefix=torch.cat((q.new_zeros((1,q.shape[1])),q.cumsum(0)))
    hold=torch.nn.functional.logsigmoid(-b)
    hp=torch.cat((b.new_zeros(1),hold.cumsum(0)))
    onset=torch.nn.functional.logsigmoid(b)
    duration=(ends-starts).to(q.dtype)
    content=(prefix[ends,1:]-prefix[starts,1:])/duration[:,None]
    return content+onset[starts,None]+(hp[ends]-hp[starts+1])[:,None]


def exact_sequence_nll(torch: Any,q: Any,b: Any,target: Sequence[int],*,tile_arcs: int=8192) -> Any:
    _validate(torch,q,b,target)
    if not isinstance(tile_arcs,int) or tile_arcs<1:raise ValueError('invalid tile size')
    unique=sorted(set(target));lookup={k:i for i,k in enumerate(unique)}
    target_unique=[lookup[k] for k in target]
    class Exact(torch.autograd.Function):
        @staticmethod
        def forward(ctx,q,b):
            st,en,_=arc_indices(len(q));starts=torch.as_tensor(st.copy(),device=q.device);ends=torch.as_tensor(en.copy(),device=q.device)
            all_arcs=[];chosen=[]
            for lo in range(0,len(st),tile_arcs):
                scores=_scores(torch,q,b,starts[lo:lo+tile_arcs],ends[lo:lo+tile_arcs])
                all_arcs.append(torch.logsumexp(scores,dim=1).cpu().double().numpy())
                chosen.append(scores[:,[k-1 for k in unique]].cpu().double().numpy())
            dp=partitions_and_marginals(q[:,0].cpu().double().numpy(),np.concatenate(all_arcs),np.concatenate(chosen),target_unique)
            if dp['loss'] < -1e-3 or not np.isfinite(dp['loss']):raise ValueError('invalid exact event loss')
            ctx.save_for_backward(q,b,starts,ends);ctx.dp=dp
            return q.new_tensor(dp['loss'])
        @staticmethod
        def backward(ctx,grad):
            q,b,starts,ends=ctx.saved_tensors;dp=ctx.dp;t=len(q)
            prefix_grad=q.new_zeros((t+1,q.shape[1]-1))
            start_grad=b.new_zeros(t);hold_delta=b.new_zeros(t+1)
            columns=torch.tensor([k-1 for k in unique],device=q.device,dtype=torch.long)
            for lo in range(0,len(starts),tile_arcs):
                s=starts[lo:lo+tile_arcs];e=ends[lo:lo+tile_arcs]
                scores=_scores(torch,q,b,s,e)
                den=torch.as_tensor(dp['den_arc'][lo:lo+tile_arcs],device=q.device,dtype=q.dtype)
                numerator=torch.as_tensor(dp['num_unique_arc'][lo:lo+tile_arcs],device=q.device,dtype=q.dtype)
                g=scores.softmax(-1)*den[:,None]
                if len(unique):g[:,columns]-=numerator
                mass=g.sum(-1)
                start_grad.index_add_(0,s,mass)
                hold_delta.index_add_(0,s+1,mass);hold_delta.index_add_(0,e,-mass)
                scaled=g/(e-s).to(q.dtype)[:,None]
                prefix_grad.index_add_(0,e,scaled);prefix_grad.index_add_(0,s,-scaled)
            dq=q.new_zeros(q.shape)
            dq[:,0]=torch.as_tensor(dp['blank_gradient'],device=q.device,dtype=q.dtype)
            dq[:,1:]=prefix_grad[1:].flip(0).cumsum(0).flip(0)
            sigmoid=b.sigmoid();db=start_grad*(1-sigmoid)-hold_delta.cumsum(0)[:t]*sigmoid
            if not torch.isfinite(dq).all() or not torch.isfinite(db).all():raise ValueError('nonfinite event gradient')
            return grad*dq,grad*db
    return Exact.apply(q,b)


def exact_viterbi(torch: Any,q: Any,b: Any,*,tile_arcs: int=8192) -> dict:
    """GPU score tiles, CPU exact Viterbi; no reference or target length."""
    _validate(torch,q,b,[])
    st,en,_=arc_indices(len(q));starts=torch.as_tensor(st.copy(),device=q.device);ends=torch.as_tensor(en.copy(),device=q.device)
    maximum=[];labels=[]
    with torch.no_grad():
        for lo in range(0,len(st),tile_arcs):
            scores=_scores(torch,q,b,starts[lo:lo+tile_arcs],ends[lo:lo+tile_arcs])
            a,k=scores.max(-1);maximum.append(a.cpu().double().numpy());labels.append(k.cpu().numpy()+1)
    a=np.concatenate(maximum);k=np.concatenate(labels);blank=q[:,0].detach().cpu().double().numpy()
    best=np.full(len(q)+1,-np.inf);best[0]=0.;back=[]
    for end in range(1,len(q)+1):
        lo=end*(end-1)//2;values=best[:end]+a[lo:lo+end];start=int(values.argmax())
        null=best[end-1]+blank[end-1]
        if null>=values[start]:best[end]=null;back.append((end-1,0))
        else:best[end]=values[start];back.append((start,int(k[lo+start])))
    end=len(q);events=[]
    while end:
        start,label=back[end-1]
        if label:events.append({'start_frame':start,'end_frame_exclusive':end,'label_id':label})
        end=start
    events.reverse()
    return {'event_ids':[r['label_id'] for r in events],'events':events,'best_path_score':float(best[-1]),'reference_conditioning':False}

"""Exact latent-event CRF: unbounded phone segments and one-frame null arcs.

This is NOT CTC collapse. Consecutive identical phone segments are separate
events. All durations 1..T participate; no observed or inferred boundary labels
are supplied to the loss. Torch is injected to keep lightweight imports safe.
"""
from __future__ import annotations
from typing import Any, Sequence


def _validate(torch: Any, frame_log_probs: Any, boundary_logits: Any, target: Sequence[int]) -> tuple[int,int]:
    q,b=frame_log_probs,boundary_logits
    if q.ndim!=2 or q.shape[0]<1 or q.shape[1]<2 or b.shape!=(q.shape[0],):
        raise ValueError('event model requires [T,K>=2] log probabilities and [T] boundary logits')
    if q.dtype not in (torch.float32,torch.float64) or b.dtype!=q.dtype or b.device!=q.device:
        raise ValueError('event evidence dtype/device mismatch')
    if not torch.isfinite(q).all() or not torch.isfinite(b).all():raise ValueError('non-finite event evidence')
    if not torch.allclose(torch.logsumexp(q,dim=-1),torch.zeros_like(b),atol=2e-5,rtol=0):
        raise ValueError('frame evidence must be normalized log probabilities')
    if len(target)>len(q) or any(isinstance(v,bool) or not isinstance(v,int) or not 0<v<q.shape[1] for v in target):
        raise ValueError('target must be nonblank labels with at most T events')
    return len(q),q.shape[1]


def log_partitions(torch: Any, frame_log_probs: Any, boundary_logits: Any, target: Sequence[int]) -> tuple[Any,Any]:
    """Return log Z(all paths), log Z(paths spelling target), differentiably."""
    q,b=frame_log_probs,boundary_logits
    total,_=_validate(torch,q,b,target);u=len(target)
    prefix=torch.cat((q.new_zeros((1,q.shape[1])),q.cumsum(0)),dim=0)
    start=torch.nn.functional.logsigmoid(b)
    hold=torch.nn.functional.logsigmoid(-b)
    hp=torch.cat((hold.new_zeros(1),hold.cumsum(0)))
    zero=q.new_zeros(());den=[zero]
    num=[torch.cat((zero.reshape(1),q.new_full((u,),float('-inf'))))]
    ids=torch.tensor(list(target),device=q.device,dtype=torch.long)
    for end in range(1,total+1):
        duration=torch.arange(end,0,-1,device=q.device,dtype=q.dtype)
        # Content is the mean log-probability over the whole candidate segment;
        # start/hold costs represent event onset vs continued occupancy.
        scores=(prefix[end,1:]-prefix[:end,1:])/duration[:,None]
        scores=scores+start[:end,None]+(hp[end]-hp[1:end+1])[:,None]
        arc=torch.stack(den)+torch.logsumexp(scores,dim=1)
        den.append(torch.logaddexp(torch.logsumexp(arc,dim=0),den[-1]+q[end-1,0]))
        reachable=min(u,end)
        null=num[-1][0]+q[end-1,0]
        if reachable:
            previous=torch.stack(num)[:,:reachable]
            chosen=scores[:,ids[:reachable]-1]
            # Each column has at least one reachable start. Avoid all -inf
            # logsumexp columns, whose gradients would otherwise become NaN.
            phone=torch.logsumexp(previous+chosen,dim=0)
            blank=num[-1][1:reachable+1]+q[end-1,0]
            current=torch.logaddexp(phone,blank)
        else:current=q.new_empty(0)
        num.append(torch.cat((null.reshape(1),current,q.new_full((u-reachable,),float('-inf')))))
    return den[-1],num[-1][u]


def sequence_nll(torch: Any, frame_log_probs: Any, boundary_logits: Any, target: Sequence[int]) -> Any:
    denominator,numerator=log_partitions(torch,frame_log_probs,boundary_logits,target)
    loss=denominator-numerator
    if not torch.isfinite(loss) or loss.detach().item() < -1e-3:
        raise ValueError('invalid segmental negative log likelihood')
    return loss


def make_event_head(torch: Any, pretrained_weight: Any, pretrained_bias: Any) -> Any:
    """Small cache-only prototype, never aliases the shared native CTC head."""
    if pretrained_weight.shape!=(428,1024) or pretrained_bias.shape!=(428,):
        raise ValueError('expected the full frozen pretrained 1024-to-428 head')
    class EventHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.content=torch.nn.Linear(1024,428,device=pretrained_weight.device,dtype=pretrained_weight.dtype)
            self.event=torch.nn.Linear(1024,1,device=pretrained_weight.device,dtype=pretrained_weight.dtype)
            with torch.no_grad():
                self.content.weight.copy_(pretrained_weight);self.content.bias.copy_(pretrained_bias)
                self.event.weight.zero_();self.event.bias.zero_()
        def forward(self,features):
            if features.ndim!=2 or features.shape[1]!=1024 or not torch.isfinite(features).all():
                raise ValueError('invalid frozen final features')
            differences=torch.cat((torch.zeros_like(features[:1]),features[1:]-features[:-1]))
            return self.content(features).log_softmax(-1),self.event(differences).squeeze(-1)
    return EventHead()

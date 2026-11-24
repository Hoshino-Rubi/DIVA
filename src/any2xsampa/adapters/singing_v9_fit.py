"""Rank-eight final-only residual on frozen V5 embeddings; no onset parameters."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .singing_adapter import SingingAdapter, ONSETS, FINALS, split_target, note_pools
from ..exporters.syllable_components import components
from ..exporters.singing_v9_rules import allowed

def classes(rules):
    states=[s for s in rules["states"] if allowed(s.split(),rules)]
    parts=[components(s.split()) for s in states]
    finals=sorted({" ".join(f) for _,f,_ in parts})
    return states,finals,[(ONSETS.index(o),finals.index(" ".join(f))) for o,f,_ in parts]

def features(score, frames, audio, state, rules):
    """Keep the published V5 joint winner's onset, including continuation overrides."""
    states,finals,parts=classes(rules)
    net=SingingAdapter(frames.shape[1]).cuda().eval().requires_grad_(False)
    net.load_state_dict(state["weights"])
    rows=score["notes"]; _,b,_,_=note_pools(rows,frames,audio)
    with torch.no_grad():
        h=net.final.pooled(((torch.as_tensor(b,device="cuda").float()-net.mean)/net.scale).clamp(-6,6))
        z=net.final.output(h)
    base=torch.full((len(rows),len(finals)),-torch.inf,device="cuda")
    for s,(oi,fi) in zip(states,parts):
        _,f,_=components(s.split())
        split=split_target(s.split())
        proxy=split[1] if split else f[0] if f[0] in FINALS else None
        if proxy is None:continue
        j=FINALS.index(proxy)
        idx=[k for k,r in enumerate(rows) if r["acoustic_prediction"]["onset"]==ONSETS[oi]]
        prior=float(state["pair_prior"][oi,j])*state["calibration"][1]
        penalty=1.0 if s not in state["witnesses"] else 0.
        base[idx,fi]=z[idx,j]+prior+state["calibration"][2]*int(f[-1] in ("t","k"))-penalty
    for i,r in enumerate(rows):
        f=r["acoustic_prediction"]["final"]
        if f not in finals or not torch.isfinite(base[i,finals.index(f)]):
            raise ValueError("Frozen onset has no playable V5 fallback")
    return h,base,finals

class FinalResidual(nn.Module):
    def __init__(self,width:int,n:int):
        super().__init__()
        self.down=nn.Linear(width,8,bias=False)
        self.up=nn.Linear(8,n,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,x):
        return self.up(self.down(x))

def infer(net,h,b,rows,finals):
    with torch.no_grad():
        pr=(b+net(h)).softmax(1);p,idx=pr.max(1)
    result=[]
    for r,k,conf in zip(rows,idx.tolist(),p.tolist(),strict=True):
        ev=r["acoustic_prediction"]
        s=" ".join(filter(None,[ev["onset"],finals[k]]))
        if ev["continuation"]:s=" ".join(r["v4c_phonemes"])
        result.append((s,conf))
    return result

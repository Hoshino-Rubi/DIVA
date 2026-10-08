"""Head-only H/E training helpers; references never enter decoding."""
from __future__ import annotations
from typing import Any
from any2xsampa.training.segmental_event import make_event_head
from any2xsampa.training.segmental_event_exact import exact_sequence_nll,exact_viterbi
from any2xsampa.training.singing_domain_losses import sequence_ctc
from any2xsampa.training.protected_adaptation import baseline_kl
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def make_head(torch: Any,state: dict,arm: str):
    if arm not in ('H','E'):raise ValueError('unknown head arm')
    if state['weight'].shape!=(428,1024) or state['bias'].shape!=(428,):raise ValueError('pretrained head geometry mismatch')
    if arm=='E':return make_event_head(torch,state['weight'],state['bias'])
    class Content(torch.nn.Module):
        def __init__(self):
            super().__init__();self.content=torch.nn.Linear(1024,428,device=state['weight'].device,dtype=state['weight'].dtype)
            with torch.no_grad():self.content.weight.copy_(state['weight']);self.content.bias.copy_(state['bias'])
        def forward(self,features):
            return self.content(features).log_softmax(-1),None
    return Content()


def loss_on_row(torch: Any,head: Any,row: dict,arm: str,spec: dict):
    features=row['features'].to('cuda');prior=row['prior'].to('cuda')
    q,b=head(features);ids=row['target_ids'];params=spec['training']
    if arm=='H':data=sequence_ctc(torch,q.unsqueeze(0),ids)
    elif arm=='E':data=exact_sequence_nll(torch,q,b,ids,tile_arcs=spec['exact_engine']['tile_arcs'])/len(ids)
    else:raise ValueError('unknown head arm')
    kl=baseline_kl(torch,q.unsqueeze(0),prior.unsqueeze(0),torch.tensor([len(q)],device=q.device),temperature=params['temperature'])
    weight=params['rendered_loss_weight'] if row['audio_role']=='rendered' else 1.
    loss=weight*data+params['kl_weight']*kl
    if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
    return loss,{'sequence_loss_per_reference':float(data.detach()),'kl':float(kl.detach())}


def predict(torch: Any,head: Any,features: Any,arm: str,spec: dict):
    with torch.no_grad():
        q,b=head(features.to('cuda'))
        if arm=='H':ids=ctc_collapse(q.argmax(-1).cpu().tolist())
        elif arm=='E':ids=exact_viterbi(torch,q,b,tile_arcs=spec['exact_engine']['tile_arcs'])['event_ids']
        else:raise ValueError('unknown head arm')
    return ids,q.detach().cpu().contiguous(),b.detach().cpu().contiguous() if b is not None else None

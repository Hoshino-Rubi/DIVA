"""Complete-sequence auxiliary CTC and masked two-view consistency."""
from __future__ import annotations
from pathlib import Path
from typing import Any,Sequence
from any2xsampa.training.limited_data_artifacts import read_json
from any2xsampa.training.protected_adaptation import baseline_kl

def attribute_inventory()->dict:
    cfg=read_json(Path(__file__).parents[1]/'config/singing_phone_attributes_v1.json')
    attrs=cfg['attributes']
    if any(len(values)!=len(attrs) for values in cfg['phones'].values()):raise ValueError('incomplete IPA attribute row')
    inventories={name:{value:j+1 for j,value in enumerate(sorted({r[i] for r in cfg['phones'].values()}))} for i,name in enumerate(attrs)}
    return {'config':cfg,'labels':inventories,'sizes':{name:len(values)+1 for name,values in inventories.items()}}

def attribute_targets(phones:Sequence[str],inventory:dict)->dict[str,list[int]]|None:
    cfg=inventory['config']
    if any(p not in cfg['phones'] for p in phones):return None
    return {name:[inventory['labels'][name][cfg['phones'][p][i]] for p in phones]
            for i,name in enumerate(cfg['attributes'])}

def required_frames(ids:Sequence[int])->int:
    return len(ids)+sum(x==y for x,y in zip(ids,ids[1:]))

def sequence_ctc(torch:Any,logits:Any,ids:Sequence[int])->Any:
    if logits.ndim!=3 or logits.shape[0]!=1 or not torch.isfinite(logits).all():raise ValueError('CTC microbatch requires finite [1,T,C]')
    if not ids or any(isinstance(i,bool) or not isinstance(i,int) or not 0<i<logits.shape[-1] for i in ids):raise ValueError('blank/invalid CTC target')
    if required_frames(ids)>logits.shape[1]:raise ValueError('insufficient CTC frames for repeated labels')
    return torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0,1),torch.tensor(ids,device=logits.device,dtype=torch.long),
        torch.tensor([logits.shape[1]],device=logits.device),torch.tensor([len(ids)],device=logits.device),blank=0,zero_infinity=False)

def two_view_kl(torch:Any,left:Any,right:Any,lengths:Any,temperature:float=2.)->Any:
    return .5*(baseline_kl(torch,left,right,lengths,temperature=temperature)+baseline_kl(torch,right,left,lengths,temperature=temperature))

def auxiliary_ctc(torch:Any,logits:dict,phones:Sequence[str],inventory:dict)->tuple[Any,list[str]]:
    targets=attribute_targets(phones,inventory)
    if targets is None:
        first=next(iter(logits.values()));return first.sum()*0.,['unknown_ipa_attribute_mapping']
    losses=[];excluded=[]
    for name,ids in targets.items():
        if required_frames(ids)>logits[name].shape[1]:excluded.append(name+':insufficient_ctc_frames')
        else:losses.append(sequence_ctc(torch,logits[name],ids))
    if not losses:return next(iter(logits.values())).sum()*0.,excluded
    return torch.stack(losses).mean(),excluded

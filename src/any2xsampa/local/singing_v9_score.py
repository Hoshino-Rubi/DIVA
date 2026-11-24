"""Frozen AB release route. Inference takes only audio, its GAME notes, and local models."""
from pathlib import Path
import torch
from ..adapters.singing_v9_asr import LocalWhisper
from ..adapters.singing_v9_fit import FinalResidual,features,infer
from ..pipeline.singing_v9_decode import combine,proposals
from ..exporters.miku_v4c import POLICY
from ..exporters.singing_v9_rules import allowed,load_rules
from .singing_score import singing_score

def score(audio,vocal,frames,game,bundle,checkpoint:Path,separated,phrases=None,*,checked_confidence=.70,prepared=None):
    parent=checkpoint.parent.parent/"cantonese-singing-v5/adapter.pt"
    base=singing_score(audio,vocal,frames,game,parent,separated,prepared=prepared)
    if phrases is None:
        raw=[dict(n,end_sec=n["start_sec"]+n["duration_sec"]) for n in game["note_candidates"]]
        phrases=LocalWhisper(bundle).decode(vocal,raw)
    rows=base["notes"];rules=load_rules()
    st=torch.load(parent,weights_only=True,map_location="cpu")
    head=torch.load(checkpoint,weights_only=True,map_location="cpu")
    h,b,f=features(base,frames,vocal,st,rules)
    if head["format"]!="v9_final_rank8" or head["finals"]!=f:raise ValueError("Incompatible V9 rules")
    net=FinalResidual(h.shape[1],len(f)).cuda().eval()
    net.load_state_dict(head["weights"])
    back=infer(net,h,b,rows,f);prop,_=proposals(phrases,rows,rules)
    seq=combine(rows,prop,back,rules,*head["blend"],
                checked_confidence=checked_confidence)
    for row,(s,src,q,jp),(fallback,conf) in zip(rows,seq,back,strict=True):
        tok=s.split()
        if not allowed(tok,rules):raise ValueError(s)
        label=" ".join(tok[:-1] if tok[-1] in ("t","k") else tok)
        row.update(v4c_phonemes=tok,lyric=jp or POLICY["base_pinyin"].get(label) or s,recognition_source=src,
                   acoustic_prediction=dict(v5=row["acoustic_prediction"],v9_confidence=q,asr_jyutping=jp,selected=s,
                       checked_protected=checked_confidence is not None and conf>=checked_confidence and fallback.split()[-1] in ('t','k')))
    base.update(recognition_mode="singing_v9",track="AB",adapter=str(checkpoint.resolve()))
    return base

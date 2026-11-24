"""Audio-only V11 route: supervise seams first, then repool V9 on the retained grid."""
import torch
from .common import read
from .cantonese_audio import POLICY
from .singing_v9_score import score as ab_score
from ..adapters.singing_v11_seams import CFG, evidence, probability
from ..music.melody_cleanup import game_rows
from ..music.singing_v10_seams import Seams
from ..music.singing_v5_melody import CFG as LEGATO
from ..exporters.syllable_components import components
from ..exporters.singing_v9_rules import allowed,load_rules


def load_features(vocal,frames,game,phrases,checkpoint,vocab,*,onsets=None):
    rows=game_rows(game,len(vocal)/16000);probe=Seams(vocal)
    v5=checkpoint.parent.parent/"cantonese-singing-v5/adapter.pt"
    states=[torch.load(p,weights_only=True,map_location="cpu") for p in (v5,checkpoint)]
    x=evidence(rows,frames,probe,phrases,vocab,*states,onsets=onsets)
    return rows,probe,x


def decisions(rows,probe,features,model):
    actions={}
    for j,x in features.items():
        if probability(model,x)<model["threshold"]:continue
        a,b=rows[j:j+2]
        cs=[probe.core(r["start_sec"],r["end_sec"],r["pitch"]) for r in (a,b)]
        if not all(c[0]>0 for c in cs):continue
        stable=[m<=CFG["stable_mad"] and s<=CFG["unstable_speed"] for _,m,s,_ in cs]
        short=min(c[3] for c in cs)<=CFG["short"]
        if abs(a["pitch"]-b["pitch"])<=CFG["max_glide_step"] and (short or not all(stable)):
            actions[j]="merge"
        elif all(stable) and not short:
            actions[j]="continue"
    return actions


def continuation(rows,indices):
    rules=load_rules(); changed=0;vetoed=0
    for j in sorted(indices):
        row=rows[j];ev=row["acoustic_prediction"]
        # A high-confidence held stop cannot simultaneously be an open vowel.
        if ev.get("checked_protected"):vetoed+=1;continue
        _,f,_=components(rows[j-1]["v4c_phonemes"])
        vowel=LEGATO["nucleus"].get(f[0]) if f else None
        if vowel and allowed([vowel],rules):
            changed+=row["v4c_phonemes"]!=[vowel]
            row.update(v4c_phonemes=[vowel],lyric="-",recognition_source="v11_continuation")
            ev.update(selected=vowel)
    return changed,vetoed

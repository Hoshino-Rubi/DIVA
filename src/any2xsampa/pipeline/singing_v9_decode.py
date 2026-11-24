"""Time-constrained monotone ASR assignment before score export; no lyric input."""
from itertools import groupby
import numpy as np
from ..music.singing_v5_melody import CFG
from ..exporters.syllable_components import components
from ..exporters.singing_v9_rules import allowed

def match(words, rows):
    # Reward matching, permit deletions on either side. No nearest-neighbor reuse.
    m,n=len(words),len(rows)
    dp=np.zeros((m+1,n+1));step=np.zeros((m+1,n+1),np.uint8)
    for i,w in enumerate(words,1):
        for j,r in enumerate(rows,1):
            d=abs(w["start"]-r["start_sec"])
            length=max(.08,w["end"]-w["start"])
            ov=max(0.,min(w["end"],r["end_sec"])-max(w["start"],r["start_sec"]))
            fit=max(.0,1-d/.36)*.64+.36*min(1.,ov/min(length,r["duration_sec"]))
            options=(dp[i-1,j],dp[i,j-1],dp[i-1,j-1]+fit if d<.36 and fit>.23 else -np.inf)
            k=int(np.argmax(options));dp[i,j]=options[k];step[i,j]=k
    i,j=m,n; matches=[]
    while i and j:
        k=step[i,j]
        if k==2:
            w=words[i-1];r=rows[j-1]
            q=w["confidence"]*np.exp(-abs(w["start"]-r["start_sec"])/.36)
            matches.append((j-1,w,float(q)))
        i-=int(k!=1); j-=int(k!=0)
    return matches[::-1]

def proposals(phrases,rows,rules):
    table=rules["jyutping"];out={}; rejected=0
    for phrase in phrases:
        words=phrase["syllables"]
        # Repeated hallucinations can be very confident even entirely outside speech.
        skip=set()
        for _,g in groupby(enumerate(words),lambda kv:kv[1]["jp"]):
            run=list(g)
            if len(run)>=5:skip.update(i for i,_ in run)
        clean=[]
        for i,w in enumerate(words):
            jp=w["jp"].rstrip("123456")
            if i in skip or jp not in table or not .035<=w["end"]-w["start"]<=2.4:
                rejected+=1;continue
            clean.append(dict(w,jp=jp))
        ix=[i for i,r in enumerate(rows) if phrase["a"]<=r["start_sec"]<phrase["b"]]
        for j,w,conf in match(clean,[rows[k] for k in ix]):
            k=ix[j];s=table[w["jp"]]
            if k not in out or conf>out[k][1]:out[k]=(s,conf,w["jp"],w["end"])
    return out,rejected

def combine(rows,props,back,rules,threshold,relative,*,checked_confidence=None):
    res=[]; last_end=-1.
    for k,(fallback,conf) in enumerate(back):
        p=props.get(k); r=rows[k]
        protected = (checked_confidence is not None and conf >= checked_confidence
                     and components(fallback.split())[2] in {"t","k"})
        open_proposal = p and components(p[0].split())[2] not in {"p","t","k","m","n","N"}
        if protected and open_proposal and p[1]>=max(threshold,relative*conf):
            res.append((fallback,"protected_checked",conf,""));last_end=-1.
            continue
        if p and p[1]>=max(threshold,relative*conf):
            s,q,jp,last_end=p
            res.append((s,"asr",q,jp));continue
        if not protected and res and res[-1][1] in ("asr","melisma") and r["start_sec"]<last_end+.06 and r["acoustic_prediction"]["continuation"]:
            _,f,_=components(res[-1][0].split());nucleus=CFG["nucleus"].get(f[0])
            if nucleus and allowed([nucleus],rules):
                res.append((nucleus,"melisma",res[-1][2],res[-1][3]));continue
        res.append((fallback,"acoustic",conf,""));last_end=-1.
    return res

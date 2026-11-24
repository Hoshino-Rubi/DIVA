"""Frozen contextual evidence for adjacent GAME events (no references or text labels)."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.special import softmax, expit
from .singing_adapter import SingingAdapter, note_pools, ONSETS
from .singing_v9_fit import features, FinalResidual
from ..music.singing_v10_seams import periodic
from ..exporters.singing_v9_rules import load_rules

CFG=json.loads((Path(__file__).resolve().parents[1]/"config/cantonese_singing_v11.json").read_text("utf8"))
NAMES=("whisper_distance","ctc_consonant_peak","entropy_left","entropy_right","final_distance",
       "same_asr_syllable","asr_both_covered","interval","slope_left","slope_right","slope_change",
       "log_variance_ratio","periodicity_left","periodicity_right","duration_left","duration_right",
       "log_duration_ratio","gap_ms","rms_dip_db","hf_flux","acoustic_valid","mad_left","mad_right")


def distance(x,y):
    return float(np.clip(1-np.dot(x,y)/max(1e-12,np.linalg.norm(x)*np.linalg.norm(y)),0,2))


@torch.inference_mode()
def distributions(rows,frames,vocal,state,head,*,onsets=None):
    from .singing_v5 import predict_v5
    adapter=SingingAdapter(frames.shape[1]).cuda().eval()
    adapter.load_state_dict(state["weights"])
    a,b,v,_=note_pools(rows,frames,vocal)
    x,_=adapter(*(torch.as_tensor(t,device="cuda") for t in (a,b,v)))
    p=x.softmax(1)
    if onsets is not None:
        confidence,idx=p.max(1)
        onsets.extend(zip((ONSETS[i] for i in idx.tolist()),confidence.tolist(),strict=True))
    entropy=-(p*x.log_softmax(1)).sum(1).cpu().numpy()
    ev=predict_v5(rows,frames,vocal,state)
    tagged=[dict(r,acoustic_prediction=e) for r,e in zip(rows,ev,strict=True)]
    h,bias,finals=features(dict(notes=tagged),frames,vocal,state,load_rules())
    if head["finals"]!=finals: raise ValueError("Final classes changed")
    net=FinalResidual(h.shape[1],len(finals)).cuda().eval()
    net.load_state_dict(head["weights"])
    return entropy,(bias+net(h)).softmax(1).cpu().numpy()


def flank(audio,row,left):
    """Three 20ms periodic frames wholly inside the 40ms note flank; no octave search."""
    end,start=round(row["end_sec"]*16000),round(row["start_sec"]*16000)
    a=max(start,end-640) if left else start
    b=end if left else min(end,start+640)
    target=16000/(440*2**((row["pitch"]-69)/12))
    values=[]; times=[]; conf=[]
    for k in range(a,b-319,80):
        l,q=periodic(audio[k:k+320].astype(float),int(target/1.28)-1,int(np.ceil(target*1.28))+1)
        if q>.5 and l>0:
            times.append((k-a)/16000);values.append(-12*np.log2(l/target));conf.append(q)
    if len(times)<2:return 0.,0.,0.
    t=np.array(times);y=np.array(values)
    return float(np.sum((t-t.mean())*(y-y.mean()))/sum((t-t.mean())**2)),float(y.var()),float(np.mean(conf))


def evidence(rows,frames,probe,phrases,vocab,state,head,*,onsets=None):
    whisper=768
    if frames.shape[1]!=whisper+len(vocab)+80:raise ValueError("Unexpected frozen frame layout")
    consonants=[i for k,i in vocab.items() if k[0] in CFG["ctc_consonant_initials"]
                and k.rstrip("12345") not in CFG["semivowel_vowel_labels"]]
    if not consonants: raise ValueError("Missing CTC consonant labels")
    ent,final=distributions(rows,frames,probe.vocal,state,head,onsets=onsets)
    words=[w for p in phrases for w in p["syllables"] if .035<=w["end"]-w["start"]<=2.4]
    def covers(r):
        mid=(r["start_sec"]+r["end_sec"])/2
        return {i for i,w in enumerate(words) if w["start"]<=mid<w["end"]}
    spans=list(map(covers,rows));result={}
    for j,(l,r) in enumerate(zip(rows,rows[1:])):
        gap=r["start_sec"]-l["end_sec"]
        if not -1e-8<=gap<=CFG["max_gap"]+1e-8:continue
        t=(l["end_sec"]+r["start_sec"])/2
        # Frozen frame centres are at (i+.5)/50, not frame left edges.
        lo,mid,hi=np.clip(np.ceil((np.array([t-.04,t,t+.04]))*50-.5).astype(int),0,len(frames))
        if not lo<mid<hi:continue
        fr=frames[lo:hi].astype(np.float32)
        wh=distance(fr[:mid-lo,:whisper].mean(0),fr[mid-lo:,:whisper].mean(0))
        ctc=softmax(fr[:,whisper:whisper+len(vocab)],axis=1)[:,consonants].sum(1).max()
        ls,lv,lq=flank(probe.vocal,l,True);rs,rv,rq=flank(probe.vocal,r,False)
        def core(n):return probe.core(n["start_sec"],n["end_sec"],n["pitch"])
        lc,rc=core(l),core(r); env=probe.at(round(t,6))
        dl,dr=l["duration_sec"],r["duration_sec"]
        x=[wh,ctc,ent[j],ent[j+1],distance(final[j],final[j+1]),bool(spans[j]&spans[j+1]),
           bool(spans[j] and spans[j+1]),abs(l["pitch"]-r["pitch"]),ls,rs,abs(ls-rs),
           np.log((lv+.005)/(rv+.005)),lq,rq,dl,dr,np.log(dl/dr),1000*gap,
           env.get("dip_db",0.),env.get("flux",0.),env["valid"],lc[1],rc[1]]
        result[j]=np.asarray(x,dtype=float)
    return result


def probability(model,x):
    if list(NAMES)!=model["features"]:raise ValueError("Incompatible seam feature order")
    z=np.clip((np.asarray(x)-model["mean"])/np.asarray(model["scale"]),-5.,5.)
    return expit(z@model["weights"]+model["bias"])

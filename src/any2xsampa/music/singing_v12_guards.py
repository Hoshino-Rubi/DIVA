"""Audio-only guards on frozen V11 decisions. Times are source seconds."""
from pathlib import Path
import json
import numpy as np
from scipy.special import softmax
from .singing_v10_seams import periodic
from ..adapters.singing_v11_seams import CFG as V11

CFG=json.loads((Path(__file__).resolve().parents[1]/"config/cantonese_singing_v12.json").read_text("utf8"))


def peak(x: np.ndarray, a: float, b: float) -> float:
    i,j=np.clip(np.ceil(np.array([a,b])*50-.5).astype(int),0,len(x))
    return float(np.max(x[i:j],initial=0.))


def note_evidence(rows,probe,frames,vocab,onsets,phrases):
    """Use calibrated CTC frame posteriors, never sum logits or count blank as speech."""
    p=softmax(frames[:,768:768+len(vocab)].astype(np.float32),axis=1)
    consonants=[i for k,i in vocab.items() if k[0] in V11["ctc_consonant_initials"]
                and k.rstrip("12345") not in V11["semivowel_vowel_labels"]]
    real=[i for k,i in vocab.items() if not k.startswith("<") and k not in ("|","??") and not k.isdigit()]
    cons=p[:,consonants].sum(1);speech=p[:,real].sum(1)
    words=[w for ph in phrases for w in ph["syllables"] if .035<=w["end"]-w["start"]<=2.4]
    out=[]
    for j,(r,(label,q)) in enumerate(zip(rows,onsets,strict=True)):
        a,b=r["start_sec"],r["end_sec"]
        wave=probe.vocal[round(16000*a):round(16000*b)]
        power=float(np.mean(wave.astype(float)**2))
        # Independent initial evidence must be inside the leading event or its own onset pool.
        prior=rows[j-1]["end_sec"] if j else 0.
        c=peak(cons,max(prior,a-.080),min(a+.075,b))
        local=peak(cons,a-.04,a+.06)
        pre=peak(cons,a-.12,a-.04)
        voice=peak(speech,a-.04,b+.04)
        has_asr=any(w["end"]>a-CFG["asr_guard_sec"] and w["start"]<b+CFG["asr_guard_sec"] for w in words)
        out.append(dict(onset=label,q=float(q),power_db=float(10*np.log10(max(1e-16,power))),
                        leading=bool((label and q>=CFG["lead_onset_conf"]) or c>=CFG["lead_ctc"]),
                        fresh_ctc=bool(local>=CFG["new_peak"] and local-pre>=CFG["peak_rise"]),
                        ctc_peak=local,ctc_speech=voice,asr=has_asr))
    return out


def stable(core):
    strength,mad,speed,_=core
    return bool(strength>0 and mad<=V11["stable_mad"] and speed<=V11["unstable_speed"])


def veto(left,right,cl,cr,el,er):
    if (el["onset"] and er["onset"] and el["onset"]!=er["onset"]
            and min(el["q"],er["q"])>=CFG["distinct_onset_conf"]):return "distinct_onsets"
    if er["fresh_ctc"]:return "new_ctc"
    if min(left["duration_sec"],right["duration_sec"])>CFG["long_sec"]+1e-8:return "both_long"
    if right["end_sec"]-left["start_sec"]>CFG["long_span"] and stable(cl) and stable(cr):return "stable_span"
    return None


def leading_action(left,ev,host_ev,dip):
    if host_ev["power_db"]-ev["power_db"]>=CFG["lead_drop_db"]:return "drop"
    if left["duration_sec"]<=CFG["ultrashort"] and not ev["leading"]:return "drop"
    if ev["leading"] and dip is not None and dip<CFG["continuous_dip"]:return "union"
    return "retain"  # Absence of evidence is not evidence for erasing this note.


def vocal_support(wave):
    """Reject deletion on periodicity OR a prominent mid-band resonance."""
    for i in range(0,len(wave)-479,160):
        x=wave[i:i+480].astype(float)
        if not np.any(x):continue
        if periodic(x,18,205)[1]>=CFG["periodic_support"]:return True
        spec=np.abs(np.fft.rfft(x*np.hanning(480),512))**2
        # Smooth over ~250 Hz; a broad F1/F2 peak is supporting evidence, not a phone label.
        bands=np.add.reduceat(spec[8:104],np.arange(0,96,8))
        if bands.max()>10**(CFG["formant_contrast_db"]/10)*max(np.median(bands),1e-16):return True
    return False


def isolated(rows,ev,probe):
    discard=[]
    for j in range(1,len(rows)-1):
        r=rows[j];a,b=r["start_sec"],r["end_sec"]
        if min(a-rows[j-1]["end_sec"],rows[j+1]["start_sec"]-b)<=CFG["standalone_gap"]:continue
        e=ev[j]
        if e["ctc_speech"]>=CFG["nonblank_prob"] or e["asr"]:continue
        voiced=[v["power_db"] for n,v in zip(rows,ev) if abs(n["start_sec"]-a)<=CFG["baseline_sec"]
                and (v["asr"] or v["ctc_speech"]>=.5)]
        if len(voiced)<3 or np.median(voiced)-e["power_db"]<CFG["bleed_drop_db"]:continue
        if not vocal_support(probe.vocal[round(a*16000):round(b*16000)]):
            discard.append(j)
    return set(discard)

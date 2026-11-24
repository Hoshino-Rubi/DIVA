"""V5 legal Cantonese joint decoding and acoustic-only melisma decisions."""
from __future__ import annotations

import numpy as np
import torch

from .singing_adapter import ONSETS, FINALS, SingingAdapter, note_pools
from any2xsampa.music.singing_v5_melody import CFG
from any2xsampa.pipeline.subphone_aligner import _periodicity


def junction(audio: np.ndarray, start: float) -> dict:
    """Contrast the boundary against its own voiced flanks (gain invariant)."""
    middle = round(start*16000)
    if middle < 1440 or middle+1440 > len(audio):
        return {"continuous":False}
    x = audio[middle-1440:middle+1440].astype(float)
    frames = np.lib.stride_tricks.sliding_window_view(x,320)[::80]
    power = np.mean(frames**2,axis=1)
    if power.max() <= np.finfo(np.float32).tiny:
        return {"continuous":False}
    flank = np.r_[0:6,27:33]
    inner = slice(7,26)
    fine = np.mean(np.lib.stride_tricks.sliding_window_view(x,128)[::64]**2,axis=1)
    dip = 10*np.log10(max(np.median(power[flank]),1e-24)/max(fine[11:32].min(),1e-24))
    spectra = abs(np.fft.rfft(frames*np.hanning(320)))
    # HF novelty relative to broadband energy in the two surrounding vowels.
    broad = np.median(np.linalg.norm(spectra[flank],axis=1))+1e-15
    flux = np.linalg.norm(np.maximum(np.diff(spectra[:,40:],axis=0),0),axis=1)/broad
    flux_ratio = float(flux[inner].max()/max(.018,np.quantile(flux[flank[:-1]],.9)))
    hf = float((spectra[inner,40:]**2).sum() / max((spectra[inner]**2).sum(),1e-24))
    # Coarse spectral bands suppress sensitivity to individual harmonics/vibrato.
    bands = np.add.reduceat(spectra**2,[0,6,12,20,32,48,72,105],axis=1)
    shapes = np.sqrt(bands/(bands.sum(axis=1,keepdims=True)+1e-24))
    shape = float(np.linalg.norm(shapes[:6].mean(0)-shapes[-6:].mean(0)))
    periodic = min(_periodicity(x[a:a+480],16000,np) for a in (720,1040,1360,1680))
    c = CFG["legato"]
    keep = dip<c["dip_db"] and flux_ratio<c["flux_ratio"] and periodic>c["min_periodicity"] and hf<c["max_hf_ratio"] and shape<c["max_shape_distance"]
    return dict(continuous=bool(keep),dip_db=float(dip),flux_ratio=flux_ratio,
                hf_ratio=hf,periodicity=float(periodic),shape_distance=shape)


def joint(x, y, mask, prior, calibration):
    onset_weight,prior_weight,checked_bias = calibration
    tails = x.new_tensor([float(f.endswith((" t"," k"))) for f in FINALS])
    score = (onset_weight*x.log_softmax(-1)[:,:,None]+y.log_softmax(-1)[:,None,:]
             +prior_weight*prior+checked_bias*tails)
    return score.masked_fill(~mask,-torch.inf).flatten(1)


def predict_v5(rows,frames,audio,state):
    from any2xsampa.exporters.miku_v4c import POLICY,require_legal
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SingingAdapter(frames.shape[1])
    model.load_state_dict(state["weights"])
    model.eval().to(device)
    mask = state["mask"].to(device)
    prior = state["pair_prior"].to(device)
    a,b,vot,ev = note_pools(rows,frames,audio)
    inputs = (a,b,vot)
    results = []
    with torch.inference_mode():
        for k in range(0,len(rows),64):
            x,y = model(*(torch.from_numpy(t[k:k+64]).to(device) for t in inputs))
            logits = joint(x,y,mask,prior,state["calibration"])
            probs,idx = logits.softmax(-1).max(1)
            for q,(win,p) in enumerate(zip(idx.tolist(),probs.tolist()),k):
                i,j = divmod(win,len(FINALS))
                s,f = ONSETS[i],FINALS[j]
                raw = " ".join(filter(None,(s,f)))
                gap = rows[q]["start_sec"]-rows[q-1]["end_sec"] if q else None
                stats = junction(audio,rows[q]["start_sec"]) if gap is not None and -1e-8<=gap<CFG["legato"]["gap_sec"]-1e-9 else {"continuous":False}
                if stats["continuous"]:
                    s = ""
                    f = CFG["nucleus"][results[-1]["final"].split()[0]]
                tokens = " ".join(filter(None,(s,f))).split()
                require_legal(tokens)
                base = " ".join(tokens[:-1] if tokens[-1] in ("t","k") else tokens)
                results.append(dict(**ev[q],onset=s,final=f,raw_syllable=raw,
                                    v4c_phonemes=tokens,lyric=POLICY["base_pinyin"][base],
                                    joint_confidence=p,continuation=stats["continuous"],junction=stats))
    return results

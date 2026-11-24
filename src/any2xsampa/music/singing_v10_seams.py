"""Source-clock GAME seam evidence. Never consult phones or a reference in the first pass."""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.fft import dct
SR = 16000
HANN = np.hanning(320)
# Triangular mel filters, excluding the log-energy coefficient in MFCC distances.
freq = np.arange(257)*SR/512
bins = 700*np.expm1(np.linspace(np.log1p(80/700), np.log1p(7800/700), 26))
MEL = np.maximum(0, np.minimum((freq-bins[:-2,None])/np.diff(bins)[:-1,None],
                               (bins[2:,None]-freq)/np.diff(bins)[1:,None]))


def periodic(frame: np.ndarray, low: int, high: int) -> tuple[float, float]:
    """Energy-normalized autocorrelation on sub-frame lags, with parabolic refinement."""
    x = frame-frame.mean()
    energy = np.r_[0., np.cumsum(x*x)]
    lags = np.arange(max(2,low), min(len(x)//2, high)+1)
    if not len(lags) or energy[-1] < 1e-20:
        return 0., 0.
    cor = np.correlate(x,x,"full")[len(x)-1+lags]
    cor /= np.sqrt(np.maximum(1e-30,energy[-1-lags]*(energy[-1]-energy[lags])))
    peaks = np.flatnonzero((cor[1:-1]>=cor[:-2]) & (cor[1:-1]>cor[2:]))+1
    # Boundary peaks do not establish periodic support for a pitch outside its band.
    if not len(peaks):
        return 0., 0.
    k = peaks[np.argmax(cor[peaks])]
    a,b,c = cor[k-1:k+2]
    delta = np.clip(.5*(a-c)/min(-1e-10,a-2*b+c),-.5,.5)
    return float(lags[k]+delta), float(np.clip(b,0,1))


class Seams:
    """One stem, one clock. Cache evidence for calibration and both merge passes."""
    def __init__(self, vocal: np.ndarray):
        self.vocal = vocal
        # Per-instance lifetime; never retain completed songs in global caches.
        self.at = lru_cache(maxsize=4096)(self._at)
        self.core = lru_cache(maxsize=4096)(self._core)

    def _at(self, t: float) -> dict:
        c = round(t*SR)
        x = self.vocal[max(0,c-640):c+640].astype(float)
        if len(x)!=1280 or np.max(abs(x)) < 1e-10:
            return {"valid":False}
        frames = np.lib.stride_tricks.sliding_window_view(x,320)[::80]
        power = np.mean(frames**2,axis=1)
        flank = min(np.median(power[:3]),np.median(power[-3:]))
        fine = np.mean(np.lib.stride_tricks.sliding_window_view(x,160)[::80]**2,axis=1)
        dip = 10*np.log10(max(flank,1e-25)/max(fine[3:-3].min(),1e-25))
        sp = abs(np.fft.rfft(frames*HANN,n=512))
        broad = max(1e-12,np.median(np.linalg.norm(sp,axis=1)))
        flux = np.max(np.linalg.norm(np.maximum(0,np.diff(sp[:,65:],axis=0)),axis=1))/broad
        mel = (sp**2)@MEL.T
        cep = dct(np.log(np.maximum(mel,1e-7*max(mel.max(),1e-25))),norm="ortho",axis=1)[:,1:13]
        # Distance over 10ms rather than sensitivity to single harmonic bins.
        distance = np.linalg.norm(cep[2:]-cep[:-2],axis=1).max()/np.sqrt(12)
        q = [periodic(x[i:i+480],16,220)[1] for i in range(0,801,80)]
        return dict(valid=True,dip_db=float(dip),flux=float(flux),mfcc_distance=float(distance),
                    periodicity=min(q[2:-2]),period_drop=max(0.,min(max(q[:3]),max(q[-3:]))-min(q[2:-2])))

    def _core(self, start: float, end: float, pitch: int) -> tuple:
        x = self.vocal[round(start*SR):round(end*SR)].astype(float)
        d = len(x)/SR
        if len(x)<480:
            return (0.,0.,0.,d)
        lag = SR/(440*2**((pitch-69)/12))
        f0, weight, times = [],[],[]
        for k in range(0,len(x)-479,160):
            f = x[k:k+480]
            l,q = periodic(f,int(lag/1.26)-1,int(np.ceil(lag*1.26))+1)
            if q>.6 and l>0:
                f0.append(-12*np.log2(l/lag))
                weight.append(q*np.sqrt(np.mean(f*f)))
                times.append(k/SR)
        if len(f0)<2 or sum(weight)<=0:
            return (0.,0.,0.,d)
        m = np.median(f0)
        speed = np.median(abs(np.diff(f0)/np.diff(times)))
        return (float(np.mean(weight)*d*len(weight)/max(1,(len(x)-480)//160+1)),
                float(np.median(abs(np.array(f0)-m))),float(speed),d)

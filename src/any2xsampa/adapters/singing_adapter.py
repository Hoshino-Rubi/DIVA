"""Note-pool attention heads for frozen singing features; checkpoint owns its training vocabulary."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from any2xsampa.pipeline.subphone_aligner import measure_subphone_boundaries

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config/cantonese_singing_v4.json").read_text(encoding="utf8"))
ONSETS = CONFIG["onsets"]
FINALS = CONFIG["nuclei"] + [v+" "+t for v in CONFIG["checked_nuclei"] for t in ("t","k")]


def split_target(tokens: list[str]) -> tuple[str, str] | None:
    if not tokens:
        return None
    initial = tokens[0] if tokens[0] in ONSETS else ""
    body = tokens[int(bool(initial)):]
    final = " ".join(body)
    if final not in FINALS:
        if len(body) == 2 and body[1] in ("n","m","N"):
            final = CONFIG["nasal_aliases"].get(body[0], "")
        elif len(body) == 2 and body[1] == "p":
            final = body[0]+" t"
        else:
            final = CONFIG["glide_aliases"].get(final, "")
    return (initial, final) if final in FINALS else None


def windows(rows: list[dict], duration: float):
    prev = 0.
    for row in rows:
        start, end = row["start_sec"], row["end_sec"]
        yield (max(0.,prev,start-.130), min(duration,start+.035),
               min(duration,start+.020), min(duration,end+.040))
        prev = end


def sample_grid(frames: np.ndarray, a: float, b: float, count: int) -> np.ndarray:
    centers = np.linspace(a,b,count,endpoint=False)+(b-a)/(2*count)
    # Use only frame centers inside the requested interval, or nearest center if sub-frame.
    lo = np.clip(int(np.ceil(a*50-.5)), 0, len(frames)-1)
    hi = np.clip(int(np.floor(b*50-.5)), lo, len(frames)-1)
    pos = np.clip(np.rint(centers*50-.5).astype(int), lo, hi)
    return frames[pos]


def note_pools(rows: list[dict], frames: np.ndarray, audio: np.ndarray):
    onset, final, vots, evidence = [], [], [], []
    for a,b,c,d in windows(rows, len(audio)/16000):
        onset.append(sample_grid(frames,a,b,8))
        final.append(sample_grid(frames,c,d,24))
        first = max(0,round((a-.020)*16000))
        last = min(len(audio),round((b+.030)*16000))
        bounds = measure_subphone_boundaries(audio[first:last],16000,a,b,onset=True,offset_sec=first/16000)
        vot = None
        if bounds.release_ms is not None and bounds.voicing_onset_ms is not None:
            vot = bounds.voicing_onset_ms - bounds.release_ms
        vots.append(0. if vot is None else float(np.clip((vot-25)/25,-1,2)))
        evidence.append(dict(onset_pool_sec=[a,b],final_pool_sec=[c,d],vot_ms=vot))
    return np.stack(onset), np.stack(final), np.asarray(vots,dtype=np.float32), evidence


class AttentivePool(nn.Module):
    def __init__(self, width: int, classes: int, steps: int):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(width,112), nn.LayerNorm(112), nn.GELU())
        self.attention = nn.Linear(112,2)
        self.position = nn.Parameter(torch.stack([torch.linspace(1,-1,steps),torch.linspace(-1,1,steps)],dim=1))
        self.output = nn.Sequential(nn.LayerNorm(224),nn.Dropout(.25),nn.Linear(224,classes))

    def pooled(self, frames):
        hidden = self.project(frames)
        weight = (self.attention(hidden) + self.position).softmax(1)
        return (weight.transpose(1,2) @ hidden).flatten(1)

    def forward(self, frames):
        return self.output(self.pooled(frames))


class SingingAdapter(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.onset = AttentivePool(width,len(ONSETS),8)
        self.final = AttentivePool(width,len(FINALS),24)
        self.register_buffer("mean",torch.zeros(width))
        self.register_buffer("scale",torch.ones(width))
        signs = [1 if s in CONFIG["aspirates"] else -1 if s in CONFIG["plain_stops"] else 0 for s in ONSETS]
        self.register_buffer("aspiration",torch.tensor(signs,dtype=torch.float32))

    def forward(self, a, b, vot):
        def norm(x):
            return ((x.float()-self.mean)/self.scale).clamp(-6,6)
        return self.onset(norm(a)) + .25*vot[:,None]*self.aspiration, self.final(norm(b))


def predict(rows: list[dict], frames: np.ndarray, audio: np.ndarray, checkpoint: Path) -> list[dict]:
    from any2xsampa.exporters.miku_v4c import is_legal, POLICY
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if (state["format"], state["onsets"], state["finals"]) != ("diva_singing_v4",ONSETS,FINALS):
        raise ValueError("Incompatible singing adapter")
    net = SingingAdapter(frames.shape[1])
    net.load_state_dict(state["weights"])
    net.eval().to(device)
    # Only training-song bases and structurally available tails can compete.
    bases = set(state["bases"])
    pairs = [(i,j) for i,s in enumerate(ONSETS) for j,f in enumerate(FINALS)
             if " ".join(filter(None,(s,f.split()[0]))) in bases and is_legal(" ".join(filter(None,(s,f))))]
    ii,jj = torch.tensor(pairs,device=device).T
    prior = state["pair_prior"].to(device)[ii,jj]
    a,b,vot,evidence = note_pools(rows,frames,audio)
    results = []
    with torch.inference_mode():
        for start in range(0,len(rows),64):
            x,y = net(*(torch.from_numpy(t[start:start+64]).to(device) for t in (a,b,vot)))
            logits = x[:,ii]+y[:,jj]+prior
            winners = logits.argmax(1).tolist()
            confidence = logits.softmax(1).amax(1).tolist()
            for idx,(win,prob) in enumerate(zip(winners,confidence),start):
                i,j = pairs[win]
                tokens = " ".join(filter(None,(ONSETS[i],FINALS[j]))).split()
                base = " ".join(tokens[:-1] if tokens[-1] in ("t","k") else tokens)
                results.append(dict(**evidence[idx], onset=ONSETS[i],final=FINALS[j],
                                    v4c_phonemes=tokens,lyric=POLICY["base_pinyin"][base],
                                    confidence=prob))
    return results

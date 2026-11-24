"""Detect short pitch jumps on either flank and against the local phrase median."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

CFG = json.loads((Path(__file__).resolve().parents[1]/"config/cantonese_singing_v5.json").read_text(encoding="utf8"))


def repair(notes: list[dict], scale: float) -> list[dict]:
    """Simultaneous decisions from the original contour, never cascade across a rest."""
    old = [n["midi"] for n in notes]
    events = []
    start = 0
    for stop in range(1,len(notes)+1):
        if stop < len(notes):
            prev = notes[stop-1]
            if notes[stop]["start_sec"]-(prev["start_sec"]+prev["duration_sec"]) <= CFG["phrase_gap_sec"]:
                continue
        for i in range(start,stop):
            row = notes[i]
            dur = row["duration_sec"]
            if dur >= CFG["short_sec"]-1e-9:
                continue
            neighbors = [j for j in (i-1,i+1) if start <= j < stop]
            if not neighbors:
                continue
            # Shift rather than shrink the window at a phrase edge.
            left = max(start,min(i-2,stop-5))
            context = list(range(left,min(stop,left+5)))
            mid = median(old[j] for j in context)
            flank = any(abs(old[i]-old[j]) > CFG["jump_semitones"] for j in neighbors)
            if not flank and abs(old[i]-mid) <= CFG["jump_semitones"]:
                continue
            # A true octave leap between two long notes cannot be within 7 of both
            # anchors. Prefer median agreement; never flip an already-correct anchor.
            candidates = neighbors
            if not flank and all(old[j]==old[i] for j in neighbors):
                candidates = [j for j in context if j!=i]
            def cost(j):
                q=old[j]
                violations=sum(abs(q-old[n])>CFG["jump_semitones"] for n in neighbors)
                return (violations + (abs(q-mid)>CFG["jump_semitones"]),
                        notes[j]["duration_sec"] < CFG["short_sec"],abs(q-mid),
                        -notes[j]["duration_sec"], abs(j-i))
            j = min(candidates,key=cost)
            notes[i]["midi"] = old[j]
            events.append(dict(index=i,start_sec=row["start_sec"],duration_sec=dur,old=old[i],
                               new=old[j],anchor=j,median=mid,flank=flank))
        start = stop
    return events

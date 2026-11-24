"""Time calibration using CTC vowel anchors and low-energy syllable recovery."""
from __future__ import annotations

import numpy as np


def emission_delay(segments: list[dict], notes: list[dict], nuclei: set[str]) -> float:
    if len(notes)<2:
        return 0.
    starts = np.array([n["start_sec"] for n in notes])
    differences = []
    for phone in segments:
        if phone["normalized_ipa"] not in nuclei:
            continue
        t = (phone["start_sec"]+phone["end_sec"])/2
        i = int(np.searchsorted(starts,t))
        nearest = min(starts[max(0,i-1):min(i+1,len(starts))],key=lambda a:abs(a-t))
        if abs(t-nearest)<=.12:
            differences.append(t-nearest)
    return float(np.median(differences)) if len(differences)>=5 else 0.


def adjust_emission_times(segments: list[dict], delay: float) -> list[dict]:
    """Only assignment coordinates move; source-clock landmarks stay in the originals."""
    return [dict(p,start_sec=p["start_sec"]-delay,end_sec=p["end_sec"]-delay) for p in segments]


def supported_centers(notes: list[dict], frames: list[dict]) -> set[int]:
    if not frames:
        return set()
    time = np.array([f["time_sec"] for f in frames])
    pitch = np.array([69+12*np.log2(f["hz"]/440) for f in frames])
    supported = set()
    for n in notes:
        a,b = np.searchsorted(time,[n["start_sec"],n["end_sec"]])
        if b-a>=2 and np.median(abs(pitch[a:b]-n["pitch"]))<=1.5:
            supported.add(n["index"])
    return supported

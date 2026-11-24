"""Local, gain-invariant acoustic landmarks on the original audio clock."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from any2xsampa.core.models import PhoneSegment, SubPhoneBoundaries


def _periodicity(frame: Any, rate: int, np: Any) -> float:
    centered = frame - np.mean(frame)
    energy = float(np.dot(centered, centered))
    if energy == 0:
        return 0.0
    spectrum = np.fft.rfft(centered, 2 * len(frame))
    corr = np.fft.irfft(np.abs(spectrum) ** 2)[:len(frame)]
    prefix = np.concatenate(([0.0], np.cumsum(centered ** 2)))
    lags = np.arange(max(1, rate // 850), min(len(frame) // 2, rate // 75) + 1)
    if not len(lags):
        return 0.0
    denominator = np.sqrt(prefix[len(frame) - lags] * (energy - prefix[lags]))
    return float(np.max(corr[lags] / np.maximum(denominator, energy * 1e-12)))


def measure_subphone_boundaries(
    samples: Any,
    rate: int,
    start_sec: float,
    end_sec: float,
    *,
    onset: bool = False,
    coda: bool = False,
    transition: bool = False,
    offset_sec: float = 0.0,
) -> SubPhoneBoundaries:
    """Estimate supported landmarks inside a phone, with milliseconds in source time."""
    import numpy as np

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    frame = max(4, round(rate * 0.010))
    hop = max(1, round(rate * 0.0025))
    centers = np.arange(frame // 2, len(values) - frame // 2, hop)
    if not len(centers):
        return SubPhoneBoundaries()
    times = centers / rate + offset_sec
    window = np.hanning(frame)
    frames = np.stack([values[i-frame//2:i-frame//2+frame] for i in centers])
    energy = np.mean(frames ** 2, axis=1)
    peak = float(np.max(energy))
    target = np.flatnonzero((times >= start_sec) & (times <= end_sec))
    if not len(target) or peak == 0:
        return SubPhoneBoundaries()
    local_floor = max(float(np.quantile(energy, 0.15)), peak * 1e-7)
    db = 10 * np.log10(np.maximum(energy, local_floor * 1e-7) / local_floor)
    magnitude = np.abs(np.fft.rfft(frames * window))
    freqs = np.fft.rfftfreq(frame, 1 / rate)
    hf = magnitude[:, freqs >= min(1800, rate * 0.25)]
    flux = np.maximum(np.diff(hf, axis=0, prepend=hf[:1]), 0).sum(axis=1)
    native_flux = max(float(np.quantile(flux, 0.55)), float(np.max(flux)) * 0.05)
    active = (db >= 6.0) | (energy >= peak * 0.25)
    voiced = np.zeros(len(centers), dtype=bool)
    periodic_frame = max(frame, round(rate * 0.024))
    for index in np.flatnonzero(active):
        first = max(0, centers[index] - periodic_frame // 2)
        last = min(len(values), first + periodic_frame)
        voiced[index] = _periodicity(values[first:last], rate, np) >= 0.82
    limits: dict[str, float] = {}
    if onset:
        search = target[times[target] <= start_sec + min(0.14, (end_sec-start_sec)*0.70)]
        for index in search:
            if index < 2 or not active[index]:
                continue
            if flux[index] <= native_flux or db[index] - db[index-2] < 4:
                continue
            limits["closure_end_ms"] = limits["release_ms"] = times[index] * 1000
            for later in target[target >= index]:
                if later + 2 < len(voiced) and voiced[later:later+3].all():
                    limits["voicing_onset_ms"] = times[later] * 1000
                    break
            break
    if coda:
        tail = target[times[target] >= start_sec + (end_sec-start_sec)*0.40]
        for index in reversed(tail):
            if index < 3 or index + 2 >= len(energy):
                continue
            preceding = float(np.max(energy[max(0, index-8):index]))
            if preceding > 0 and np.max(energy[index:index+3]) <= preceding * 0.125:
                if np.any(voiced[max(0, index-10):index]) and not np.any(voiced[index:index+3]):
                    limits["coda_closure_ms"] = times[index] * 1000
    if transition and len(target) >= 8:
        inner = target[len(target)//4:-len(target)//4]
        spectrum = magnitude / np.maximum(magnitude.sum(axis=1, keepdims=True), np.finfo(float).tiny)
        changes = np.abs(np.diff(spectrum, axis=0, prepend=spectrum[:1])).sum(axis=1)
        stable = inner[voiced[inner] & voiced[inner-1]]
        if len(stable):
            winner = int(stable[np.argmax(changes[stable])])
            if changes[winner] > max(0.12, float(np.median(changes[stable])) * 1.5):
                limits["vowel_transition_ms"] = times[winner] * 1000
    return SubPhoneBoundaries(**limits)


def align_subphones(path: str | Path, segments: list[PhoneSegment]) -> None:
    """Read small windows directly from the vocal audio and annotate stop/diphthong evidence."""
    import soundfile as sf

    from any2xsampa.core.io import load_json

    config = load_json(Path(__file__).resolve().parents[1] / "config" / "cantonese_subphones.json")
    with sf.SoundFile(path) as source:
        rate = source.samplerate
        for segment in segments:
            symbol = segment.normalized_ipa
            start, end = segment.time_range.start_sec, segment.time_range.end_sec
            onset = symbol in config["onsets"]
            coda = symbol in config["codas"]
            transition = symbol in config["transitions"]
            if not (onset or coda or transition):
                continue
            first = max(0, round((start - 0.035) * rate))
            last = min(source.frames, round((end + 0.035) * rate))
            if first >= last:
                continue
            source.seek(first)
            values = source.read(last - first, dtype="float32", always_2d=True)
            segment.subphone_boundaries = measure_subphone_boundaries(
                values, rate, start, end, onset=onset, coda=coda,
                transition=transition, offset_sec=first/rate,
            )

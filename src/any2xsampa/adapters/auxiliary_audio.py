"""Bounded derived audio with an explicit, sample-based original clock."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from any2xsampa.adapters.audio_provenance import describe_audio_file
from any2xsampa.training.limited_data_artifacts import finite_number, sha256_file


def read_stereo_context(path: Path, expected_hash: str, start: float, end: float, *, sample_rate: int = 44100) -> tuple[Any, dict]:
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf

    start = finite_number(start, "context start", minimum=0)
    end = finite_number(end, "context end", minimum=0)
    if not 0 < end - start <= 30.000001 or sample_rate not in {16000, 44100}:
        raise ValueError("auxiliary context duration/rate outside its bounds")
    source = describe_audio_file(path)
    if source["sha256"] != expected_hash or end > source["duration_sec"] + 1e-6:
        raise ValueError("context source hash or bounds mismatch")
    rate = source["sample_rate"]
    first, last = round(start * rate), round(end * rate)
    with sf.SoundFile(path) as stream:
        if stream.channels not in {1, 2}:
            raise ValueError("separator supports mono/stereo only")
        stream.seek(first)
        audio = stream.read(last - first, dtype="float32", always_2d=True)
    if len(audio) != last - first or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("truncated or non-finite source context")
    channel_policy = "preserve_stereo"
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
        channel_policy = "duplicate_mono_for_stereo_separator"
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor, axis=0).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("non-finite resampled context")
    return np.ascontiguousarray(audio.T), {
        "source": source, "source_start_sample": first, "source_end_sample_exclusive": last,
        "requested_start_sec": start, "requested_end_sec": end,
        "effective_original_start_sec": first / rate, "effective_original_end_sec": last / rate,
        "derived_sample_rate": sample_rate, "derived_samples": int(audio.shape[0]),
        "derived_duration_sec": len(audio) / sample_rate, "channel_policy": channel_policy,
        "resample_policy": "scipy_resample_poly_v1", "sample_rounding": "python_round_matches_legacy_producer",
        "clock_equation": "original_sec = effective_original_start_sec + derived_sample / derived_sample_rate",
        "no_time_stretch_or_compensation": True, "full_source_audio_copied": False,
    }


def write_stereo_float(path: Path, audio: Any, sample_rate: int) -> dict:
    import numpy as np
    import soundfile as sf

    if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] < 1 or not np.isfinite(audio).all():
        raise ValueError("derived stereo audio must be finite and nonempty")
    if sample_rate != 44100 or audio.shape[1] > sample_rate * 30 + 2:
        raise ValueError("derived audio exceeds its fixed rate/duration budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        sf.write(stream, audio.T, sample_rate, format="WAV", subtype="FLOAT")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "sample_rate": sample_rate,
            "samples": int(audio.shape[1]), "channels": 2, "subtype": "FLOAT",
            "peak_abs": float(np.max(np.abs(audio))), "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))}


def vocal_energy(audio: Any, sample_rate: int, policy: Mapping[str, Any]) -> dict:
    import numpy as np

    values = np.asarray(audio)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("invalid vocal waveform for energy evidence")
    frame, hop = round(sample_rate * policy["energy_frame_sec"]), round(sample_rate * policy["energy_hop_sec"])
    if min(frame, hop) < 1:
        raise ValueError("energy geometry must be positive")
    power = np.mean(values.astype(np.float64) ** 2, axis=0)
    sums = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, max(0, len(power) - frame + 1), hop, dtype=int)
    rms = np.sqrt(np.maximum(0, (sums[starts + frame] - sums[starts]) / frame))
    peak = float(rms.max()) if rms.size else 0.0
    relative = 20 * np.log10(np.maximum(rms, 1e-12) / max(peak, 1e-12))
    return {"frames": [{"start_sec": int(first) / sample_rate, "end_sec": int(first + frame) / sample_rate,
                        "rms": float(value), "relative_peak_db": float(db)} for first, value, db in zip(starts, rms, relative)],
            "peak_rms": peak, "has_signal": peak > 1e-8, "phone_silence_truth": False}

"""Optional audio-only YIN period diagnostics, never phonetic or voicing truth."""
from __future__ import annotations

from array import array
from copy import deepcopy
import os
from typing import Any

PARAMETERS = {
    "frame_seconds": 0.05, "hop_seconds": 0.01, "center": False,
    "fmin_hz": 65, "fmax_hz": 1600, "trough_threshold": 0.1,
    "active_frame_minimum_rms_FS": 1e-4,
    "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
    "maximum_window_seconds": 20,
    "PCM_unit_conversion": "signed_int16_counts/32768; no fitted gain",
}


def _dependencies():
    # Only this explicitly requested diagnostic uses optional installed packages.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    import numpy as np
    import librosa
    return np, librosa.yin


def analyze_period(pcm: array, sample_rate: int) -> dict[str, Any]:
    """Observe complete frames without target pitch, IPA, references or padding.

    Short trailing samples are reported, not padded into a new period frame.
    The separate physical audit still measures every sample in the window.
    """
    if not isinstance(pcm, array) or pcm.typecode != "h" or pcm.itemsize != 2 or not pcm:
        raise ValueError("Period diagnostic requires nonempty signed PCM16 counts")
    if type(sample_rate) is not int or sample_rate <= 3200:
        raise ValueError("Period diagnostic requires sample rate above 3200 Hz")
    length, hop = round(sample_rate * 0.05), round(sample_rate * 0.01)
    frame_count = 1 + (len(pcm) - length) // hop if len(pcm) >= length else 0
    result: dict[str, Any] = {
        "f0_status": "unknown", "f0_quantiles_hz": None,
        "f0_estimate_frames": 0, "parameters": deepcopy(PARAMETERS),
        "frame_samples": length, "hop_samples": hop, "frame_count": frame_count,
        "unframed_tail_samples": len(pcm) - ((frame_count - 1) * hop + length)
        if frame_count else len(pcm),
        "pitch_truth": None, "voicing_truth": None, "pronunciation_correct": None,
        "uncertainty": "YIN periods may have octave/aperiodicity errors; they do not establish voicing or phone identity",
    }
    if not any(pcm):
        return {**result, "f0_status": "unknown_digital_silence"}
    if not frame_count:
        return {**result, "f0_status": "unknown_no_complete_frames"}
    if len(pcm) > 20 * sample_rate:
        return {**result, "f0_status": "unknown_window_exceeds_20_seconds"}
    try:
        np, yin = _dependencies()
        samples = np.asarray(pcm, dtype=np.int16).astype(np.float64) / 32768.0
        frames = np.lib.stride_tricks.sliding_window_view(samples, length)[::hop]
        active = np.sqrt(np.mean(frames * frames, axis=1)) >= 1e-4
        result["active_frames"] = int(active.sum())
        if not active.any():
            return {**result, "f0_status": "unknown_no_active_complete_frames"}
        estimates = np.asarray(yin(samples, sr=sample_rate, fmin=65, fmax=1600,
                                  frame_length=length, hop_length=hop,
                                  center=False, trough_threshold=0.1))
        if estimates.shape != active.shape:
            raise ValueError("YIN frame count differs from complete source frames")
        selected = estimates[active & np.isfinite(estimates) & (estimates > 0)]
        result["f0_estimate_frames"] = int(selected.size)
        if selected.size:
            result.update(f0_status="period_estimate_unqualified",
                          f0_quantiles_hz=dict(zip(("p05", "p25", "p50", "p75", "p95"),
                              map(float, np.quantile(selected, PARAMETERS["quantiles"])))))
        return result
    except ImportError as exc:
        return {**result, "f0_status": "unavailable", "f0_error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {**result, "f0_status": "unknown_estimation_error",
                "f0_error": f"{type(exc).__name__}: {exc}"}

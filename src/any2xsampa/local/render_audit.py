"""Read-only, whole-window PCM evidence; never a per-phone pronunciation test."""

from __future__ import annotations

import array
from collections.abc import Callable
from copy import deepcopy
import hashlib
import io
import math
from pathlib import Path
import sys
import wave


MAX_WAV_BYTES = 128 * 1024 * 1024
PCM_SCALE = 32768
CLIP_THRESHOLD = 0.999
ACTIVITY_THRESHOLD = 1e-4
FRAME_MS = 20
MIN_NOTE_MS = 40
MIN_ACTIVE_FRACTION = 0.5
ROLES = ("voiced_nucleus", "unvoiced_or_transient", "glide_or_nasal", "unknown")


def _validate_schedule(schedule: dict) -> None:
    if not isinstance(schedule, dict):
        raise ValueError("schedule must be a dict")
    if schedule.get("schema_version") != "render_audit_schedule_v1":
        raise ValueError("schedule schema_version must be render_audit_schedule_v1")
    if schedule.get("clock") != "rendered_wave_samples":
        raise ValueError("schedule clock must be rendered_wave_samples")
    rate = schedule.get("sample_rate")
    if type(rate) is not int or rate <= 0:
        raise ValueError("schedule sample_rate must be a positive integer, not bool")
    windows = schedule.get("windows")
    if not isinstance(windows, list):
        raise ValueError("schedule windows must be a list")
    seen: set[str] = set()
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("each schedule window must be a dict")
        identifier = window.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("window id must be a nonempty string")
        if identifier in seen:
            raise ValueError(f"duplicate window id: {identifier}")
        seen.add(identifier)
        if window.get("kind") not in ("note", "rest"):
            raise ValueError(f"window {identifier}: kind must be note or rest")
        start, end = window.get("start_sample"), window.get("end_sample")
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"window {identifier}: sample bounds must be integers, not bool")
        if start < 0 or end <= start:
            raise ValueError(f"window {identifier}: invalid half-open sample bounds")
        for name in ("raw_ipa", "export_tokens"):
            value = window.get(name)
            if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
                raise ValueError(f"window {identifier}: {name} must be a list of strings")
        for name in ("placeholder", "approximate"):
            if type(window.get(name)) is not bool:
                raise ValueError(f"window {identifier}: {name} must be bool")
        if window.get("role") not in ROLES:
            raise ValueError(f"window {identifier}: invalid role")
        if "pitch_midi" in window and (type(window["pitch_midi"]) is not int
                                       or not 0 <= window["pitch_midi"] <= 127):
            raise ValueError(f"window {identifier}: pitch_midi must be an integer from 0 to 127")
    ordered = sorted(windows, key=lambda w: w["start_sample"])
    for previous, current in zip(ordered, ordered[1:]):
        if current["start_sample"] < previous["end_sample"]:
            raise ValueError(f"overlapping windows: {previous['id']}, {current['id']}")


def _read_wave(path: Path) -> tuple[array.array, int, bytes]:
    # One bounded filesystem read; parsing and SHA use the very same bytes.
    with path.open("rb") as source:
        contents = source.read(MAX_WAV_BYTES + 1)
    if len(contents) > MAX_WAV_BYTES:
        raise ValueError("WAV exceeds the 128 MiB file limit")
    try:
        with wave.open(io.BytesIO(contents), "rb") as source:
            if (source.getnchannels() != 1 or source.getsampwidth() != 2
                    or source.getcomptype() != "NONE"):
                raise ValueError("WAV must be mono, uncompressed PCM16")
            rate, count = source.getframerate(), source.getnframes()
            if rate <= 0:
                raise ValueError("WAV sample_rate must be positive")
            # The extra requested frame exposes odd/truncated data, rather than
            # silently dropping a byte after the declared complete samples.
            pcm = source.readframes(count + 1)
            if len(pcm) != count * 2:
                raise ValueError("WAV has truncated or incomplete PCM16 samples")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"invalid mono PCM16 WAV: {exc}") from exc
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples, rate, contents


def _stats(samples: memoryview) -> dict:
    count = len(samples)
    square_sum = peak = clipped = zero = 0
    clip_count_threshold = math.ceil(CLIP_THRESHOLD * PCM_SCALE)
    for sample in samples:
        magnitude = abs(sample)
        square_sum += sample * sample
        peak = max(peak, magnitude)
        clipped += magnitude >= clip_count_threshold
        zero += sample == 0
    return {
        "sample_count": count,
        "rms": math.sqrt(square_sum / count) / PCM_SCALE,
        "peak": peak / PCM_SCALE,
        "clipped_samples": clipped,
        "clip_fraction": clipped / count,
        "digital_zero": zero == count,
        "digital_zero_samples": zero,
        "digital_zero_fraction": zero / count,
    }


def _levels(samples: memoryview, rate: int) -> dict:
    frame_samples = max(1, rate * FRAME_MS // 1000)
    count = len(samples)
    frame_count = (count + frame_samples - 1) // frame_samples
    active_frames = 0
    last_frame: dict = {}
    for start in range(0, count, frame_samples):
        last_frame = _stats(samples[start:start + frame_samples])
        active_frames += last_frame["rms"] >= ACTIVITY_THRESHOLD
    return {
        **_stats(samples),
        "activity": {
            "frame_samples": frame_samples,
            "frame_count": frame_count,
            "active_frames": active_frames,
            "active_fraction": active_frames / frame_count,
            "last_frame_samples": last_frame["sample_count"],
        },
        "edges": {
            "first_20ms": _stats(samples[:frame_samples]),
            "last_20ms": _stats(samples[max(0, count - frame_samples):]),
            "last_activity_frame": last_frame,
        },
    }


def _physical_status(window: dict, levels: dict, rate: int) -> tuple[str, list[str]]:
    if window["placeholder"]:
        return "unknown", ["placeholder_request"]
    if window["kind"] == "rest":
        return "unknown", ["rest_digital_zero_is_reported_separately"]
    if window["role"] == "unknown":
        return "unknown", ["unknown_role"]
    if window["role"] == "unvoiced_or_transient":
        return ("energy_available", ["energy_only_no_phone_judgement"]) \
            if levels["activity"]["active_frames"] else (
                "unknown", ["no_transient_energy_observed"])
    if levels["sample_count"] * 1000 <= rate * MIN_NOTE_MS:
        return "unknown", ["duration_not_over_40ms"]
    reasons = []
    if levels["rms"] < ACTIVITY_THRESHOLD:
        reasons.append("rms_below_threshold")
    if levels["activity"]["active_fraction"] < MIN_ACTIVE_FRACTION:
        reasons.append("active_fraction_below_threshold")
    if levels["clipped_samples"]:
        reasons.append("clipped_samples")
    return ("level_fail", reasons) if reasons else ("level_pass", [])


def _period_observation(
    samples: memoryview, rate: int, analyzer: Callable[[array.array, int], dict]
) -> dict:
    try:
        # Isolate analyzer writes from all other windows and physical evidence.
        observation = analyzer(array.array("h", samples), rate)
        if not isinstance(observation, dict):
            raise ValueError("period_analyzer must return a dict")
        return deepcopy(observation)
    except Exception as exc:
        return {"f0_status": "unknown", "f0_error": f"{type(exc).__name__}: {exc}"}


def _audit_row(
    window: dict, samples: memoryview, rate: int,
    analyzer: Callable[[array.array, int], dict] | None,
) -> dict:
    start, end = window["start_sample"], window["end_sample"]
    available = max(0, min(end, len(samples)) - start)
    missing = end > len(samples)
    row = {
        "request": deepcopy(window),
        "id": window["id"], "kind": window["kind"],
        "start_sample": start, "end_sample": end,
        "missing": missing,
        "missing_samples": end - start - available,
        "available_samples": available,
        "physical_emission_status": "unknown",
        "reasons": ["missing_window"] if missing else [],
        "levels": None,
        "rest_digital_zero": None,
        "strict_zero": None,
        "phone_emission": "unknown",
        "pronunciation_correct": None,
        "period_observation": {"f0_status": "not_requested"},
    }
    if missing:
        return row
    whole_window = samples[start:end]
    levels = _levels(whole_window, rate)
    row["levels"] = levels
    row["physical_emission_status"], row["reasons"] = _physical_status(window, levels, rate)
    if window["kind"] == "rest":
        row["rest_digital_zero"] = levels["digital_zero"]
        row["strict_zero"] = levels["digital_zero"]
    elif analyzer is not None:
        row["period_observation"] = _period_observation(whole_window, rate, analyzer)
    return row


def _summary(rows: list[dict]) -> dict:
    notes = [row for row in rows if row["kind"] == "note"]
    rests = [row for row in rows if row["kind"] == "rest"]
    return {
        "windows": len(rows), "notes": len(notes), "rests": len(rests),
        "missing_windows": sum(row["missing"] for row in rows),
        "missing_notes": sum(row["missing"] for row in notes),
        "missing_rests": sum(row["missing"] for row in rests),
        "placeholder_notes": sum(row["request"]["placeholder"] for row in notes),
        "note_status_counts": {
            status: sum(row["physical_emission_status"] == status for row in notes)
            for status in ("level_pass", "level_fail", "energy_available", "unknown")
        },
        "digital_zero_rests": sum(row["rest_digital_zero"] is True for row in rests),
        "nonzero_rests": sum(row["rest_digital_zero"] is False for row in rests),
        "unknown_rests": sum(row["rest_digital_zero"] is None for row in rests),
        "clipped_windows": sum(bool(row["levels"] and row["levels"]["clipped_samples"])
                               for row in rows),
        "period_error_notes": sum(bool(row["period_observation"].get("f0_error"))
                                  for row in notes),
    }


def audit_wave(
    wave_path: Path, schedule: dict, *,
    period_analyzer: Callable[[array.array, int], dict] | None = None,
) -> dict:
    """Audit complete half-open windows at the original rate, preserving order/N.

    Each row contains an independent ``request`` copy and immutable window
    coordinates. ``physical_emission_status`` is level_pass, level_fail,
    energy_available or unknown. Rest digital silence is a separate exact bool.
    Missing windows have no partial measurements. Placeholder requests remain
    unknown, with fallback levels still available. Period observations are
    copied verbatim; an analyzer error leaves all physical evidence intact.
    """
    _validate_schedule(schedule)
    if period_analyzer is not None and not callable(period_analyzer):
        raise ValueError("period_analyzer must be callable or None")
    samples, rate, contents = _read_wave(Path(wave_path))
    if schedule["sample_rate"] != rate:
        raise ValueError("schedule sample_rate does not match the rendered WAV clock")
    rows = [_audit_row(window, memoryview(samples), rate, period_analyzer)
            for window in schedule["windows"]]
    return {
        "schema_version": "render_audit_v1",
        "clock": "rendered_wave_samples",
        "sample_rate": rate, "samples": len(samples), "wav_bytes": len(contents),
        "wav_sha256": hashlib.sha256(contents).hexdigest(),
        "parameters": {
            "max_wav_bytes": MAX_WAV_BYTES,
            "pcm_scale": PCM_SCALE, "clip_threshold": CLIP_THRESHOLD,
            "activity_frame_ms": FRAME_MS,
            "activity_frame_samples": max(1, rate * FRAME_MS // 1000),
            "activity_frame_rounding": "floor_with_minimum_one_sample",
            "activity_threshold_rms": ACTIVITY_THRESHOLD,
            "activity_fraction_basis": "frames_including_partial_tail",
            "minimum_note_duration_ms_exclusive": MIN_NOTE_MS,
            "level_rms_threshold": ACTIVITY_THRESHOLD,
            "level_active_fraction_threshold": MIN_ACTIVE_FRACTION,
            "level_requires_no_clipping": True,
            "level_eligible_roles": ["voiced_nucleus", "glide_or_nasal"],
        },
        "scope": (
            "Whole-note/window physical PCM evidence only, without per-phone localization. "
            "A level pass does not establish voicing, phone emission or correct pronunciation. "
            "Placeholder requests remain unknown; their levels describe fallback audio only. "
            "Nonzero rest samples are not a judgement of audible noise. "
            "Period analyzers receive only copied PCM16 samples and the original sample rate; "
            "no target pitch, labels or references are supplied or compared."
        ),
        "rows": rows, "summary": _summary(rows),
    }

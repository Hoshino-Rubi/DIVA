"""Independent audio/score clock checks; no recognizer logits enter fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from any2xsampa.training.limited_data_artifacts import finite_number, sha256_object, verify_file


def _rank(scores: list[dict], separation: float) -> dict[str, Any]:
    if not scores:
        raise ValueError("empty clock score curve")
    ordered = sorted(scores, key=lambda row: (-row["similarity"], abs(row["offset_sec"]), row["offset_sec"]))
    best = ordered[0]
    runner = next((r for r in ordered if abs(r["offset_sec"] - best["offset_sec"]) >= separation), None)
    return {"best": best, "separated_runner_up": runner,
            "margin": best["similarity"] - runner["similarity"] if runner else None,
            "curve": sorted(scores, key=lambda row: row["offset_sec"])}


def common_support_offset(original: Any, rendered: Any, clips: Sequence[Mapping[str, Any]], *, factor: int, policy: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    x, y = np.asarray(original), np.asarray(rendered)
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("invalid clock pooling factor")
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or min(x.shape[1], y.shape[1]) < 4:
        raise ValueError("incompatible clock descriptors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite clock descriptors")
    x, y = x[:, ::factor], y[:, ::factor]
    x = x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=0, keepdims=True), 1e-12)
    step = factor * policy["descriptor_hop_samples"] / policy["descriptor_sample_rate"]
    maximum = int(policy["maximum_offset_sec"] / step)
    support = np.arange(maximum, min(x.shape[1], y.shape[1] - maximum), dtype=np.int64)
    for clip in clips:
        start = finite_number(clip["start_sec"], "original clip start", minimum=0)
        end = finite_number(clip["end_sec"], "original clip end", minimum=0)
        if end <= start:
            raise ValueError("invalid original clip interval")
        pad = policy["fit_clip_exclusion_padding_sec"]
        support = support[(support * step < start - pad) | (support * step >= end + pad)]
    if len(support) < policy["minimum_common_fit_frames"]:
        raise ValueError("insufficient common fit support outside fixed original clips")
    if (2 * maximum + 1) * len(support) > policy["maximum_frame_lag_products"]:
        raise ValueError("clock lag search exceeds resource limit")
    scores = []
    query = x[:, support]
    for lag in range(-maximum, maximum + 1):
        scores.append({"lag_frames": lag, "offset_sec": lag * step,
                       "similarity": float(np.sum(query * y[:, support + lag], axis=0).mean()),
                       "fit_frames": len(support)})
    ranked = _rank(scores, policy["runner_up_separation_sec"])
    best_lag = ranked["best"]["lag_frames"]
    validations = []
    radius = round(policy["local_validation_radius_sec"] / step)
    for clip in clips:
        lo = int(np.ceil(clip["start_sec"] / step))
        hi = int(np.ceil(clip["end_sec"] / step))
        if lo < 0 or hi > x.shape[1] or hi <= lo:
            raise ValueError("validation clip exceeds source descriptor")
        local = []
        for lag in range(best_lag - radius, best_lag + radius + 1):
            if lo + lag < 0 or hi + lag > y.shape[1]:
                continue
            local.append({"offset_sec": lag * step,
                          "similarity": float(np.sum(x[:, lo:hi] * y[:, lo + lag:hi + lag], axis=0).mean())})
        best = _rank(local, policy["runner_up_separation_sec"])["best"] if local else None
        validations.append({"example_id": clip["example_id"], "best": best,
                            "deviation_from_fit_sec": abs(best["offset_sec"] - ranked["best"]["offset_sec"]) if best else None,
                            "used_to_refit_offset": False})
    return {"factor": factor, "frame_seconds": step, **ranked,
            "common_support_frames": len(support), "support_original_frame_indices": support.tolist(),
            "support_sha256": sha256_object(support.tolist()), "fit_support_identical_for_all_lags": True,
            "validation": validations, "used_phone_predictions": False,
            "strict_waveform_context_independence_claimed": False}


def validate_offset_probes(probes: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    if [row["factor"] for row in probes] != policy["factors"]:
        raise ValueError("offset probe resolutions differ from locked policy")
    offsets = [finite_number(row["best"]["offset_sec"], "fitted offset") for row in probes]
    differences = max(offsets) - min(offsets)
    reasons = []
    if differences > policy["maximum_resolution_difference_sec"]:
        reasons.append("independent_resolution_offset_disagreement")
    if any(row["margin"] is None or finite_number(row["margin"], "similarity margin") < policy["minimum_separated_similarity_margin"] for row in probes):
        reasons.append("nonlocal_offset_ambiguity")
    if any(v["deviation_from_fit_sec"] is None or finite_number(v["deviation_from_fit_sec"], "validation offset deviation", minimum=0) > policy["maximum_local_offset_deviation_sec"] for row in probes for v in row["validation"]):
        reasons.append("constant_offset_not_stable_across_fixed_clips")
    observations = [(finite_number(row["best"]["offset_sec"], "fitted offset"), finite_number(row["frame_seconds"], "frame quantum", minimum=1e-9)) for row in probes]
    observations += [(finite_number(v["best"]["offset_sec"], "validation offset"), finite_number(row["frame_seconds"], "frame quantum", minimum=1e-9)) for row in probes for v in row["validation"] if v["best"]]
    low = min(value - quantum / 2 for value, quantum in observations)
    high = max(value + quantum / 2 for value, quantum in observations)
    return {"passed": not reasons, "failure_reasons": reasons, "point_offset_sec": offsets[0],
            "resolution_difference_sec": differences, "observed_offset_envelope_sec": [low, high],
            "envelope_is_statistical_confidence_interval": False, "score_based_offset_selection": False}


def rendered_activity(audio: Mapping[str, Any], *, step_sec: float) -> tuple[Any, float]:
    import numpy as np
    import soundfile as sf

    path = Path(audio["path"])
    verify_file(path, audio["sha256"])
    step_sec = finite_number(step_sec, "activity step", minimum=1e-4)
    with sf.SoundFile(path) as stream:
        rate = stream.samplerate
        if rate != audio["sample_rate"] or len(stream) != audio["frames"] or stream.channels != audio["channels"]:
            raise ValueError("rendered activity source metadata changed")
        hop = round(step_sec * rate)
        values = []
        while True:
            block = stream.read(hop * 256, dtype="float32", always_2d=True)
            if not len(block):
                break
            if not np.isfinite(block).all():
                raise ValueError("non-finite rendered waveform")
            mono = block.mean(axis=1).astype(np.float64)
            full = len(mono) // hop
            if full:
                values.extend(np.sqrt(np.mean(mono[:full * hop].reshape(full, hop) ** 2, axis=1)).tolist())
            if len(mono) % hop:
                values.append(float(np.sqrt(np.mean(mono[full * hop:] ** 2))))
    rms = np.asarray(values, dtype=np.float64)
    if not rms.size or rms.max() <= 0:
        raise ValueError("silent rendered waveform cannot validate a score clock")
    return rms, hop / rate


def score_activity_offset(rms: Any, notes: Sequence[Mapping[str, Any]], step: float, policy: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(rms, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or values.max() <= 0:
        raise ValueError("invalid rendered activity")
    step = finite_number(step, "activity step", minimum=1e-6)
    if np.any(values < 0):
        raise ValueError("negative rendered RMS")
    centers = (np.arange(len(values)) + .5) * step
    score = np.zeros(len(values), dtype=bool)
    for note in notes:
        lo = finite_number(note["start_sec"], "score note start", minimum=0)
        hi = finite_number(note["end_sec"], "score note end", minimum=0)
        if hi <= lo:
            raise ValueError("invalid score note interval")
        score |= (centers >= lo) & (centers < hi)
    maximum = round(policy["maximum_offset_sec"] / step)
    support = np.arange(maximum, len(values) - maximum)
    if len(support) < 10:
        raise ValueError("score clock activity has insufficient common support")
    query = score[support].astype(np.float64)
    centered = query - query.mean()
    query_norm = float(np.linalg.norm(centered))
    if query_norm == 0:
        raise ValueError("score activity needs both notes and rests to identify an offset")
    fits = []
    for ratio in policy["activity_peak_ratios"]:
        ratio = finite_number(ratio, "activity threshold", minimum=1e-6)
        if ratio >= 1:
            raise ValueError("activity threshold must be below the peak")
        actual = values >= values.max() * ratio
        curve = []
        for lag in range(-maximum, maximum + 1):
            target = actual[support + lag].astype(np.float64)
            target -= target.mean()
            denominator = query_norm * np.linalg.norm(target)
            similarity = float(np.dot(centered, target) / denominator) if denominator else 0.0
            curve.append({"offset_sec": lag * step, "similarity": similarity, "fit_frames": len(support)})
        ranked = _rank(curve, 1.0)
        fits.append({"peak_ratio": ratio, **ranked})
    offsets = [f["best"]["offset_sec"] for f in fits]
    reasons = []
    if any(f["best"]["similarity"] < policy["minimum_mcc"] for f in fits):
        reasons.append("rendered_activity_does_not_support_score_clock")
    if max(offsets) - min(offsets) > policy["maximum_threshold_offset_spread_sec"]:
        reasons.append("rendered_score_offset_threshold_instability")
    return {"schema_version": "rendered_score_activity_clock_v1", "equation": "rendered = score_absolute + offset",
            "fits": fits, "point_offset_sec": offsets[0], "passed": not reasons, "failure_reasons": reasons,
            "observed_offset_envelope_sec": [min(offsets) - step/2, max(offsets) + step/2],
            "descriptor_step_sec": step, "score_pitch_or_phones_used": False,
            "phone_boundaries_validated": False, "human_reviewed": False}

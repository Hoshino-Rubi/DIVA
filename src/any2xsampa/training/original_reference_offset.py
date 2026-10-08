"""Audio-only clock-offset diagnostics fitted outside frozen evaluation clips."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from any2xsampa.training.limited_data_artifacts import finite_number


def offset_probe(original: Any, rendered: Any, clips: Sequence[Mapping[str, Any]], *, factor: int, frame_seconds: float, policy: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    x, y = np.asarray(original), np.asarray(rendered)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or min(x.shape[1], y.shape[1]) < 4:
        raise ValueError("invalid offset descriptor dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite offset descriptor")
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("invalid offset pooling factor")
    frame_seconds = finite_number(frame_seconds, "descriptor frame duration", minimum=1e-9)
    x, y = x[:, ::factor], y[:, ::factor]
    step = frame_seconds * factor
    x = x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=0, keepdims=True), 1e-12)
    times = np.arange(x.shape[1]) * step
    fit_mask = np.ones(x.shape[1], dtype=bool)
    for clip in clips:
        start = finite_number(clip["start_sec"], "validation clip start", minimum=0)
        end = finite_number(clip["end_sec"], "validation clip end", minimum=0)
        if end <= start:
            raise ValueError("invalid validation clip interval")
        fit_mask &= ~((times >= start) & (times < end))
    fit_count = int(fit_mask.sum())
    if fit_count < 4:
        raise ValueError("no independent fit frames outside evaluation clips")
    max_lag = int(policy["maximum_offset_sec"] / step)
    if (2 * max_lag + 1) * x.shape[1] > policy["maximum_lag_frame_products"]:
        raise ValueError("offset search exceeds the resource budget")
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        lo, hi = max(0, -lag), min(x.shape[1], y.shape[1] - lag)
        if hi <= lo:
            continue
        mask = fit_mask[lo:hi]
        count = int(mask.sum())
        if count < fit_count * policy["minimum_fit_overlap_fraction"]:
            continue
        similarities = np.sum(x[:, lo:hi] * y[:, lo + lag:hi + lag], axis=0)
        scores.append({"lag_frames": lag, "offset_sec": lag * step, "cosine_similarity": float(similarities[mask].mean()), "fit_frames": count})
    scores.sort(key=lambda row: (-row["cosine_similarity"], abs(row["lag_frames"]), row["lag_frames"]))
    if not scores:
        raise ValueError("no sufficiently covered offset candidate")
    best = scores[0]
    alternatives = [r for r in scores[1:] if abs(r["offset_sec"] - best["offset_sec"]) >= policy["competitor_separation_sec"]]
    runner = alternatives[0] if alternatives else None
    validations = []
    radius = round(policy["validation_local_radius_sec"] / step)
    for clip in clips:
        lo, hi = np.searchsorted(times, [clip["start_sec"], clip["end_sec"]])
        local = []
        for lag in range(best["lag_frames"] - radius, best["lag_frames"] + radius + 1):
            if lo + lag < 0 or hi + lag > y.shape[1] or hi <= lo:
                continue
            local.append({"offset_sec": lag * step, "cosine_similarity": float(np.sum(x[:, lo:hi] * y[:, lo + lag:hi + lag], axis=0).mean())})
        local.sort(key=lambda r: (-r["cosine_similarity"], abs(r["offset_sec"] - best["offset_sec"])))
        validations.append({"example_id": clip["example_id"], "local_best": local[0] if local else None,
                            "fixed_fit_offset_sec": best["offset_sec"], "fit_used_this_clip": False})
    return {"factor": factor, "best": best, "separated_runner_up": runner,
            "absolute_similarity_margin": best["cosine_similarity"] - runner["cosine_similarity"] if runner else None,
            "available_fit_frames": fit_count, "excluded_validation_frames": int((~fit_mask).sum()),
            "validation": validations, "human_reviewed": False, "official_evaluation_eligible": False}

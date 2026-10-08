"""Audio-only clock probes and a note-count ceiling for frozen predictions."""

from __future__ import annotations

import math
from typing import Any, Sequence


def note_count_ceiling(reference_ids: Sequence[str], predicted_count: int) -> dict[str, Any]:
    if not reference_ids or len(set(reference_ids)) != len(reference_ids):
        raise ValueError("reference note IDs must be nonempty and unique")
    if isinstance(predicted_count, bool) or not isinstance(predicted_count, int) or predicted_count < 0:
        raise ValueError("predicted note count must be a nonnegative integer")
    maximum = min(len(reference_ids), predicted_count)
    return {
        "reference_notes": len(reference_ids),
        "predicted_notes": predicted_count,
        "maximum_one_to_one_matches": maximum,
        "maximum_recall": maximum / len(reference_ids),
        "minimum_false_negatives": len(reference_ids) - maximum,
        "assumes_every_predicted_note_is_exact": True,
        "pitch_or_boundary_accuracy_inferred": False,
    }


def _lag_curve(original: Any, rendered: Any, frames: Any, lags: range) -> list[dict[str, float | int]]:
    import numpy as np

    x = original[:, frames]
    x = x / np.maximum(np.linalg.norm(x, axis=0), 1e-12)
    rows = []
    for lag in lags:
        target = frames + lag
        if target[0] < 0 or target[-1] >= rendered.shape[1]:
            continue
        y = rendered[:, target]
        y = y / np.maximum(np.linalg.norm(y, axis=0), 1e-12)
        rows.append({"lag_frames": lag, "similarity": float(np.mean(np.sum(x * y, axis=0)))})
    if not rows:
        raise ValueError("no lag covers the fixed source support")
    return rows


def _best(curve: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    return sorted(curve, key=lambda row: (-row["similarity"], abs(row["lag_frames"]), row["lag_frames"]))[0]


def audit_local_lag(
    original: Any, rendered: Any, *, start_sec: float, end_sec: float,
    expected_lag_sec: float, frame_sec: float, radius_sec: float = 4.0,
) -> dict[str, Any]:
    """Fit on outer audio thirds and check the withheld middle, never model notes."""
    import numpy as np

    x, y = np.asarray(original), np.asarray(rendered)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2
            or min(x.shape[1], y.shape[1]) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid audio-only descriptors")
    values = (start_sec, end_sec, expected_lag_sec, frame_sec, radius_sec)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("clock arguments must be finite numbers")
    if start_sec < 0 or end_sec - start_sec < 6 or frame_sec <= 0 or radius_sec <= 0:
        raise ValueError("invalid fixed clip or lag interval")
    first = math.ceil((start_sec + .5) / frame_sec)
    last = math.floor((end_sec - .5) / frame_sec)
    if first < 0 or last >= x.shape[1] or last - first < 30:
        raise ValueError("audio clip is outside cached descriptors")
    split1, split2 = first + (last - first) // 3, first + 2 * (last - first) // 3
    training = np.r_[first:split1, split2:last + 1]
    withheld = np.arange(split1, split2)
    center = round(expected_lag_sec / frame_sec)
    radius = math.ceil(radius_sec / frame_sec)
    lags = range(center - radius, center + radius + 1)
    fit_curve = _lag_curve(x, y, training, lags)
    hold_curve = _lag_curve(x, y, withheld, lags)
    fit, withheld_best = _best(fit_curve), _best(hold_curve)
    runner = next((r for r in sorted(fit_curve, key=lambda r: -r["similarity"])
                   if abs(r["lag_frames"] - fit["lag_frames"]) * frame_sec >= .5), None)
    return {
        "equation": "rendered_sec = original_sec + lag_sec",
        "candidate_lag_sec": fit["lag_frames"] * frame_sec,
        "withheld_best_lag_sec": withheld_best["lag_frames"] * frame_sec,
        "fit_similarity": fit["similarity"],
        "withheld_similarity_at_fit": next(r["similarity"] for r in hold_curve if r["lag_frames"] == fit["lag_frames"]),
        "withheld_lag_disagreement_sec": abs(withheld_best["lag_frames"] - fit["lag_frames"]) * frame_sec,
        "separated_runner_up_margin": fit["similarity"] - runner["similarity"] if runner else None,
        "at_search_edge": abs(fit["lag_frames"] - center) >= radius - 1,
        "fit_frames": len(training), "withheld_frames": len(withheld),
        "frame_step_sec": frame_sec, "search_center_sec": expected_lag_sec,
        "search_radius_sec": radius_sec,
        "source_reference_used_only_to_bound_phrase_search": True,
        "model_predictions_used": False,
        "note_boundary_precision_established": False,
        "clock_approved": False,
    }

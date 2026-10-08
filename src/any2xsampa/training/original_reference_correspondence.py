"""Bounded whole-song timing proposals independent of recognized phones."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from any2xsampa.training.limited_data_artifacts import finite_number, read_json, verify_file


POLICY_PATH = Path(__file__).resolve().parents[1] / "config/original_reference_evaluation_v1.json"


def load_policy() -> dict[str, Any]:
    policy = read_json(POLICY_PATH)
    if policy.get("schema_version") != "original_reference_evaluation_policy_v1":
        raise ValueError("unsupported original-reference policy")
    return policy


def _pooled(values: Any, factor: int) -> Any:
    import numpy as np

    starts = np.arange(0, values.shape[1], factor)
    counts = np.minimum(factor, values.shape[1] - starts)
    return np.add.reduceat(values, starts, axis=1) / counts[None, :]


def global_path(original: Any, rendered: Any, *, factor: int, policy: Mapping[str, Any]) -> dict[str, Any]:
    import librosa
    import numpy as np

    if librosa.__version__ != "0.11.0":
        raise RuntimeError("global correspondence requires existing librosa 0.11.0")
    x, y = np.asarray(original), np.asarray(rendered)
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("invalid pooling factor")
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or min(x.shape[1], y.shape[1]) < 2:
        raise ValueError("invalid descriptor dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite descriptor")
    cells = ((x.shape[1] + factor - 1) // factor) * ((y.shape[1] + factor - 1) // factor)
    if cells > policy["maximum_cost_matrix_cells"]:
        raise ValueError("correspondence exceeds the cost matrix resource budget")
    x, y = _pooled(x, factor), _pooled(y, factor)
    if min(x.shape[1], y.shape[1]) < 2:
        raise ValueError("pooled descriptor too short")
    x = x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=0, keepdims=True), 1e-12)
    cost = np.clip(1.0 - x.T @ y, 0.0, 2.0).astype(np.float64)
    # A normalized-progress band preserves order without equating the two clocks.
    progress_y = np.linspace(0, 1, y.shape[1])
    for i, progress_x in enumerate(np.linspace(0, 1, x.shape[1])):
        cost[i, np.abs(progress_y - progress_x) > policy["relative_progress_band"]] = np.inf
    accumulated, reverse = librosa.sequence.dtw(
        C=cost, step_sizes_sigma=np.asarray(policy["steps"]),
        weights_mul=np.asarray(policy["step_cost_weights"]),
        weights_add=np.asarray(policy["step_additive_penalties"]), subseq=False,
    )
    path = reverse[::-1]
    if not np.isfinite(accumulated[-1, -1]) or tuple(path[0]) != (0, 0):
        raise ValueError("whole-song correspondence has no complete path")
    step_sec = factor * policy["hop_length"] / policy["sample_rate"]
    center_sec = (factor - 1) * policy["hop_length"] / (2 * policy["sample_rate"])
    anchors = []
    for i in np.unique(path[:, 0]):
        js = path[path[:, 0] == i, 1]
        anchors.append({
            "original_sec": float(i * step_sec + center_sec),
            "rendered_sec": float(np.median(js) * step_sec + center_sec),
            "rendered_min_sec": float(js.min() * step_sec + center_sec),
            "rendered_max_sec": float(js.max() * step_sec + center_sec),
        })
    return {
        "factor": factor, "cost_matrix_cells": int(cells),
        "normalized_cost": float(accumulated[-1, -1] / (x.shape[1] + y.shape[1] - 1)),
        "anchors": anchors, "alignment_status": "machine_proposed",
        "human_reviewed": False, "phone_model_outputs_used": False,
    }


def map_clip(path: Mapping[str, Any], start: float, end: float) -> dict[str, float]:
    import numpy as np

    start = finite_number(start, "original start", minimum=0)
    end = finite_number(end, "original end", minimum=0)
    anchors = path["anchors"]
    xs = [row["original_sec"] for row in anchors]
    if end <= start or start < xs[0] or end > xs[-1]:
        raise ValueError("original clip is outside the timing path")
    return {
        "rendered_start_sec": float(np.interp(start, xs, [r["rendered_sec"] for r in anchors])),
        "rendered_end_sec": float(np.interp(end, xs, [r["rendered_sec"] for r in anchors])),
        "start_path_spread_sec": float(np.interp(start, xs, [r["rendered_max_sec"] - r["rendered_min_sec"] for r in anchors])),
        "end_path_spread_sec": float(np.interp(end, xs, [r["rendered_max_sec"] - r["rendered_min_sec"] for r in anchors])),
    }


def match_cached_song(song: Mapping[str, Any], old: Mapping[str, Any], clips: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    if song["split"] == "test":
        raise ValueError("sealed test songs cannot enter correspondence development")
    if old["song_id"] != song["song_id"] or old["split"] != song["split"]:
        raise ValueError("descriptor song/split mismatch")
    for role in ("original", "rendered"):
        if old[f"{role}_audio_sha256"] != song[f"{role}_audio"]["sha256"]:
            raise ValueError("stale descriptor audio hash")
    descriptor = old["descriptor_artifact"]
    path = Path(descriptor["path"])
    verify_file(path, descriptor["sha256"])
    if path.stat().st_size > 64 * 1024 ** 2:
        raise ValueError("descriptor cache exceeds resource budget")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"original", "rendered"}:
            raise ValueError("descriptor cache roles differ")
        original, rendered = archive["original"], archive["rendered"]
    for role, values in (("original", original), ("rendered", rendered)):
        expected = 1 + int(song[f"{role}_audio"]["duration_sec"] * policy["sample_rate"] / policy["hop_length"])
        if values.ndim != 2 or values.shape[0] != 12 or abs(values.shape[1] - expected) > 2:
            raise ValueError("descriptor timebase differs from its source audio")
    paths = [global_path(original, rendered, factor=factor, policy=policy) for factor in policy["downsample_factors"]]
    rows = []
    for clip in clips:
        if clip["song_id"] != song["song_id"] or clip["split"] != song["split"] or clip["audio_role"] != "original":
            raise ValueError("clip song/split/role mismatch")
        candidates = [{"factor": p["factor"], **map_clip(p, clip["start_sec"], clip["end_sec"])} for p in paths]
        disagreement = max(max(c[key] for c in candidates) - min(c[key] for c in candidates)
                           for key in ("rendered_start_sec", "rendered_end_sec"))
        rows.append({
            "example_id": clip["example_id"], "original_start_sec": clip["start_sec"],
            "original_end_sec": clip["end_sec"], "candidates": candidates,
            "resolution_disagreement_sec": disagreement,
            "resolution_consistent": disagreement <= policy["maximum_resolution_disagreement_sec"],
            "alignment_status": "machine_proposed", "human_reviewed": False,
            "official_evaluation_eligible": False,
        })
    return {"song_id": song["song_id"], "split": song["split"], "descriptor_artifact": descriptor,
            "paths": paths, "clips": rows, "phone_predictions_used": False, "test_songs_opened": False}

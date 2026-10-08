"""Independent melody-based correspondence proposals, never acoustic labels."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from any2xsampa.training.limited_data_artifacts import sha256_file, verify_file


SAMPLE_RATE = 22050
HOP_LENGTH = 512
MATCH_POLICY = {
    "schema_version": "audio_section_correspondence_policy_v2",
    "descriptor": "librosa_0.11.0_chroma_cens",
    "sample_rate": SAMPLE_RATE, "hop_length": HOP_LENGTH,
    "cost": "cosine", "steps": [[1, 1], [1, 2], [2, 1]],
    "step_cost_weights": [1.0, 1.0, 2.0],
    "normalization": "cost_per_consumed_original_frame",
    "subsequence": True, "candidate_count": 3,
    "candidate_endpoint_separation_sec": 18.0,
    "ambiguity_relative_cost_gap": 0.1,
    "human_confirmation_required": True,
    "phone_model_output_used": False,
}


def audio_descriptor(audio: Mapping[str, Any]) -> Any:
    import librosa
    import numpy as np

    if librosa.__version__ != "0.11.0":
        raise RuntimeError("section correspondence requires the pinned existing librosa 0.11.0")
    path = Path(audio["path"])
    verify_file(path, audio["sha256"])
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("invalid waveform for correspondence")
    descriptor = librosa.feature.chroma_cens(y=waveform, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    if not np.isfinite(descriptor).all():
        raise ValueError("non-finite correspondence descriptor")
    return descriptor.astype(np.float32)


def propose_correspondence(query: Any, reference: Any, *, query_start_sec: float, frame_seconds: float = HOP_LENGTH / SAMPLE_RATE) -> dict[str, Any]:
    import librosa
    import numpy as np

    x, y = np.asarray(query), np.asarray(reference)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or min(x.shape[1], y.shape[1]) < 2:
        raise ValueError("correspondence expects compatible non-empty feature sequences")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(query_start_sec):
        raise ValueError("non-finite correspondence input")
    if frame_seconds <= 0 or query_start_sec < 0:
        raise ValueError("invalid correspondence timebase")
    x = x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=0, keepdims=True), 1e-12)
    cost = np.maximum(0, 1 - x.T @ y).astype(np.float64)
    step_sizes = np.asarray(MATCH_POLICY["steps"], dtype=np.int64)
    accumulated, steps = librosa.sequence.dtw(
        C=cost, step_sizes_sigma=step_sizes, subseq=True,
        weights_mul=np.asarray(MATCH_POLICY["step_cost_weights"]),
        backtrack=False, return_steps=True,
    )
    endpoint_costs = accumulated[-1] / x.shape[1]
    available = endpoint_costs.copy()
    candidates = []
    separation = max(1, round(18.0 / frame_seconds))
    for rank in range(1, 4):
        endpoint = int(np.argmin(available))
        if not np.isfinite(available[endpoint]):
            break
        path = librosa.sequence.dtw_backtracking(
            steps, step_sizes_sigma=step_sizes, subseq=True, start=endpoint,
        )[::-1]
        source_times = query_start_sec + path[:, 0] * frame_seconds
        reference_times = path[:, 1] * frame_seconds
        coefficients = np.polyfit(source_times, reference_times, 1)
        residual = reference_times - np.polyval(coefficients, source_times)
        pairs = [
            {"original_sec": float(source_times[i]), "rendered_sec": float(reference_times[i])}
            for i in sorted(set([*range(0, len(path), 10), len(path) - 1]))
        ]
        candidates.append({
            "rank": rank,
            "normalized_cost": float(endpoint_costs[endpoint]),
            "mean_path_cost": float(np.mean(cost[path[:, 0], path[:, 1]])),
            "rendered_start_sec": float(reference_times[0]),
            "rendered_end_sec": float(reference_times[-1] + frame_seconds),
            "source_span_coverage_fraction": float((path[-1, 0] - path[0, 0] + 1) / x.shape[1]),
            "matched_source_frame_fraction": float(len(np.unique(path[:, 0])) / x.shape[1]),
            "affine_original_to_rendered": {"slope": float(coefficients[0]), "offset_sec": float(coefficients[1])},
            "affine_residual_p95_sec": float(np.quantile(np.abs(residual), 0.95)),
            "path_anchors": pairs,
            "alignment_status": "machine_proposed",
            "human_reviewed": False,
        })
        available[max(0, endpoint - separation):endpoint + separation + 1] = np.inf
    gap = None
    if len(candidates) >= 2:
        first, second = candidates[0]["normalized_cost"], candidates[1]["normalized_cost"]
        gap = (second - first) / max(abs(second), 1e-12)
    return {
        "candidates": candidates, "relative_cost_gap": gap,
        "ambiguous": gap is None or gap < 0.1,
        "status": "machine_proposed" if candidates else "no_candidate",
        "original_phone_truth_established": False,
        "training_or_accuracy_evaluation_allowed": False,
    }


def match_song(song: Mapping[str, Any], clips: list[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    import numpy as np

    if song["split"] == "test":
        raise ValueError("sealed test songs are not opened for correspondence development")
    before = time.monotonic()
    original = audio_descriptor(song["original_audio"])
    rendered = audio_descriptor(song["rendered_audio"])
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path = output_dir / f"{song['song_id']}-descriptors.npz"
    if descriptor_path.exists():
        raise FileExistsError(descriptor_path)
    with descriptor_path.open("xb") as stream:
        np.savez_compressed(stream, original=original, rendered=rendered)
    results = []
    frame_seconds = HOP_LENGTH / SAMPLE_RATE
    for clip in clips:
        first = round(clip["start_sec"] / frame_seconds)
        last = min(original.shape[1], round(clip["end_sec"] / frame_seconds))
        results.append({
            "example_id": clip["example_id"],
            "requested_original_start_sec": clip["start_sec"],
            "requested_original_end_sec": clip["end_sec"],
            **propose_correspondence(original[:, first:last], rendered, query_start_sec=first * frame_seconds),
        })
    return {
        "song_id": song["song_id"], "split": song["split"], "policy": MATCH_POLICY,
        "original_audio_sha256": song["original_audio"]["sha256"],
        "rendered_audio_sha256": song["rendered_audio"]["sha256"],
        "descriptor_artifact": {"path": str(descriptor_path.resolve()), "sha256": sha256_file(descriptor_path)},
        "clips": results, "elapsed_seconds": time.monotonic() - before,
        "labels_promoted_or_modified": False,
    }

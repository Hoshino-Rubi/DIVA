from __future__ import annotations

from typing import Any

from any2xsampa.adapters.recognizer_support import (
    _normalize_symbol_scores_frame,
    _probability_to_logit,
    _safe_mean,
    _softmax_symbol_scores,
    _sorted_symbol_scores,
)
from any2xsampa.core.models import PhoneCandidate


def frame_trace(
    metadata: dict[str, Any],
    trace_key: str,
    score_key: str,
) -> tuple[list[dict[str, float]], str | None, float | None]:
    trace_payload = metadata.get(trace_key)
    trace_map = trace_payload if isinstance(trace_payload, dict) else {}
    frames_payload = trace_map.get("frames", trace_map.get(trace_key))
    if frames_payload is None:
        frames_payload = metadata.get(trace_key)
    if not isinstance(frames_payload, list):
        return [], None, None

    frames: list[dict[str, float]] = []
    for frame in frames_payload:
        normalized = _normalize_symbol_scores_frame(frame, score_key=score_key)
        if normalized:
            frames.append(normalized)

    source = str(trace_map.get("source", trace_key))
    frame_hop_sec = None
    try:
        raw_hop = trace_map.get("frame_hop_sec", metadata.get("frame_hop_sec"))
        if raw_hop is not None and not isinstance(raw_hop, bool):
            frame_hop_sec = float(raw_hop)
    except (TypeError, ValueError):
        frame_hop_sec = None
    return frames, source, frame_hop_sec


def derive_posterior_frames(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, float]], str, float | None]:
    posterior_frames, posterior_source, frame_hop_sec = frame_trace(
        metadata=metadata,
        trace_key="posterior_trace",
        score_key="confidence",
    )
    if posterior_frames:
        return posterior_frames, posterior_source or "posterior_trace", frame_hop_sec

    posterior = metadata.get("posterior")
    if isinstance(posterior, list) and posterior:
        normalized = _normalize_symbol_scores_frame(
            {"candidates": posterior},
            score_key="confidence",
        )
        if normalized:
            return [normalized], "posterior", frame_hop_sec

    return (
        [{candidate.symbol: candidate.confidence for candidate in candidates if candidate.symbol}],
        "candidate_phones",
        frame_hop_sec,
    )


def derive_logit_frames(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, float]], str, float | None]:
    logit_frames, logit_source, frame_hop_sec = frame_trace(
        metadata=metadata,
        trace_key="logit_trace",
        score_key="logit",
    )
    if logit_frames:
        return logit_frames, logit_source or "logit_trace", frame_hop_sec

    posterior_frames, posterior_source, frame_hop_sec = derive_posterior_frames(
        candidates,
        metadata,
    )
    derived_frames: list[dict[str, float]] = []
    for frame in posterior_frames:
        if frame:
            derived_frames.append(
                {
                    symbol: _probability_to_logit(value)
                    for symbol, value in frame.items()
                }
            )
    return derived_frames, f"{posterior_source}:derived_logits", frame_hop_sec


def posterior_summary(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    top_phone: str,
) -> dict[str, Any]:
    explicit_summary_payload = metadata.get("posterior_summary")
    explicit_summary = explicit_summary_payload if isinstance(explicit_summary_payload, dict) else {}
    if explicit_summary:
        return {
            "available": bool(explicit_summary.get("available", True)),
            "source": str(explicit_summary.get("source", "posterior_summary")),
            "frame_count": int(explicit_summary.get("frame_count", 0) or 0),
            "frame_hop_sec": (
                round(float(explicit_summary["frame_hop_sec"]), 4)
                if explicit_summary.get("frame_hop_sec") is not None
                else None
            ),
            "top_phone_mean_posterior": round(
                float(explicit_summary.get("top_phone_mean_posterior", candidates[0].confidence if candidates else 0.0)),
                3,
            ),
            "top_phone_peak_posterior": round(
                float(explicit_summary.get("top_phone_peak_posterior", candidates[0].confidence if candidates else 0.0)),
                3,
            ),
            "runner_up_mean_posterior": round(float(explicit_summary.get("runner_up_mean_posterior", 0.0)), 3),
            "mean_margin": round(float(explicit_summary.get("mean_margin", 0.0)), 3),
            "peak_margin": round(float(explicit_summary.get("peak_margin", 0.0)), 3),
            "frame_vote_ratio": round(float(explicit_summary.get("frame_vote_ratio", 0.0)), 3),
            "temporal_consistency": round(float(explicit_summary.get("temporal_consistency", 0.0)), 3),
            "interior_mean_margin": round(
                float(explicit_summary.get("interior_mean_margin", explicit_summary.get("mean_margin", 0.0))),
                3,
            ),
            "interior_vote_ratio": round(
                float(explicit_summary.get("interior_vote_ratio", explicit_summary.get("frame_vote_ratio", 0.0))),
                3,
            ),
            "entry_margin": round(
                float(explicit_summary.get("entry_margin", explicit_summary.get("peak_margin", 0.0))),
                3,
            ),
            "exit_margin": round(
                float(explicit_summary.get("exit_margin", explicit_summary.get("peak_margin", 0.0))),
                3,
            ),
            "edge_competition_ratio": round(float(explicit_summary.get("edge_competition_ratio", 0.0)), 3),
        }

    frames, source, frame_hop_sec = derive_posterior_frames(candidates, metadata)
    if not frames:
        return {"available": False, "source": source or "none"}

    margins: list[float] = []
    interior_margins: list[float] = []
    top_values: list[float] = []
    runner_values: list[float] = []
    top_votes = 0
    interior_votes = 0
    current_run = 0
    longest_run = 0
    edge_competition: list[float] = []
    frame_count = len(frames)
    edge_indices = {0, frame_count - 1}
    interior_start = 1 if frame_count >= 3 else 0
    interior_end = frame_count - 1 if frame_count >= 3 else frame_count
    entry_margin = 0.0
    exit_margin = 0.0
    for index, frame in enumerate(frames):
        ordered = _sorted_symbol_scores(frame)
        if not ordered:
            continue
        winning_symbol, winning_value = ordered[0]
        runner_value = ordered[1][1] if len(ordered) > 1 else 0.0
        top_value = frame.get(top_phone, 0.0)
        margin = max(0.0, top_value - max((value for symbol, value in ordered if symbol != top_phone), default=0.0))
        top_values.append(top_value)
        runner_values.append(runner_value if winning_symbol == top_phone else winning_value)
        margins.append(margin)
        if index in edge_indices:
            edge_competition.append(max(0.0, min(1.0, 1.0 - margin)))
            if index == 0:
                entry_margin = margin
            if index == frame_count - 1:
                exit_margin = margin
        if interior_start <= index < interior_end:
            interior_margins.append(margin)
            if winning_symbol == top_phone:
                interior_votes += 1
        if winning_symbol == top_phone:
            top_votes += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    interior_frame_count = max(1, interior_end - interior_start)
    return {
        "available": True,
        "source": source,
        "frame_count": frame_count,
        "frame_hop_sec": round(frame_hop_sec, 4) if frame_hop_sec is not None else None,
        "top_phone_mean_posterior": round(sum(top_values) / max(1, len(top_values)), 3),
        "top_phone_peak_posterior": round(max(top_values) if top_values else 0.0, 3),
        "runner_up_mean_posterior": round(sum(runner_values) / max(1, len(runner_values)), 3),
        "mean_margin": round(sum(margins) / max(1, len(margins)), 3),
        "peak_margin": round(max(margins) if margins else 0.0, 3),
        "frame_vote_ratio": round(top_votes / max(1, frame_count), 3),
        "temporal_consistency": round(longest_run / max(1, frame_count), 3),
        "interior_mean_margin": round(_safe_mean(interior_margins, _safe_mean(margins, 0.0)), 3),
        "interior_vote_ratio": round(interior_votes / max(1, interior_frame_count), 3),
        "entry_margin": round(entry_margin or _safe_mean(margins, 0.0), 3),
        "exit_margin": round(exit_margin or _safe_mean(margins, 0.0), 3),
        "edge_competition_ratio": round(_safe_mean(edge_competition, 0.0), 3),
    }


def logit_summary(
    candidates: list[PhoneCandidate],
    metadata: dict[str, Any],
    top_phone: str,
) -> dict[str, Any]:
    explicit_summary_payload = metadata.get("logit_summary")
    explicit_summary = explicit_summary_payload if isinstance(explicit_summary_payload, dict) else {}
    if explicit_summary:
        return {
            "available": bool(explicit_summary.get("available", True)),
            "source": str(explicit_summary.get("source", "logit_summary")),
            "frame_count": int(explicit_summary.get("frame_count", 0) or 0),
            "top_phone_mean_logit": round(float(explicit_summary.get("top_phone_mean_logit", 0.0)), 3),
            "top_phone_peak_logit": round(float(explicit_summary.get("top_phone_peak_logit", 0.0)), 3),
            "runner_up_mean_logit": round(float(explicit_summary.get("runner_up_mean_logit", 0.0)), 3),
            "mean_margin": round(float(explicit_summary.get("mean_margin", 0.0)), 3),
            "peak_margin": round(float(explicit_summary.get("peak_margin", 0.0)), 3),
            "interior_mean_margin": round(
                float(explicit_summary.get("interior_mean_margin", explicit_summary.get("mean_margin", 0.0))),
                3,
            ),
            "entry_margin": round(
                float(explicit_summary.get("entry_margin", explicit_summary.get("peak_margin", 0.0))),
                3,
            ),
            "exit_margin": round(
                float(explicit_summary.get("exit_margin", explicit_summary.get("peak_margin", 0.0))),
                3,
            ),
        }

    frames, source, _ = derive_logit_frames(candidates, metadata)
    if not frames:
        return {"available": False, "source": source or "none"}

    margins: list[float] = []
    interior_margins: list[float] = []
    top_values: list[float] = []
    runner_values: list[float] = []
    frame_count = len(frames)
    interior_start = 1 if frame_count >= 3 else 0
    interior_end = frame_count - 1 if frame_count >= 3 else frame_count
    entry_margin = 0.0
    exit_margin = 0.0
    for index, frame in enumerate(frames):
        ordered = _sorted_symbol_scores(frame)
        if not ordered:
            continue
        top_value = frame.get(top_phone, ordered[0][1])
        runner_value = max((value for symbol, value in ordered if symbol != top_phone), default=top_value)
        top_values.append(top_value)
        runner_values.append(runner_value)
        margin = top_value - runner_value
        margins.append(margin)
        if index == 0:
            entry_margin = margin
        if index == frame_count - 1:
            exit_margin = margin
        if interior_start <= index < interior_end:
            interior_margins.append(margin)

    return {
        "available": True,
        "source": source,
        "frame_count": len(frames),
        "top_phone_mean_logit": round(sum(top_values) / max(1, len(top_values)), 3),
        "top_phone_peak_logit": round(max(top_values) if top_values else 0.0, 3),
        "runner_up_mean_logit": round(sum(runner_values) / max(1, len(runner_values)), 3),
        "mean_margin": round(sum(margins) / max(1, len(margins)), 3),
        "peak_margin": round(max(margins) if margins else 0.0, 3),
        "interior_mean_margin": round(_safe_mean(interior_margins, _safe_mean(margins, 0.0)), 3),
        "entry_margin": round(entry_margin or _safe_mean(margins, 0.0), 3),
        "exit_margin": round(exit_margin or _safe_mean(margins, 0.0), 3),
    }


def infer_candidates_from_metadata(metadata: dict[str, Any]) -> list[PhoneCandidate]:
    posterior = metadata.get("posterior")
    if isinstance(posterior, list):
        normalized = _normalize_symbol_scores_frame({"candidates": posterior}, score_key="confidence")
        if normalized:
            return [
                PhoneCandidate(symbol=symbol, confidence=round(value, 3))
                for symbol, value in _sorted_symbol_scores(normalized)
            ]

    posterior_frames, _, _ = frame_trace(metadata, trace_key="posterior_trace", score_key="confidence")
    if posterior_frames:
        aggregate: dict[str, list[float]] = {}
        for frame in posterior_frames:
            for symbol, value in frame.items():
                aggregate.setdefault(symbol, []).append(value)
        if aggregate:
            averaged = {symbol: (sum(values) / len(values)) for symbol, values in aggregate.items()}
            return [
                PhoneCandidate(symbol=symbol, confidence=round(value, 3))
                for symbol, value in _sorted_symbol_scores(averaged)
            ]

    logit_frames, _, _ = frame_trace(metadata, trace_key="logit_trace", score_key="logit")
    if logit_frames:
        aggregate_logits: dict[str, list[float]] = {}
        for frame in logit_frames:
            for symbol, value in frame.items():
                aggregate_logits.setdefault(symbol, []).append(value)
        if aggregate_logits:
            mean_logits = {symbol: (sum(values) / len(values)) for symbol, values in aggregate_logits.items()}
            posterior_like = _softmax_symbol_scores(mean_logits)
            return [
                PhoneCandidate(symbol=symbol, confidence=round(value, 3))
                for symbol, value in _sorted_symbol_scores(posterior_like)
            ]
    return []


def resolve_evidence_kind(
    posterior_like_available: bool,
    logit_like_available: bool,
) -> str:
    if posterior_like_available and logit_like_available:
        return "posterior_and_logit_adapter_like"
    if posterior_like_available:
        return "posterior_adapter_like"
    if logit_like_available:
        return "logit_adapter_like"
    return "candidate_confidence_adapter"


def build_adapter_diagnostics(
    *,
    adapter_kind: str,
    backend_kind: str,
    source: str,
    aggregation: str,
    candidate_source: str,
    candidate_aggregation: str | None = None,
    posterior_like_available: bool,
    logit_like_available: bool,
    approximation_only: bool,
    evidence_kind: str | None = None,
    posterior_source: object = None,
    logit_source: object = None,
    frame_count: int | None = None,
    frame_hop_sec: object = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "adapter_kind": adapter_kind,
        "backend_kind": backend_kind,
        "evidence_kind": evidence_kind
        or resolve_evidence_kind(posterior_like_available, logit_like_available),
        "candidate_source": candidate_source,
        "posterior_like_available": posterior_like_available,
        "logit_like_available": logit_like_available,
        "posterior_source": posterior_source,
        "logit_source": logit_source,
        "candidate_aggregation": candidate_aggregation or aggregation,
        "aggregation": aggregation,
        "approximation_only": approximation_only,
        "source": source,
    }
    if frame_count is not None:
        payload["frame_count"] = frame_count
    if frame_hop_sec is not None:
        payload["frame_hop_sec"] = frame_hop_sec
    return payload


__all__ = [
    "build_adapter_diagnostics",
    "derive_logit_frames",
    "derive_posterior_frames",
    "frame_trace",
    "infer_candidates_from_metadata",
    "logit_summary",
    "posterior_summary",
    "resolve_evidence_kind",
]

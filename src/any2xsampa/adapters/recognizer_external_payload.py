from __future__ import annotations

from typing import Any, Callable

from any2xsampa.adapters.recognizer_support import (
    _candidate_payload_from_scores,
    _merge_text_list,
    _normalize_symbol_scores_frame,
    _normalize_trace_payload,
)


def _normalize_external_summary(
    payload: object,
    *,
    summary_kind: str,
    default_source: str,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    summary: dict[str, Any] = {"available": bool(payload.get("available", True))}
    summary["source"] = str(payload.get("source", default_source))
    numeric_keys = (
        "frame_count",
        "frame_hop_sec",
        "top_phone_mean_posterior",
        "top_phone_peak_posterior",
        "runner_up_mean_posterior",
        "mean_margin",
        "peak_margin",
        "frame_vote_ratio",
        "temporal_consistency",
        "top_phone_mean_logit",
        "top_phone_peak_logit",
        "runner_up_mean_logit",
    )
    for key in numeric_keys:
        value = payload.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            summary[key] = float(value)
        except (TypeError, ValueError):
            continue
    if summary_kind == "posterior":
        summary.setdefault("top_phone_mean_posterior", 0.0)
        summary.setdefault("runner_up_mean_posterior", 0.0)
        summary.setdefault("mean_margin", 0.0)
        summary.setdefault("peak_margin", 0.0)
        summary.setdefault("frame_vote_ratio", 0.0)
        summary.setdefault("temporal_consistency", 0.0)
    else:
        summary.setdefault("top_phone_mean_logit", 0.0)
        summary.setdefault("runner_up_mean_logit", 0.0)
        summary.setdefault("mean_margin", 0.0)
        summary.setdefault("peak_margin", 0.0)
    return summary


def _external_json_acoustic_evidence(
    segment_payload: dict[str, Any],
    *,
    backend_name: str,
    source_label: str,
    build_adapter_diagnostics: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata_payload = segment_payload.get("metadata")
    metadata = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}
    summary_payload = segment_payload.get("segment_summary", segment_payload.get("acoustic_summary"))
    summary_map = summary_payload if isinstance(summary_payload, dict) else {}
    acoustic_payload = metadata.get("acoustic_evidence")
    acoustic_map = dict(acoustic_payload) if isinstance(acoustic_payload, dict) else {}

    direct_numeric_fields = (
        "acoustic_confidence",
        "timing_reliability",
        "frame_consistency",
        "evidence_quality",
        "segment_likelihood",
        "candidate_lead",
        "boundary_transition_strength",
        "boundary_softness",
        "ownership_ambiguity",
    )
    for key in direct_numeric_fields:
        value = segment_payload.get(key, summary_map.get(key))
        if value is not None:
            acoustic_map[key] = value
            metadata[key] = value

    boundary_payload = (
        segment_payload.get("boundary_uncertainty")
        or summary_map.get("boundary_uncertainty")
        or acoustic_map.get("boundary_uncertainty")
    )
    if isinstance(boundary_payload, dict):
        acoustic_map["boundary_uncertainty"] = {str(key): boundary_payload[key] for key in boundary_payload}

    note_alignment_payload = (
        segment_payload.get("note_alignment")
        or segment_payload.get("continuation_profile")
        or segment_payload.get("continuation")
        or summary_map.get("note_alignment")
        or acoustic_map.get("note_alignment")
    )
    if isinstance(note_alignment_payload, dict):
        note_alignment = dict(acoustic_map.get("note_alignment", {}))
        note_alignment.update({str(key): note_alignment_payload[key] for key in note_alignment_payload})
        acoustic_map["note_alignment"] = note_alignment

    ownership_payload = (
        segment_payload.get("ownership")
        or segment_payload.get("ownership_summary")
        or summary_map.get("ownership")
        or acoustic_map.get("ownership")
    )
    if isinstance(ownership_payload, dict):
        ownership = dict(acoustic_map.get("ownership", {}))
        ownership.update({str(key): ownership_payload[key] for key in ownership_payload})
        acoustic_map["ownership"] = ownership

    continuation_profile_payload = segment_payload.get("continuation_profile") or summary_map.get("continuation_profile")
    if isinstance(continuation_profile_payload, dict):
        acoustic_map["continuation_profile"] = {
            str(key): continuation_profile_payload[key] for key in continuation_profile_payload
        }

    boundary_profile_payload = segment_payload.get("boundary_profile") or summary_map.get("boundary_profile")
    if isinstance(boundary_profile_payload, dict):
        acoustic_map["boundary_profile"] = {
            str(key): boundary_profile_payload[key] for key in boundary_profile_payload
        }

    candidate_distribution_payload = (
        segment_payload.get("candidate_distribution") or summary_map.get("candidate_distribution")
    )
    if isinstance(candidate_distribution_payload, dict):
        acoustic_map["candidate_distribution"] = {
            str(key): candidate_distribution_payload[key] for key in candidate_distribution_payload
        }

    posterior_summary_payload = segment_payload.get("posterior_summary") or summary_map.get("posterior_summary")
    posterior_summary = _normalize_external_summary(
        posterior_summary_payload,
        summary_kind="posterior",
        default_source=f"{backend_name}:posterior_summary",
    )
    if posterior_summary is not None:
        metadata["posterior_summary"] = posterior_summary

    logit_summary_payload = segment_payload.get("logit_summary") or summary_map.get("logit_summary")
    logit_summary = _normalize_external_summary(
        logit_summary_payload,
        summary_kind="logit",
        default_source=f"{backend_name}:logit_summary",
    )
    if logit_summary is not None:
        metadata["logit_summary"] = logit_summary

    adapter_payload = segment_payload.get("adapter_diagnostics")
    if not isinstance(adapter_payload, dict):
        adapter_payload = summary_map.get("adapter_diagnostics")
    adapter_map = dict(adapter_payload) if isinstance(adapter_payload, dict) else {}
    default_aggregation = str(
        adapter_map.get("aggregation", adapter_map.get("candidate_aggregation", "external_segment_candidates"))
    )
    acoustic_map["adapter_diagnostics"] = build_adapter_diagnostics(
        adapter_kind=str(adapter_map.get("adapter_kind", backend_name)),
        backend_kind=str(adapter_map.get("backend_kind", backend_name)),
        source=str(adapter_map.get("source", source_label)),
        aggregation=default_aggregation,
        candidate_source=str(adapter_map.get("candidate_source", f"{backend_name}:candidate_phones")),
        candidate_aggregation=str(adapter_map.get("candidate_aggregation", default_aggregation)),
        posterior_like_available=bool(adapter_map.get("posterior_like_available", False)),
        logit_like_available=bool(adapter_map.get("logit_like_available", False)),
        approximation_only=bool(adapter_map.get("approximation_only", False)),
        evidence_kind=str(adapter_map.get("evidence_kind", "")) or None,
        frame_hop_sec=adapter_map.get("frame_hop_sec"),
    )

    acoustic_map["notes"] = _merge_text_list(
        _merge_text_list([], acoustic_map.get("notes")),
        segment_payload.get("notes", summary_map.get("notes")),
    )
    return metadata, acoustic_map, summary_map


def _normalize_external_segment_payload(
    segment_payload: dict[str, Any],
    *,
    backend_name: str,
    source_mode: str,
    source_label: str,
    index: int,
    build_adapter_diagnostics: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    start_sec = segment_payload.get("start_sec")
    end_sec = segment_payload.get("end_sec")
    if start_sec is None or end_sec is None:
        raise ValueError(f"Recognizer segment {index} is missing start_sec/end_sec.")

    metadata, acoustic_map, _ = _external_json_acoustic_evidence(
        segment_payload,
        backend_name=backend_name,
        source_label=source_label,
        build_adapter_diagnostics=build_adapter_diagnostics,
    )
    metadata.setdefault("recognizer_backend_family", backend_name.replace("-", "_"))
    metadata.setdefault("source", source_label)
    metadata.setdefault("source_mode", source_mode)

    posterior_payload = segment_payload.get("posterior", segment_payload.get("posteriors"))
    if posterior_payload is not None and "posterior" not in metadata:
        posterior_map = _normalize_symbol_scores_frame(posterior_payload, score_key="confidence")
        if posterior_map:
            metadata["posterior"] = _candidate_payload_from_scores(posterior_map, score_key="confidence")
            acoustic_map["adapter_diagnostics"]["posterior_like_available"] = True
            acoustic_map["adapter_diagnostics"]["candidate_source"] = f"{backend_name}:posterior"

    posterior_trace_payload = segment_payload.get("posterior_trace", segment_payload.get("posterior_frames"))
    posterior_trace = _normalize_trace_payload(
        posterior_trace_payload,
        score_key="confidence",
        default_source=f"{backend_name}:posterior_trace",
    )
    if posterior_trace is not None:
        metadata["posterior_trace"] = posterior_trace
        acoustic_map["adapter_diagnostics"]["posterior_like_available"] = True
        acoustic_map["adapter_diagnostics"]["candidate_source"] = str(
            posterior_trace.get("source", f"{backend_name}:posterior_trace")
        )
        if posterior_trace.get("frame_hop_sec") is not None:
            acoustic_map["adapter_diagnostics"]["frame_hop_sec"] = posterior_trace["frame_hop_sec"]
        if acoustic_map["adapter_diagnostics"]["candidate_aggregation"] == "external_segment_candidates":
            acoustic_map["adapter_diagnostics"]["candidate_aggregation"] = "mean_frame_confidence"
        acoustic_map["adapter_diagnostics"]["aggregation"] = acoustic_map["adapter_diagnostics"]["candidate_aggregation"]
    elif "posterior" in metadata or "posterior_summary" in metadata:
        acoustic_map["adapter_diagnostics"]["posterior_like_available"] = True
        acoustic_map["adapter_diagnostics"]["candidate_source"] = "posterior"

    logits_payload = segment_payload.get("logits", segment_payload.get("logit_scores"))
    logits_trace = _normalize_trace_payload(
        logits_payload,
        score_key="logit",
        default_source=f"{backend_name}:segment_logits",
    )
    if logits_trace is not None:
        metadata["logit_trace"] = logits_trace
        acoustic_map["adapter_diagnostics"]["logit_like_available"] = True
        if not acoustic_map["adapter_diagnostics"]["posterior_like_available"]:
            acoustic_map["adapter_diagnostics"]["candidate_source"] = str(
                logits_trace.get("source", f"{backend_name}:segment_logits")
            )

    logit_trace_payload = segment_payload.get("logit_trace", segment_payload.get("logit_frames"))
    logit_trace = _normalize_trace_payload(
        logit_trace_payload,
        score_key="logit",
        default_source=f"{backend_name}:logit_trace",
    )
    if logit_trace is not None:
        metadata["logit_trace"] = logit_trace
        acoustic_map["adapter_diagnostics"]["logit_like_available"] = True
        if (
            logit_trace.get("frame_hop_sec") is not None
            and acoustic_map["adapter_diagnostics"].get("frame_hop_sec") is None
        ):
            acoustic_map["adapter_diagnostics"]["frame_hop_sec"] = logit_trace["frame_hop_sec"]
        if not acoustic_map["adapter_diagnostics"]["posterior_like_available"]:
            acoustic_map["adapter_diagnostics"]["candidate_source"] = str(
                logit_trace.get("source", f"{backend_name}:logit_trace")
            )
        if acoustic_map["adapter_diagnostics"]["candidate_aggregation"] == "external_segment_candidates":
            acoustic_map["adapter_diagnostics"]["candidate_aggregation"] = "mean_frame_logit"
        acoustic_map["adapter_diagnostics"]["aggregation"] = acoustic_map["adapter_diagnostics"]["candidate_aggregation"]
    elif "logit_trace" in metadata or "logit_summary" in metadata:
        acoustic_map["adapter_diagnostics"]["logit_like_available"] = True

    if (
        acoustic_map["adapter_diagnostics"]["posterior_like_available"]
        and acoustic_map["adapter_diagnostics"]["logit_like_available"]
    ):
        acoustic_map["adapter_diagnostics"]["evidence_kind"] = "posterior_and_logit_adapter_like"
    elif acoustic_map["adapter_diagnostics"]["posterior_like_available"]:
        acoustic_map["adapter_diagnostics"]["evidence_kind"] = "posterior_adapter_like"
    elif acoustic_map["adapter_diagnostics"]["logit_like_available"]:
        acoustic_map["adapter_diagnostics"]["evidence_kind"] = "logit_adapter_like"

    metadata["acoustic_evidence"] = acoustic_map
    return {
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "confidence": segment_payload.get("confidence"),
        "candidate_phones": list(segment_payload.get("candidate_phones", [])),
        "metadata": metadata,
    }


def _normalize_json_payload(
    payload: dict[str, Any],
    *,
    source_mode: str,
    backend_name: str,
    source_label: str,
    build_adapter_diagnostics: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    segments_payload = payload.get("segments")
    if not isinstance(segments_payload, list) or not segments_payload:
        raise ValueError("Recognizer payload must contain a non-empty 'segments' list.")

    normalized_segments: list[dict[str, Any]] = []
    for index, segment_payload in enumerate(segments_payload):
        if not isinstance(segment_payload, dict):
            raise ValueError(f"Recognizer segment {index} must be an object.")
        normalized_segments.append(
            _normalize_external_segment_payload(
                segment_payload,
                backend_name=backend_name,
                source_mode=source_mode,
                source_label=source_label,
                index=index,
                build_adapter_diagnostics=build_adapter_diagnostics,
            )
        )
    return {"segments": normalized_segments}

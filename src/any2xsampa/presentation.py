from __future__ import annotations

import json
from typing import Any


def augment_pipeline_payload(payload: dict[str, Any]) -> dict[str, Any]:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return payload
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        metadata = segment.get("metadata")
        if not isinstance(metadata, dict):
            continue
        metadata["future_aligner_provenance_view"] = build_provenance_view(segment)
    return payload


def build_error_payload(
    message: str,
    *,
    error_code: str,
    error_kind: str,
    status_code: int | None = None,
    exit_code: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": message,
        "error_code": error_code,
        "error_kind": error_kind,
    }
    if status_code is not None:
        payload["status_code"] = status_code
    if exit_code is not None:
        payload["exit_code"] = exit_code
    if details:
        payload["details"] = dict(details)
    return payload


def build_transport_error_payload(
    message: str,
    *,
    transport: str,
    error_code: str,
    error_kind: str,
    status_code: int | None = None,
    exit_code: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_details = {"transport": transport}
    if details:
        merged_details.update(details)
    return build_error_payload(
        message,
        error_code=error_code,
        error_kind=error_kind,
        status_code=status_code,
        exit_code=exit_code,
        details=merged_details,
    )


def classify_transport_exception(
    exc: Exception,
    *,
    transport: str,
) -> tuple[str, str]:
    if isinstance(exc, FileNotFoundError):
        return f"{transport}.file_not_found", "file_not_found"
    if isinstance(exc, json.JSONDecodeError):
        return f"{transport}.invalid_json", "invalid_json"
    if isinstance(exc, KeyError):
        return f"{transport}.not_found", "not_found"
    return f"{transport}.validation_error", "validation_error"


def build_transport_exception_payload(
    exc: Exception,
    *,
    transport: str,
    message: str | None = None,
    status_code: int | None = None,
    exit_code: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error_code, error_kind = classify_transport_exception(exc, transport=transport)
    return build_transport_error_payload(
        message or str(exc),
        transport=transport,
        error_code=error_code,
        error_kind=error_kind,
        status_code=status_code,
        exit_code=exit_code,
        details=details,
    )


def build_status_capabilities(
    *,
    supported_dialects: object = None,
    supported_export_formats: object = None,
    supported_note_context_formats: object = None,
    supported_recognizer_modes: object = None,
    api_features: object = None,
) -> dict[str, list[str]]:
    capabilities = {
        "supported_dialects": _unique_text_list(supported_dialects),
        "supported_export_formats": _unique_text_list(supported_export_formats),
        "supported_note_context_formats": _unique_text_list(supported_note_context_formats),
    }
    recognizer_modes = _unique_text_list(supported_recognizer_modes)
    if recognizer_modes:
        capabilities["supported_recognizer_modes"] = recognizer_modes
    features = _unique_text_list(api_features)
    if features:
        capabilities["api_features"] = features
    return capabilities


def with_status_capabilities(
    payload: dict[str, Any],
    *,
    api_features: object = None,
) -> dict[str, Any]:
    nested = payload.get("capabilities")
    nested_map = nested if isinstance(nested, dict) else {}
    capabilities = build_status_capabilities(
        supported_dialects=nested_map.get("supported_dialects", payload.get("supported_dialects")),
        supported_export_formats=nested_map.get("supported_export_formats", payload.get("supported_export_formats")),
        supported_note_context_formats=nested_map.get(
            "supported_note_context_formats",
            payload.get("supported_note_context_formats"),
        ),
        supported_recognizer_modes=nested_map.get(
            "supported_recognizer_modes",
            payload.get("supported_recognizer_modes"),
        ),
        api_features=nested_map.get("api_features", payload.get("api_features")) if api_features is None else api_features,
    )
    merged = dict(payload)
    merged.update(capabilities)
    merged["capabilities"] = capabilities
    return merged


def extract_status_capabilities(payload: dict[str, Any]) -> dict[str, list[str]]:
    return with_status_capabilities(payload).get("capabilities", {})


def build_provenance_view(segment: dict[str, Any]) -> dict[str, Any]:
    metadata = segment.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}
    hint = metadata_map.get("future_aligner_sequence_hint")
    hint_map = hint if isinstance(hint, dict) else {}
    items = _normalize_items(
        metadata_map.get("future_aligner_provenance_items")
        if isinstance(metadata_map.get("future_aligner_provenance_items"), list)
        else hint_map.get("future_aligner_provenance_items")
    )
    accepted = [item for item in items if item["accepted"]]
    rejected = [item for item in items if not item["accepted"]]
    left = [item for item in items if item["support_direction"] == "left"]
    right = [item for item in items if item["support_direction"] == "right"]
    rejection_reasons: list[str] = []
    for item in rejected:
        reason = item["rejection_reason"]
        if reason and reason not in rejection_reasons:
            rejection_reasons.append(reason)
    aligned_note_id = _text(hint_map.get("aligned_note_id")) or _text(metadata_map.get("aligned_note_id"))
    summary = _text(metadata_map.get("future_aligner_provenance_summary")) or _text(
        metadata_map.get("future_aligner_evidence_summary")
    )
    fallback_used = bool(
        hint_map.get("aligner_fallback_used", False) or metadata_map.get("future_aligner_fallback_used", False)
    )
    has_provenance = bool(items)
    if not has_provenance:
        message = "No future aligner provenance was generated for this segment."
    elif fallback_used and not accepted:
        message = "Conservative fallback: note ownership was not stabilized; rejected or insufficient evidence is shown."
    elif rejected:
        message = f"Accepted {len(accepted)} evidence items; rejected {len(rejected)} competing or insufficient items."
    else:
        message = f"Accepted {len(accepted)} evidence items for note stabilization."
    return {
        "has_provenance": has_provenance,
        "fallback_used": fallback_used,
        "aligned_note_id": aligned_note_id,
        "summary": summary,
        "message": message,
        "left_evidence": left,
        "right_evidence": right,
        "accepted_evidence": accepted,
        "rejected_evidence": rejected,
        "rejection_reasons": rejection_reasons,
    }


def _normalize_items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "evidence_type": _text(entry.get("evidence_type")) or "unknown",
                "source_segment_index": _int(entry.get("source_segment_index")),
                "source_note_id": _text(entry.get("source_note_id")),
                "support_direction": _text(entry.get("support_direction")) or "window",
                "support_score": round(_float(entry.get("support_score")) or 0.0, 3),
                "accepted": bool(entry.get("accepted", False)),
                "rejection_reason": _text(entry.get("rejection_reason")),
                "source_track_id": _text(entry.get("source_track_id")),
                "source_part_id": _text(entry.get("source_part_id")),
                "notes": _text_list(entry.get("notes")),
            }
        )
    return normalized


def _text(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = _text(entry)
        if text is not None:
            items.append(text)
    return items


def _unique_text_list(value: object) -> list[str]:
    items: list[str] = []
    if not isinstance(value, list):
        return items
    for entry in value:
        text = _text(entry)
        if text is not None and text not in items:
            items.append(text)
    return items

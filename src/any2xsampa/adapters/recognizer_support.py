from __future__ import annotations

import math
from typing import Any


def _default_articulatory_features(symbol: str) -> dict[str, str]:
    feature_map = {
        "a": {"type": "vowel", "height": "open", "backness": "front"},
        "@": {"type": "vowel", "height": "mid", "backness": "central"},
        "aa": {"type": "vowel", "height": "open", "backness": "front"},
        "i": {"type": "vowel", "height": "close", "backness": "front"},
        "l": {"type": "consonant", "manner": "lateral_approximant", "place": "alveolar"},
        "N": {"type": "consonant", "manner": "nasal", "place": "velar"},
        "ng": {"type": "consonant", "manner": "nasal", "place": "velar"},
        "s": {"type": "consonant", "manner": "fricative", "place": "alveolar"},
        "sil": {"type": "silence"},
        "ts": {"type": "consonant", "manner": "affricate", "place": "alveolar"},
        "ts_h": {"type": "consonant", "manner": "aspirated_affricate", "place": "alveolar"},
        "u": {"type": "vowel", "height": "close", "backness": "back"},
    }
    return feature_map.get(symbol, {"type": "unknown"})


def _default_continuation_bias(symbol: str, features: dict[str, str]) -> float:
    phone_type = features.get("type")
    manner = features.get("manner")
    if phone_type == "silence":
        return 0.08
    if phone_type == "vowel":
        return 0.9
    if manner in {"nasal", "lateral_approximant"}:
        return 0.72
    if manner in {"fricative", "affricate", "aspirated_affricate"}:
        return 0.38
    if symbol in {"w", "j"}:
        return 0.7
    return 0.5


def _normalize_unit_interval(value: object, default: float) -> float:
    if value is None or isinstance(value, bool):
        return max(0.0, min(default, 1.0))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return max(0.0, min(default, 1.0))
    return max(0.0, min(numeric, 1.0))


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _normalize_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _probability_to_logit(value: float) -> float:
    clamped = max(1e-4, min(value, 1.0 - 1e-4))
    return math.log(clamped / (1.0 - clamped))


def _softmax_symbol_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    peak = max(scores.values())
    exp_scores = {symbol: math.exp(score - peak) for symbol, score in scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return {symbol: value / total for symbol, value in exp_scores.items()}


def _sorted_symbol_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _normalize_symbol_scores_frame(
    frame: object,
    score_key: str,
) -> dict[str, float]:
    if isinstance(frame, dict):
        if isinstance(frame.get("candidates"), list):
            normalized: dict[str, float] = {}
            for item in frame["candidates"]:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol", "")).strip()
                if not symbol:
                    continue
                try:
                    normalized[symbol] = float(item.get(score_key, item.get("confidence", 0.0)))
                except (TypeError, ValueError):
                    continue
            return normalized
        nested = frame.get(score_key)
        if isinstance(nested, dict):
            return _normalize_symbol_scores_frame(nested, score_key)
        if isinstance(nested, list):
            return _normalize_symbol_scores_frame({"candidates": nested}, score_key)
        normalized = {}
        for key, value in frame.items():
            if key in {"start_sec", "end_sec", "top_phone", "frame_index"}:
                continue
            if isinstance(value, bool):
                continue
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized
    if isinstance(frame, list):
        return _normalize_symbol_scores_frame({"candidates": frame}, score_key)
    return {}


def _candidate_payload_from_scores(
    scores: dict[str, float],
    *,
    score_key: str,
) -> list[dict[str, float | str]]:
    items: list[dict[str, float | str]] = []
    for symbol, value in _sorted_symbol_scores(scores):
        items.append({"symbol": symbol, score_key: round(value, 3)})
    return items


def _normalize_trace_payload(
    payload: object,
    *,
    score_key: str,
    default_source: str,
) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        frames_payload = payload.get("frames")
        if isinstance(frames_payload, list):
            trace_payload = {str(key): payload[key] for key in payload}
            trace_payload.setdefault("source", default_source)
            return trace_payload
        normalized = _normalize_symbol_scores_frame(payload, score_key=score_key)
        if normalized:
            return {
                "source": default_source,
                "frames": [{"candidates": _candidate_payload_from_scores(normalized, score_key=score_key)}],
            }
        return None
    if isinstance(payload, list):
        normalized = _normalize_symbol_scores_frame(payload, score_key=score_key)
        if normalized:
            return {
                "source": default_source,
                "frames": [{"candidates": _candidate_payload_from_scores(normalized, score_key=score_key)}],
            }
    return None


def _merge_text_list(target: list[str], payload: object) -> list[str]:
    items = list(target)
    if not isinstance(payload, list):
        return items
    for entry in payload:
        text = str(entry).strip()
        if text and text not in items:
            items.append(text)
    return items


def _normalize_float_mapping(
    payload: object,
    *,
    keys: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, float] = {}
    for key in keys:
        value = payload.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized

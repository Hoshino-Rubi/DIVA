"""Model-free candidate-4 completion selection and Jyutping conversion.

Preserves the frozen completion policy, ``jyutping_acoustic.v1`` diagnostics,
and fixed-raw projection. No model, audio, reference, or training access.
Phones are segmental evidence only: missing tone, timing, and acoustic
repetition evidence are never inferred. Repeated tokens are never removed.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
import unicodedata
from itertools import groupby
from pathlib import Path
from typing import Any, Iterator


__all__ = ["completion_gate", "choose_prediction", "convert_jyutping_output", "project"]

_GATE = {"status": "ok", "eos_observed": True, "truncated_or_incomplete": False,
         "truncation_status": "complete_with_eos"}
_RULES = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "jyutping_acoustic_ipa.json")
    .read_text(encoding="utf-8")
)
_SYLLABLE = re.compile(r"([A-Za-z]+)([0-9]*)")
_TONED_RUN = re.compile(r"(?:[A-Za-z]+[1-6])+")
_TONED_SYLLABLE = re.compile(r"[A-Za-z]+[1-6]")


def completion_gate(raw: dict[str, Any] | None) -> bool:
    """An empty decoded sequence is legal; absent/unknown flags never pass."""
    return (isinstance(raw, dict) and raw.get("status") == "ok"
            and raw.get("eos_observed") is True
            and raw.get("truncated_or_incomplete") is False
            and raw.get("truncation_status") == "complete_with_eos")


def choose_prediction(
    primary_raw: dict[str, Any] | None, base_raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Select by completion only, preserving the frozen routing sidecar.

    ``selected_model=None`` and ``reason=no_complete_prediction`` explicitly
    signal failure. The caller must emit the frozen ``<UNK:completion>``
    failure placeholder, never project a rejected decode or assume silence.
    """
    primary_ok, base_ok = completion_gate(primary_raw), completion_gate(base_raw)
    return {"selected_model": "small" if primary_ok else "base" if base_ok else None,
            "reason": "primary_complete" if primary_ok else "fallback_base_complete" if base_ok else "no_complete_prediction",
            "completion_gate": {model: {"passed": passed,
                                        "observed": {key: deepcopy((raw or {}).get(key)) for key in _GATE}}
                                for model, raw, passed in (("small", primary_raw, primary_ok),
                                                           ("base", base_raw, base_ok))}}


def _character_kind(char: str) -> str:
    if char.isspace():
        return "space"
    if char.isalnum() or char == "_" or unicodedata.category(char).startswith("M"):
        return "word"
    return "other"


def _native_tokens(text: str) -> Iterator[tuple[str, int, int]]:
    groups = groupby(enumerate(text), key=lambda pair: _character_kind(pair[1]))
    for kind, group in groups:
        positions = list(group)
        if kind == "space":
            continue
        start = positions[0][0]
        native = "".join(char for _, char in positions)
        if _TONED_RUN.fullmatch(native):
            for match in _TONED_SYLLABLE.finditer(native):
                yield match[0], start + match.start(), start + match.end()
        else:
            yield native, start, start + len(native)


def _unknown(item: dict[str, Any], reason: str, evidence: Any) -> None:
    item["unknown"].append({"reason": reason, "evidence": evidence})


def _candidates(syllable: str) -> list[tuple[str, str, bool]]:
    if syllable in _RULES["syllabic_nasals"]:
        return [("", syllable, True)]
    return [
        (onset, syllable[len(onset):], False)
        for onset in _RULES["onsets"]
        if syllable.startswith(onset)
        and syllable[len(onset):] in _RULES["rimes"]
    ]


def _variant(phones: list[str], component: str) -> dict[str, Any]:
    return {
        "IPA": "".join(phones),
        "phones": phones,
        "component": component,
        "source": "jyutping_scheme",
    }


def _map_syllable(item: dict[str, Any], candidate: tuple[str, str, bool]) -> None:
    onset, rime, nasal = candidate
    initial = _RULES["onsets"][onset]
    final = _RULES["syllabic_nasals" if nasal else "rimes"][rime]
    onset_phones = [initial["phone"]] if initial["phone"] else []
    final_phones = list(final["phones"])
    item.update(
        status="mapped",
        parts={"onset": onset, "rime": rime, "syllabic_nasal": nasal},
        phones=onset_phones + final_phones,
        canonicalIPA="".join(onset_phones + final_phones),
        sourceIPA={"pycantonese": initial["sourceIPA"] + final["sourceIPA"]},
    )
    for phone in initial.get("variants", []):
        item["phonetic_variants"].append(_variant([phone] + final_phones, "onset"))
    for phones in final.get("variants", []):
        item["phonetic_variants"].append(_variant(onset_phones + list(phones), "rime"))


def _convert_token(native: str, start: int, end: int) -> dict[str, Any]:
    item: dict[str, Any] = {
        "native_syllable": native,
        "span": {"start_char": start, "end_char": end},
        "normalized": None,
        "status": "unknown",
        "canonicalIPA": None,
        "phones": [],
        "parts": None,
        "tone": "unknown",
        "sourceIPA": {},
        "phonetic_variants": [],
        "acoustic_repetition_evidence": None,
        "unknown": [],
    }
    match = _SYLLABLE.fullmatch(native)
    if match is None:
        reason = "isolated_number" if native.isnumeric() else "non_jyutping_token"
        _unknown(item, reason, native)
        return item
    syllable, tone = match[1].lower(), match[2]
    item["normalized"] = syllable + tone
    if tone and tone not in {"1", "2", "3", "4", "5", "6"}:
        _unknown(item, "invalid_tone", tone)
        return item
    if tone:
        item["tone"] = int(tone)
    else:
        _unknown(item, "tone_absent", native)
    candidates = _candidates(syllable)
    if len(candidates) != 1:
        reason = "ambiguous_syllable" if candidates else "unsupported_or_unseparated_syllable"
        _unknown(item, reason, {"native": native, "candidates": candidates})
        return item
    _map_syllable(item, candidates[0])
    return item


def _mark_repetitions(items: list[dict[str, Any]]) -> None:
    occurrences: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        if item["canonicalIPA"] is not None:
            occurrences.setdefault(item["normalized"], []).append(index)
    for indices in occurrences.values():
        if len(indices) > 1:
            for index in indices:
                items[index]["acoustic_repetition_evidence"] = "unknown"
                _unknown(items[index], "acoustic_repetition_unresolved", list(indices))


def convert_jyutping_output(text: str) -> dict[str, Any]:
    """Return lossless token diagnostics, with no model calls or invented times.

    ``mapped`` describes segmental conversion only; missing tone stays unknown.
    Repeated syllables retain mapped status and deterministic IPA; only their
    acoustic repetition evidence is unknown. Unknown
    punctuation/non-Jyutping tokens also occupy slots in ``syllables``.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    items = [_convert_token(*token) for token in _native_tokens(text)]
    _mark_repetitions(items)
    counts = {
        "tokens": len(items),
        "mapped": sum(item["status"] == "mapped" for item in items),
        "unknown": sum(item["status"] == "unknown" for item in items),
        "unknown_tone": sum(item["tone"] == "unknown" for item in items),
    }
    status = "mapped"
    if not items or counts["unknown"] == len(items):
        status = "unknown"
    elif any(item["unknown"] for item in items):
        status = "partial"
    return {
        "schema_version": "jyutping_acoustic.v1",
        "mapping_version": _RULES["schema_version"],
        "mapping_sources": dict(_RULES["sources"]),
        "input": text,
        "status": status,
        "counts": counts,
        "syllables": items,
        "unknown": [] if items else [{"reason": "empty_output", "evidence": text}],
    }


def project(text: str) -> dict[str, Any]:
    """Return the frozen phone/syllable projection and full diagnostics.

    Unicode P* punctuation is retained separately. Every other unmapped token
    contributes one ``<UNK:token_index>`` at its original sequence position.
    """
    converted = convert_jyutping_output(text)
    syllables, phones, unknown, punctuation = [], [], [], []
    for index, item in enumerate(converted["syllables"]):
        native = item["native_syllable"]
        if item["phones"]:
            mapped = {"native": native, "normalized": item["normalized"], "phones": item["phones"],
                      "canonicalIPA": item["canonicalIPA"], "status": item["status"]}
        elif native and all(unicodedata.category(char).startswith("P") for char in native):
            punctuation.append({"index": index, "native": native})
            continue
        else:
            marker = f"<UNK:{index}>"
            mapped = {"native": native, "normalized": item["normalized"], "phones": [marker],
                      "canonicalIPA": marker, "status": "unknown"}
            unknown.append(marker)
        syllables.append(mapped)
        phones.extend(mapped["phones"])
    return {"syllables": syllables, "scored_tokens": [item["canonicalIPA"] for item in syllables],
            "phones": phones, "unknown_tokens": unknown, "punctuation": punctuation,
            "conversion": converted}

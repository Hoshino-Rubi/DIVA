from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from importlib import resources
import re
from typing import Any


ONSET_ORDER = (
    "gw",
    "kw",
    "ng",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "z",
    "c",
    "s",
    "j",
    "w",
)
CODAS = ("ng", "m", "n", "p", "t", "k")
NUCLEUS_ORDER = (
    "eoi",
    "yu",
    "oe",
    "eo",
    "aa",
    "ai",
    "au",
    "ei",
    "oi",
    "ou",
    "ui",
    "iu",
    "a",
    "e",
    "i",
    "o",
    "u",
    "m",
    "ng",
)
ROMANIZATION_NORMALIZATIONS = (
    ("yue", "jyu"),
    ("yu", "jyu"),
    ("yun", "jyun"),
    ("yut", "jyut"),
)


class JyutpingParseError(ValueError):
    """Raised when a Jyutping syllable cannot be mapped to phone evidence."""


class MikuV4CPhonemeMapper:
    """Map Cantonese internal phones to Miku V4C phoneme candidates.

    The returned payload is phoneme evidence for a draft/export consumer. It is
    not a recognizer truth layer and does not use X-SAMPA as an intermediate.
    """

    def __init__(
        self,
        phoneme_map: Mapping[str, Mapping[str, object]],
        empirical: Mapping[str, object] | None = None,
    ) -> None:
        self.phoneme_map = {str(key): dict(value) for key, value in phoneme_map.items()}
        self._empirical = dict(empirical or {})

    @classmethod
    def for_cantonese(cls) -> "MikuV4CPhonemeMapper":
        payload = _load_cantonese_resource_json("miku_v4c_phoneme_map.json")
        empirical_payload = _load_cantonese_resource_json("miku_v4c_empirical_phonemes.json")
        raw_map = payload.get("phoneme_map", {})
        if not isinstance(raw_map, dict):
            raw_map = {}
        empirical = empirical_payload.get("empirical_audit", {})
        if not isinstance(empirical, dict):
            empirical = {}
        return cls(raw_map, empirical=empirical)

    def empirical_audit(self) -> dict[str, object]:
        """Return the empirical Miku V4C VSQX note-p audit data."""

        return deepcopy(self._empirical)

    def map_jyutping(self, jyutping: str) -> dict[str, Any]:
        parsed = parse_jyutping(jyutping)
        ipa: list[str] = []
        v4c: list[str] = []
        notes: list[str] = list(parsed["notes"])
        for role, phone in parsed["role_phones"]:
            key = _map_key(role, phone)
            entry = self.phoneme_map.get(key)
            if entry is None:
                ipa.append(phone)
                v4c.append(phone)
                notes.append(f"no V4C mapping for {key}; preserved internal phone as fallback")
                continue
            ipa.extend(_string_list(entry.get("ipa")))
            v4c.extend(_string_list(entry.get("v4c")))
            notes.extend(_string_list(entry.get("notes")))
        notes.extend(self._empirical_notes_for_v4c(v4c))
        return {
            "jyutping": parsed["jyutping"],
            "tone": parsed["tone"],
            "internal_phones": list(parsed["internal_phones"]),
            "ipa": ipa,
            "v4c_phonemes": v4c,
            "notes": _dedupe_notes(notes),
        }

    def map_internal(self, phones: list[str]) -> tuple[list[str], list[str]]:
        mapped: list[str] = []
        notes: list[str] = []
        for index, phone in enumerate(phones):
            role = _guess_role(index, phone, phones)
            entry = self.phoneme_map.get(_map_key(role, phone), self.phoneme_map.get(phone))
            if entry is None:
                mapped.append(phone)
                notes.append(f"no V4C mapping for {phone}; preserved internal phone as fallback")
                continue
            mapped.extend(_string_list(entry.get("v4c")))
            notes.extend(_string_list(entry.get("notes")))
        notes.extend(self._empirical_notes_for_v4c(mapped))
        return mapped, _dedupe_notes(notes)

    def _empirical_notes_for_v4c(self, v4c: list[str]) -> list[str]:
        phrase = " ".join(v4c)
        if not phrase:
            return []

        notes: list[str] = []
        invalid = _empirical_invalid_phrases(self._empirical)
        if phrase in invalid or (phrase == "@N" and "bare @N" in invalid):
            notes.append(f"empirical V4C audit marks {phrase!r} invalid in VSQX note p context")

        validated = _empirical_validated_phrases(self._empirical)
        if phrase in validated:
            notes.append(f"empirical V4C audit validates {phrase!r} in VSQX note p context")
        return notes


def parse_jyutping(raw: str) -> dict[str, Any]:
    normalized = str(raw).strip().lower().replace(" ", "")
    if not normalized:
        raise JyutpingParseError("empty Jyutping syllable")

    normalized, normalization_notes = _normalize_romanization(normalized)
    match = re.fullmatch(r"([a-z]+)([1-6]?)", normalized)
    if match is None:
        raise JyutpingParseError(f"invalid Jyutping syllable: {raw!r}")
    body = match.group(1)
    tone = match.group(2) or None

    notes = list(normalization_notes)
    if body in {"m", "ng"}:
        role_phones = [("syllabic", body)]
        notes.append(f"syllabic {body} handled as a complete Cantonese syllable")
        return _parsed_payload(normalized, tone, role_phones, notes)

    onset, rest = _consume_onset(body)
    nucleus, coda = _split_final(rest)
    if not nucleus:
        raise JyutpingParseError(f"no nucleus found in Jyutping syllable: {raw!r}")

    role_phones: list[tuple[str, str]] = []
    if onset:
        role_phones.append(("onset", onset))
    role_phones.append(("nucleus", nucleus))
    if coda:
        role_phones.append(("coda", coda))

    if onset in {"gw", "kw"}:
        notes.append(f"{onset} onset preserved as labialized velar evidence")
    if onset in {"n", "l"}:
        notes.append("n/l onset may need dialect or singer-specific variant review")
    if coda in {"p", "t", "k"}:
        notes.append(f"entering-tone coda -{coda} preserved as unreleased stop evidence")
    if nucleus in {"oe", "eo", "eoi", "yu"}:
        notes.append(f"{nucleus} requires V4C approximation strategy")
    return _parsed_payload(normalized, tone, role_phones, notes)


def jyutping_to_internal_phones(jyutping: str) -> tuple[list[str], list[str]]:
    parsed = parse_jyutping(jyutping)
    return list(parsed["internal_phones"]), list(parsed["notes"])


def internal_to_v4c(phones: list[str]) -> tuple[list[str], list[str]]:
    return MikuV4CPhonemeMapper.for_cantonese().map_internal(phones)


def _load_cantonese_resource_json(filename: str) -> dict[str, object]:
    with resources.files("any2xsampa.cantonese.resources").joinpath(filename).open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return {}


def _guess_role(index: int, phone: str, phones: list[str]) -> str:
    if len(phones) == 1 and phone in {"m", "ng"}:
        return "syllabic"
    if index == 0 and phone in set(ONSET_ORDER):
        return "onset"
    if index == len(phones) - 1 and phone in set(CODAS):
        return "coda"
    return "nucleus"


def _parsed_payload(
    jyutping: str,
    tone: str | None,
    role_phones: list[tuple[str, str]],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "jyutping": jyutping,
        "tone": tone,
        "role_phones": role_phones,
        "internal_phones": [phone for _, phone in role_phones],
        "notes": _dedupe_notes(notes),
    }


def _normalize_romanization(value: str) -> tuple[str, list[str]]:
    for source, target in ROMANIZATION_NORMALIZATIONS:
        if value.startswith(source):
            normalized = target + value[len(source) :]
            if normalized != value:
                return normalized, [f"normalized non-Jyutping {source}* spelling to {target}* for yue/yun/yut strategy"]
    return value, []


def _consume_onset(body: str) -> tuple[str | None, str]:
    for onset in ONSET_ORDER:
        if body.startswith(onset):
            rest = body[len(onset) :]
            if rest:
                return onset, rest
    return None, body


def _split_final(rest: str) -> tuple[str, str | None]:
    for coda in CODAS:
        if rest.endswith(coda) and len(rest) > len(coda):
            nucleus = rest[: -len(coda)]
            if nucleus in NUCLEUS_ORDER:
                return nucleus, coda
    if rest in NUCLEUS_ORDER:
        return rest, None
    for nucleus in sorted(NUCLEUS_ORDER, key=len, reverse=True):
        if rest.startswith(nucleus):
            tail = rest[len(nucleus) :]
            return nucleus, tail or None
    return rest, None


def _map_key(role: str, phone: str) -> str:
    if role == "onset":
        return f"onset_{phone}"
    if role == "coda":
        return f"coda_{phone}"
    return phone


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _empirical_invalid_phrases(empirical: Mapping[str, object]) -> set[str]:
    phrases: set[str] = set()
    for audit in empirical.values():
        if not isinstance(audit, Mapping):
            continue
        phrases.update(_string_list(audit.get("invalid_in_vsqx_note_p_context")))
    return phrases


def _empirical_validated_phrases(empirical: Mapping[str, object]) -> set[str]:
    phrases: set[str] = set()
    for audit in empirical.values():
        if not isinstance(audit, Mapping):
            continue
        validated = audit.get("validated_in_vsqx_note_p_context")
        if not isinstance(validated, Mapping):
            continue
        for values in validated.values():
            phrases.update(_string_list(values))
    return phrases


def _dedupe_notes(notes: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for note in notes:
        if note and note not in seen:
            seen.add(note)
            result.append(note)
    return result


__all__ = [
    "JyutpingParseError",
    "MikuV4CPhonemeMapper",
    "internal_to_v4c",
    "jyutping_to_internal_phones",
    "parse_jyutping",
]

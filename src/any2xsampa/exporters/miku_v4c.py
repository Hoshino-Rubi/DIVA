"""Project *whole* Cantonese rimes onto an allowlisted Miku V4C syllable.

The reference token n-grams document the source material. Recombining those
n-grams would admit strings that never formed a playable base syllable.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

_CONFIG = Path(__file__).resolve().parents[1] / "config"
POLICY = json.loads((_CONFIG / "miku_v4c_syllables_v3.json").read_text(encoding="utf-8"))
_IPA = json.loads((_CONFIG / "jyutping_acoustic_ipa.json").read_text(encoding="utf-8"))
_BASES = frozenset(POLICY["base_syllables"])
_TAILS = frozenset(POLICY["allowed_stop_tails"])
_F = POLICY["final_features"]
_O = POLICY["onset_features"]


def _sequence(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(value.split())
    if not isinstance(value, Sequence) or not all(
        isinstance(token, str) and token and not any(c.isspace() for c in token)
        for token in value
    ):
        return ()
    return tuple(value)


def is_legal(tokens: str | Sequence[str]) -> bool:
    parts = _sequence(tokens)
    if parts and parts[-1] in _TAILS:
        parts = parts[:-1]
    return bool(parts) and " ".join(parts) in _BASES


def require_legal(tokens: str | Sequence[str]) -> None:
    if not is_legal(tokens):
        raise ValueError(f"Miku V4C syllable outside the playable whitelist: {tokens!r}")


def _onset_cost(target: str, other: str) -> float:
    if target == other:
        return 0.0
    if not target or not other:
        return 5.0
    manner, place, aspiration = _O[target]
    m, p, a = _O[other]
    return 2.2 * (manner != m) + 1.8 * (place != p) + 2.4 * (aspiration != a) + .2


def _final_cost(target: str, other: str) -> float:
    if target == other:
        return 0.0
    height, backness, rounded, coda = _F[target]
    h, b, r, c = _F[other]
    return .7 * abs(height - h) + .8 * abs(backness - b) + 1.0 * (rounded != r) + 2.2 * (coda != c) + .35


_BASE_PAIRS = tuple((s.split()[0] if " " in s else "", s.split()[-1], s) for s in sorted(_BASES))


@lru_cache(maxsize=2048)
def _project(onset: str, final: str) -> tuple[str, ...]:
    proxy = POLICY["glide_finals"].get(onset, {}).get(final, POLICY["finals"][final])
    vowel, *tail = proxy.split()
    initial = POLICY["onsets"][onset]
    desired = " ".join(x for x in (initial, vowel) if x)
    if desired not in _BASES:
        # Only distances between whole PHG2 finals are defined. Break feature
        # ties by explicit token identity then lexicographic order, never by
        # individual song frequency.
        desired = min(_BASE_PAIRS, key=lambda p: (
            _onset_cost(initial, p[0]) + _final_cost(vowel, p[1]),
            initial != p[0], vowel != p[1], p[2],
        ))[2]
    result = (*desired.split(), *tail)
    require_legal(result)
    return result


def project_jyutping(syllable: str) -> list[str]:
    """Map a decoded closed-set candidate, accepting optional tone 1–6."""
    match = re.fullmatch(r"([a-z]+)[1-6]?", syllable.strip().lower())
    if not match:
        raise ValueError(f"Invalid Jyutping syllable: {syllable!r}")
    word = match[1]
    if word in _IPA["syllabic_nasals"]:
        return list(_project("", word))
    for onset in sorted(POLICY["onsets"], key=len, reverse=True):
        final = word[len(onset):]
        if word.startswith(onset) and final in _IPA["rimes"]:
            return list(_project(onset, final))
    raise ValueError(f"Unknown Cantonese final in {syllable!r}")


def _ipa_key(ipa: str) -> str:
    # Remove notation variants only. A vowel cluster is always looked up as
    # an entire rime. Length neutralization is a secondary lookup below.
    return unicodedata.normalize("NFC", ipa).translate(str.maketrans("", "", " ̚͜͡ˈˌ/"))


_IPA_FINALS = {}
for _rime, _data in _IPA["rimes"].items():
    for _phones in (_data["phones"], *_data.get("variants", [])):
        _IPA_FINALS.setdefault(_ipa_key("".join(_phones)), _rime)
    _IPA_FINALS.setdefault(_ipa_key(_data["sourceIPA"]), _rime)
_IPA_FINALS.update(POLICY["ipa_final_aliases"])
_IPA_ONSETS = dict(POLICY["ipa_onset_aliases"])
for _jyutping, _data in _IPA["onsets"].items():
    for _symbol in (_data["phone"], _data["sourceIPA"], *_data.get("variants", [])):
        _IPA_ONSETS[_ipa_key(_symbol)] = _jyutping
for _jyutping, _symbol in POLICY["onsets"].items():
    if _symbol:
        _IPA_ONSETS.setdefault(_symbol, _jyutping)
_SORTED_IPA_ONSETS = tuple(sorted(_IPA_ONSETS, key=len, reverse=True))


def project_ipa(phones: Sequence[str] | str) -> list[str]:
    """Map a single recognised syllable. Missing evidence calls for decoding."""
    raw = phones if isinstance(phones, str) else "".join(phones)
    key = _ipa_key(raw)
    if not key:
        raise ValueError("Missing acoustic syllable")
    if key in ("m", "ŋ"):
        key += "̩"
    for onset in _SORTED_IPA_ONSETS:
        if not key.startswith(onset):
            continue
        final = key[len(onset):]
        rime = _IPA_FINALS.get(final) or _IPA_FINALS.get(final.replace("ː", ""))
        if rime is not None and (not onset or rime not in _IPA["syllabic_nasals"]):
            return list(_project(_IPA_ONSETS[onset], rime))
    raise ValueError(f"No acoustic final for V4C projection: {phones!r}")

def reference_inventory(files: Sequence[Path]) -> dict:
    """Rebuild the observational part of the policy offline, without granting legality."""
    from collections import Counter
    from xml.etree import ElementTree as ET

    counts: dict[str, int] = {}
    histograms = [Counter(), Counter(), Counter()]
    for path in sorted(files):
        root = ET.fromstring(path.read_text(encoding="utf-8-sig"))
        notes = root.findall(".//{*}vsPart/{*}note")
        counts[path.name] = len(notes)
        for note in notes:
            tokens = (note.findtext("{*}p") or "").split()
            if not tokens:
                raise ValueError(f"Missing note phonemes in {path.name}")
            histograms[0][" ".join(tokens)] += 1
            for size, histogram in ((2, histograms[1]), (3, histograms[2])):
                histogram.update(" ".join(tokens[i:i+size]) for i in range(len(tokens)-size+1))
    return dict(source_note_counts=counts, observed_sequences=dict(histograms[0]),
                within_note_bigrams=dict(histograms[1]), within_note_trigrams=dict(histograms[2]))

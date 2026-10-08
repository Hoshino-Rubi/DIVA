"""Cantonese lyric-to-phone lattice reconstruction.

Phones and IPA remain the truth layer.  Optional annotated VSQX evidence can
add target-export candidates but never replaces the canonical phone fields.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from any2xsampa.cantonese.types import CantoneseCandidate, CantoneseSyllableResolution
from any2xsampa.vocaloid.phoneme_mapper import MikuV4CPhonemeMapper

RESOLVER_SCHEMA_VERSION = "cantonese_phoneme_lattice_v1"


def load_lexicon_file(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw = payload.get("entries", payload) if isinstance(payload, dict) else {}
    return _normalize_lexicon(raw)


def load_phoneme_evidence_file(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("phoneme evidence must be a JSON object")
    return payload


def _normalize_lexicon(value: object) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, list[str]] = {}
    for key, raw_values in value.items():
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        result[str(key)] = [str(item).strip() for item in values if str(item).strip()]
    return result


def _default_lexicon() -> dict[str, list[str]]:
    with resources.files("any2xsampa.cantonese.resources").joinpath("jyutping_lexicon.recovered.json").open(
        "r", encoding="utf-8"
    ) as stream:
        payload = json.load(stream)
    return _normalize_lexicon(payload.get("entries", {}))


class _EvidenceCandidate:
    """Candidate-like serializer for target-only evidence extensions."""

    __slots__ = ("_payload",)

    def __init__(self, canonical: CantoneseCandidate, *, tokens: list[str], score: float, notes: list[str], extras: dict[str, Any]) -> None:
        payload = canonical.to_dict()
        payload.update({
            "v4c_phonemes": list(tokens),
            "score": round(score, 3),
            "source": "annotated_vsqx_evidence",
            "notes": list(notes),
        })
        payload.update(extras)
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class CantonesePhonemeResolver:
    def __init__(
        self,
        lexicon: Mapping[str, object] | None = None,
        *,
        phoneme_evidence: Mapping[str, object] | None = None,
        mapper: MikuV4CPhonemeMapper | None = None,
    ) -> None:
        self.lexicon = _normalize_lexicon(lexicon) if lexicon is not None else _default_lexicon()
        self.phoneme_evidence = dict(phoneme_evidence or {})
        self.mapper = mapper or MikuV4CPhonemeMapper.for_cantonese()

    def resolve_syllable(
        self,
        text: str,
        *,
        index: int = 0,
        jyutping_override: str | None = None,
    ) -> CantoneseSyllableResolution:
        override = jyutping_override.strip() if jyutping_override else None
        pronunciations = [override] if override else list(self.lexicon.get(text, []))
        candidates: list[CantoneseCandidate] = []
        for rank, jyutping in enumerate(pronunciations):
            mapped = self.mapper.map_jyutping(jyutping)
            candidates.append(CantoneseCandidate(
                jyutping=str(mapped["jyutping"]),
                internal_phones=list(mapped["internal_phones"]),
                ipa=list(mapped["ipa"]),
                v4c_phonemes=list(mapped["v4c_phonemes"]),
                score=round(1.0 - rank * 0.05, 6),
                source="jyutping_override" if override else "recovered_lexicon",
                notes=list(mapped.get("notes", [])),
            ))
        notes: list[str] = []
        if not candidates:
            notes.append("no recovered Jyutping entry; no phone candidate inferred")
        resolution = CantoneseSyllableResolution(index=index, text=text, override=override, candidates=candidates, notes=notes)
        self._add_character_evidence(resolution)
        return resolution

    def resolve_text(
        self,
        text: str,
        *,
        jyutping_override: str | list[str] | None = None,
    ) -> dict[str, Any]:
        units = [character for character in str(text) if not character.isspace()]
        if isinstance(jyutping_override, str):
            overrides = [item for item in jyutping_override.split() if item]
        elif isinstance(jyutping_override, list):
            overrides = [str(item).strip() for item in jyutping_override if str(item).strip()]
        else:
            overrides = []
        if overrides and len(overrides) != len(units):
            raise ValueError(f"Jyutping override count {len(overrides)} does not match lyric unit count {len(units)}")
        syllables = [
            self.resolve_syllable(unit, index=index, jyutping_override=overrides[index] if overrides else None)
            for index, unit in enumerate(units)
        ]
        payload: dict[str, Any] = {
            "schema_version": RESOLVER_SCHEMA_VERSION,
            "text": str(text),
            "dialect": "cantonese",
            "syllables": [syllable.to_dict() for syllable in syllables],
            "user_jyutping_required": False,
            "truth_layer": "phones_ipa_articulatory_features",
            "xsampa_role": "export_only_not_internal_truth",
        }
        self._add_phrase_evidence(str(text), syllables, payload)
        character_rows = []
        for syllable in syllables:
            sequences = [
                " ".join(candidate.to_dict()["v4c_phonemes"])
                for candidate in syllable.candidates
                if candidate.to_dict().get("source") == "annotated_vsqx_evidence"
            ]
            if sequences:
                character_rows.append({"character": syllable.text, "phoneme_sequences": sequences})
        if character_rows:
            payload["target_phoneme_character_evidence"] = character_rows
            payload["syllables"] = [syllable.to_dict() for syllable in syllables]
        return payload

    def _add_character_evidence(self, syllable: CantoneseSyllableResolution) -> None:
        characters = self.phoneme_evidence.get("characters", {})
        rows = characters.get(syllable.text, []) if isinstance(characters, Mapping) else []
        if not isinstance(rows, list) or not syllable.candidates:
            return
        canonical = syllable.candidates[0]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            tokens = row.get("phoneme_tokens", [])
            if not isinstance(tokens, list) or not tokens:
                continue
            syllable.candidates.append(_EvidenceCandidate(
                canonical,
                tokens=[str(token) for token in tokens],
                score=float(row.get("candidate_score", 0.5)),
                notes=["target-only annotated VSQX evidence; canonical phone truth preserved"],
                extras={"target_only_evidence": True, "evidence_quality_unscored": True},
            ))  # type: ignore[arg-type]

    def _add_phrase_evidence(
        self,
        text: str,
        syllables: list[CantoneseSyllableResolution],
        payload: dict[str, Any],
    ) -> None:
        raw_phrases = self.phoneme_evidence.get("phrases", [])
        if not isinstance(raw_phrases, list):
            return
        groups: list[dict[str, Any]] = []
        for row in raw_phrases:
            if not isinstance(row, Mapping) or str(row.get("lyrics", "")) != text:
                continue
            by_character = row.get("phonemes_by_character", [])
            if not isinstance(by_character, list) or len(by_character) != len(syllables):
                continue
            evidence_group = str(row.get("evidence_id") or row.get("segment_id") or "phrase-evidence")
            for syllable, tokens in zip(syllables, by_character):
                if not syllable.candidates or not isinstance(tokens, list):
                    continue
                canonical = syllable.candidates[0]
                syllable.candidates.append(_EvidenceCandidate(
                    canonical,
                    tokens=[str(token) for token in tokens],
                    score=float(row.get("candidate_score", 0.5)),
                    notes=["coherent phrase target evidence; canonical phone truth preserved"],
                    extras={"target_only_evidence": True, "evidence_group": evidence_group},
                ))  # type: ignore[arg-type]
            groups.append({"evidence_group": evidence_group, "lyrics": text, "candidate_score": row.get("candidate_score")})
        if groups:
            payload["target_phoneme_evidence_groups"] = groups
            payload["syllables"] = [syllable.to_dict() for syllable in syllables]


__all__ = ["CantonesePhonemeResolver", "RESOLVER_SCHEMA_VERSION", "load_lexicon_file", "load_phoneme_evidence_file"]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CantoneseCandidate:
    jyutping: str
    internal_phones: list[str]
    ipa: list[str]
    v4c_phonemes: list[str]
    score: float
    source: str
    notes: list[str] = field(default_factory=list)
    candidate_role: str | None = None
    failure_kind_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "jyutping": self.jyutping,
            "internal_phones": list(self.internal_phones),
            "ipa": list(self.ipa),
            "v4c_phonemes": list(self.v4c_phonemes),
            "score": round(self.score, 3),
            "source": self.source,
            "notes": list(self.notes),
        }
        if self.candidate_role:
            payload["candidate_role"] = self.candidate_role
        if self.failure_kind_hint:
            payload["failure_kind_hint"] = self.failure_kind_hint
        return payload


@dataclass(frozen=True, slots=True)
class CantoneseSyllableResolution:
    index: int
    text: str
    override: str | None
    candidates: list[CantoneseCandidate]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "override": self.override,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "notes": list(self.notes),
        }

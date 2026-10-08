from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CorpusPhone:
    """Unified phone-level open-corpus contract.

    ``phone`` is the corpus/IPA/articulatory truth token. X-SAMPA is an
    export-only concern and is intentionally not part of this contract.
    """

    phone: str
    ipa: str | None = None
    timing: dict[str, float] = field(default_factory=dict)
    note_context: dict[str, Any] = field(default_factory=dict)
    lyrics: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    model_output: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CorpusUtterance:
    """Unified utterance-level open-corpus contract."""

    utterance_id: str
    phones: list[CorpusPhone]
    ipa: list[str | None] = field(default_factory=list)
    timing: list[dict[str, float]] = field(default_factory=list)
    note_context: list[dict[str, Any]] = field(default_factory=list)
    lyrics: str | list[str] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    model_output: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "phones": [phone.to_dict() for phone in self.phones],
            "ipa": list(self.ipa),
            "timing": [dict(item) for item in self.timing],
            "note_context": [dict(item) for item in self.note_context],
            "lyrics": self.lyrics,
            "provenance": dict(self.provenance),
            "model_output": self.model_output,
        }

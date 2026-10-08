from __future__ import annotations

from pathlib import Path

from any2xsampa.core.interfaces import PhoneNormalizer
from any2xsampa.core.io import load_json
from any2xsampa.core.models import PhoneSegment


class ConfigurablePhoneNormalizer(PhoneNormalizer):
    def __init__(self, alias_config_path: Path) -> None:
        self.aliases = load_json(alias_config_path)

    def normalize(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        for segment in segments:
            top_symbol = segment.candidate_phones[0].symbol if segment.candidate_phones else "sil"
            segment.normalized_ipa = self.aliases.get(top_symbol, top_symbol)
            notes = segment.metadata.setdefault("normalization_notes", [])
            if segment.normalized_ipa != top_symbol:
                notes.append(f"alias normalized: {top_symbol} -> {segment.normalized_ipa}")
            else:
                notes.append("no alias normalization applied")
            segment.metadata["normalized_from"] = top_symbol
        return segments

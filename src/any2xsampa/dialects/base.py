from __future__ import annotations

from pathlib import Path

from any2xsampa.core.interfaces import DialectPlugin
from any2xsampa.core.io import load_json
from any2xsampa.core.models import DialectConstraintResult, PhoneSegment


class InventoryDialectPlugin(DialectPlugin):
    def __init__(self, config_path: Path) -> None:
        config = load_json(config_path)
        self.name = str(config["dialect"])
        self._allowed = set(config["allowed_ipa"])
        self._fallback_map = dict(config.get("fallback_map", {}))

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "type": "inventory_constraint"}

    def apply(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        for segment in segments:
            ipa = segment.normalized_ipa or "sil"
            adjusted = self._fallback_map.get(ipa, ipa)
            accepted = adjusted in self._allowed
            notes = []
            if adjusted != ipa:
                notes.append(f"fallback applied: {ipa} -> {adjusted}")
            if accepted:
                notes.append("phone allowed by dialect inventory")
            else:
                notes.append("phone not found in dialect inventory")
            segment.dialect_result = DialectConstraintResult(
                dialect=self.name,
                accepted=accepted,
                adjusted_ipa=adjusted,
                notes=notes,
            )
            segment.metadata["dialect_inventory_hit"] = accepted
            segment.normalized_ipa = adjusted
        return segments

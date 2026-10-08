from __future__ import annotations

from pathlib import Path

from any2xsampa.core.interfaces import XSampaExporter
from any2xsampa.core.io import load_json
from any2xsampa.core.models import PhoneSegment


class JsonXSampaExporter(XSampaExporter):
    def __init__(self, mapping_path: Path) -> None:
        self.mapping = load_json(mapping_path)

    def export(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        for segment in segments:
            ipa = segment.normalized_ipa or "sil"
            segment.xsampa = self.mapping.get(ipa, ipa)
            segment.metadata["xsampa_exported_from"] = ipa
        return segments

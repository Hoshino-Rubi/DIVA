from pathlib import Path

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.exporters.xsampa import JsonXSampaExporter


def test_xsampa_exporter_maps_ipa_symbols() -> None:
    exporter = JsonXSampaExporter(
        Path(__file__).resolve().parents[1] / "src" / "any2xsampa" / "config" / "ipa_to_xsampa.json"
    )
    segments = [
        PhoneSegment(
            time_range=TimeRange(0.0, 0.1),
            candidate_phones=[PhoneCandidate(symbol="ng", confidence=0.8)],
            confidence=0.8,
            normalized_ipa="N",
        )
    ]

    exported = exporter.export(segments)

    assert exported[0].xsampa == "N"

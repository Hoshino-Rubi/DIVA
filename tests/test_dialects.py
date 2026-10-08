from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.dialects.registry import DialectRegistry


def test_dialect_registry_loads_cantonese_and_taigi() -> None:
    registry = DialectRegistry()

    assert registry.get("cantonese").describe()["name"] == "cantonese"
    assert registry.get("taigi").describe()["name"] == "taigi"
    assert registry.get("taiwanese").describe()["name"] == "taigi"


def test_cantonese_dialect_fallback_maps_ts_h_to_inventory_form() -> None:
    registry = DialectRegistry()
    segment = PhoneSegment(
        TimeRange(0.0, 0.1),
        [PhoneCandidate("ts_h", 0.9)],
        0.9,
        normalized_ipa="ts_h",
    )

    [adjusted] = registry.get("cantonese").apply([segment])

    assert adjusted.normalized_ipa == "ts"
    assert adjusted.dialect_result is not None
    assert adjusted.dialect_result.accepted is True
    assert adjusted.metadata["dialect_inventory_hit"] is True
    assert "fallback applied: ts_h -> ts" in adjusted.dialect_result.notes


def test_taiwanese_alias_uses_taigi_fallback_map() -> None:
    registry = DialectRegistry()
    segment = PhoneSegment(
        TimeRange(0.0, 0.1),
        [PhoneCandidate("ts", 0.9)],
        0.9,
        normalized_ipa="ts",
    )

    [adjusted] = registry.get("taiwanese").apply([segment])

    assert adjusted.normalized_ipa == "ts_h"
    assert adjusted.dialect_result is not None
    assert adjusted.dialect_result.dialect == "taigi"
    assert adjusted.dialect_result.accepted is True
    assert adjusted.metadata["dialect_inventory_hit"] is True
    assert "fallback applied: ts -> ts_h" in adjusted.dialect_result.notes

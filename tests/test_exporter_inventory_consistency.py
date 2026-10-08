from __future__ import annotations

import json
from pathlib import Path

from any2xsampa.exporters.inventory import (
    AliasCandidate,
    AliasInventoryValidator,
    AliasSelectionContext,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _selection_context() -> AliasSelectionContext:
    return AliasSelectionContext(
        current_role="nucleus",
        note_midi=64,
        segment_duration_ms=80.0,
        timing_bucket="short",
        release_context="short",
        note_context_applied=True,
        note_context_source="test_case",
    )


def _consistency_candidates() -> list[AliasCandidate]:
    return [
        AliasCandidate(alias="a low", alias_type="plain", role="nucleus"),
        AliasCandidate(alias="a power", alias_type="plain", role="nucleus"),
    ]


def test_json_inventory_accepts_direct_midi_note_range_fields(tmp_path: Path) -> None:
    validator = AliasInventoryValidator(
        _write_json(
            tmp_path / "direct_midi_inventory.json",
            {
                "aliases": ["a low", "a high"],
                "alias_metadata": {
                    "a low": {"min_note_midi": 48, "max_note_midi": 59},
                    "a high": {"min_note_midi": 60, "max_note_midi": 71},
                },
            },
        )
    )

    result = validator.validate_candidates(
        [
            AliasCandidate(alias="a low", alias_type="plain", role="nucleus"),
            AliasCandidate(alias="a high", alias_type="plain", role="nucleus"),
        ],
        selection_context=AliasSelectionContext(
            current_role="nucleus",
            note_midi=64,
            note_context_applied=True,
            note_context_source="test_case",
        ),
    )

    assert result.selected_alias == "a high"
    assert result.note_range_match is True
    assert result.selected_alias_metadata is not None
    assert result.selected_alias_metadata["min_note_midi"] == 60
    assert result.selected_alias_metadata["max_note_midi"] == 71
    assert "note_range_match=60-71" in result.ranking_features


def test_json_and_singer_inventory_normalize_behavior_metadata_consistently(tmp_path: Path) -> None:
    json_inventory = _write_json(
        tmp_path / "inventory.json",
        {
            "aliases": ["a low", "a power"],
            "alias_metadata": {
                "a low": {
                    "note_range": "C3..B3",
                    "timing_group": "long",
                    "release_type": "legato",
                },
                "a power": {
                    "subbank": "power",
                    "color": "power",
                    "note_range": {"min_note": "C4", "max_note": "B4"},
                    "timing_group": "tight",
                    "release_type": "staccato",
                },
            },
        },
    )
    singer_root = tmp_path / "singer"
    _write_text(
        singer_root / "character.yaml",
        "\n".join(
            [
                "name: Consistency Singer",
                "note_range: C3..B3",
                "timing_group: long",
                "release_type: legato",
                "subbanks:",
                "  - name: Power",
                "    dir: Power",
                "    color: Power",
                "    note_range: C4..B4",
                "    timing_group: tight",
                "    release_type: staccato",
                "",
            ]
        ),
    )
    _write_text(singer_root / "oto.ini", "a.wav=a low,0,0,0,0,0\n")
    _write_text(singer_root / "Power" / "oto.ini", "a.wav=a power,0,0,0,0,0\n")

    json_result = AliasInventoryValidator(json_inventory).validate_candidates(
        _consistency_candidates(),
        selection_context=_selection_context(),
    )
    singer_result = AliasInventoryValidator(singer_root).validate_candidates(
        _consistency_candidates(),
        selection_context=_selection_context(),
    )

    assert json_result.selected_alias == "a power"
    assert singer_result.selected_alias == "a power"
    assert json_result.note_range_match is True
    assert singer_result.note_range_match is True
    assert json_result.timing_match is True
    assert singer_result.timing_match is True
    assert json_result.release_match is True
    assert singer_result.release_match is True
    assert json_result.selected_alias_metadata == singer_result.selected_alias_metadata
    assert json_result.selected_alias_metadata == {
        "subbank": "power",
        "color": "power",
        "min_note": "c4",
        "max_note": "b4",
        "min_note_midi": 60,
        "max_note_midi": 71,
        "timing": "short",
        "release": "short",
    }

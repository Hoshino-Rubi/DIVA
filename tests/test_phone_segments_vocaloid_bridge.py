from __future__ import annotations

from pathlib import Path

import pytest

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.exporters.engine import MappingEnginePhoneExporter
from any2xsampa.vocaloid import draft_from_phone_segments


def _segment(symbol: str, start: float = 0.0, end: float = 0.5) -> PhoneSegment:
    return PhoneSegment(
        time_range=TimeRange(start, end),
        candidate_phones=[PhoneCandidate(symbol, 0.9)],
        confidence=0.9,
        normalized_ipa=symbol,
        metadata={"checkpoint_sha256": "a" * 64, "backend_kind": "fixture-ctc"},
    )


def _export(segments: list[PhoneSegment]) -> list[PhoneSegment]:
    config = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "any2xsampa"
        / "config"
        / "exporter_vocaloid_like.json"
    )
    return MappingEnginePhoneExporter("vocaloid_like", config).export(segments)


def test_phone_segment_bridge_uses_timing_mapping_and_checkpoint_provenance() -> None:
    draft = draft_from_phone_segments(
        _export([_segment("a", 0.25, 0.75), _segment("n", 0.75, 1.0)]),
        tempo=120.0,
        pitches=[60, 62],
        lyrics=["la", "na"],
    )
    notes = draft.tracks[0].notes
    assert [(note.start_tick, note.duration_ticks) for note in notes] == [
        (240, 480),
        (720, 240),
    ]
    assert [note.phonemes for note in notes] == [["a"], ["n"]]
    assert notes[0].provenance["normalized_ipa"] == "a"
    assert notes[0].provenance["recognizer"]["checkpoint_sha256"] == "a" * 64
    assert draft.provenance["engine_phonemes_are_export_only"] is True


def test_phone_segment_bridge_rejects_unmapped_and_reserved_truth() -> None:
    with pytest.raises(ValueError, match="approved Vocaloid mapping"):
        draft_from_phone_segments(_export([_segment("not-mapped")]))
    reserved = _segment("a")
    reserved.candidate_phones[0].symbol = "<blank>"
    with pytest.raises(ValueError, match="invalid phone/IPA truth"):
        draft_from_phone_segments(_export([reserved]))


def test_phone_segment_bridge_rejects_unsafe_miku_token() -> None:
    segment = _segment("a")
    _export([segment])
    assert segment.engine_export is not None
    segment.engine_export.engine_phone = "i\\"
    with pytest.raises(ValueError, match="silent/unsafe"):
        draft_from_phone_segments([segment])


def test_phone_segment_bridge_maps_ipa_script_g_to_vocaloid_g() -> None:
    segment = _export([_segment("ɡ")])[0]
    assert segment.engine_export is not None
    assert segment.engine_export.engine_phone == "g"
    assert segment.engine_export.unmapped is False
    assert draft_from_phone_segments([segment]).tracks[0].notes[0].phonemes == ["g"]

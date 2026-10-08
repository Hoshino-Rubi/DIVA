from __future__ import annotations

from pathlib import Path

import pytest

from any2xsampa.cantonese.g2p import (
    CantoneseG2PError,
    ExternalCommandCantoneseG2PAdapter,
    PyCantoneseG2PAdapter,
    ReviewedLexiconCantoneseG2PAdapter,
    resolve_lyrics_timeline,
)
from any2xsampa.lyrics.timeline import parse_lyrics_timeline


def _timeline(text: str, *, jyutping: str | None = None):
    item = {"index": 0, "text": text, "start_sec": 0.0, "end_sec": 0.5}
    if jyutping:
        item["jyutping"] = jyutping
    return parse_lyrics_timeline(
        {
            "schema_version": "lyrics_timeline_v1",
            "audio_sha256": "0" * 64,
            "source": "manual_alignment",
            "items": [item],
            "provenance": {},
        }
    )


def test_reviewed_lexicon_preserves_phone_truth_candidates() -> None:
    resolved, summary = resolve_lyrics_timeline(_timeline("喜", jyutping="hei2"))
    assert resolved[0].candidate.jyutping == "hei2"
    assert resolved[0].candidate.internal_phones
    assert resolved[0].candidate.ipa
    assert summary["oov_items"] == []


def test_oov_blocks_instead_of_filtering() -> None:
    with pytest.raises(CantoneseG2PError) as exc_info:
        resolve_lyrics_timeline(_timeline("𠀀"), adapter=ReviewedLexiconCantoneseG2PAdapter())
    assert exc_info.value.error_kind == "unsupported_lyrics_evidence"
    assert exc_info.value.details["oov_items"][0]["text"] == "𠀀"


def test_pycantonese_adapter_is_version_and_license_pinned() -> None:
    resolved, summary = resolve_lyrics_timeline(_timeline("喜"), adapter=PyCantoneseG2PAdapter())
    assert resolved[0].candidate.jyutping == "hei2"
    assert resolved[0].candidate.source == "pycantonese_5.0.0"
    assert summary["adapter"]["license_id"] == "MIT"
    assert summary["adapter"]["wheel_sha256"] == "8f7bdc1e9a37b5e164cd57f22aecae8532d4d0b74ff5b6f5e28480a657fa53f9"


def test_external_g2p_rejects_nc_license(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    with pytest.raises(CantoneseG2PError) as exc_info:
        ExternalCommandCantoneseG2PAdapter(
            ["g2p"], license_id="NC", model_path=model, model_revision="r1"
        )
    assert exc_info.value.error_kind == "g2p_license_unverified"


def test_invalid_timeline_jyutping_is_blocked() -> None:
    with pytest.raises(CantoneseG2PError) as exc_info:
        resolve_lyrics_timeline(_timeline("喜", jyutping="not-a-jyutping"))
    assert exc_info.value.error_kind == "unsupported_lyrics_evidence"

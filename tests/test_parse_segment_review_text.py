from __future__ import annotations

import pytest

from tools.parse_segment_review_text import parse_text


def test_free_form_text_is_preserved_and_stays_pending() -> None:
    packet = {"packet_sha256": "p" * 64, "items": [{"review_id": "B01"}], "test_songs_opened": False}
    result = parse_text(packet, "B01：切点在‘住’字中间，无法确认具体音素")
    row = result["responses"][0]
    assert row["status"] == "pending"
    assert "无法确认" in row["review_statement"]
    assert row["phones"] == []


def test_structured_text_parses_explicit_phones_only() -> None:
    packet = {"packet_sha256": "p" * 64, "items": [{"review_id": "C03"}], "test_songs_opened": False}
    result = parse_text(packet, "C03:status=human_confirmed;boundary=truncated_start;notes=n1,n2;phones=n1|1|a|truncated_start,n2|0|t|full;说明=明确听到")
    row = result["responses"][0]
    assert row["status"] == "human_confirmed"
    assert row["note_ids"] == ["n1", "n2"]
    assert row["phones"][0]["phone"] == "a"


def test_unknown_id_rejected() -> None:
    packet = {"packet_sha256": "p" * 64, "items": [{"review_id": "B01"}], "test_songs_opened": False}
    with pytest.raises(ValueError, match="unknown review ID"):
        parse_text(packet, "B99:pending")

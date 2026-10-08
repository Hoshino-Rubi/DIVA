from __future__ import annotations

import pytest

from any2xsampa.adapters.forced_aligner import is_machine_review_draft_payload
from any2xsampa.lyrics.timeline import canonical_timeline_sha256, parse_lyrics_timeline
from tools.rebalance_singing_phone_review_draft import (
    TRANSFORM_NAME,
    RebalanceError,
    rebalance_singing_phone_review_draft,
)


def _timeline():
    return parse_lyrics_timeline(
        {
            "schema_version": "lyrics_timeline_v1",
            "audio_sha256": "a" * 64,
            "source": "manual_alignment",
            "items": [
                {"index": 0, "text": "A", "start_sec": 1.0, "end_sec": 1.5},
                {"index": 1, "text": "B", "start_sec": 2.0, "end_sec": 3.4},
            ],
            "provenance": {},
        }
    )


def _draft() -> dict[str, object]:
    timeline = _timeline()
    rows = []
    for lyric_index, phones, bounds in (
        (0, ("k", "a"), ((1.0, 1.45), (1.45, 1.5))),
        (1, ("ts", "u", "ng"), ((2.0, 3.0), (3.0, 3.05), (3.05, 3.4))),
    ):
        for phone, (start, end) in zip(phones, bounds, strict=True):
            rows.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "normalized_ipa": phone,
                    "confidence": 0.5,
                    "candidate_phones": [{"symbol": phone, "confidence": 0.5}],
                    "metadata": {"lyric_item_index": lyric_index},
                }
            )
    return {
        "schema_version": "forced_phone_alignment_v1",
        "vocal_stem_sha256": "a" * 64,
        "lyrics_timeline_sha256": canonical_timeline_sha256(timeline),
        "alignment_source": "fixture_machine_review",
        "model_output": True,
        "model_hash": "b" * 64,
        "product_accepted": False,
        "status": "machine_draft_pending_human_phone_review",
        "segments": rows,
        "review": {"status": "pending_human_phone_review"},
        "provenance": {},
    }


def test_rebalances_pathological_phone_ratios_and_keeps_prior_bounds() -> None:
    result = rebalance_singing_phone_review_draft(
        _draft(), timeline=_timeline(), source_alignment_sha256="c" * 64
    )
    segments = result["segments"]
    assert [row["start_sec"] for row in segments] == pytest.approx(
        [1.0, 1.09, 2.0, 2.16, 3.18]
    )
    assert [row["end_sec"] for row in segments] == pytest.approx(
        [1.09, 1.5, 2.16, 3.18, 3.4]
    )
    assert segments[2]["metadata"]["prior_review_suggestion_end_sec"] == 3.0
    assert segments[2]["metadata"]["boundary_transform"] == TRANSFORM_NAME
    assert result["product_accepted"] is False
    assert result["provenance"]["boundary_transform_acoustic_truth"] is False
    assert result["provenance"]["previous_review_draft_sha256"] == "c" * 64
    assert is_machine_review_draft_payload(result) is True


def test_rejects_accepted_or_wrong_timeline_inputs() -> None:
    accepted = _draft()
    accepted["product_accepted"] = True
    with pytest.raises(RebalanceError, match="unaccepted machine review draft"):
        rebalance_singing_phone_review_draft(
            accepted, timeline=_timeline(), source_alignment_sha256="c" * 64
        )

    other = _timeline().to_dict()
    other["items"][1]["end_sec"] = 3.5
    with pytest.raises(RebalanceError, match="different lyrics timeline"):
        rebalance_singing_phone_review_draft(
            _draft(),
            timeline=parse_lyrics_timeline(other),
            source_alignment_sha256="c" * 64,
        )

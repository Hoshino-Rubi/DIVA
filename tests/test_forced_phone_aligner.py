from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from any2xsampa.adapters.forced_aligner import (
    HUMAN_REVIEW_PENDING_DETACHED_ACCEPTANCE_STATUS,
    MACHINE_REVIEW_DRAFT_STATUS,
    PENDING_HUMAN_PHONE_REVIEW_STATUS,
    ExternalCommandForcedPhoneAligner,
    FixtureForcedPhoneAligner,
    ForcedPhoneAlignerError,
    compare_forced_and_ctc_alignment,
    is_human_reviewed_alignment_payload,
    is_machine_review_draft_payload,
    lyrics_timeline_sha256,
)
from any2xsampa.cantonese.g2p import resolve_lyrics_timeline
from any2xsampa.lyrics.timeline import parse_lyrics_timeline, sha256_file


def _wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)
    return path


def _timeline(audio: Path):
    return parse_lyrics_timeline(
        {
            "schema_version": "lyrics_timeline_v1",
            "audio_sha256": sha256_file(audio),
            "source": "manual_alignment",
            "items": [
                {"index": 0, "text": "喜", "start_sec": 0.0, "end_sec": 0.5, "jyutping": "hei2"},
                {"index": 1, "text": "歡", "start_sec": 0.5, "end_sec": 1.0, "jyutping": "fun1"},
            ],
            "provenance": {},
        }
    )


def _alignment(stem: Path, timeline) -> dict[str, object]:
    return {
        "schema_version": "forced_phone_alignment_v1",
        "vocal_stem_sha256": sha256_file(stem),
        "lyrics_timeline_sha256": lyrics_timeline_sha256(timeline),
        "alignment_source": "fixture_test_aligner",
        "model_output": True,
        "model_hash": "1" * 64,
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 0.45,
                "normalized_ipa": "e",
                "confidence": 0.95,
                "candidate_phones": [{"symbol": "e", "confidence": 0.95}],
                "metadata": {"alignment_source": "fixture_test_aligner", "lyric_item_index": 0, "direct_overlap_sec": 0.45, "phone_overlap_ratio": 1.0, "alignment_confidence": 0.95},
            },
            {
                "start_sec": 0.5,
                "end_sec": 0.95,
                "normalized_ipa": "u",
                "confidence": 0.9,
                "candidate_phones": [{"symbol": "u", "confidence": 0.9}],
                "metadata": {"alignment_source": "fixture_test_aligner", "lyric_item_index": 1, "direct_overlap_sec": 0.45, "phone_overlap_ratio": 1.0, "alignment_confidence": 0.9},
            },
        ],
    }


def test_loads_hash_bound_forced_phone_truth(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline = _timeline(stem)
    resolved, _ = resolve_lyrics_timeline(timeline)
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(_alignment(stem, timeline)), encoding="utf-8")

    result = FixtureForcedPhoneAligner(path).align(stem, timeline, resolved)

    assert len(result.segments) == 2
    assert result.segments[0].metadata["lyric_item_index"] == 0
    assert result.metadata["source_mode"] == "real"


def test_rejects_missing_lyric_item_coverage(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline = _timeline(stem)
    resolved, _ = resolve_lyrics_timeline(timeline)
    payload = _alignment(stem, timeline)
    payload["segments"] = payload["segments"][:1]  # type: ignore[index]
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ForcedPhoneAlignerError) as exc_info:
        FixtureForcedPhoneAligner(path).align(stem, timeline, resolved)
    assert exc_info.value.error_kind == "missing_phone_evidence"


def test_machine_review_draft_cannot_be_parsed_as_accepted_alignment(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline = _timeline(stem)
    resolved, _ = resolve_lyrics_timeline(timeline)
    payload = _alignment(stem, timeline)
    payload["review"] = {
        "status": "pending_human_phone_review",
        "scope": "full_song",
        "reviewed_range_sec": [0.0, 2.0],
    }
    payload["provenance"] = {
        "status": "machine_draft_pending_human_phone_review",
        "product_accepted": False,
    }
    path = tmp_path / "machine-review-draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ForcedPhoneAlignerError, match="machine phone-review drafts") as exc_info:
        FixtureForcedPhoneAligner(path).align(stem, timeline, resolved)
    assert exc_info.value.error_kind == "unqualified_forced_phone_evidence"


@pytest.mark.parametrize(
    "marker",
    [
        {"status": MACHINE_REVIEW_DRAFT_STATUS},
        {"review": {"status": PENDING_HUMAN_PHONE_REVIEW_STATUS}},
        {"product_accepted": False},
        {"provenance": {"product_accepted": False}},
    ],
)
def test_machine_review_draft_markers_are_rejected_even_when_partial(
    tmp_path: Path,
    marker: dict[str, object],
) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline = _timeline(stem)
    resolved, _ = resolve_lyrics_timeline(timeline)
    payload = _alignment(stem, timeline)
    payload.update(marker)

    assert is_machine_review_draft_payload(payload) is True
    path = tmp_path / "machine-review-draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForcedPhoneAlignerError, match="machine phone-review drafts"):
        FixtureForcedPhoneAligner(path).align(stem, timeline, resolved)


def test_reviewed_alignment_round_trips_hash_and_detached_status(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline = _timeline(stem)
    resolved, _ = resolve_lyrics_timeline(timeline)
    payload = _alignment(stem, timeline)
    payload["review"] = {"status": "human_reviewed", "reviewer_id": "reviewer-1"}
    payload["provenance"] = {
        "status": HUMAN_REVIEW_PENDING_DETACHED_ACCEPTANCE_STATUS,
        "product_accepted": False,
        "human_reviewed": True,
    }

    assert is_human_reviewed_alignment_payload(payload) is True
    assert is_machine_review_draft_payload(payload) is False
    path = tmp_path / "reviewed-alignment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = FixtureForcedPhoneAligner(path).align(stem, timeline, resolved)

    round_trip = result.to_dict()
    assert round_trip["vocal_stem_sha256"] == sha256_file(stem)
    assert round_trip["lyrics_timeline_sha256"] == lyrics_timeline_sha256(timeline)
    assert round_trip["status"] == HUMAN_REVIEW_PENDING_DETACHED_ACCEPTANCE_STATUS
    assert round_trip["product_accepted"] is False
    assert round_trip["review"] == payload["review"]
    assert round_trip["provenance"]["status"] == HUMAN_REVIEW_PENDING_DETACHED_ACCEPTANCE_STATUS


def test_rejects_timeline_hash_and_fabricated_overlap(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline = _timeline(stem)
    resolved, _ = resolve_lyrics_timeline(timeline)
    payload = _alignment(stem, timeline)
    payload["lyrics_timeline_sha256"] = "0" * 64
    path = tmp_path / "bad-hash.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForcedPhoneAlignerError) as exc_info:
        FixtureForcedPhoneAligner(path).align(stem, timeline, resolved)
    assert exc_info.value.error_kind == "forced_phone_lyrics_hash_mismatch"

    payload = _alignment(stem, timeline)
    payload["segments"][0]["metadata"]["direct_overlap_sec"] = 0.1  # type: ignore[index]
    path = tmp_path / "bad-overlap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForcedPhoneAlignerError) as exc_info:
        FixtureForcedPhoneAligner(path).align(stem, timeline, resolved)
    assert exc_info.value.error_kind == "invalid_forced_phone_evidence"


def test_rejects_unsorted_candidate_confidence(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline = _timeline(stem)
    resolved, _ = resolve_lyrics_timeline(timeline)
    payload = _alignment(stem, timeline)
    payload["segments"][0]["candidate_phones"] = [  # type: ignore[index]
        {"symbol": "e", "confidence": 0.2},
        {"symbol": "x", "confidence": 0.9},
    ]
    path = tmp_path / "bad-candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForcedPhoneAlignerError) as exc_info:
        FixtureForcedPhoneAligner(path).align(stem, timeline, resolved)
    assert exc_info.value.error_kind == "invalid_forced_phone_evidence"


def test_external_aligner_missing_stem_has_structured_error(tmp_path: Path) -> None:
    timeline = _timeline(_wav(tmp_path / "timeline-audio.wav"))
    resolved, _ = resolve_lyrics_timeline(timeline)
    with pytest.raises(ForcedPhoneAlignerError) as exc_info:
        ExternalCommandForcedPhoneAligner(["aligner.exe"]).align(
            tmp_path / "missing-stem.wav", timeline, resolved
        )
    assert exc_info.value.error_kind == "missing_vocal_stem_evidence"


def test_ctc_comparison_uses_union_coverage(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline = _timeline(stem)
    resolved, _ = resolve_lyrics_timeline(timeline)
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(_alignment(stem, timeline)), encoding="utf-8")
    forced = FixtureForcedPhoneAligner(path).align(stem, timeline, resolved).segments
    report = compare_forced_and_ctc_alignment(forced, [forced[0], forced[0]])
    assert report["overlap_sec"] == 0.45
    assert report["forced_covered_by_ctc_ratio"] == 0.5

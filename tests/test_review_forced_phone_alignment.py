from __future__ import annotations

import json
import threading
import urllib.request
import wave
from pathlib import Path

import pytest

from any2xsampa.adapters.forced_alignment_acceptance import load_forced_phone_acceptance
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.lyrics.timeline import (
    canonical_timeline_sha256,
    parse_lyrics_timeline,
    sha256_file,
)
from tools.review_forced_phone_alignment import (
    ACCEPTANCE_SCHEMA_VERSION,
    ALIGNMENT_SCHEMA_VERSION,
    BROWSER_DRAFT_SCHEMA_VERSION,
    DEFAULT_ACCEPTANCE_NAME,
    DEFAULT_REVIEWED_ALIGNMENT_NAME,
    FULL_SONG_SCOPE,
    HUMAN_REVIEW_ROUTE,
    ReviewError,
    create_server,
    load_review_inputs,
    render_review_page,
    save_review,
)


def _wav(path: Path, duration_sec: float = 12.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * duration_sec))
    return path


def _alignment(path: Path, stem: Path) -> Path:
    payload = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "vocal_stem_sha256": sha256_file(stem),
        "alignment_source": "fixture_review_test",
        "model_output": True,
        "model_hash": "1" * 64,
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 6.0,
                "normalized_ipa": "e",
                "confidence": 0.95,
                "candidate_phones": [{"symbol": "e", "confidence": 0.95}],
                "metadata": {
                    "alignment_source": "fixture_review_test",
                    "lyric_item_index": 0,
                    "direct_overlap_sec": 6.0,
                    "phone_overlap_ratio": 1.0,
                    "alignment_confidence": 0.95,
                },
            },
            {
                "start_sec": 6.0,
                "end_sec": 12.0,
                "normalized_ipa": "u",
                "confidence": 0.9,
                "candidate_phones": [{"symbol": "u", "confidence": 0.9}],
                "metadata": {
                    "alignment_source": "fixture_review_test",
                    "lyric_item_index": 1,
                    "direct_overlap_sec": 6.0,
                    "phone_overlap_ratio": 1.0,
                    "alignment_confidence": 0.9,
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _request(source: dict[str, object], *, reviewed: bool = True) -> dict[str, object]:
    return {
        "reviewer_id": "reviewer-1",
        "review_method": "auditory_acoustic_phone_boundary_review",
        "reviewed_at": "2026-08-15T12:00:00+08:00",
        "reviewed_range_sec": [0.0, 12.0],
        "segments": [
            {
                "index": row["index"],
                "symbol": row["symbol"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "lyric_item_index": row["lyric_item_index"],
                "reviewed": reviewed,
            }
            for row in source["segments"]  # type: ignore[index]
        ],
    }


def test_save_emits_hash_bound_human_acceptance(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    source = load_review_inputs(stem, machine)
    result = save_review(tmp_path / "out", source, _request(source))

    reviewed_alignment = Path(str(result["reviewed_alignment_path"]))
    acceptance_path = Path(str(result["acceptance_path"]))
    assert reviewed_alignment.name == DEFAULT_REVIEWED_ALIGNMENT_NAME
    assert acceptance_path.name == DEFAULT_ACCEPTANCE_NAME
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    assert acceptance["schema_version"] == ACCEPTANCE_SCHEMA_VERSION
    assert acceptance["route"] == HUMAN_REVIEW_ROUTE
    assert acceptance["review"]["review_status"] == "approved"
    assert acceptance["review"]["phone_boundaries_reviewed"] is True
    assert acceptance["vocal_stem_sha256"] == sha256_file(stem)
    assert acceptance["alignment_artifact_sha256"] == sha256_file(reviewed_alignment)
    assert acceptance["alignment_artifact_path"] == DEFAULT_REVIEWED_ALIGNMENT_NAME
    assert acceptance["provenance"]["machine_source_segment_count"] == 2
    assert acceptance["provenance"]["reviewed_segment_count"] == 2
    assert acceptance["provenance"]["human_row_edits_permitted"] is True

    reviewed = json.loads(reviewed_alignment.read_text(encoding="utf-8"))
    parsed_segments = tuple(
        PhoneSegment(
            TimeRange(item["start_sec"], item["end_sec"]),
            [PhoneCandidate(item["normalized_ipa"], item["confidence"])],
            item["confidence"],
            normalized_ipa=item["normalized_ipa"],
            metadata=item["metadata"],
        )
        for item in reviewed["segments"]
    )
    load_forced_phone_acceptance(
        acceptance_path,
        alignment_path=reviewed_alignment,
        vocal_stem=stem,
        model_hash="1" * 64,
        segments=parsed_segments,
    )


def test_short_clip_selects_complete_phone_rows_from_full_song_draft(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav", duration_sec=30.0)
    machine = _alignment(tmp_path / "machine.json", stem)
    payload = json.loads(machine.read_text(encoding="utf-8"))
    payload["segments"] = [
        {
            **payload["segments"][index % 2],
            "start_sec": float(index * 6),
            "end_sec": float((index + 1) * 6),
            "metadata": {
                **payload["segments"][index % 2]["metadata"],
                "lyric_item_index": index,
            },
        }
        for index in range(5)
    ]
    machine.write_text(json.dumps(payload), encoding="utf-8")

    source = load_review_inputs(
        stem,
        machine,
        reviewed_range_sec=[6.0, 18.0],
    )

    assert source["reviewed_range_sec"] == [6.0, 18.0]
    assert [row["start_sec"] for row in source["segments"]] == [6.0, 12.0]
    assert [row["end_sec"] for row in source["segments"]] == [12.0, 18.0]


def test_short_clip_rejects_window_that_cuts_phone_boundary(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav", duration_sec=20.0)
    machine = _alignment(tmp_path / "machine.json", stem)

    with pytest.raises(ReviewError, match="cuts through a phone segment"):
        load_review_inputs(
            stem,
            machine,
            reviewed_range_sec=[1.0, 11.0],
        )


@pytest.mark.parametrize(
    ("case_name", "segments"),
    [
        (
            "added",
            [
                {"symbol": "e", "start_sec": 0.0, "end_sec": 3.0, "lyric_item_index": 0},
                {"symbol": "i", "start_sec": 3.0, "end_sec": 6.0, "lyric_item_index": 0},
                {"symbol": "u", "start_sec": 6.0, "end_sec": 12.0, "lyric_item_index": 1},
            ],
        ),
        (
            "deleted",
            [
                {"symbol": "e", "start_sec": 0.0, "end_sec": 12.0, "lyric_item_index": 0},
            ],
        ),
    ],
)
def test_save_accepts_added_or_deleted_rows_with_final_hashes(
    tmp_path: Path,
    case_name: str,
    segments: list[dict[str, object]],
) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    source = load_review_inputs(stem, machine)
    request = _request(source)
    request["segments"] = [
        {"index": index, **row, "reviewed": True}
        for index, row in enumerate(segments)
    ]

    result = save_review(tmp_path / f"out-{case_name}", source, request)
    reviewed_alignment = Path(str(result["reviewed_alignment_path"]))
    acceptance_path = Path(str(result["acceptance_path"]))
    reviewed = json.loads(reviewed_alignment.read_text(encoding="utf-8"))
    assert len(reviewed["segments"]) == len(segments)
    assert reviewed["provenance"]["machine_source_segment_count"] == 2
    assert reviewed["provenance"]["reviewed_segment_count"] == len(segments)
    assert reviewed["provenance"]["machine_source_is_truth"] is False
    assert all(row["metadata"]["alignment_confidence"] == 1.0 for row in reviewed["segments"])
    parsed_segments = tuple(
        PhoneSegment(
            TimeRange(item["start_sec"], item["end_sec"]),
            [PhoneCandidate(item["normalized_ipa"], item["confidence"])],
            item["confidence"],
            normalized_ipa=item["normalized_ipa"],
            metadata=item["metadata"],
        )
        for item in reviewed["segments"]
    )
    load_forced_phone_acceptance(
        acceptance_path,
        alignment_path=reviewed_alignment,
        vocal_stem=stem,
        model_hash="1" * 64,
        segments=parsed_segments,
    )


def test_save_rejects_unreviewed_added_row_and_empty_request(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    source = load_review_inputs(stem, machine)

    added_unreviewed = _request(source)
    added_unreviewed["segments"] = [
        {
            "index": 0,
            "symbol": "e",
            "start_sec": 0.0,
            "end_sec": 3.0,
            "lyric_item_index": 0,
            "reviewed": True,
        },
        {
            "index": 1,
            "symbol": "i",
            "start_sec": 3.0,
            "end_sec": 6.0,
            "lyric_item_index": 0,
            "reviewed": False,
        },
        {
            "index": 2,
            "symbol": "u",
            "start_sec": 6.0,
            "end_sec": 12.0,
            "lyric_item_index": 1,
            "reviewed": True,
        },
    ]
    with pytest.raises(ReviewError, match="not been explicitly reviewed"):
        save_review(tmp_path / "out-added-unreviewed", source, added_unreviewed)

    empty = _request(source)
    empty["segments"] = []
    with pytest.raises(ReviewError, match="at least one phone row"):
        save_review(tmp_path / "out-empty", source, empty)


def test_save_rejects_unreviewed_short_range_overlap_and_overwrite(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    source = load_review_inputs(stem, machine)

    with pytest.raises(ReviewError, match="not been explicitly reviewed"):
        save_review(tmp_path / "out-unreviewed", source, _request(source, reviewed=False))

    too_short = _request(source)
    too_short["reviewed_range_sec"] = [0.0, 9.9]
    with pytest.raises(ReviewError, match="10-20"):
        save_review(tmp_path / "out-short", source, too_short)

    overlap = _request(source)
    overlap["segments"][1]["start_sec"] = 5.0  # type: ignore[index]
    with pytest.raises(ReviewError, match="overlapping"):
        save_review(tmp_path / "out-overlap", source, overlap)

    out = tmp_path / "out-existing"
    out.mkdir()
    (out / DEFAULT_ACCEPTANCE_NAME).write_text("keep", encoding="utf-8")
    with pytest.raises(ReviewError, match="overwrite"):
        save_review(out, source, _request(source))
    assert (out / DEFAULT_ACCEPTANCE_NAME).read_text(encoding="utf-8") == "keep"

    with pytest.raises(ReviewError, match="safe file name"):
        save_review(tmp_path / "out-traversal", source, _request(source), reviewed_alignment_name="../escape.json")


def test_server_serves_page_audio_and_requires_review(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    server = create_server(stem, machine, tmp_path / "out", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/", timeout=2).read().decode("utf-8")
        assert "waveform" in page and "Reviewed" in page and "/api/save" in page
        assert 'id="first-unreviewed"' in page
        assert 'id="previous-unreviewed"' in page
        assert 'id="next-unreviewed"' in page
        assert 'id="play-active"' in page
        assert 'id="discard-draft"' in page
        assert "function markDirty(tr)" in page
        assert "tr.querySelector('.reviewed').checked=false" in page
        assert "function markAllDirty()" in page
        assert 'class="group-reviewed"' in page
        assert 'class="icon-button group-play"' in page
        assert "function currentGroups()" in page
        assert "function markGroupDirty(tr,reconfigure=false)" in page
        assert "function setGroupReviewed(tr,checked)" in page
        assert "function playGroup(tr)" in page
        assert "groupForRow(tr).rows.forEach" in page
        assert "groups.filter(group=>group.rows.every" in page
        assert "function findUnreviewed(direction,fromStart=false)" in page
        assert "const token=++playbackToken" in page
        assert "if(token!==playbackToken)return" in page
        assert BROWSER_DRAFT_SCHEMA_VERSION in page
        assert "'any2xsampa.phone-review.'+source.review_session_id" in page
        assert "draft?.review_session_id!==source.review_session_id" in page
        assert "const restored={...row,reviewed:false}" in page
        assert "window.addEventListener('beforeunload'" in page
        assert "allowUnload=true;removeLocalDraft()" in page
        assert "confirm-all" not in page
        with urllib.request.urlopen(base + "/audio", timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("audio/")
        payload = json.dumps(_request(server.source, reviewed=False)).encode("utf-8")
        request = urllib.request.Request(base + "/api/save", data=payload, method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=2)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_browser_draft_identity_changes_with_bound_evidence(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    source = load_review_inputs(stem, machine)
    first_page = render_review_page(source).decode("utf-8")
    changed_source = dict(source)
    changed_source["alignment_sha256"] = "2" * 64
    changed_page = render_review_page(changed_source).decode("utf-8")

    marker = '"review_session_id":"'
    first_session = first_page.split(marker, 1)[1].split('"', 1)[0]
    changed_session = changed_page.split(marker, 1)[1].split('"', 1)[0]
    assert len(first_session) == 64
    assert len(changed_session) == 64
    assert first_session != changed_session


def test_short_save_starts_hash_validated_full_song_review_server(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    timeline_payload = {
        "schema_version": "lyrics_timeline_v1",
        "audio_sha256": sha256_file(stem),
        "source": "manual_alignment",
        "items": [
            {"index": 0, "text": "求", "jyutping": "kau4", "start_sec": 0.0, "end_sec": 6.0},
            {"index": 1, "text": "留", "jyutping": "lau4", "start_sec": 6.0, "end_sec": 12.0},
        ],
        "provenance": {"review_fixture": True},
    }
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps(timeline_payload, ensure_ascii=False), encoding="utf-8")
    machine_payload = json.loads(machine.read_text(encoding="utf-8"))
    machine_payload["lyrics_timeline_sha256"] = canonical_timeline_sha256(
        parse_lyrics_timeline(timeline_payload)
    )
    machine.write_text(json.dumps(machine_payload), encoding="utf-8")

    server = create_server(
        stem,
        machine,
        tmp_path / "short-out",
        lyrics_timeline_path=timeline,
        port=0,
        next_full_alignment_path=machine,
        next_full_output_dir=tmp_path / "full-out",
        next_full_port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        payload = json.dumps(_request(server.source)).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/save",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert result["next_review_started"] is True
        assert result["next_review_scope"] == FULL_SONG_SCOPE
        with urllib.request.urlopen(result["next_review_url"], timeout=3) as response:
            full_page = response.read().decode("utf-8")
        assert "0 / 2 reviewed" in full_page
        assert server.next_full_server is not None
        assert server.next_full_server.source["review_scope"] == FULL_SONG_SCOPE
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_hash_bound_lyrics_context_is_displayed_and_session_bound(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    timeline_payload = {
        "schema_version": "lyrics_timeline_v1",
        "audio_sha256": sha256_file(stem),
        "source": "manual_alignment",
        "items": [
            {"index": 0, "text": "求", "jyutping": "kau4", "start_sec": 0.0, "end_sec": 6.0},
            {"index": 1, "text": "留", "jyutping": "lau4", "start_sec": 6.0, "end_sec": 12.0},
        ],
        "provenance": {"review_fixture": True},
    }
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps(timeline_payload, ensure_ascii=False), encoding="utf-8")
    machine_payload = json.loads(machine.read_text(encoding="utf-8"))
    machine_payload["lyrics_timeline_sha256"] = canonical_timeline_sha256(
        parse_lyrics_timeline(timeline_payload)
    )
    machine.write_text(json.dumps(machine_payload), encoding="utf-8")

    source = load_review_inputs(stem, machine, lyrics_timeline_path=timeline)
    assert source["segments"][0]["lyric_text"] == "求"  # type: ignore[index]
    assert source["segments"][0]["jyutping"] == "kau4"  # type: ignore[index]
    page = render_review_page(source).decode("utf-8")
    assert "求" in page
    assert "Review evidence" in page
    assert source["audio_sha256"] in page
    assert source["alignment_sha256"] in page
    assert source["lyrics_timeline_sha256"] in page

    edited = _request(source)
    edited["segments"][0]["start_sec"] = 0.5  # type: ignore[index]
    edited["segments"][0]["end_sec"] = 5.5  # type: ignore[index]
    saved = save_review(tmp_path / "out-recomputed-overlap", source, edited)
    reviewed = json.loads(
        Path(str(saved["reviewed_alignment_path"])).read_text(encoding="utf-8")
    )
    metadata = reviewed["segments"][0]["metadata"]
    assert metadata["direct_overlap_sec"] == 5.0
    assert metadata["phone_overlap_ratio"] == 1.0
    assert (
        reviewed["provenance"]["status"]
        == "human_reviewed_phone_alignment_pending_detached_acceptance_validation"
    )
    assert reviewed["provenance"]["product_accepted"] is False

    wrong_lyric = _request(source)
    wrong_lyric["segments"][0]["lyric_item_index"] = 1  # type: ignore[index]
    with pytest.raises(ReviewError, match="does not overlap lyric item"):
        save_review(tmp_path / "out-wrong-lyric", source, wrong_lyric)

    timeline.write_text(timeline.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="timeline changed"):
        save_review(tmp_path / "out-changed-timeline", source, _request(source))


def test_lyrics_context_rejects_wrong_canonical_timeline(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    machine_payload = json.loads(machine.read_text(encoding="utf-8"))
    machine_payload["lyrics_timeline_sha256"] = "0" * 64
    machine.write_text(json.dumps(machine_payload), encoding="utf-8")
    timeline = tmp_path / "timeline.json"
    timeline.write_text(
        json.dumps(
            {
                "schema_version": "lyrics_timeline_v1",
                "audio_sha256": sha256_file(stem),
                "source": "manual_alignment",
                "items": [
                    {"index": 0, "text": "求", "start_sec": 0.0, "end_sec": 12.0},
                ],
                "provenance": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReviewError, match="different lyrics timeline"):
        load_review_inputs(stem, machine, lyrics_timeline_path=timeline)


def test_hash_bound_lyrics_context_requires_timeline_path(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    machine_payload = json.loads(machine.read_text(encoding="utf-8"))
    machine_payload["lyrics_timeline_sha256"] = "1" * 64
    machine.write_text(json.dumps(machine_payload), encoding="utf-8")

    with pytest.raises(ReviewError, match="no lyrics timeline was supplied"):
        load_review_inputs(stem, machine)


def test_lyrics_context_rejects_timeline_bound_to_different_audio(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    timeline_audio = _wav(tmp_path / "source.wav", duration_sec=13.0)
    machine = _alignment(tmp_path / "machine.json", stem)
    timeline_payload = {
        "schema_version": "lyrics_timeline_v1",
        "audio_sha256": sha256_file(timeline_audio),
        "source": "manual_alignment",
        "items": [
            {"index": 0, "text": "瘙?, ", "start_sec": 0.0, "end_sec": 6.0},
            {"index": 1, "text": "??, ", "start_sec": 6.0, "end_sec": 12.0},
        ],
        "provenance": {"review_fixture": True},
    }
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps(timeline_payload, ensure_ascii=False), encoding="utf-8")
    machine_payload = json.loads(machine.read_text(encoding="utf-8"))
    machine_payload["lyrics_timeline_sha256"] = canonical_timeline_sha256(
        parse_lyrics_timeline(timeline_payload)
    )
    machine.write_text(json.dumps(machine_payload), encoding="utf-8")

    with pytest.raises(ReviewError, match="separate source audio"):
        load_review_inputs(stem, machine, lyrics_timeline_path=timeline)

    with pytest.raises(ReviewError, match="does not match"):
        load_review_inputs(
            stem,
            machine,
            lyrics_timeline_path=timeline,
            lyrics_timeline_audio_path=stem,
        )

    source = load_review_inputs(
        stem,
        machine,
        lyrics_timeline_path=timeline,
        lyrics_timeline_audio_path=timeline_audio,
    )
    assert source["audio_sha256"] == sha256_file(stem)
    assert source["lyrics_timeline_audio_sha256"] == sha256_file(timeline_audio)

    _wav(timeline_audio, duration_sec=14.0)
    with pytest.raises(ReviewError, match="timeline changed"):
        save_review(tmp_path / "out-changed-source-audio", source, _request(source))


def test_full_song_review_requires_and_embeds_short_clip_predecessor(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    short_source = load_review_inputs(stem, machine)
    short_result = save_review(tmp_path / "short", short_source, _request(short_source))
    predecessor = Path(str(short_result["acceptance_path"]))

    with pytest.raises(ReviewError, match="requires a short-clip acceptance"):
        load_review_inputs(stem, machine, review_scope=FULL_SONG_SCOPE)

    full_source = load_review_inputs(
        stem,
        machine,
        review_scope=FULL_SONG_SCOPE,
        short_clip_acceptance_path=predecessor,
    )
    full_result = save_review(tmp_path / "full", full_source, _request(full_source))
    full_acceptance_path = Path(str(full_result["acceptance_path"]))
    full_acceptance = json.loads(full_acceptance_path.read_text(encoding="utf-8"))
    assert full_acceptance["review"]["scope"] == FULL_SONG_SCOPE
    assert full_acceptance["review"]["short_clip_predecessor"] == {
        "acceptance_path": str(predecessor.resolve()),
        "acceptance_sha256": sha256_file(predecessor),
    }

    reviewed_alignment = Path(str(full_result["reviewed_alignment_path"]))
    reviewed = json.loads(reviewed_alignment.read_text(encoding="utf-8"))
    parsed_segments = tuple(
        PhoneSegment(
            TimeRange(item["start_sec"], item["end_sec"]),
            [PhoneCandidate(item["normalized_ipa"], item["confidence"])],
            item["confidence"],
            normalized_ipa=item["normalized_ipa"],
            metadata=item["metadata"],
        )
        for item in reviewed["segments"]
    )
    load_forced_phone_acceptance(
        full_acceptance_path,
        alignment_path=reviewed_alignment,
        vocal_stem=stem,
        model_hash="1" * 64,
        segments=parsed_segments,
    )


def test_full_song_save_revalidates_mutated_short_clip_predecessor(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    machine = _alignment(tmp_path / "machine.json", stem)
    short_source = load_review_inputs(stem, machine)
    short_result = save_review(tmp_path / "short", short_source, _request(short_source))
    predecessor = Path(str(short_result["acceptance_path"]))
    full_source = load_review_inputs(
        stem,
        machine,
        review_scope=FULL_SONG_SCOPE,
        short_clip_acceptance_path=predecessor,
    )

    predecessor_alignment = Path(str(short_result["reviewed_alignment_path"]))
    predecessor_alignment.write_text(
        predecessor_alignment.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReviewError, match="short-clip predecessor") as exc_info:
        save_review(tmp_path / "full-mutated-predecessor", full_source, _request(full_source))
    assert exc_info.value.error_kind == "invalid_predecessor"

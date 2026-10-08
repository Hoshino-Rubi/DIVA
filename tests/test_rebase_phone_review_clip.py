from __future__ import annotations

import json
from pathlib import Path
import struct
import wave

import pytest

from any2xsampa.lyrics.timeline import canonical_timeline_sha256, parse_lyrics_timeline, sha256_file
from tools.rebase_phone_review_clip import RebaseError, rebase_phone_review_clip, write_rebased_review_draft
from tools.review_forced_phone_alignment import load_review_inputs


RATE = 1000


def _wav(path: Path, values: list[int]) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(struct.pack(f"<{len(values)}h", *values))
    return path


def _timeline(audio: Path, *, full: bool) -> dict[str, object]:
    items = (
        [
            {"index": 0, "text": "前", "jyutping": "cin4", "start_sec": 0.0, "end_sec": 1.0},
            {"index": 1, "text": "求", "jyutping": "kau4", "start_sec": 10.0, "end_sec": 16.0},
            {"index": 2, "text": "留", "jyutping": "lau4", "start_sec": 16.0, "end_sec": 22.0},
        ]
        if full
        else [
            {"index": 0, "text": "求", "jyutping": "kau4", "start_sec": 0.0, "end_sec": 6.0},
            {"index": 1, "text": "留", "jyutping": "lau4", "start_sec": 6.0, "end_sec": 12.0},
        ]
    )
    return {
        "schema_version": "lyrics_timeline_v1",
        "audio_sha256": sha256_file(audio),
        "source": "manual_alignment",
        "items": items,
        "provenance": {"fixture": True},
    }


def _alignment(path: Path, clip: Path, timeline: dict[str, object]) -> Path:
    payload = {
        "schema_version": "forced_phone_alignment_v1",
        "vocal_stem_sha256": sha256_file(clip),
        "lyrics_timeline_sha256": canonical_timeline_sha256(parse_lyrics_timeline(timeline)),
        "alignment_source": "fixture_machine_clip",
        "model_output": True,
        "model_hash": "1" * 64,
        "segments": [
            {
                "candidate_phones": [{"symbol": "k", "confidence": 0.8}],
                "confidence": 0.8,
                "normalized_ipa": "k",
                "metadata": {"lyric_item_index": 0},
                "start_sec": 0.0,
                "end_sec": 6.0,
            },
            {
                "candidate_phones": [{"symbol": "l", "confidence": 0.7}],
                "confidence": 0.7,
                "normalized_ipa": "l",
                "metadata": {"lyric_item_index": 1},
                "start_sec": 6.0,
                "end_sec": 12.0,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rebases_exact_pcm_clip_to_full_stem_review_window(tmp_path: Path) -> None:
    full_values = [index % 2000 - 1000 for index in range(30 * RATE)]
    full = _wav(tmp_path / "full.wav", full_values)
    clip = _wav(tmp_path / "clip.wav", full_values[10 * RATE : 22 * RATE])
    clip_timeline_payload = _timeline(clip, full=False)
    full_timeline_payload = _timeline(full, full=True)
    clip_timeline = tmp_path / "clip-timeline.json"
    full_timeline = tmp_path / "full-timeline.json"
    clip_timeline.write_text(json.dumps(clip_timeline_payload, ensure_ascii=False), encoding="utf-8")
    full_timeline.write_text(json.dumps(full_timeline_payload, ensure_ascii=False), encoding="utf-8")
    alignment = _alignment(tmp_path / "clip-alignment.json", clip, clip_timeline_payload)

    payload = rebase_phone_review_clip(
        clip_audio=clip,
        full_audio=full,
        clip_alignment=alignment,
        clip_timeline=clip_timeline,
        full_timeline=full_timeline,
        offset_sec=10.0,
        lyric_index_offset=1,
    )
    assert payload["vocal_stem_sha256"] == sha256_file(full)
    assert payload["segments"][0]["start_sec"] == 10.0
    assert payload["segments"][1]["end_sec"] == 22.0
    assert payload["segments"][0]["metadata"]["lyric_item_index"] == 1
    assert payload["provenance"]["review_draft_rebase"]["pcm_crop"]["exact_pcm_match"] is True

    output = tmp_path / "rebased.json"
    report = write_rebased_review_draft(output, payload)
    assert report["sha256"] == sha256_file(output)
    source = load_review_inputs(
        full,
        output,
        lyrics_timeline_path=full_timeline,
        reviewed_range_sec=[10.0, 22.0],
    )
    assert source["reviewed_range_sec"] == [10.0, 22.0]
    assert source["segments"][0]["lyric_text"] == "求"  # type: ignore[index]

    with pytest.raises(RebaseError, match="overwrite"):
        write_rebased_review_draft(output, payload)


def test_rejects_clip_that_is_not_exact_full_stem_pcm(tmp_path: Path) -> None:
    full_values = [0] * (30 * RATE)
    full = _wav(tmp_path / "full.wav", full_values)
    clip_values = full_values[10 * RATE : 22 * RATE]
    clip_values[100] = 1
    clip = _wav(tmp_path / "clip.wav", clip_values)
    clip_timeline_payload = _timeline(clip, full=False)
    full_timeline_payload = _timeline(full, full=True)
    clip_timeline = tmp_path / "clip-timeline.json"
    full_timeline = tmp_path / "full-timeline.json"
    clip_timeline.write_text(json.dumps(clip_timeline_payload, ensure_ascii=False), encoding="utf-8")
    full_timeline.write_text(json.dumps(full_timeline_payload, ensure_ascii=False), encoding="utf-8")
    alignment = _alignment(tmp_path / "clip-alignment.json", clip, clip_timeline_payload)

    with pytest.raises(RebaseError, match="does not exactly match"):
        rebase_phone_review_clip(
            clip_audio=clip,
            full_audio=full,
            clip_alignment=alignment,
            clip_timeline=clip_timeline,
            full_timeline=full_timeline,
            offset_sec=10.0,
            lyric_index_offset=1,
        )

from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest

from any2xsampa.lyrics.timeline import (
    LyricsTimelineError,
    load_lyrics_timeline,
    parse_lyrics_timeline,
    sha256_file,
    timeline_from_asr_transcript,
)


def _wav(path: Path, seconds: float = 2.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _payload(audio: Path) -> dict[str, object]:
    return {
        "schema_version": "lyrics_timeline_v1",
        "audio_sha256": sha256_file(audio),
        "source": "manual_alignment",
        "items": [
            {"index": 0, "text": "喜", "start_sec": 0.0, "end_sec": 0.5, "jyutping": "hei2"},
            {"index": 1, "text": "歡", "start_sec": 0.5, "end_sec": 1.0, "jyutping": "fun1"},
        ],
        "provenance": {"timestamp_source": "manual"},
    }


def test_loads_valid_hash_bound_timeline(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "audio.wav")
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(_payload(audio), ensure_ascii=False), encoding="utf-8")

    timeline = load_lyrics_timeline(path, audio_path=audio)

    assert timeline.text == "喜歡"
    assert timeline.items[0].jyutping == "hei2"


@pytest.mark.parametrize("mutation", ["overlap", "bad_hash", "out_of_bounds"])
def test_rejects_invalid_timeline_evidence(tmp_path: Path, mutation: str) -> None:
    audio = _wav(tmp_path / "audio.wav")
    payload = _payload(audio)
    if mutation == "overlap":
        payload["items"][1]["start_sec"] = 0.4  # type: ignore[index]
    elif mutation == "bad_hash":
        payload["audio_sha256"] = "0" * 64
    else:
        payload["items"][1]["end_sec"] = 3.0  # type: ignore[index]
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(LyricsTimelineError):
        load_lyrics_timeline(path, audio_path=audio)


def test_qwen_text_without_timestamps_cannot_become_timeline(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "audio.wav")
    with pytest.raises(LyricsTimelineError) as exc_info:
        timeline_from_asr_transcript(
            {"schema_version": "asr_transcript_v1", "text": "喜歡", "segments": [], "provenance": {}},
            audio_path=audio,
        )
    assert exc_info.value.error_kind == "missing_lyrics_timing_evidence"


def test_timestamped_single_syllable_asr_segments_convert(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "audio.wav")
    provenance = {"timestamps_requested": True, "truth_layer": "asr_evidence_not_phone_truth", "audio_sha256": sha256_file(audio)}
    timeline = timeline_from_asr_transcript(
        {
            "schema_version": "asr_transcript_v1",
            "text": "喜歡",
            "segments": [
                {"text": "喜", "start_sec": 0.0, "end_sec": 0.5, "confidence": 0.9},
                {"text": "歡", "start_sec": 0.5, "end_sec": 1.0, "confidence": 0.8},
            ],
            "provenance": provenance,
        },
        audio_path=audio,
    )
    assert timeline.source == "qwen3_asr"
    assert parse_lyrics_timeline(timeline.to_dict()).text == "喜歡"


def test_nested_qwen_timestamp_items_convert(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "audio.wav")
    timeline = timeline_from_asr_transcript(
        {
            "schema_version": "asr_transcript_v1",
            "text": "你好",
            "segments": [
                {
                    "index": 0,
                    "text": "你好",
                    "items": [
                        {"index": 0, "text": "你", "start_sec": 0.0, "end_sec": 0.31},
                        {"index": 1, "text": "好", "start_sec": 0.31, "end_sec": 0.62},
                    ],
                }
            ],
            "provenance": {"timestamps_requested": True},
            "provenance": {"timestamps_requested": True, "truth_layer": "asr_evidence_not_phone_truth", "audio_sha256": sha256_file(audio)},
        },
        audio_path=audio,
    )
    assert timeline.text == "你好"
    assert [item.end_sec for item in timeline.items] == [0.31, 0.62]


def test_asr_audio_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "audio.wav")
    with pytest.raises(LyricsTimelineError) as exc_info:
        timeline_from_asr_transcript(
            {
                "schema_version": "asr_transcript_v1",
                "text": "喜",
                "segments": [{"text": "喜", "start_sec": 0.0, "end_sec": 0.5}],
                "provenance": {"audio_sha256": "1" * 64, "truth_layer": "asr_evidence_not_phone_truth"},
            },
            audio_path=audio,
        )
    assert exc_info.value.error_kind == "lyrics_timing_audio_hash_mismatch"


def test_timeline_rejects_coerced_timing_and_missing_index(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "audio.wav")
    payload = _payload(audio)
    payload["items"][0]["start_sec"] = "0.0"  # type: ignore[index]
    with pytest.raises(LyricsTimelineError):
        parse_lyrics_timeline(payload)


def test_timeline_rejects_any_overlap_and_audio_tail_overrun(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "audio.wav", seconds=1.0)
    payload = _payload(audio)
    payload["items"][1]["start_sec"] = 0.499999999  # type: ignore[index]
    with pytest.raises(LyricsTimelineError):
        parse_lyrics_timeline(payload)

    payload = _payload(audio)
    payload["items"][1]["end_sec"] = 1.000000001  # type: ignore[index]
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(LyricsTimelineError) as exc_info:
        load_lyrics_timeline(path, audio_path=audio)
    assert exc_info.value.error_kind == "lyrics_timing_out_of_bounds"


def test_asr_timeline_rejects_out_of_bounds_and_non_wav(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "audio.wav", seconds=1.0)
    payload = {
        "schema_version": "asr_transcript_v1",
        "text": "喜",
        "segments": [{"text": "喜", "start_sec": 1.0, "end_sec": 2.0}],
        "provenance": {"truth_layer": "asr_evidence_not_phone_truth", "audio_sha256": sha256_file(audio)},
    }
    with pytest.raises(LyricsTimelineError) as exc_info:
        timeline_from_asr_transcript(payload, audio_path=audio)
    assert exc_info.value.error_kind == "lyrics_timing_out_of_bounds"

    garbage = tmp_path / "garbage.wav"
    garbage.write_bytes(b"not-a-wav")
    with pytest.raises(LyricsTimelineError) as exc_info:
        timeline_from_asr_transcript(payload, audio_path=garbage)
    assert exc_info.value.error_kind == "invalid_lyrics_timing_evidence"
    payload = _payload(audio)
    del payload["items"][0]["index"]  # type: ignore[index]
    with pytest.raises(LyricsTimelineError):
        parse_lyrics_timeline(payload)

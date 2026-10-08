from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import wave


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_cantonese_singing_phrase_ctc_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_cantonese_singing_phrase_ctc_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_wav(path: Path, *, duration_sec: float = 20.0, sample_rate: int = 16_000, channels: int = 1) -> None:
    frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames * channels)


def _row(
    *,
    note_index: int,
    start: float,
    end: float,
    ready: bool = True,
    target: list[str] | None = None,
    manual: list[str] | None = None,
    segment_id: str = "qre-p1",
    audio_paths: list[str] | None = None,
) -> dict[str, object]:
    target = target if target is not None else ["p", "a"]
    manual = manual if manual is not None else ["p", "a"]
    return {
        "training_ready": ready,
        "segment_id": segment_id,
        "variant_group": "qre-lines-01-02",
        "singer_id": "fixture-singer",
        "song_id": "fixture-song",
        "phrase_id": "fixture-phrase",
        "note_index": note_index,
        "audio_paths": audio_paths or [],
        "source_vsqx": "fixture.vsqx",
        "source_vsqx_sha256": "a" * 64,
        "source_corpus": "fixture-corpus.json",
        "source_corpus_sha256": "b" * 64,
        "start_sec": start,
        "end_sec": end,
        "v4c_tokens": manual,
        "target_phone_tokens": target,
        "internal_ipa_tokens": list(target),
        "articulatory_features": [{"class": "fixture"} for _ in target],
        "label_provenance": {
            "schema_version": "human_phone_label_v1",
            "kind": "human",
            "source_id": "fixture",
            "source_revision": "r1",
            "annotation_sha256": "c" * 64,
            "policy_version": "fixture-v1",
            "annotator_id": "tester",
            "review_status": "approved",
        },
        "phoneme_lock": True,
    }


def _write_manifest(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    return path


def test_phrase_windows_preserve_manual_tokens_and_record_16k_read_policy(tmp_path: Path) -> None:
    wav = tmp_path / "qre.wav"
    _write_wav(wav, duration_sec=8.0, sample_rate=48_000, channels=2)
    rows = [
        _row(note_index=1, start=0.2, end=1.2, audio_paths=[str(wav)], manual=["uAN"]),
        _row(note_index=2, start=1.25, end=2.3, audio_paths=[str(wav)], manual=["x", "AU"]),
        _row(note_index=3, start=2.35, end=3.8, audio_paths=[str(wav)], manual=["p_h", "a"]),
        _row(note_index=4, start=3.9, end=5.3, audio_paths=[str(wav)], manual=["k_h", "a"]),
    ]
    payload = MODULE.build_dataset(
        manifest_path=_write_manifest(tmp_path, rows),
        output_root=tmp_path / "out",
    )

    assert payload["schema_version"] == "cantonese_singing_phrase_ctc_dataset_v2"
    assert payload["example_count"] == 1
    example = payload["examples"][0]
    assert example["note_count"] == 4
    assert example["human_v4c_tokens_by_note"] == [["uAN"], ["x", "AU"], ["p_h", "a"], ["k_h", "a"]]
    assert example["internal_ipa_tokens"] == example["target_phone_tokens"]
    assert example["source_audio_sample_rate"] == 48_000
    assert example["model_audio_sample_rate"] == 16_000
    assert example["model_audio_channels"] == 1
    assert example["lyrics_used_as_recognizer_input"] is False
    assert example["label_provenance"]["schema_version"] == "human_phone_label_v1"
    assert example["label_provenance"]["review_status"] == "approved"
    assert len(example["label_provenance"]["annotation_sha256"]) == 64
    assert example["left_context_sec"] == 0.15
    assert example["right_context_sec"] == 0.15
    assert Path(payload["artifact_dir"]).name == payload["dataset_hash"]
    assert (Path(payload["artifact_dir"]) / "dataset_lock.json").is_file()


def test_phrase_windows_preserve_original_audio_identity_for_leakage_checks(tmp_path: Path) -> None:
    wav = tmp_path / "qre.wav"
    _write_wav(wav, duration_sec=4.0)
    row = _row(note_index=1, start=0.2, end=1.2, audio_paths=[str(wav)])
    row["original_audio_path"] = str(tmp_path / "original.wav")
    row["original_audio_sha256"] = "d" * 64
    row["original_audio_metadata"] = {"duration_sec": 4.0}
    payload = MODULE.build_dataset(
        manifest_path=_write_manifest(tmp_path, [row]),
        output_root=tmp_path / "out",
    )
    example = payload["examples"][0]
    assert example["original_audio_path"] == str(tmp_path / "original.wav")
    assert example["original_audio_sha256"] == "d" * 64


def test_unresolved_and_large_gap_rows_break_streams_without_rejecting_manual_annotations(tmp_path: Path) -> None:
    wav = tmp_path / "qre.wav"
    _write_wav(wav, duration_sec=12.0)
    audio = [str(wav)]
    rows = [
        _row(note_index=1, start=0.1, end=1.1, audio_paths=audio),
        _row(note_index=2, start=1.2, end=2.3, audio_paths=audio),
        _row(note_index=3, start=2.4, end=2.8, ready=False, target=[], manual=["UNRESOLVED"], audio_paths=audio),
        _row(note_index=4, start=2.9, end=4.0, audio_paths=audio),
        _row(note_index=5, start=4.1, end=5.3, audio_paths=audio),
        _row(note_index=6, start=6.2, end=7.3, audio_paths=audio),
        _row(note_index=7, start=7.4, end=8.7, audio_paths=audio),
    ]
    payload = MODULE.build_dataset(
        manifest_path=_write_manifest(tmp_path, rows),
        output_root=tmp_path / "out",
    )

    reasons = {row["reason"] for row in payload["exclusions"]}
    break_reasons = {row["reason"] for row in payload["continuity_breaks"]}
    assert "unresolved_note_break" in reasons
    assert "unresolved_note_break" in break_reasons
    assert "gap_exceeds_max" in break_reasons
    assert payload["example_count"] == 3
    unresolved = next(row for row in payload["exclusions"] if row["reason"] == "unresolved_note_break")
    assert unresolved["human_v4c_tokens_by_note"] == [["UNRESOLVED"]]
    assert unresolved["manual_annotation_rejected"] is False


def test_tail_uses_at_most_fifty_percent_overlap_when_merge_would_exceed_eight_seconds(tmp_path: Path) -> None:
    wav = tmp_path / "qre.wav"
    _write_wav(wav, duration_sec=12.0)
    audio = [str(wav)]
    rows = [
        _row(note_index=1, start=0.0, end=3.0, audio_paths=audio),
        _row(note_index=2, start=3.05, end=6.05, audio_paths=audio),
        _row(note_index=3, start=6.1, end=9.1, audio_paths=audio),
    ]
    payload = MODULE.build_dataset(
        manifest_path=_write_manifest(tmp_path, rows),
        output_root=tmp_path / "out",
    )

    assert payload["example_count"] == 2
    first, second = payload["examples"]
    overlap = max(0.0, min(first["note_end_sec"], second["note_end_sec"]) - max(first["note_start_sec"], second["note_start_sec"]))
    ratio = overlap / min(first["note_span_duration_sec"], second["note_span_duration_sec"])
    assert ratio <= 0.5 + 1e-9
    assert first["note_ids"][-1] == second["note_ids"][0]


def test_qre_rejects_multiple_audio_candidates_and_midi_is_never_accepted(tmp_path: Path) -> None:
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _write_wav(wav_a)
    _write_wav(wav_b)
    rows = [_row(note_index=1, start=0.1, end=1.2, audio_paths=[str(wav_a), str(wav_b)])]
    try:
        MODULE.build_dataset(
            manifest_path=_write_manifest(tmp_path, rows),
            output_root=tmp_path / "out",
        )
    except ValueError as exc:
        assert "latest WAV" in str(exc)
    else:
        raise AssertionError("QRE with multiple WAV candidates must be rejected")

    midi_rows = [_row(note_index=1, start=0.1, end=1.2, audio_paths=[str(tmp_path / "part.mid")])]
    try:
        MODULE.build_dataset(
            manifest_path=_write_manifest(tmp_path, midi_rows),
            output_root=tmp_path / "out-midi",
        )
    except ValueError as exc:
        assert "WAV" in str(exc)
    else:
        raise AssertionError("MIDI must never be accepted as CTC audio")

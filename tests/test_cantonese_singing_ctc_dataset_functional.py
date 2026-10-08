from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import wave


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_cantonese_singing_ctc_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_cantonese_singing_ctc_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_note_dataset_keeps_groups_and_human_phone_truth(tmp_path: Path) -> None:
    audio_paths = [tmp_path / "audio-a.wav", tmp_path / "audio-b.wav"]
    for audio_index, audio in enumerate(audio_paths, start=1):
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(bytes([audio_index, 0]) * 32_000)
    rows = []
    for note, (group, token) in enumerate((("song-a", "a"), ("song-b", "b")), start=1):
        rows.append(
            {
                "training_ready": True,
                "segment_id": f"qre-{group}",
                "variant_group": group,
                "singer_id": f"singer-{note}",
                "song_id": group,
                "phrase_id": f"phrase-{note}",
                "note_index": note,
                "audio_paths": [str(audio_paths[note - 1])],
                "start_sec": 0.2 * note,
                "end_sec": 0.2 * note + 0.15,
                "v4c_tokens": [token.upper()],
                "target_phone_tokens": [token],
                "internal_ipa_tokens": [token],
                "articulatory_features": [{"phone": token, "class": "fixture"}],
                "label_provenance": {
                    "schema_version": "human_phone_label_v1",
                    "kind": "human",
                    "source_id": "fixture",
                    "source_revision": "r1",
                    "annotation_sha256": f"{note}" * 64,
                    "policy_version": "fixture-v1",
                    "annotator_id": "tester",
                    "review_status": "approved",
                },
                "phoneme_lock": True,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    payload = MODULE.build_dataset(manifest_path=manifest, output_root=tmp_path / "out")
    assert payload["example_count"] == 2
    assert payload["group_count"] == 2
    assert len(payload["folds"]) == 2
    assert all(row["human_authored_annotation_evidence"] is True for row in payload["examples"])
    assert all(row["target_phone_tokens"] for row in payload["examples"])
    assert all(fold["leakage_free"] is True for fold in payload["folds"])


def test_note_dataset_does_not_split_groups_sharing_source_evidence() -> None:
    groups = {
        "song-a": [{"audio_sha256": "f" * 64, "source_vsqx": "same.vsqx", "singer_id": "s1", "song_id": "a", "phrase_id": "p1", "target_phone_tokens": ["a"]}],
        "song-b": [{"audio_sha256": "f" * 64, "source_vsqx": "same.vsqx", "singer_id": "s2", "song_id": "b", "phrase_id": "p2", "target_phone_tokens": ["b"]}],
    }
    folds = MODULE._build_group_folds(groups)
    assert folds == []


def test_note_dataset_rejects_training_ready_without_human_provenance(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "training_ready": True,
                        "segment_id": "unsafe",
                        "variant_group": "unsafe",
                        "singer_id": "s",
                        "song_id": "song",
                        "phrase_id": "phrase",
                        "note_index": 1,
                        "audio_paths": [str(audio)],
                        "start_sec": 0.0,
                        "end_sec": 0.5,
                        "target_phone_tokens": ["a"],
                        "internal_ipa_tokens": ["a"],
                        "articulatory_features": [{"class": "vowel"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    try:
        MODULE.build_dataset(manifest_path=manifest, output_root=tmp_path / "out")
    except ValueError as exc:
        assert "label_provenance" in str(exc)
    else:
        raise AssertionError("training_ready must not manufacture human provenance")

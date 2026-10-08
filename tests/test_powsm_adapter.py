"""Evidence integrity and normalization tests for the isolated challenger."""
from __future__ import annotations
import json
from pathlib import Path
import wave

import numpy as np
import pytest
from safetensors.numpy import save_file

from any2xsampa.adapters.audio_provenance import describe_audio_file
from any2xsampa.adapters.powsm import (
    MODEL_ID, REVISION, WEIGHT_SHA256, PowsmPosteriorRecognizer,
    decoded_tokens, emission_payload, token_phone, adjacent_file,
)
from any2xsampa.training.limited_data_artifacts import sha256_file

LABELS = ["<blank>", "<unk>", "/i/", "/t͡sʰ/", "unwrapped", "/n/"]


def values(ids: list[int]) -> np.ndarray:
    result = np.full((len(ids), len(LABELS)), -8.0, dtype=np.float32)
    result[np.arange(len(ids)), ids] = 8.0
    return result


def payload(ids: list[int], *, end: float = 1.0) -> dict:
    return emission_payload(values(ids), LABELS, audio={"duration_sec": 2.0},
        crop={"effective_start_sec": 0.0, "effective_end_sec": end},
        identity={}, runtime={})


@pytest.mark.parametrize(("ids", "expected"), [
    ([0, 0, 2, 2, 0, 2, 0], ["i", "i"]),
    ([0, 0, 3, 3, 0], ["t͡sʰ"]),
    ([0, 0, 0, 0], []),
])
def test_ctc_blank_repeats_and_empty(ids, expected):
    result = payload(ids)
    assert result["decoding"]["prediction_phones"] == expected
    assert len(result["segments"]) == len(expected)


def test_no_phonetic_alias_or_diacritic_guess():
    assert token_phone("/t͡sʰ/") == "t͡sʰ"
    assert token_phone("/a\u0301/") == "á"
    assert token_phone("/iː/") == "iː"
    assert token_phone("text") is None
    result = decoded_tokens([0, 1, 1, 4, 3], LABELS)
    assert result["prediction_phones"] == ["<nonphone:unwrapped>", "t͡sʰ"]
    assert result["special_emissions"] == ["<unk>"]
    assert result["nonphone_emissions"] == ["unwrapped"]


def test_sentencepiece_separator_is_not_a_phone_or_guessed_text():
    labels = LABELS + ["▁", "▁text"]
    result = decoded_tokens([0, 6, 2, 7, 0], labels)
    assert result["prediction_phones"] == ["i", "<nonphone:▁text>"]
    assert result["separator_emissions"] == ["▁"]
    assert result["collapsed_raw_labels"] == ["▁", "/i/", "▁text"]
    legacy = decoded_tokens([0, 6, 2, 7, 0], labels, legacy=True)
    assert legacy["prediction_phones"][0] == "<nonphone:▁>"


def test_padding_emission_not_deleted_to_improve_score():
    result = payload([0, 0, 2, 0, 5, 5, 0], end=0.06)
    assert result["decoding"]["prediction_phones"] == ["i", "n"]
    assert len(result["segments"]) == 1
    assert result["decoding"]["untimed_phone_emissions"][0]["phone"] == "n"
    assert result["decoding"]["padding_emissions_retained_in_scoring"] is True


def test_prefix_not_given_fake_audio_time():
    result = payload([2, 0, 5, 0])
    assert result["decoding"]["untimed_phone_emissions"][0]["reason"] == "prefix_position"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_illegal_logits_rejected(bad):
    data = values([0, 0, 2, 0])
    data[2, 2] = bad
    with pytest.raises(ValueError, match="invalid logits"):
        emission_payload(data, LABELS, audio={"duration_sec": 2},
            crop={"effective_start_sec": 0, "effective_end_sec": 1}, identity={}, runtime={})


def test_source_time_offset_is_preserved():
    result = emission_payload(values([0, 0, 2, 0]), LABELS, audio={"duration_sec": 15},
        crop={"effective_start_sec": 10, "effective_end_sec": 11}, identity={}, runtime={})
    assert result["segments"][0]["start_sec"] == pytest.approx(10.01)
    assert result["segments"][0]["end_sec"] == pytest.approx(10.05)
    assert result["segments"][0]["boundary_uncertainty"]["phone_duration_verified"] is False


def make_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    audio = tmp_path / "source.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 32000)
    raw, vocab, posterior = [tmp_path / name for name in ("logits.safetensors", "vocabulary.json", "posterior.json")]
    save_file({"frame_logits": values([0, 0, 2, 2, 0, 3, 0])}, str(raw))
    vocab.write_text(json.dumps(LABELS), encoding="utf-8")
    identity = {"model_id": MODEL_ID, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
                "vocabulary_sha256": sha256_file(vocab)}
    result = emission_payload(values([0, 0, 2, 2, 0, 3, 0]), LABELS,
        audio=describe_audio_file(audio), crop={"effective_start_sec": 0, "effective_end_sec": 1},
        identity=identity, runtime={"fixture_test": True})
    result["raw_evidence"] = {"file": raw.name, "sha256": sha256_file(raw), "vocabulary_file": vocab.name}
    posterior.write_text(json.dumps(result), encoding="utf-8")
    return audio, posterior, result


def test_fixture_replays_raw_evidence(tmp_path):
    audio, posterior, _ = make_fixture(tmp_path)
    real = PowsmPosteriorRecognizer(posterior).recognize(audio)
    fixture = PowsmPosteriorRecognizer(posterior, fixture=True).recognize(audio)
    assert [s.normalized_ipa for s in real] == ["i", "t͡sʰ"]
    assert [(s.time_range, s.candidate_phones) for s in real] == [(s.time_range, s.candidate_phones) for s in fixture]
    with pytest.raises(ValueError, match="audio only"):
        PowsmPosteriorRecognizer(posterior).recognize(audio, lyrics="")


def test_modified_projection_rejected(tmp_path):
    audio, posterior, result = make_fixture(tmp_path)
    result["segments"][0]["start_sec"] += 0.02
    posterior.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its raw logits"):
        PowsmPosteriorRecognizer(posterior).recognize(audio)


def test_modified_audio_and_model_identity_rejected(tmp_path):
    audio, posterior, result = make_fixture(tmp_path)
    result["producer"]["model_metadata"]["revision"] = "0" * 40
    posterior.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected POWSM"):
        PowsmPosteriorRecognizer(posterior).recognize(audio)
    result["producer"]["model_metadata"]["revision"] = REVISION
    posterior.write_text(json.dumps(result), encoding="utf-8")
    with audio.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="audio hash mismatch"):
        PowsmPosteriorRecognizer(posterior).recognize(audio)


def test_external_raw_evidence_path_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        adjacent_file(tmp_path / "result.json", "../outside.safetensors")


def test_raw_tensor_hash_mismatch_rejected(tmp_path):
    audio, posterior, _ = make_fixture(tmp_path)
    with (tmp_path / "logits.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PowsmPosteriorRecognizer(posterior).recognize(audio)


def test_float64_rounding_allowed_but_material_change_rejected(tmp_path):
    audio, posterior, result = make_fixture(tmp_path)
    result["segments"][0]["confidence"] -= 1e-15
    posterior.write_text(json.dumps(result), encoding="utf-8")
    assert len(PowsmPosteriorRecognizer(posterior).recognize(audio)) == 2
    result["segments"][0]["confidence"] -= 1e-5
    posterior.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its raw logits"):
        PowsmPosteriorRecognizer(posterior).recognize(audio)


@pytest.mark.parametrize("change", [
    {"lyrics": "forbidden"}, {"song_id": "king-of-karaoke", "split": "test"},
    {"song_id": "shalong", "split": "train"}, {"audio_role": "rendered"},
])
def test_audio_only_worker_rejects_label_leakage_and_wrong_roles(change):
    from any2xsampa.adapters.powsm_runtime import PowsmRuntime
    row = {"example_id": "one", "song_id": "promise", "split": "train", "audio_role": "original",
        "audio_path": "not-opened.wav", "audio_sha256": "a" * 64, "start_sec": 0.0, "end_sec": 8.0}
    row.update(change)
    uninitialized = object.__new__(PowsmRuntime)
    with pytest.raises(ValueError):
        uninitialized.encode(row)

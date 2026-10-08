from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest
np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")

from any2xsampa.adapters.audio_provenance import describe_audio_file
from any2xsampa.adapters.phoneticxeus import (
    PhoneticXeusPosteriorRecognizer, convolution_geometry, ctc_collapse, emission_payload,
)
from any2xsampa.training.limited_data_artifacts import sha256_file, write_json_new


VOCAB = {"<blank>": 0, "y": 1, "k\u02b0": 2, "<unk>": 3}


def payload(ids: list[int], audio: dict | None = None) -> dict:
    logits = np.full((len(ids), len(VOCAB)), -5.0, dtype=np.float32)
    logits[np.arange(len(ids)), ids] = 5
    return emission_payload(
        logits, VOCAB, input_audio=audio or {},
        crop={"effective_start_sec": 0.5, "effective_end_sec": 1.5},
        model_metadata={"fixture": True},
        geometry={"frame_count": len(ids), "stride_samples": 320, "receptive_field_samples": 400},
        runtime={"fixture": True},
    )


def wav(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(64000))
    return path


def store(tmp_path: Path, value: dict) -> Path:
    from safetensors.numpy import save_file

    evidence = tmp_path / "evidence.safetensors"
    logits = np.full((value["decoding"]["frame_count"], len(VOCAB)), -5.0, dtype=np.float32)
    logits[:, 0] = 5
    for row in value["segments"]:
        trace = row["raw_label_trace"]
        logits[trace["start_frame"]:trace["end_frame_exclusive"]] = -5
        logits[trace["start_frame"]:trace["end_frame_exclusive"], trace["label_id"]] = 5
    save_file({"frame_logits": logits}, str(evidence))
    vocabulary_path = tmp_path / "vocabulary.json"
    write_json_new(vocabulary_path, VOCAB)
    value["producer"]["model_metadata"]["vocabulary_sha256"] = sha256_file(vocabulary_path)
    value["raw_evidence"] = {"file": evidence.name, "sha256": sha256_file(evidence)}
    value["raw_evidence"]["vocabulary_file"] = vocabulary_path.name
    path = tmp_path / "posterior.json"
    write_json_new(path, value)
    return path


def test_ctc_repetition_blank_and_special_tokens() -> None:
    assert ctc_collapse([1, 1, 0, 1, 2, 2, 3]) == [1, 1, 2, 3]
    assert ctc_collapse([0, 0, 0]) == []
    with pytest.raises(ValueError):
        ctc_collapse([True])


def test_geometry_matches_wav2vec_frontend() -> None:
    shape = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]
    assert convolution_geometry(shape, 16000) == {"frame_count": 49, "stride_samples": 320, "receptive_field_samples": 400}
    with pytest.raises(ValueError, match="too short"):
        convolution_geometry(shape, 10)


def test_rich_ipa_event_windows_and_special_evidence() -> None:
    value = payload([1, 1, 0, 1, 2, 3])
    assert [row["candidate_phones"][0]["symbol"] for row in value["segments"]] == ["y", "y", "k\u02b0"]
    assert value["decoding"]["collapsed_raw_ids"] == [1, 1, 2, 3]
    assert value["decoding"]["special_emissions"] == ["<unk>"]
    assert value["segments"][0]["start_sec"] > 0.5
    assert value["segments"][0]["end_sec"] < value["segments"][1]["start_sec"]
    assert all(not row["boundary_uncertainty"]["phone_duration_verified"] for row in value["segments"])
    assert all(not row["posterior_summary"]["renormalized_after_truncation"] for row in value["segments"])


def test_codec_rejects_nonfinite_and_wrong_shape() -> None:
    kwargs = dict(vocabulary=VOCAB, input_audio={}, crop={"effective_start_sec": 0, "effective_end_sec": 1}, model_metadata={}, geometry={"frame_count": 1, "stride_samples": 320, "receptive_field_samples": 400}, runtime={})
    with pytest.raises(ValueError, match="non-finite"):
        emission_payload(np.array([[0, 1, 2, np.nan]]), **kwargs)
    with pytest.raises(ValueError, match="non-empty"):
        emission_payload(np.zeros((0, 4)), **kwargs)


def test_real_and_fixture_replay_preserve_phone_contract(tmp_path: Path) -> None:
    audio = wav(tmp_path)
    path = store(tmp_path, payload([1, 0, 2], describe_audio_file(audio)))
    real = PhoneticXeusPosteriorRecognizer(path).recognize(audio)
    fixture = PhoneticXeusPosteriorRecognizer(path, fixture=True).recognize(audio)
    assert [(row.time_range, row.candidate_phones, row.confidence) for row in real] == [(row.time_range, row.candidate_phones, row.confidence) for row in fixture]
    assert [row.candidate_phones[0].symbol for row in real] == ["y", "k\u02b0"]
    assert all(row.metadata["audio_binding"]["verified"] for row in real)
    with pytest.raises(ValueError, match="audio-only"):
        PhoneticXeusPosteriorRecognizer(path).recognize(audio, lyrics="forbidden")


def test_empty_emissions_are_not_stub_output(tmp_path: Path) -> None:
    audio = wav(tmp_path)
    path = store(tmp_path, payload([0, 0, 0], describe_audio_file(audio)))
    assert PhoneticXeusPosteriorRecognizer(path).recognize(audio) == []


def test_audio_and_evidence_hash_mismatches_fail(tmp_path: Path) -> None:
    audio = wav(tmp_path)
    path = store(tmp_path, payload([1], describe_audio_file(audio)))
    (tmp_path / "evidence.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PhoneticXeusPosteriorRecognizer(path).recognize(audio)
    value = json.loads(path.read_text())
    value["producer"]["input_audio"]["sha256"] = "f" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="audio hash mismatch"):
        PhoneticXeusPosteriorRecognizer(path).recognize(audio)


def test_tampered_candidates_do_not_pass_raw_replay(tmp_path: Path) -> None:
    audio = wav(tmp_path)
    path = store(tmp_path, payload([1, 0, 2], describe_audio_file(audio)))
    value = json.loads(path.read_text())
    value["segments"][0]["candidate_phones"][0]["symbol"] = "u"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="candidates differ"):
        PhoneticXeusPosteriorRecognizer(path).recognize(audio)


def test_sealed_posterior_is_not_opened(tmp_path: Path) -> None:
    audio = wav(tmp_path)
    value = payload([1], describe_audio_file(audio))
    value["test_outputs_sealed"] = True
    path = store(tmp_path, value)
    with pytest.raises(ValueError, match="sealed"):
        PhoneticXeusPosteriorRecognizer(path).recognize(audio)

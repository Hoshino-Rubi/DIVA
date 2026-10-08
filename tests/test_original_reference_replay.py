from __future__ import annotations

from pathlib import Path
import wave

import pytest

np = pytest.importorskip("numpy")
safetensors = pytest.importorskip("safetensors.numpy")

from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256, emission_payload
from any2xsampa.training.limited_data_artifacts import sha256_file, write_json_new
from any2xsampa.training.original_reference_evaluation import replay_prediction


@pytest.fixture
def evidence(tmp_path: Path) -> tuple[dict, dict, Path, Path, dict]:
    audio = tmp_path / "original.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    vocabulary = {"<blank>": 0, "k\u02b0": 1, "y": 2}
    vocab_path = tmp_path / "vocabulary.json"
    write_json_new(vocab_path, vocabulary)
    logits = np.array([[3, 0, 0], [0, 3, 0], [0, 3, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=np.float32)
    raw_path = tmp_path / "evidence.safetensors"
    safetensors.save_file({"frame_logits": logits}, str(raw_path))
    model = {"model_id": MODEL_ID, "revision": REVISION, "weights_sha256": WEIGHT_SHA256, "vocabulary_sha256": sha256_file(vocab_path)}
    payload = emission_payload(logits, vocabulary, input_audio={"sha256": sha256_file(audio), "sample_rate": 16000},
        crop={"requested_start_sec": 0.0, "requested_end_sec": 1.0, "effective_start_sec": 0.0, "effective_end_sec": 1.0},
        model_metadata=model, geometry={"frame_count": 6, "stride_samples": 320, "receptive_field_samples": 400}, runtime={})
    identity = {"example_id": "shalong-original-01", "song_id": "shalong", "split": "dev", "audio_role": "original", "test_outputs_sealed": False}
    payload.update(identity)
    payload["raw_evidence"] = {"file": raw_path.name, "sha256": sha256_file(raw_path)}
    posterior = tmp_path / "posterior.json"
    write_json_new(posterior, payload)
    row = {**identity, "start_sec": 0.0, "end_sec": 1.0, "audio_path": str(audio), "audio_sha256": sha256_file(audio)}
    entry = {**identity, "posterior_path": str(posterior), "posterior_sha256": sha256_file(posterior),
             "evidence_path": str(raw_path), "evidence_sha256": sha256_file(raw_path)}
    return row, entry, vocab_path, posterior, payload


def test_real_fixture_contract_and_ctc_blank_repetition(evidence) -> None:
    row, entry, vocab, _, _ = evidence
    result = replay_prediction(row, entry, vocab)
    assert result["phones"] == ["k\u02b0", "k\u02b0", "y"]
    assert result["segments"] == 3
    assert result["real_fixture_match"]


def test_sealed_file_rejected_before_read(evidence) -> None:
    row, entry, vocab, _, _ = evidence
    entry.update(test_outputs_sealed=True, posterior_path="does-not-exist.json")
    with pytest.raises(ValueError, match="sealed"):
        replay_prediction(row, entry, vocab)


def test_time_offset_mismatch(evidence) -> None:
    row, entry, vocab, _, _ = evidence
    row["start_sec"] = 0.01
    with pytest.raises(ValueError, match="crop time"):
        replay_prediction(row, entry, vocab)


def test_wrong_original_audio_hash(evidence) -> None:
    row, entry, vocab, _, _ = evidence
    row["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audio hash"):
        replay_prediction(row, entry, vocab)


def test_stale_raw_cache_binding(evidence) -> None:
    row, entry, vocab, _, _ = evidence
    entry["evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw cache"):
        replay_prediction(row, entry, vocab)


def test_reference_conditioned_posterior_rejected(evidence, tmp_path: Path) -> None:
    row, entry, vocab, _, payload = evidence
    payload["producer"]["lyrics_used"] = True
    changed = tmp_path / "conditioned-posterior.json"
    write_json_new(changed, payload)
    entry.update(posterior_path=str(changed), posterior_sha256=sha256_file(changed))
    with pytest.raises(ValueError, match="audio-only"):
        replay_prediction(row, entry, vocab)


def test_nonfinite_posterior_rejected(evidence) -> None:
    row, entry, vocab, posterior, _ = evidence
    with posterior.open("w", encoding="utf-8") as stream:
        stream.write('{"bad":NaN}')
    entry["posterior_sha256"] = sha256_file(posterior)
    with pytest.raises(ValueError, match="non-finite"):
        replay_prediction(row, entry, vocab)

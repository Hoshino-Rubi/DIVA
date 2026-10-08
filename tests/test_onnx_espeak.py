from __future__ import annotations

import json
from pathlib import Path
import struct
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from any2xsampa.adapters.onnx_espeak import (
    OnnxEspeakCTCUniversalPhoneRecognizer as Recognizer,
    OnnxEspeakResourceError as ResourceError,
)


class FakeSession:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits
        self.feed: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_values")]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        del outputs
        self.feed = feed
        return [self.logits]


def _write_resources(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "model_int8.onnx"
    model.write_bytes(b"candidate-test-model")
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps({"<pad>": 0, "a": 1, "b": 2, "s": 3, "q": 4}),
        encoding="utf-8",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": "test_mapping_v1",
                "profile": "test",
                "blank_labels": ["<pad>"],
                "mappings": {"a": ["a"], "p": ["b"], "s": ["s"]},
            }
        ),
        encoding="utf-8",
    )
    return model, vocab, mapping


def _write_wav(path: Path) -> None:
    samples = [int(5000 * np.sin(2 * np.pi * 220 * index / 16000)) for index in range(1600)]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def test_mocked_ctc_inference_preserves_phone_contract_and_ignores_lyrics(tmp_path: Path) -> None:
    model, vocab, mapping = _write_resources(tmp_path)
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    logits = np.asarray(
        [
            [
                [8.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0, 0.0],
                [0.0, 7.0, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 8.0],
            ]
        ],
        dtype=np.float32,
    )
    fake = FakeSession(logits)
    recognizer = Recognizer(
        model,
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="cpu",
        session_factory=lambda path, providers: fake,
    )

    segments = recognizer.recognize(audio, lyrics="must not guide recognition")

    assert [segment.candidate_phones[0].symbol for segment in segments] == ["a", "q"]
    assert segments[0].time_range.start_sec == pytest.approx(0.02)
    assert segments[0].time_range.end_sec == pytest.approx(0.06)
    assert segments[1].time_range.start_sec == pytest.approx(0.08)
    assert segments[1].time_range.end_sec == pytest.approx(0.1)
    assert all(segment.normalized_ipa == "" and segment.xsampa == "" for segment in segments)
    assert segments[0].metadata["source_mode"] == "real"
    assert segments[0].metadata["real_model_output"] is True
    assert segments[0].metadata["lyrics_guided"] is False
    assert segments[0].metadata["lyrics_input_ignored"] is True
    assert segments[0].metadata["posterior_trace"]["frames"]
    assert segments[0].metadata["raw_label_trace"]["frames"]
    assert fake.feed is not None and fake.feed["input_values"].shape == (1, 1600)


def test_configured_label_normalization_preserves_raw_trace(tmp_path: Path) -> None:
    model = tmp_path / "model_int8.onnx"
    model.write_bytes(b"candidate-test-model")
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"<pad>": 0, "iɑ5": 1}), encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": "test_mapping_v1",
                "profile": "test",
                "blank_labels": ["<pad>"],
                "normalization": {
                    "strip_trailing_tone_digits": True,
                    "aliases": {"iɑ": "a"},
                },
                "mappings": {"a": ["a"]},
            }
        ),
        encoding="utf-8",
    )
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    fake = FakeSession(np.asarray([[[0.0, 8.0], [8.0, 0.0]]], dtype=np.float32))
    recognizer = Recognizer(
        model,
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="cpu",
        session_factory=lambda path, providers: fake,
    )

    segment = recognizer.recognize(audio)[0]

    assert segment.candidate_phones[0].symbol == "a"
    raw = segment.metadata["raw_label_trace"]["frames"][0]["candidates"][0]
    assert raw["label"] == "iɑ5"
    assert raw["mapped_symbol"] == "a"


def test_missing_model_fails_clearly_without_stub(tmp_path: Path) -> None:
    _, vocab, mapping = _write_resources(tmp_path)
    recognizer = Recognizer(
        tmp_path / "missing.onnx",
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="cpu",
    )

    with pytest.raises(ResourceError, match="model resource not found"):
        recognizer._load_runtime()


def test_missing_vocabulary_fails_before_inference(tmp_path: Path) -> None:
    model, _, mapping = _write_resources(tmp_path)
    recognizer = Recognizer(
        model,
        vocab_path=tmp_path / "missing-vocab.json",
        phone_mapping_path=mapping,
        device="cpu",
    )

    with pytest.raises(ResourceError, match="vocabulary file not found"):
        recognizer._load_runtime()


def test_sha256_mismatch_is_a_hard_resource_error(tmp_path: Path) -> None:
    model, vocab, mapping = _write_resources(tmp_path)
    recognizer = Recognizer(
        model,
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="cpu",
        expected_model_sha256="0" * 64,
    )

    with pytest.raises(ResourceError, match="SHA-256 mismatch"):
        recognizer._load_runtime()


def test_chunked_prediction_projects_frames_to_global_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, vocab, mapping = _write_resources(tmp_path)
    recognizer = Recognizer(
        model,
        vocab_path=vocab,
        phone_mapping_path=mapping,
        device="cpu",
        chunk_duration_sec=1.0,
        chunk_overlap_sec=0.2,
    )
    calls: list[int] = []

    def fake_predict(samples: np.ndarray, sample_rate: int, **_: object) -> list[dict[str, int]]:
        del sample_rate
        calls.append(len(samples))
        return [{"start_index": 0, "end_index": len(samples)}]

    monkeypatch.setattr(recognizer, "_predict_frames", fake_predict)
    projected = recognizer._predict_chunked_frames(np.zeros(32000, dtype=np.float32), 16000)

    assert len(calls) == 3
    assert [item["_chunk_index"] for item in projected] == [0, 1, 2]
    assert [(item["start_index"], item["end_index"]) for item in projected] == [
        (0, 16000),
        (12800, 28800),
        (25600, 32000),
    ]

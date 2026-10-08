from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from any2xsampa.adapters import checkpoint_loader
from any2xsampa.adapters import recognizer_cantonese_singing_ctc as ctc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path) -> Path:
    (root / "backbone").mkdir(parents=True)
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    vocab = root / "vocab.json"
    vocab.write_text(json.dumps({"phones": ["<blank>", "a", "b", "c"]}), encoding="utf-8")
    license_manifest = root / "license_manifest.json"
    entries = [
        {
            "asset_id": name,
            "asset_sha256": _sha256(path),
            "source": "synthetic:test",
            "revision": "r1",
            "license_id": "LicenseRef-Synthetic-Test",
            "license_source": "local:test-fixture",
            "status": "approved",
            "approved_uses": ["inference"],
        }
        for name, path in (("checkpoint", checkpoint), ("vocabulary", vocab))
    ]
    license_manifest.write_text(
        json.dumps({"schema_version": "corpus_license_manifest_v1", "entries": entries}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "recognizer_bundle_v1",
        "backbone": {"family": "fixture", "model_id": "fixture", "revision": "test", "path": "backbone"},
        "checkpoint": {"path": "checkpoint.pt", "sha256": _sha256(checkpoint)},
        "vocab": {"path": "vocab.json", "sha256": _sha256(vocab), "blank_id": 0},
        "license_manifest": {"path": "license_manifest.json", "sha256": _sha256(license_manifest)},
        "audio": {"sample_rate": 16_000, "chunk_sec": 8.0, "overlap_sec": 1.0},
        "confidence_calibration": {"temperature": 1.0},
        "milestone": "experimental",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _checkpoint_payload() -> dict[str, Any]:
    return {
        "schema_version": "ranker_checkpoint_v1",
        "vocab": ["<blank>", "a"],
        "dataset_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "split_identity": "singer-group-01",
        "seed": 7,
        "configuration": {"hidden_size": 8},
        "torch_version": "2.13.0",
        "cuda_version": "12.6",
        "state_dict": {"head.weight": [1.0]},
    }


def test_checkpoint_loader_validates_provenance_and_expected_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ranker.json"
    checkpoint.write_text(json.dumps(_checkpoint_payload()), encoding="utf-8")
    loaded = checkpoint_loader.load_trained_checkpoint(
        checkpoint,
        expected_schema="ranker_checkpoint_v1",
        expected_vocabulary=("<blank>", "a"),
    )
    assert loaded.metadata.split_identity == "singer-group-01"
    assert loaded.metadata.seed == 7
    assert loaded.state_dict["head.weight"] == [1.0]

    altered = dict(_checkpoint_payload())
    altered["dataset_sha256"] = "c" * 64
    checkpoint.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(checkpoint_loader.CheckpointLoadError, match="dataset SHA-256"):
        checkpoint_loader.load_trained_checkpoint(
            checkpoint,
            expected_dataset_sha256="a" * 64,
        )


def test_checkpoint_loader_rejects_missing_or_empty_trained_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ranker.json"
    missing = _checkpoint_payload()
    del missing["state_dict"]
    checkpoint.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(checkpoint_loader.CheckpointLoadError, match="must be non-empty"):
        checkpoint_loader.load_trained_checkpoint(checkpoint)

    empty = _checkpoint_payload()
    empty["state_dict"] = {}
    checkpoint.write_text(json.dumps(empty), encoding="utf-8")
    with pytest.raises(checkpoint_loader.CheckpointLoadError, match="must be non-empty"):
        checkpoint_loader.load_trained_checkpoint(checkpoint)


def test_bundle_hash_and_safe_path_validation(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    loaded = ctc.load_recognizer_bundle(bundle)
    assert loaded.phones == ("<blank>", "a", "b", "c")
    assert loaded.sample_rate == 16_000

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["checkpoint"]["path"] = "../checkpoint.pt"
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="safe relative path"):
        ctc.load_recognizer_bundle(bundle)


class _FakeAudio:
    def __init__(self, count: int) -> None:
        self.shape = (count,)

    def __getitem__(self, item: slice) -> "_FakeAudio":
        start = item.start or 0
        stop = min(self.shape[0], item.stop if item.stop is not None else self.shape[0])
        return _FakeAudio(max(0, stop - start))


class _FakeRuntime:
    def __init__(self, calls: list[tuple[int, int]]) -> None:
        self.calls = calls

    def infer(self, audio: _FakeAudio, *, sample_rate: int) -> tuple[list[list[float]], float]:
        self.calls.append((audio.shape[0], sample_rate))
        return [
            [8.0, -8.0, -8.0, -8.0],
            [-8.0, 8.0, -8.0, -8.0],
            [-8.0, 8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0, -8.0],
            [-8.0, -8.0, 8.0, -8.0],
        ], 0.1


def test_real_adapter_decodes_ctc_runs_and_does_not_use_lyrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    audio_path = tmp_path / "vocal.wav"
    audio_path.write_bytes(b"fixture-audio")
    fake_audio = _FakeAudio(16_000 * 9)
    monkeypatch.setattr(ctc, "_load_mono_audio", lambda _path, target_rate: fake_audio)
    calls: list[tuple[int, int]] = []
    recognizer = ctc.CantoneseSingingCTCRecognizer(
        bundle,
        device="cpu",
        runtime_factory=lambda _bundle, _device: _FakeRuntime(calls),
    )

    segments = recognizer.recognize(audio_path, lyrics="lyrics must remain evidence")

    assert len(calls) == 2
    assert all(rate == 16_000 for _, rate in calls)
    assert [segment.candidate_phones[0].symbol for segment in segments] == ["a", "b", "a", "b"]
    assert all(segment.metadata["real_model_output"] is True for segment in segments)
    assert all(segment.metadata["lyrics_used_as_recognizer_input"] is False for segment in segments)
    assert all(segment.metadata["lyrics_used"] is False for segment in segments)
    assert all(len(segment.candidate_phones) <= 3 for segment in segments)
    assert all(left.time_range.end_sec <= right.time_range.start_sec for left, right in zip(segments, segments[1:]))


def test_ctc_decoder_preserves_repeat_after_blank() -> None:
    runs = ctc._decode_chunk_logits(
        [
            [5.0, 0.0, -1.0, -2.0],
            [0.0, 5.0, -1.0, -2.0],
            [5.0, 0.0, -1.0, -2.0],
            [-1.0, -2.0, 5.0, 4.0],
        ],
        phones=["<blank>", "a", "b", "c"],
        blank_id=0,
        temperature=1.0,
        chunk_index=0,
        chunk_offset_sec=0.0,
        frame_hop_sec=0.1,
        audio_end_sec=1.0,
    )
    assert [run.symbol for run in runs] == ["a", "b"]
    assert len(runs[-1].candidates) == 3


def _run(symbol: str, start: float, end: float, confidence: float, chunk: int = 0) -> ctc._DecodedRun:
    return ctc._DecodedRun(
        start,
        end,
        ((symbol, confidence),),
        chunk,
        0,
        1,
        0.1,
    )


def test_ctc_overlap_merge_preserves_nested_prefix_and_suffix() -> None:
    merged = ctc._merge_overlapping_runs(
        [_run("a", 0.0, 8.0, 0.7), _run("b", 7.0, 7.5, 0.9, 1)]
    )
    assert [(row.symbol, row.start_sec, row.end_sec) for row in merged] == [
        ("a", 0.0, 7.0),
        ("b", 7.0, 7.5),
        ("a", 7.5, 8.0),
    ]


def test_ctc_overlap_merge_is_deterministic_for_equal_and_three_way_runs() -> None:
    equal = ctc._merge_overlapping_runs(
        [_run("a", 0.0, 2.0, 0.8, 0), _run("b", 1.0, 3.0, 0.8, 1)]
    )
    assert [(row.symbol, row.start_sec, row.end_sec) for row in equal] == [
        ("a", 0.0, 2.0),
        ("b", 2.0, 3.0),
    ]
    three_way = ctc._merge_overlapping_runs(
        [
            _run("a", 0.0, 5.0, 0.5, 0),
            _run("b", 1.0, 4.0, 0.9, 1),
            _run("c", 2.0, 3.0, 0.95, 2),
        ]
    )
    assert [(row.symbol, row.start_sec, row.end_sec) for row in three_way] == [
        ("a", 0.0, 1.0),
        ("b", 1.0, 2.0),
        ("c", 2.0, 3.0),
        ("b", 3.0, 4.0),
        ("a", 4.0, 5.0),
    ]


def test_frozen_backbone_hash_is_bound_to_bundle_weights(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path / "bundle")
    weights = bundle_path / "backbone" / "pytorch_model.bin"
    weights.write_bytes(b"backbone-weights")
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["backbone"]["weights_path"] = "backbone/pytorch_model.bin"
    manifest["backbone"]["sha256"] = _sha256(weights)
    license_manifest = bundle_path / "license_manifest.json"
    license_payload = json.loads(license_manifest.read_text(encoding="utf-8"))
    license_payload["entries"].append(
        {
            "asset_id": "backbone",
            "asset_sha256": _sha256(weights),
            "source": "synthetic:test",
            "revision": "r1",
            "license_id": "LicenseRef-Synthetic-Test",
            "license_source": "local:test-fixture",
            "status": "approved",
            "approved_uses": ["inference"],
        }
    )
    license_manifest.write_text(json.dumps(license_payload), encoding="utf-8")
    manifest["license_manifest"]["sha256"] = _sha256(license_manifest)
    (bundle_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = ctc.load_recognizer_bundle(bundle_path)
    payload = {"configuration": {"backbone_frozen": True, "backbone_sha256": _sha256(weights)}}
    ctc._validate_frozen_backbone_identity(payload, bundle)
    payload["configuration"]["backbone_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        ctc._validate_frozen_backbone_identity(payload, bundle)


def test_bundle_requires_an_approved_license_manifest_schema(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    license_manifest = bundle / "license_manifest.json"
    license_manifest.write_text(json.dumps({"schema_version": "", "entries": []}), encoding="utf-8")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["license_manifest"]["sha256"] = _sha256(license_manifest)
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema is unsupported"):
        ctc.load_recognizer_bundle(bundle)


def test_bundle_rejects_empty_or_uncovered_license_entries(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    license_manifest = bundle / "license_manifest.json"
    license_manifest.write_text(
        json.dumps({"schema_version": "corpus_license_manifest_v1", "entries": []}),
        encoding="utf-8",
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["license_manifest"]["sha256"] = _sha256(license_manifest)
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty entries"):
        ctc.load_recognizer_bundle(bundle)

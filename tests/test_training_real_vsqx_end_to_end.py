from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from any2xsampa.adapters import recognizer_cantonese_singing_ctc as recognizer
from any2xsampa.exporters.engine import MappingEnginePhoneExporter
from any2xsampa.training import cantonese_singing_ctc as training
from any2xsampa.vocaloid import draft_from_phone_segments, export_vsqx_from_template


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _human_truth(tokens: list[str]) -> dict[str, object]:
    return {
        "target_phone_tokens": tokens,
        "internal_ipa_tokens": list(tokens),
        "articulatory_features": [{"class": "fixture"} for _ in tokens],
        "human_authored_annotation_evidence": True,
        "label_provenance": {
            "schema_version": "human_phone_label_v1",
            "kind": "human",
            "source_id": "fixture",
            "source_revision": "r1",
            "annotation_sha256": "d" * 64,
            "policy_version": "fixture-v1",
            "annotator_id": "tester",
            "review_status": "approved",
        },
    }


class _Audio:
    shape = (16_000,)

    def __getitem__(self, _item: slice) -> "_Audio":
        return self


class _Runtime:
    def infer(self, _audio: _Audio, *, sample_rate: int) -> tuple[list[list[float]], float]:
        assert sample_rate == 16_000
        return [[-8.0, 8.0, -8.0], [8.0, -8.0, -8.0], [-8.0, -8.0, 8.0]], 0.2


def _template(path: Path) -> Path:
    namespace = recognizer.__dict__.get("VSQ4_NS", "http://www.yamaha.co.jp/vocaloid/schema/vsq4/")
    root = ET.Element(f"{{{namespace}}}vsq4")
    master = ET.SubElement(root, f"{{{namespace}}}masterTrack")
    ET.SubElement(master, f"{{{namespace}}}resolution").text = "480"
    track = ET.SubElement(root, f"{{{namespace}}}vsTrack")
    ET.SubElement(track, f"{{{namespace}}}vsPart")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_dataset_checkpoint_real_adapter_to_template_vsqx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": "cantonese_singing_ctc_dataset_v1",
                "examples": [
                    {"example_id": "train", "group_id": "song-a", **_human_truth(["a", "n"])},
                    {"example_id": "valid", "group_id": "song-b", **_human_truth(["a"])},
                ],
                "folds": [{"validation_group": "song-b", "train_groups": ["song-a"]}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "features.pt"
    features = {
        "train": torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
        "valid": torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
    }
    training.write_feature_cache(dataset_path=dataset, features=features, output_path=cache, model_sha256="c" * 64)
    training.train_ctc_head(
        dataset_path=dataset,
        feature_cache=cache,
        output_root=tmp_path / "trained",
        epochs=1,
        patience=1,
        batch_size=1,
        device_name="cpu",
    )
    checkpoint = tmp_path / "trained" / "model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    bundle = tmp_path / "bundle"
    (bundle / "backbone").mkdir(parents=True)
    bundled_checkpoint = bundle / "checkpoint.pt"
    bundled_checkpoint.write_bytes(checkpoint.read_bytes())
    vocab = bundle / "vocab.json"
    vocab.write_text(json.dumps({"phones": payload["vocabulary"]}), encoding="utf-8")
    license_manifest = bundle / "license_manifest.json"
    license_manifest.write_text(
        json.dumps(
            {
                "schema_version": "corpus_license_manifest_v1",
                "entries": [
                    {
                        "asset_id": name,
                        "asset_sha256": _sha(path),
                        "source": "synthetic:test",
                        "revision": "r1",
                        "license_id": "LicenseRef-Synthetic-Test",
                        "license_source": "local:test-fixture",
                        "status": "approved",
                        "approved_uses": ["inference"],
                    }
                    for name, path in (("checkpoint", bundled_checkpoint), ("vocabulary", vocab))
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "recognizer_bundle_v1",
        "backbone": {"family": "wavlm", "model_id": "toy", "revision": "local", "path": "backbone"},
        "checkpoint": {"path": "checkpoint.pt", "sha256": _sha(bundled_checkpoint)},
        "vocab": {"path": "vocab.json", "sha256": _sha(vocab), "blank_id": 0},
        "license_manifest": {"path": "license_manifest.json", "sha256": _sha(license_manifest)},
        "audio": {"sample_rate": 16_000, "chunk_sec": 8.0, "overlap_sec": 1.0},
        "confidence_calibration": {"temperature": 1.0},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fixture")
    monkeypatch.setattr(recognizer, "_load_mono_audio", lambda _path, target_rate: _Audio())
    adapter = recognizer.CantoneseSingingCTCRecognizer(
        bundle, device="cpu", runtime_factory=lambda _bundle, _device: _Runtime()
    )
    segments = adapter.recognize(audio_path)
    assert [segment.candidate_phones[0].symbol for segment in segments] == ["a", "n"]
    for segment in segments:
        segment.normalized_ipa = segment.candidate_phones[0].symbol
    config = Path(training.__file__).resolve().parents[1] / "config" / "exporter_vocaloid_like.json"
    segments = MappingEnginePhoneExporter("vocaloid_like", config).export(segments)
    draft = draft_from_phone_segments(segments, pitches=[60, 62], lyrics=["la", "na"])
    output = export_vsqx_from_template(draft, tmp_path / "result.vsqx", template_path=_template(tmp_path / "template.vsqx"))
    tree = ET.parse(output)
    rendered = [node.text for node in tree.getroot().iter() if node.tag.endswith("}p")]
    assert rendered == ["a", "n"]

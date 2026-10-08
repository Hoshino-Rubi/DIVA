from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from any2xsampa.training.acoustic_features import (
    FrozenWavLMFeatureExtractor,
    FrozenXLSRFeatureExtractor,
    LibrosaLogMelFeatureExtractor,
    OnnxEspeakLogitFeatureExtractor,
)


def test_log_mel_feature_identity_is_stable() -> None:
    first = LibrosaLogMelFeatureExtractor().metadata()
    second = LibrosaLogMelFeatureExtractor().metadata()
    assert first == second
    assert len(str(first["model_sha256"])) == 64
    assert first["pretrained_model"] is False


def test_onnx_feature_extractor_rejects_invalid_cpu_thread_limit() -> None:
    try:
        OnnxEspeakLogitFeatureExtractor(device="cpu", cpu_thread_limit=0)
    except ValueError as exc:
        assert "cpu_thread_limit" in str(exc)
    else:
        raise AssertionError("invalid thread limits must fail before inference")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wavlm_manifest(tmp_path: Path) -> Path:
    files = {
        "README.md": (
            "model_card",
            "# Fixture\n\nThe official license can be found "
            "[here](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)\n",
        ),
        "MODEL_INFO.json": (
            "upstream_model_metadata",
            json.dumps(
                {
                    "id": "fixture/wavlm",
                    "sha": "1" * 40,
                    "private": False,
                    "gated": False,
                }
            ),
        ),
        "config.json": (
            "model_config",
            json.dumps({"model_type": "wavlm", "hidden_size": 4, "num_hidden_layers": 2}),
        ),
        "preprocessor_config.json": (
            "preprocessor_config",
            json.dumps({"sampling_rate": 16000}),
        ),
        "pytorch_model.bin": ("model_weights", "fixture-weights"),
        "UPSTREAM_LICENSE": (
            "license",
            "Creative Commons Attribution-ShareAlike 3.0 Unported\n",
        ),
    }
    entries = []
    for name, (role, content) in files.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        entries.append(
            {
                "path": name,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = tmp_path / "backbone_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "pretrained_acoustic_backbone_manifest_v1",
                "candidate_id": "fixture",
                "family": "wavlm",
                "repository": "fixture/wavlm",
                "revision": "1" * 40,
                "retrieval": {
                    "direct_network_fallback": False,
                    "local_files_only_runtime": True,
                },
                "feature_contract": {
                    "sample_rate": 16000,
                    "hidden_size": 4,
                },
                "license": {
                    "identifier": "CC-BY-SA-3.0",
                    "non_commercial_restriction": False,
                    "evidence_kind": "model_card_explicit_link",
                    "text_marker": "Attribution-ShareAlike 3.0 Unported",
                    "evidence_path": "README.md",
                    "license_path": "UPSTREAM_LICENSE",
                    "source_url": "https://github.com/microsoft/UniSpeech/blob/main/LICENSE",
                },
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_wavlm_snapshot_metadata_is_hash_bound_without_loading_runtime(tmp_path: Path) -> None:
    manifest = _wavlm_manifest(tmp_path)
    metadata = FrozenWavLMFeatureExtractor(manifest, hidden_layer=2).metadata()
    assert metadata["model_sha256"] == _sha256(tmp_path / "pytorch_model.bin")
    assert metadata["revision"] == "1" * 40
    assert metadata["license"] == "CC-BY-SA-3.0"
    assert metadata["local_files_only"] is True
    assert metadata["lyrics_used_as_acoustic_input"] is False


def test_wavlm_snapshot_rejects_tampered_weights(tmp_path: Path) -> None:
    manifest = _wavlm_manifest(tmp_path)
    (tmp_path / "pytorch_model.bin").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        FrozenWavLMFeatureExtractor(manifest)


def test_wavlm_extractor_rejects_cpu_without_falling_back(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit CUDA"):
        FrozenWavLMFeatureExtractor(tmp_path / "missing.json", device="cpu")


def test_xlsr_snapshot_requires_structured_license_and_wav2vec2_config(tmp_path: Path) -> None:
    manifest_path = _wavlm_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["family"] = "wav2vec2_xls_r"
    manifest["license"].update(
        {
            "identifier": "Apache-2.0",
            "upstream_identifier": "apache-2.0",
            "evidence_kind": "structured_upstream_metadata",
            "text_marker": "Apache License",
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model_type": "wav2vec2", "hidden_size": 4, "num_hidden_layers": 2}),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "MODEL_INFO.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cardData"] = {"license": "apache-2.0"}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    license_path = tmp_path / "UPSTREAM_LICENSE"
    license_path.write_text("Apache License\nVersion 2.0\n", encoding="utf-8")
    for entry in manifest["files"]:
        path = tmp_path / entry["path"]
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = FrozenXLSRFeatureExtractor(manifest_path, hidden_layer=1).metadata()
    assert result["family"] == "wav2vec2_xls_r"
    assert result["license"] == "Apache-2.0"
    assert result["hidden_layer"] == 1

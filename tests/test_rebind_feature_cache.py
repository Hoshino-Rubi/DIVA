from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from any2xsampa.training import cantonese_singing_ctc as ctc


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "rebind_feature_cache_dataset.py"
SPEC = importlib.util.spec_from_file_location("rebind_feature_cache_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CHECKPOINT_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "rebind_ctc_checkpoint_dataset.py"
CHECKPOINT_SPEC = importlib.util.spec_from_file_location("rebind_ctc_checkpoint_dataset", CHECKPOINT_MODULE_PATH)
assert CHECKPOINT_SPEC is not None and CHECKPOINT_SPEC.loader is not None
CHECKPOINT_MODULE = importlib.util.module_from_spec(CHECKPOINT_SPEC)
CHECKPOINT_SPEC.loader.exec_module(CHECKPOINT_MODULE)


def _truth() -> dict[str, object]:
    return {
        "target_phone_tokens": ["a", "i"],
        "internal_ipa_tokens": ["a", "i"],
        "articulatory_features": [{"class": "vowel"}, {"class": "vowel"}],
        "human_authored_annotation_evidence": True,
        "label_provenance": {
            "schema_version": "human_phone_label_v1",
            "kind": "human",
            "source_id": "fixture",
            "source_revision": "r1",
            "annotation_sha256": "a" * 64,
            "policy_version": "fixture",
            "annotator_id": "tester",
            "review_status": "approved",
        },
    }


def _write_dataset(path: Path, *, crop_end: float = 1.0, marker: str = "old") -> Path:
    payload = {
        "schema_version": "cantonese_singing_ctc_dataset_v1",
        "marker": marker,
        "examples": [
            {
                "example_id": "song-1",
                "group_id": "song",
                "audio_path": "rendered.wav",
                "audio_sha256": "b" * 64,
                "crop_start_sec": 0.0,
                "crop_end_sec": crop_end,
                "model_audio_sample_rate": 16000,
                "model_audio_channels": 1,
                **_truth(),
            }
        ],
        "folds": [{"fold_index": 0, "validation_group": "song", "train_groups": ["other"]}],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_cache(dataset: Path, path: Path, *, features: dict[str, object] | None = None) -> None:
    torch = pytest.importorskip("torch")
    values = features or {"song-1": torch.ones((4, 2), dtype=torch.float32)}
    ctc.write_feature_cache(
        dataset_path=dataset,
        features=values,
        output_path=path,
        model_sha256="c" * 64,
    )


def test_rebind_accepts_provenance_only_dataset_change(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old.json")
    new = _write_dataset(tmp_path / "new.json", marker="new")
    cache = tmp_path / "cache.pt"
    output = tmp_path / "rebound.pt"
    _write_cache(old, cache)

    summary = MODULE.rebind_feature_cache(old, new, cache, output)

    assert summary["example_count"] == 1
    rebound = __import__("torch").load(output, map_location="cpu", weights_only=True)
    assert rebound["dataset_sha256"] == summary["to_dataset_sha256"]
    assert rebound["features_sha256"] == summary["feature_tensor_sha256"]
    assert rebound["dataset_rebind"]["verified"] is True


def test_rebind_rejects_changed_acoustic_crop(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old.json")
    new = _write_dataset(tmp_path / "new.json", crop_end=1.5, marker="new")
    cache = tmp_path / "cache.pt"
    _write_cache(old, cache)
    with pytest.raises(ValueError, match="acoustic identities"):
        MODULE.rebind_feature_cache(old, new, cache, tmp_path / "rebound.pt")


def _checkpoint_dataset(path: Path, *, marker: str) -> Path:
    torch = pytest.importorskip("torch")
    payload = {
        "schema_version": "cantonese_singing_ctc_dataset_v1",
        "marker": marker,
        "examples": [
            {
                "example_id": "train-1",
                "group_id": "train",
                "audio_path": "train.wav",
                "audio_sha256": "1" * 64,
                "crop_start_sec": 0.0,
                "crop_end_sec": 1.0,
                "model_audio_sample_rate": 16000,
                "model_audio_channels": 1,
                **_truth(),
            },
            {
                "example_id": "valid-1",
                "group_id": "valid",
                "audio_path": "valid.wav",
                "audio_sha256": "2" * 64,
                "crop_start_sec": 0.0,
                "crop_end_sec": 1.0,
                "model_audio_sample_rate": 16000,
                "model_audio_channels": 1,
                **_truth(),
            },
        ],
        "folds": [{"fold_index": 0, "validation_group": "valid", "train_groups": ["train"]}],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_checkpoint_rebind_recomputes_split_and_preserves_weights(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    old_dataset = _checkpoint_dataset(tmp_path / "old-checkpoint-dataset.json", marker="old")
    new_dataset = _checkpoint_dataset(tmp_path / "new-checkpoint-dataset.json", marker="new")
    old_cache = tmp_path / "old-cache.pt"
    new_cache = tmp_path / "new-cache.pt"
    _write_cache(
        old_dataset,
        old_cache,
        features={
            "train-1": torch.ones((4, 2), dtype=torch.float32),
            "valid-1": torch.ones((4, 2), dtype=torch.float32),
        },
    )
    MODULE.rebind_feature_cache(old_dataset, new_dataset, old_cache, new_cache)
    trained = ctc.train_ctc_head(
        dataset_path=old_dataset,
        feature_cache=old_cache,
        output_root=tmp_path / "trained",
        epochs=1,
        batch_size=1,
        patience=1,
        device_name="cpu",
    )
    result = CHECKPOINT_MODULE.rebind_checkpoint(
        old_dataset,
        new_dataset,
        old_cache,
        new_cache,
        Path(trained["checkpoint"]),
        Path(trained["checkpoint"]).with_name("metrics.json"),
        tmp_path / "rebound-checkpoint",
    )
    assert result["weights_unchanged"] is True
    assert result["dataset_sha256"] != trained["dataset_sha256"]
    payload = torch.load(result["output_checkpoint"], map_location="cpu", weights_only=True)
    assert payload["checkpoint_rebind"]["verified"] is True
    assert payload["model_sha256"] == trained["model_sha256"]
    _, loaded = ctc.load_ctc_head(
        result["output_checkpoint"],
        device_name="cpu",
        expected_dataset_sha256=result["dataset_sha256"],
        expected_backbone_sha256=payload["configuration"]["backbone_sha256"],
        expected_split_sha256=result["split_sha256"],
    )
    assert loaded["dataset_sha256"] == result["dataset_sha256"]

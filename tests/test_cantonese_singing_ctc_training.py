from __future__ import annotations

import json
from pathlib import Path

import pytest

from any2xsampa.training import cantonese_singing_ctc as ctc


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
            "annotation_sha256": "a" * 64,
            "policy_version": "fixture-v1",
            "annotator_id": "tester",
            "review_status": "approved",
        },
    }


def _dataset(path: Path) -> Path:
    examples = [
        {"example_id": "a1", "group_id": "song-a", **_human_truth(["a", "b"])},
        {"example_id": "a2", "group_id": "song-a", **_human_truth(["b", "a"])},
        {"example_id": "b1", "group_id": "song-b", **_human_truth(["a"])},
        {"example_id": "b2", "group_id": "song-b", **_human_truth(["b"])},
    ]
    payload = {
        "schema_version": "cantonese_singing_ctc_dataset_v1",
        "examples": examples,
        "folds": [{"fold_index": 0, "validation_group": "song-b", "train_groups": ["song-a"]}],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _features(torch: object) -> dict[str, object]:
    return {
        "a1": torch.tensor([[2, 0, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]], dtype=torch.float32),
        "a2": torch.tensor([[0, 2, 0, 0], [0, 2, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]], dtype=torch.float32),
        "b1": torch.tensor([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0]], dtype=torch.float32),
        "b2": torch.tensor([[0, 2, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]], dtype=torch.float32),
    }


def test_ctc_collapse_and_edit_distance_match_recovered_contract() -> None:
    assert ctc._ctc_collapse([0, 1, 1, 0, 1, 2, 2, 0], blank_id=0) == [1, 1, 2]
    assert ctc._levenshtein([1, 2, 3], [1, 4, 3]) == 1


def test_cuda_training_checkpoint_and_reload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA acceptance requires the configured RTX environment")
    dataset = _dataset(tmp_path / "dataset.json")
    cache = tmp_path / "features.pt"
    ctc.write_feature_cache(dataset_path=dataset, features=_features(torch), output_path=cache, model_sha256="a" * 64)
    report = ctc.train_ctc_head(
        dataset_path=dataset,
        feature_cache=cache,
        output_root=tmp_path / "out",
        epochs=1,
        batch_size=2,
        patience=1,
        seed=7,
        device_name="cuda:0",
    )
    assert report["device"] == "cuda:0"
    assert report["epochs_completed"] == 1
    assert report["split_identity"].startswith("fold:")
    assert len(report["split_identity"]) == 69
    assert report["validation"]["example_count"] == 2
    first, payload = ctc.load_ctc_head(tmp_path / "out" / "model.pt", device_name="cuda:0")
    second, _ = ctc.load_ctc_head(tmp_path / "out" / "model.pt", device_name="cuda:0")
    probe = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]], device="cuda:0")
    with torch.inference_mode():
        assert torch.equal(first(probe), second(probe))
    assert payload["vocabulary"] == ["<blank>", "a", "b"]
    assert payload["configuration"]["backbone_frozen"] is True
    assert payload["lyrics_used_as_recognizer_input"] is False


def test_loader_rejects_tampered_model_hash(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset(tmp_path / "dataset.json")
    cache = tmp_path / "features.pt"
    ctc.write_feature_cache(dataset_path=dataset, features=_features(torch), output_path=cache, model_sha256="b" * 64)
    ctc.train_ctc_head(
        dataset_path=dataset,
        feature_cache=cache,
        output_root=tmp_path / "out",
        epochs=1,
        batch_size=2,
        patience=1,
        seed=8,
        device_name="cpu",
    )
    source = tmp_path / "out" / "model.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["model_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="model-state SHA-256 mismatch"):
        ctc.load_ctc_head(tampered)


def test_loader_rejects_tampered_head_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset(tmp_path / "dataset.json")
    cache = tmp_path / "features.pt"
    ctc.write_feature_cache(dataset_path=dataset, features=_features(torch), output_path=cache, model_sha256="d" * 64)
    ctc.train_ctc_head(
        dataset_path=dataset,
        feature_cache=cache,
        output_root=tmp_path / "out",
        epochs=1,
        batch_size=2,
        patience=1,
        seed=10,
        device_name="cpu",
    )
    payload = torch.load(tmp_path / "out" / "model.pt", map_location="cpu", weights_only=True)
    first_name = sorted(payload["head_state_dict"])[0]
    payload["head_state_dict"][first_name].view(-1)[0] += 1.0
    tampered = tmp_path / "tampered-state.pt"
    torch.save(payload, tampered)

    with pytest.raises(ValueError, match="model-state SHA-256 mismatch"):
        ctc.load_ctc_head(tampered)


def test_feature_cache_rejects_nonfinite_and_repeated_target_shortcuts(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "example_id": "repeat",
                        "group_id": "g1",
                        **_human_truth(["a", "a"]),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shorter than CTC target"):
        ctc.write_feature_cache(
            dataset_path=dataset,
            features={"repeat": torch.ones((2, 4))},
            output_path=tmp_path / "short.pt",
            model_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="NaN or infinity"):
        ctc.write_feature_cache(
            dataset_path=dataset,
            features={"repeat": torch.tensor([[1.0, 0.0], [float("nan"), 0.0], [1.0, 0.0]])},
            output_path=tmp_path / "nan.pt",
            model_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="overflow float16"):
        ctc.write_feature_cache(
            dataset_path=dataset,
            features={"repeat": torch.full((3, 2), 70_000.0)},
            output_path=tmp_path / "overflow.pt",
            model_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="valid feature frames are shorter"):
        ctc.write_feature_cache(
            dataset_path=dataset,
            features={"repeat": torch.ones((4, 2))},
            valid_frames={"repeat": 2},
            output_path=tmp_path / "truncated.pt",
            model_sha256="c" * 64,
        )


def test_feature_cache_rejects_blank_targets_and_existing_outputs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    blank_dataset = tmp_path / "blank.json"
    blank_dataset.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "example_id": "blank",
                        "group_id": "g1",
                        **_human_truth(["<blank>"]),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reserved <blank>"):
        ctc.write_feature_cache(
            dataset_path=blank_dataset,
            features={"blank": torch.ones((2, 2))},
            output_path=tmp_path / "blank.pt",
            model_sha256="e" * 64,
        )

    dataset = _dataset(tmp_path / "dataset.json")
    destination = tmp_path / "features.pt"
    ctc.write_feature_cache(
        dataset_path=dataset,
        features=_features(torch),
        output_path=destination,
        model_sha256="f" * 64,
    )
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        ctc.write_feature_cache(
            dataset_path=dataset,
            features=_features(torch),
            output_path=destination,
            model_sha256="f" * 64,
        )
    assert destination.read_bytes() == original


def test_training_rejects_tampered_feature_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset(tmp_path / "dataset.json")
    cache = tmp_path / "features.pt"
    ctc.write_feature_cache(
        dataset_path=dataset,
        features=_features(torch),
        output_path=cache,
        model_sha256="1" * 64,
    )
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    payload["features"]["a1"].view(-1)[0] += 1.0
    torch.save(payload, cache)

    with pytest.raises(ValueError, match="tensor SHA-256 mismatch"):
        ctc.train_ctc_head(
            dataset_path=dataset,
            feature_cache=cache,
            output_root=tmp_path / "out",
            epochs=1,
            patience=1,
            device_name="cpu",
        )


def test_split_rejects_shared_audio_or_source_evidence() -> None:
    dataset = {
        "folds": [{"validation_group": "valid", "train_groups": ["train"]}],
    }
    examples = [
        {"example_id": "a", "group_id": "train", "audio_sha256": "1" * 64},
        {"example_id": "b", "group_id": "valid", "audio_sha256": "1" * 64},
    ]
    with pytest.raises(ValueError, match="source evidence overlaps"):
        ctc._split(dataset, examples, 0)


def test_split_rejects_unassigned_groups() -> None:
    dataset = {"folds": [{"validation_group": "valid", "train_groups": ["train"]}]}
    examples = [
        {"example_id": "a", "group_id": "train"},
        {"example_id": "b", "group_id": "valid"},
        {"example_id": "c", "group_id": "omitted"},
    ]
    with pytest.raises(ValueError, match="assign every dataset group"):
        ctc._split(dataset, examples, 0)


def test_examples_reject_conflicting_human_and_pseudo_truth() -> None:
    row = {
        "example_id": "conflict",
        "group_id": "song",
        **_human_truth(["a"]),
        "pseudo_label": True,
    }
    with pytest.raises(ValueError, match="both human and pseudo"):
        ctc._examples({"examples": [row]})


def test_examples_reject_incomplete_pseudo_truth_and_hash_lineage() -> None:
    row = {
        "schema_version": "any2xsampa_pseudo_phone_label_v1",
        "example_id": "pseudo",
        "group_id": "s/song/phrase",
        "singer_id": "s",
        "song_id": "song",
        "phrase_id": "phrase",
        "target_phone_tokens": ["a"],
        "phone_tokens": ["a"],
        "internal_ipa_tokens": ["a"],
        "normalized_ipa": ["a"],
        "articulatory_features": [{"class": "vowel"}],
        "label_source": "pseudo",
        "pseudo_label": True,
        "model_output": True,
        "human_truth_overwritten": False,
        "source_candidate_sha256": "a" * 64,
        "pseudo_label_sha256": "b" * 64,
        "model_provenance": {
            "model_id": "fixture",
            "revision": "r1",
            "source": "checkpoint",
            "checkpoint_sha256": "1" * 64,
            "inference_config_sha256": "2" * 64,
            "input_audio_sha256": "3" * 64,
            "vocabulary_sha256": "4" * 64,
            "calibration_sha256": "5" * 64,
            "policy_version": "v1",
        },
    }
    assert ctc._examples({"examples": [row]})[0]["target_phone_tokens"] == ["a"]
    incomplete = dict(row)
    incomplete["articulatory_features"] = []
    with pytest.raises(ValueError, match="phone/IPA/articulatory truth is incomplete"):
        ctc._examples({"examples": [incomplete]})
    bad_hash = dict(row)
    bad_hash["model_provenance"] = dict(row["model_provenance"])
    bad_hash["model_provenance"]["checkpoint_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        ctc._examples({"examples": [bad_hash]})


def test_training_rejects_foreign_dataset_schema(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset(tmp_path / "dataset.json")
    cache = tmp_path / "features.pt"
    ctc.write_feature_cache(
        dataset_path=dataset,
        features=_features(torch),
        output_path=cache,
        model_sha256="9" * 64,
    )
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload["schema_version"] = "foreign_dataset_v1"
    dataset.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported CTC dataset schema"):
        ctc.train_ctc_head(
            dataset_path=dataset,
            feature_cache=cache,
            output_root=tmp_path / "out",
            epochs=1,
            patience=1,
            device_name="cpu",
        )


def test_loader_binds_expected_checkpoint_identities(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset(tmp_path / "dataset.json")
    cache = tmp_path / "features.pt"
    ctc.write_feature_cache(
        dataset_path=dataset,
        features=_features(torch),
        output_path=cache,
        model_sha256="8" * 64,
    )
    ctc.train_ctc_head(
        dataset_path=dataset,
        feature_cache=cache,
        output_root=tmp_path / "out",
        epochs=1,
        batch_size=2,
        patience=1,
        seed=31,
        device_name="cpu",
    )
    checkpoint = tmp_path / "out" / "model.pt"
    _, payload = ctc.load_ctc_head(checkpoint)
    ctc.load_ctc_head(
        checkpoint,
        expected_dataset_sha256=payload["dataset_sha256"],
        expected_backbone_sha256="8" * 64,
        expected_vocabulary_sha256=payload["vocabulary_sha256"],
        expected_configuration_sha256=payload["configuration_sha256"],
        expected_split_sha256=payload["split_sha256"],
    )
    with pytest.raises(ValueError, match="dataset SHA-256 does not match"):
        ctc.load_ctc_head(checkpoint, expected_dataset_sha256="0" * 64)


def test_prepare_ctc_dataset_assigns_identity_safe_train_and_validation() -> None:
    examples = [
        {
            "example_id": "one",
            "group_id": "legacy-one",
            "singer_id": "singer-one",
            "song_id": "song-one",
            "phrase_id": "phrase-one",
            **_human_truth(["a"]),
        },
        {
            "example_id": "two",
            "group_id": "legacy-two",
            "singer_id": "singer-two",
            "song_id": "song-two",
            "phrase_id": "phrase-two",
            **_human_truth(["b"]),
        },
    ]

    prepared = ctc.prepare_ctc_dataset(examples, validation_ratio=0.5, seed=29)
    normalized = ctc._examples(prepared)
    train, validation, identity = ctc._split(prepared, normalized, 0)

    assert len(train) == 1
    assert len(validation) == 1
    assert identity.startswith("manifest:")
    assert {row["split"] for row in prepared["examples"]} == {"train", "val"}
    assert prepared["dataset_hash"]

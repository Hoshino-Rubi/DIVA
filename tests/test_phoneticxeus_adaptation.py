from __future__ import annotations

import copy
from pathlib import Path
import random

import pytest

from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256, ctc_collapse
from any2xsampa.training.limited_data_artifacts import sha256_object
from any2xsampa.training.limited_data_reference import DATASET_SCHEMA, SPLITS
from any2xsampa.training.phone_error_metrics import align_phone_edits, development_gate, phone_metrics
from any2xsampa.training.phoneticxeus_adaptation import make_residual_adapter, required_ctc_frames, validate_feature_index


def test_phone_metrics_insert_delete_substitute_and_unclipped_per() -> None:
    value = phone_metrics([(["a", "b", "c"], ["a", "x", "c", "z"])])
    assert (value["substitutions"], value["deletions"], value["insertions"]) == (1, 0, 1)
    assert value["per"] == 2 / 3
    assert phone_metrics([(["i"], ["y", "p", "t"])])["per"] == 3
    assert phone_metrics([(["i", "i"], [])])["deletions"] == 2
    assert phone_metrics([([], ["i"])])["per"] is None
    assert phone_metrics([])["per"] is None


def test_edit_alignment_reconstructs_sequences() -> None:
    rng = random.Random(8123)
    for _ in range(100):
        a = [rng.choice(["i", "y", "k\u02b0"]) for _ in range(rng.randrange(20))]
        b = [rng.choice(["i", "y", "k\u02b0"]) for _ in range(rng.randrange(20))]
        trace = align_phone_edits(a, b)
        assert [x for _, x, _ in trace if x is not None] == a
        assert [y for _, _, y in trace if y is not None] == b
        assert phone_metrics([(a, b)])["edit_distance"] == sum(op != "match" for op, _, _ in trace)


def test_special_predictions_are_errors_not_silently_dropped() -> None:
    value = phone_metrics([(["y"], ["y", "<unk>", "<sos>"])])
    assert value["insertions"] == 2
    assert value["predicted_phones"] == 3


def test_group_metrics_preserve_aspiration_and_rounding() -> None:
    value = phone_metrics([(["y", "k\u02b0", "f", "h"], ["u", "k", "s", "s"])])
    assert value["substitutions"] == 4
    assert all(row["reference_count"] == 1 and row["per"] == 1 for row in value["groups"].values())


def test_development_gate_requires_consistency_and_unchanged_references() -> None:
    base = phone_metrics([(["i"] * 100, ["i"] * 70)])
    good = phone_metrics([(["i"] * 100, ["i"] * 80)])
    candidates = [{"seed": seed, "metrics": good} for seed in (17, 29, 43)]
    result = development_gate(base, candidates)
    assert result["passed"] and result["selected_seed"] == 17
    assert not result["automatic_model_promotion"]
    assert not development_gate(base, candidates[:1])["passed"]
    with pytest.raises(ValueError, match="unique"):
        development_gate(base, [candidates[0], candidates[0]])
    changed = copy.deepcopy(candidates)
    changed[0]["metrics"]["reference_phones"] -= 1
    with pytest.raises(ValueError, match="different development"):
        development_gate(base, changed)


def test_group_regression_blocks_apparent_aggregate_improvement() -> None:
    reference = ["y"] * 20 + ["i"] * 100
    baseline = phone_metrics([(reference, ["y"] * 20 + ["i"] * 50)])
    adapted = phone_metrics([(reference, ["u"] * 20 + ["i"] * 100)])
    result = development_gate(baseline, [{"seed": seed, "metrics": adapted} for seed in (17, 29, 43)])
    assert not result["passed"]
    assert all("front_rounded" in row["group_regressions"] for row in result["seeds"])


def test_zero_initialized_adapter_and_frozen_pretrained_head() -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    adapter = make_residual_adapter(torch, width=16, bottleneck=4)
    head = torch.nn.Linear(16, 6).requires_grad_(False)
    original_head = {name: value.clone() for name, value in head.state_dict().items()}
    features = torch.randn(2, 7, 16)
    assert torch.equal(adapter(features), features)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    head(adapter(features)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(adapter(features), features)
    assert all(torch.equal(value, original_head[name]) for name, value in head.state_dict().items())
    assert all(value.grad is None for value in head.parameters())


def fixture_cache() -> tuple[dict, dict]:
    row = {
        "example_id": "a", "song_id": "freedom-flower", "split": "train",
        "audio_role": "rendered", "label_scope": "derived_engine_reference",
        "audio_sha256": "a" * 64, "start_sec": 1.0, "end_sec": 2.0,
        "eligible": True, "training_eligible": True,
        "model_target_ids": [4, 4, 5], "model_target_ipa": ["i", "i", "y"],
    }
    dataset = {"schema_version": DATASET_SCHEMA, "split_sha256": sha256_object(SPLITS), "examples": [row], "bindings": {"vocabulary": {"sha256": "b" * 64}}}
    dataset["dataset_hash"] = sha256_object(dataset)
    entry = {key: row[key] for key in ("example_id", "song_id", "split", "audio_role", "audio_sha256", "start_sec", "end_sec")}
    entry.update(frames=49, test_outputs_sealed=False, vocabulary_sha256="b" * 64, model_snapshot_lock_sha256="c" * 64)
    index = {
        "schema_version": "phoneticxeus_feature_index_v1", "purpose": "rendered-features",
        "source_dataset_hash": dataset["dataset_hash"], "split_sha256": dataset["split_sha256"],
        "model_metadata": {"model_id": MODEL_ID, "revision": REVISION, "weights_sha256": WEIGHT_SHA256, "snapshot_lock_sha256": "c" * 64}, "entries": [entry],
    }
    return index, dataset


@pytest.mark.parametrize("field,value", [
    ("start_sec", 0.0), ("audio_sha256", "f" * 64),
    ("audio_role", "original"), ("song_id", "shalong"),
    ("vocabulary_sha256", "d" * 64), ("test_outputs_sealed", True),
    ("model_snapshot_lock_sha256", "e" * 64), ("frames", 3),
])
def test_cache_misbinding_is_rejected(field: str, value: object) -> None:
    index, dataset = fixture_cache()
    assert validate_feature_index(index, dataset)
    index["entries"][0][field] = value
    with pytest.raises(ValueError):
        validate_feature_index(index, dataset)


def test_cache_requires_exact_subset_and_dataset_identity() -> None:
    index, dataset = fixture_cache()
    index["entries"].append(copy.deepcopy(index["entries"][0]))
    with pytest.raises(ValueError, match="exactly"):
        validate_feature_index(index, dataset)
    index, dataset = fixture_cache()
    index["source_dataset_hash"] = "f" * 64
    with pytest.raises(ValueError, match="stale"):
        validate_feature_index(index, dataset)


def test_required_ctc_frames_accounts_for_repeated_targets() -> None:
    assert required_ctc_frames([4, 4, 5, 5, 5]) == 8
    assert ctc_collapse([4, 0, 4, 5, 0, 5, 0, 5]) == [4, 4, 5, 5, 5]

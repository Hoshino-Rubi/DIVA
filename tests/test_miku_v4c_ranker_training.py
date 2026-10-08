from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from any2xsampa.training import miku_v4c_ranker as ranker  # noqa: E402
from any2xsampa.training.ranker_dataset import build_dataset  # noqa: E402
from tools.train_miku_v4c_phoneme_ranker import main as train_main  # noqa: E402


def _fixtures(root: Path) -> tuple[Path, Path]:
    train = root / "train.jsonl"
    vocab = root / "vocab.json"
    manifest = {
        "schema_version": "miku-v4c-ranker-dataset-v2",
        "algorithm": "fixture",
        "seed": 17,
        "candidates": [
            {"candidate_id": "positive", "identity_group": "singer/song/phrase-a", "split": "train"},
            {"candidate_id": "negative", "identity_group": "singer/song/phrase-b", "split": "train"},
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (root / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def row(candidate_id: str, group: str, tokens: list[str], label: int, score: float) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "identity_group": group,
            "group_id": group,
            "split": "train",
            "split_manifest_sha256": manifest["manifest_sha256"],
            "training_admissible": True,
            "truth_layer": "phones-ipa-articulatory",
            "truth_ipa_complete": True,
            "truth_features_complete": True,
            "lyrics": "ab" if label else "ba",
            "phoneme_tokens": tokens,
            "phones": [
                {"phone": token, "ipa": token, "features": {"class": "fixture"}}
                for token in tokens
            ],
            "label": label,
            "score": score,
            "label_provenance_status": "complete",
            "label_provenance": {
                "kind": "pseudo",
                "model_id": "fixture-ranker-labeler",
                "model_revision": "r1",
                "generated_at_utc": "2026-08-11T00:00:00Z",
                "source": {"source_id": candidate_id, "revision": "r1", "sha256": "a" * 64},
            },
        }

    rows = [
        row("positive", "singer/song/phrase-a", ["a", "b"], 1, 0.95),
        row("negative", "singer/song/phrase-b", ["b", "a"], 0, 0.10),
    ]
    train.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    vocab.write_text(
        json.dumps(
            {
                "schema_version": "miku-v4c-ranker-vocab-v2",
                "chars": ["a", "b"],
                "phoneme_tokens": ["a", "b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return train, vocab


def _verified_cuda_available() -> bool:
    return bool(
        torch.cuda.is_available()
        and str(torch.__version__).startswith("2.13.0+cu126")
        and str(torch.version.cuda) == "12.6"
        and "RTX 4090" in torch.cuda.get_device_name(0)
    )


@pytest.mark.skipif(not _verified_cuda_available(), reason="verified RTX 4090 CUDA stack unavailable")
def test_ranker_cuda_epoch_checkpoint_and_reload_are_deterministic(tmp_path: Path) -> None:
    train_path, vocab_path = _fixtures(tmp_path)
    output = tmp_path / "ranker_out"
    metrics = ranker.train_ranker(
        train_jsonl=train_path,
        vocab=vocab_path,
        output_dir=output,
        epochs=1,
        batch_size=2,
        device="cuda:0",
        seed=17,
        embedding_dim=8,
        hidden_dim=12,
        dropout=0.0,
    )

    assert metrics["train_size"] == 2
    metadata = metrics["checkpoint_metadata"]
    assert metadata["runtime"]["torch_version"].startswith("2.13.0+cu126")
    assert metadata["runtime"]["cuda_version"] == "12.6"
    assert "RTX 4090" in metadata["runtime"]["device_name"]
    assert metadata["dataset"]["train_sha256"]
    assert metadata["split"]["identity"]
    assert metadata["split"]["hash"]
    assert metadata["model"]["config_sha256"]

    expected_vocab_hash = ranker.RankerVocab.from_json(vocab_path).fingerprint()
    model_a, _ = ranker.load_ranker_checkpoint(
        output / "model.pt",
        device="cuda:0",
        expected_vocab_hash=expected_vocab_hash,
        expected_split_hash=metadata["split"]["hash"],
    )
    model_b, _ = ranker.load_ranker_checkpoint(
        output / "model.pt",
        device="cuda:0",
        expected_vocab_hash=expected_vocab_hash,
        expected_split_hash=metadata["split"]["hash"],
    )
    vocab = ranker.RankerVocab.from_json(vocab_path)
    dataset = ranker.PhonemeRankerJsonlDataset(train_path, vocab)
    batch = ranker.phoneme_ranker_collate([dataset[0], dataset[1]])
    batch = {key: value.to("cuda:0") for key, value in batch.items()}
    with torch.inference_mode():
        output_a = model_a(batch["char_ids"], batch["phoneme_ids"], batch["char_mask"], batch["phoneme_mask"])
        output_b = model_b(batch["char_ids"], batch["phoneme_ids"], batch["char_mask"], batch["phoneme_mask"])
    assert torch.equal(output_a, output_b)


def test_ranker_cpu_cli_is_explicit_and_reloadable(tmp_path: Path) -> None:
    train_path, vocab_path = _fixtures(tmp_path)
    output = tmp_path / "ranker_out"
    assert train_main(
        [
            "--train-jsonl",
            str(train_path),
            "--vocab",
            str(vocab_path),
            "--output-dir",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--seed",
            "3",
        ]
    ) == 0
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["device"] == "cpu"
    assert metrics["epochs"] == 1
    model, payload = ranker.load_ranker_checkpoint(output / "model.pt", device="cpu")
    assert model.training is False
    assert payload["schema_version"] == ranker.SCHEMA_VERSION


def test_ranker_loader_rejects_tampered_metadata(tmp_path: Path) -> None:
    train_path, vocab_path = _fixtures(tmp_path)
    output = tmp_path / "ranker_out"
    ranker.train_ranker(
        train_jsonl=train_path,
        vocab=vocab_path,
        output_dir=output,
        epochs=1,
        batch_size=2,
        device="cpu",
        seed=5,
    )
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    checkpoint["metadata"]["model"]["config_sha256"] = "0" * 64
    tampered = output / "tampered.pt"
    torch.save(checkpoint, tampered)
    with pytest.raises(ValueError, match="model config hash"):
        ranker.load_ranker_checkpoint(tampered, device="cpu")


def test_ranker_loader_rejects_tampered_model_state(tmp_path: Path) -> None:
    train_path, vocab_path = _fixtures(tmp_path)
    output = tmp_path / "ranker_out"
    ranker.train_ranker(
        train_jsonl=train_path,
        vocab=vocab_path,
        output_dir=output,
        epochs=1,
        batch_size=2,
        device="cpu",
        seed=9,
    )
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    first_name = sorted(checkpoint["model_state_dict"])[0]
    checkpoint["model_state_dict"][first_name].view(-1)[0] += 1.0
    tampered = output / "tampered-state.pt"
    torch.save(checkpoint, tampered)

    with pytest.raises(ValueError, match="model-state SHA-256 mismatch"):
        ranker.load_ranker_checkpoint(tampered, device="cpu")


def test_ranker_training_rejects_quarantined_rows_and_unbound_split_claims(tmp_path: Path) -> None:
    train_path, vocab_path = _fixtures(tmp_path)
    row = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    row["training_admissible"] = False
    quarantined = tmp_path / "quarantined.jsonl"
    quarantined.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="quarantined or not admissible"):
        ranker.train_ranker(
            train_jsonl=quarantined,
            vocab=vocab_path,
            output_dir=tmp_path / "quarantined-out",
        )

    manifest = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="caller-provided split hash"):
        ranker.train_ranker(
            train_jsonl=train_path,
            vocab=vocab_path,
            output_dir=tmp_path / "forged-out",
            split_hash="f" * 64,
            split_manifest=tmp_path / "split_manifest.json",
        )
    assert manifest["manifest_sha256"] != "f" * 64


def test_ranker_builder_output_trains_with_bound_manifest(tmp_path: Path) -> None:
    candidates = []
    for index, (tokens, score) in enumerate(((["a", "b"], 0.95), (["b", "a"], 0.10))):
        candidates.append(
            {
                "candidate_id": f"candidate-{index}",
                "singer_id": f"singer-{index}",
                "song_id": f"song-{index}",
                "phrase_id": f"phrase-{index}",
                "lyrics": "ab",
                "phoneme_tokens": tokens,
                "ipa_tokens": tokens,
                "articulatory_features": [{"class": "fixture"} for _ in tokens],
                "qwen_score": score,
                "training_data_candidate": True,
                "rendered_wav": "fixture.wav",
                "label_provenance": {
                    "kind": "pseudo",
                    "model_id": "fixture",
                    "model_revision": "r1",
                    "generated_at_utc": "2026-08-11T00:00:00Z",
                    "source": {"source_id": f"source-{index}", "revision": "r1", "sha256": "b" * 64},
                },
            }
        )
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"scored_candidates": candidates}), encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    build_dataset(source, dataset_root, val_ratio=0.0, test_ratio=0.0, seed=23)

    metrics = ranker.train_ranker(
        train_jsonl=dataset_root / "train.jsonl",
        vocab=dataset_root / "vocab.json",
        split_manifest=dataset_root / "split_manifest.json",
        output_dir=tmp_path / "trained",
        epochs=1,
        batch_size=2,
        device="cpu",
        seed=23,
    )

    assert metrics["checkpoint_metadata"]["split"]["hash"] == json.loads(
        (dataset_root / "split_manifest.json").read_text(encoding="utf-8")
    )["manifest_sha256"]

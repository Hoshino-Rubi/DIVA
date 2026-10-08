from __future__ import annotations

from pathlib import Path

import pytest

from any2xsampa.adapters.miku_v4c_ranker import MikuV4CRankerAdapter
from any2xsampa.training import miku_v4c_ranker as ranker
from tests.test_miku_v4c_ranker_training import _fixtures


def _candidate(candidate_id: str, tokens: list[str]) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "lyrics": "ab",
        "phoneme_tokens": tokens,
        "truth_layer": "phones-ipa-articulatory",
        "phones": [
            {"phone": token, "ipa": token, "features": {"class": "fixture"}}
            for token in tokens
        ],
    }


def test_product_ranker_adapter_scores_and_ranks_checkpoint_candidates(tmp_path: Path) -> None:
    train, vocab = _fixtures(tmp_path)
    output = tmp_path / "ranker"
    ranker.train_ranker(
        train_jsonl=train,
        vocab=vocab,
        output_dir=output,
        epochs=1,
        batch_size=2,
        device="cpu",
        seed=31,
    )
    adapter = MikuV4CRankerAdapter(output / "model.pt")

    ranked = adapter.score([_candidate("one", ["a", "b"]), _candidate("two", ["b", "a"])])

    assert [row.rank for row in ranked] == [1, 2]
    assert {row.candidate_id for row in ranked} == {"one", "two"}
    assert all(0.0 <= row.score <= 1.0 for row in ranked)
    assert all(row.model_output is True for row in ranked)
    assert all(len(row.checkpoint_sha256) == 64 for row in ranked)


def test_product_ranker_adapter_rejects_export_or_unknown_phone_truth(tmp_path: Path) -> None:
    train, vocab = _fixtures(tmp_path)
    output = tmp_path / "ranker"
    ranker.train_ranker(train_jsonl=train, vocab=vocab, output_dir=output, device="cpu")
    adapter = MikuV4CRankerAdapter(output / "model.pt")

    export = _candidate("export", ["a"])
    export["truth_layer"] = "xsampa"
    with pytest.raises(ValueError, match="unsupported truth layer"):
        adapter.score([export])
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        adapter.score([_candidate("unknown", ["not-in-vocab"])])
    with pytest.raises(ValueError, match="reserved vocabulary"):
        adapter.score([_candidate("padding", ["<pad>"])])

from __future__ import annotations

import json

import pytest

from any2xsampa.data.deterministic_split import (
    DuplicateExampleError,
    IdentityCollisionError,
    InvalidRatiosError,
    MissingIdentityError,
    SplitError,
    SplitRecord,
    split_records,
)


def _records() -> list[SplitRecord]:
    return [
        SplitRecord("e01", "singer-a", "song-1", "phrase-1"),
        SplitRecord("e02", "singer-a", "song-2", "phrase-2"),
        SplitRecord("e03", "singer-b", "song-1", "phrase-3"),
        SplitRecord("e04", "singer-c", "song-3", "phrase-4"),
        SplitRecord("e05", "singer-d", "song-4", "phrase-5"),
        SplitRecord("e06", "singer-e", "song-5", "phrase-6"),
    ]


def test_split_is_order_invariant_and_prevents_identity_leakage() -> None:
    records = _records()
    first = split_records(records, ratios=(0.5, 0.25, 0.25), seed=17)
    second = split_records(reversed(records), ratios=(0.5, 0.25, 0.25), seed=17)
    assert dict(first.assignments) == dict(second.assignments)
    assert first.manifest["manifest_hash"] == second.manifest["manifest_hash"]
    for field in ("singer_id", "song_id", "phrase_id"):
        seen: dict[str, str] = {}
        for record in records:
            identity = getattr(record, field).casefold()
            split = first.assignments[record.example_id]
            assert seen.setdefault(identity, split) == split


def test_split_manifest_is_stable_json() -> None:
    result = split_records(_records(), ratios={"train": 1.0, "validation": 0.0, "test": 0.0})
    encoded = json.dumps(result.manifest, sort_keys=True, separators=(",", ":"))
    assert '"manifest_hash"' in encoded
    assert result.ids_for("train") == tuple(sorted(result.assignments))


def test_split_keeps_identical_content_and_source_hashes_together() -> None:
    records = [
        SplitRecord("a", "s1", "song-a", "p-a", content_hash="a" * 64),
        SplitRecord("b", "s2", "song-b", "p-b", content_hash="a" * 64),
        SplitRecord(
            "c",
            "s3",
            "song-c",
            "p-c",
            metadata={"source_hashes": ["source:shared"]},
        ),
        SplitRecord(
            "d",
            "s4",
            "song-d",
            "p-d",
            metadata={"source_hashes": ["source:shared"]},
        ),
        SplitRecord("e", "s5", "song-e", "p-e"),
    ]
    result = split_records(records, ratios=(0.6, 0.4, 0.0), seed=3)
    assert result.assignments["a"] == result.assignments["b"]
    assert result.assignments["c"] == result.assignments["d"]
    assert result.manifest["algorithm_version"] == "connected-components-sha256/v3"


def test_split_rejects_missing_duplicate_and_colliding_identities() -> None:
    with pytest.raises(MissingIdentityError):
        split_records([{"example_id": "e1", "singer_id": "s1", "song_id": "song", "phrase_id": ""}])
    with pytest.raises(DuplicateExampleError):
        split_records([SplitRecord("e1", "s1", "a", "p1"), SplitRecord("E1", "s2", "b", "p2")])
    records = [SplitRecord("e1", "Singer A", "a", "p1"), SplitRecord("e2", "singer a", "b", "p2")]
    with pytest.raises(IdentityCollisionError):
        split_records(records, ratios=(0.5, 0.5, 0.0))
    assert split_records(records, ratios=(1.0, 0.0, 0.0), collision_policy="merge").manifest["group_count"] == 1
    with pytest.raises(InvalidRatiosError):
        split_records(_records(), ratios=(0.7, 0.1, 0.1))
    with pytest.raises(MissingIdentityError):
        split_records([SplitRecord("e1", None, "song", "phrase")])  # type: ignore[arg-type]
    with pytest.raises(SplitError, match="cannot populate every active split"):
        split_records(
            [SplitRecord("e1", "same", "song", "phrase")],
            ratios=(0.5, 0.5, 0.0),
        )

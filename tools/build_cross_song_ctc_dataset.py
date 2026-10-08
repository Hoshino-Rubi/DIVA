from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


SCHEMA_VERSION = "cantonese_singing_ctc_dataset_v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    examples = payload.get("examples") if isinstance(payload, Mapping) else None
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"phrase dataset contains no examples: {path}")
    return [dict(row) for row in examples if isinstance(row, Mapping)]


def build_datasets(dataset_paths: list[Path], output: Path) -> dict[str, Any]:
    if len(dataset_paths) < 2:
        raise ValueError("cross-song dataset requires at least two phrase datasets")
    examples = [row for path in dataset_paths for row in _load(path)]
    ids = [str(row.get("example_id") or "") for row in examples]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("cross-song dataset requires unique example IDs")
    groups = sorted({str(row.get("group_id") or "") for row in examples})
    if len(groups) != len(dataset_paths) or any(not value for value in groups):
        raise ValueError("each phrase dataset must contribute one non-empty song group")
    evidence_by_group: dict[str, set[str]] = {group: set() for group in groups}
    for row in examples:
        group = str(row["group_id"])
        for field in (
            "audio_sha256",
            "original_audio_sha256",
            "source_vsqx_sha256",
            "source_corpus_sha256",
        ):
            value = str(row.get(field) or "").strip()
            if value:
                evidence_by_group[group].add(f"{field}:{value}")
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            if evidence_by_group[left] & evidence_by_group[right]:
                raise ValueError("cross-song source evidence overlaps")
    folds = [
        {
            "fold_index": index,
            "validation_group": validation,
            "validation_groups": [validation],
            "train_groups": [group for group in groups if group != validation],
            "leakage_free": True,
        }
        for index, validation in enumerate(reversed(groups))
    ]
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "source_phrase_datasets": [str(path.resolve()) for path in dataset_paths],
        "examples": sorted(examples, key=lambda row: str(row["example_id"])),
        "folds": folds,
        "truth_policy": {
            "phones_ipa_articulatory_are_truth": True,
            "human_v4c_tokens_preserved": True,
            "recognizer_contract_projection_is_lossy": True,
            "lyrics_used_as_acoustic_input": False,
        },
    }
    payload = {**unsigned, "dataset_hash": _sha(unsigned)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def build_dataset(first: Path, second: Path, output: Path) -> dict[str, Any]:
    """Backward-compatible two-song entry point used by existing tests/callers."""

    return build_datasets([first, second], output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build leakage-safe leave-one-song-out CTC folds.")
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = build_datasets(args.datasets, args.output)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset_hash": payload["dataset_hash"], "example_count": len(payload["examples"]), "folds": payload["folds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

from any2xsampa.training.cantonese_singing_ctc import prepare_ctc_dataset


def _objects(value: object, *, source: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if not all(isinstance(row, Mapping) for row in value):
            raise ValueError(f"CTC input list must contain only objects: {source}")
        return [dict(row) for row in value]
    if not isinstance(value, Mapping):
        raise ValueError(f"CTC input must be an object or list: {source}")
    if isinstance(value.get("examples"), list):
        return _objects(value["examples"], source=source)
    if isinstance(value.get("decisions"), list):
        result = [
            dict(row["pseudo_label"])
            for row in value["decisions"]
            if isinstance(row, Mapping)
            and row.get("status") == "accepted"
            and isinstance(row.get("pseudo_label"), Mapping)
        ]
        if not result:
            raise ValueError(f"self-training manifest contains no accepted pseudo labels: {source}")
        return result
    raise ValueError(f"CTC input object has no examples or self-training decisions: {source}")


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.casefold() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return _objects(rows, source=source)
    try:
        return _objects(json.loads(text), source=source)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return _objects(rows, source=source)


def write_dataset(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"CTC dataset output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe human/pseudo phone truth for Cantonese singing CTC training."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        examples = [row for source in args.inputs for row in load_examples(source)]
        payload = prepare_ctc_dataset(
            examples,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        write_dataset(args.output, payload)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in payload.items() if key != "examples"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a fixture-first Miku V4C phoneme-ranker dataset.

This command is intentionally dependency-free.  It accepts the historical
``scored_candidates``/``candidates``/``rows`` JSON shapes and emits normalized
JSONL plus an auditable grouped-split manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from any2xsampa.training.ranker_dataset import DatasetBuildError, build_dataset
except ImportError:
    try:
        from .ranker_dataset import DatasetBuildError, build_dataset
    except ImportError:  # direct ``python -B build_miku_v4c_ranker_dataset.py``
        from ranker_dataset import DatasetBuildError, build_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-candidates", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--positive-threshold", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-score-label", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return_code, summary = build_dataset(
            args.scored_candidates,
            args.output_root,
            positive_threshold=args.positive_threshold,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            trust_score_label=args.trust_score_label,
        )
    except (DatasetBuildError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

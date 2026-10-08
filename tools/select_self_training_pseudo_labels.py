from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from any2xsampa.training.self_training import build_self_training_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select auditable phone pseudo-labels without overwriting human truth.")
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume-manifest", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(args.candidates.read_text(encoding="utf-8-sig"))
        candidates = payload.get("candidates") if isinstance(payload, dict) else payload
        if not isinstance(candidates, list):
            raise ValueError("candidate JSON must be a list or an object containing candidates")
        manifest = build_self_training_batch(
            candidates=candidates,
            output_root=args.output_root,
            min_confidence=args.min_confidence,
            seed=args.seed,
            resume_manifest=args.resume_manifest,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

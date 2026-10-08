"""Dedicated worker for one V/MV held-song job."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.input_supervision_training import train_job


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--variant", choices=["V", "MV"], required=True)
    parser.add_argument("--held-song", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(run):
        raise ValueError("training output must stay inside the input-supervision run")
    train_job(run, args.variant, args.held_song, args.seed, output)

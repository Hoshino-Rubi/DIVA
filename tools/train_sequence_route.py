"""Thin isolated worker for one predeclared continuation fold/seed."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.sequence_route_training import train_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=["control", "sequence_risk"], required=True)
    parser.add_argument("--held-song")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_job(args.run.resolve(), args.arm, args.held_song, args.seed, args.output.resolve())

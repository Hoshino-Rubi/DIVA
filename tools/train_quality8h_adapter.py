from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.protected_adaptation import run_protected_adaptation


def main() -> int:
    parser = argparse.ArgumentParser(description="One approved CTC+KL seed, with protected development selection.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=[17, 29, 43])
    args = parser.parse_args()
    run = args.run_root.resolve()
    if not run.is_relative_to(ROOT / ".state/runs"):
        parser.error("training outputs must remain under .state/runs")
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    with ProcessResourceMonitor(run / "training-v2" / f"seed{args.seed}-monitor", timeout_seconds=1500):
        result = run_protected_adaptation(run, args.seed, ROOT)
    print(f"seed={args.seed}; admissible={result['checkpoints']['best_admissible'] is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

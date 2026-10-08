from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.phoneticxeus_adaptation import run_adaptation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train only the predeclared small residual adapter.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[17, 29, 43])
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / ".state/runs"):
        parser.error("training artifacts must remain under .state/runs")
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor

    with ProcessResourceMonitor(args.output_dir.parent / f"{args.output_dir.name}-monitor", timeout_seconds=1500):
        report = run_adaptation(args.dataset, args.features, args.output_dir.resolve(), args.seed)
    print(f"baseline_dev_per={report['baseline_dev']['per']}; adapted_dev_per={report['adapted_dev']['per']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

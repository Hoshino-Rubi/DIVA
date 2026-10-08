from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.adapters.qwen_timing_runtime import QwenAuxiliaryRuntime
from any2xsampa.training.limited_data_artifacts import write_json_new


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all actual Qwen parameters without another inference/optimization trial.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--role", choices=["asr", "timestamps"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run) or output.exists():
        parser.error("loading verification requires a new output directory in the run")
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    with ProcessResourceMonitor(output, timeout_seconds=600):
        runtime = QwenAuxiliaryRuntime(run, output / "runtime-cache", args.role)
        report = {"schema_version": "quality8h_qwen_loading_verification_v1", "role": args.role,
                  "model": runtime.identity, "verification": runtime.loading_verification,
                  "new_transcriptions_or_timestamps_generated": False, "parameters_frozen": True,
                  "test_audio_loaded": False, "optimization_performed": False}
        write_json_new(output / "verification.json", report)
        print(runtime.loading_verification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

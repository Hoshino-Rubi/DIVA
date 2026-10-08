from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import sha256_file, write_json_new  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the isolated runtime without loading a model or allocating CUDA tensors.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / ".state/runs"):
        parser.error("environment evidence must remain under .state/runs")
    import torch

    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    if torch.__version__ != "2.8.0+cu126" or importlib.metadata.version("torchaudio") != "2.8.0+cu126":
        raise RuntimeError("isolated Torch/torchaudio pair changed")
    record = {
        "schema_version": "phoneticxeus_environment_lock_v1",
        "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
        "base_prefix": sys.base_prefix, "platform": platform.platform(),
        "torch_runtime_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_initialized": torch.cuda.is_initialized(),
        "packages": dict(sorted(versions.items(), key=lambda item: item[0].lower())),
        "requirements_sha256": sha256_file(ROOT / "tools/requirements/phoneticxeus-py312.txt"),
        "model_loaded": False,
    }
    print(f"environment_sha256={write_json_new(args.output, record)}; packages={len(versions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

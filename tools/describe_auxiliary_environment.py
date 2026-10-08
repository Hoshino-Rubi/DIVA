"""Record the isolated auxiliary dependency graph and executable source hashes."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    run = args.run_root.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or Path(sys.prefix).resolve() != run / "aux-runtime":
        raise ValueError("must run inside this experiment's auxiliary environment")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("auxiliary Python must be 3.12")
    import torch
    if torch.__version__ != "2.8.0+cu126" or torch.version.cuda != "12.6":
        raise ValueError("unexpected auxiliary Torch/CUDA")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if args.verify_existing:
        locked = read_json(run / "aux-environment.json")
        if locked["packages"] != packages or locked["python"] != sys.version:
            raise ValueError("auxiliary dependency graph changed after it was locked")
        verify_file(run / "aux-requirements-lock.txt", locked["requirements_lock_sha256"])
        for name, digest in locked["inference_source_files"].items():
            verify_file(run / "aux-runtime" / name, digest)
        write_json_new(run / "aux-environment-verification.json", {"schema_version": "quality8h_aux_environment_verification_v1",
            "environment_sha256": sha256_file(run / "aux-environment.json"), "packages_identical": True,
            "inference_source_files_identical": True, "cuda_initialized": torch.cuda.is_initialized()})
        print("auxiliary dependency graph and source files unchanged")
        return 0
    files = {}
    for name in ("qwen-asr", "rotary-embedding-torch", "beartype", "einops"):
        distribution = importlib.metadata.distribution(name)
        for member in distribution.files or []:
            if str(member).endswith(".py"):
                path = Path(distribution.locate_file(member)).resolve()
                files[str(path.relative_to(run / "aux-runtime"))] = sha256_file(path)
    lock = run / "aux-requirements-lock.txt"
    with lock.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(f"{name}=={version}" for name, version in sorted(packages.items(), key=lambda p: p[0].lower())) + "\n")
    report = {"schema_version": "quality8h_aux_environment_v1", "executable": sys.executable,
              "python": sys.version, "platform": platform.platform(), "packages": packages, "inference_source_files": files,
              "torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_initialized": torch.cuda.is_initialized(),
              "requirements_lock_sha256": sha256_file(lock), "model_loaded": False}
    print(f"environment_sha256={write_json_new(run / 'aux-environment.json', report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

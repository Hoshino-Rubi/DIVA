from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from any2xsampa.adapters.recognizer_producer_runtime import (
    describe_onnx_espeak_runtime,
    prepare_onnx_espeak_runtime_bundle,
    write_onnx_espeak_bundle_state,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a distributable onnx-espeak runtime bundle layout and write bundle state metadata."
    )
    parser.add_argument(
        "--runtime-root",
        default=str(_repo_root() / "runtime" / "onnx-espeak"),
        help="Bundle runtime root to prepare.",
    )
    parser.add_argument(
        "--bundle-name",
        default="onnx-espeak-runtime",
        help="Bundle name recorded in the manifest.",
    )
    parser.add_argument(
        "--external-python",
        default=None,
        help="Optional external Python 3.11 path to record as the current runtime-backed worker.",
    )
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve(strict=False)
    bundle_layout = prepare_onnx_espeak_runtime_bundle(runtime_root, bundle_name=args.bundle_name)

    previous_runtime_root = os.environ.get("ANY2XSAMPA_ONNX_ESPEAK_RUNTIME")
    previous_runtime_python = os.environ.get("ANY2XSAMPA_ONNX_ESPEAK_PYTHON")
    try:
        os.environ["ANY2XSAMPA_ONNX_ESPEAK_RUNTIME"] = str(runtime_root)
        if args.external_python:
            os.environ["ANY2XSAMPA_ONNX_ESPEAK_PYTHON"] = args.external_python
        runtime_diagnostics = describe_onnx_espeak_runtime()
    finally:
        if previous_runtime_root is None:
            os.environ.pop("ANY2XSAMPA_ONNX_ESPEAK_RUNTIME", None)
        else:
            os.environ["ANY2XSAMPA_ONNX_ESPEAK_RUNTIME"] = previous_runtime_root
        if args.external_python:
            if previous_runtime_python is None:
                os.environ.pop("ANY2XSAMPA_ONNX_ESPEAK_PYTHON", None)
            else:
                os.environ["ANY2XSAMPA_ONNX_ESPEAK_PYTHON"] = previous_runtime_python

    bundle_state = write_onnx_espeak_bundle_state(runtime_root, runtime_diagnostics=runtime_diagnostics)
    print(
        json.dumps(
            {
                "runtime_root": str(runtime_root),
                "manifest_path": bundle_layout["manifest_path"],
                "worker_entrypoint": bundle_layout["worker_entrypoint"],
                "status_file": bundle_layout["status_file"],
                "bundle_status": bundle_state["bundle_status"],
                "runtime_status": bundle_state["runtime_status"],
                "ready_components": bundle_state["ready_components"],
                "missing_components": bundle_state["missing_components"],
                "external_runtime": bundle_state["external_runtime"],
                "model_resources": bundle_state["model_resources"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

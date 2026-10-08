from __future__ import annotations

"""Compatibility exports for the original external producer entrypoint.

The producer implementation lives in :mod:`posterior_json_producer`; this
module only preserves the historical import surface used by the command-line
worker and recovered integrations.  Keeping this as a re-export prevents
recognizer parsing code from becoming the owner of producer/runtime logic.
"""

from any2xsampa.adapters.onnx_espeak_runtime import (
    build_onnx_espeak_runtime_diagnostics_snapshot,
    describe_onnx_espeak_runtime,
    write_onnx_espeak_runtime_diagnostics,
)
from any2xsampa.adapters.posterior_json_producer import (
    SCHEMA_VERSION,
    ScoreAwarePosteriorJSONProducer,
    available_producer_backend_names,
)

__all__ = [
    "SCHEMA_VERSION",
    "ScoreAwarePosteriorJSONProducer",
    "available_producer_backend_names",
    "build_onnx_espeak_runtime_diagnostics_snapshot",
    "describe_onnx_espeak_runtime",
    "write_onnx_espeak_runtime_diagnostics",
]

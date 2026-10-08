"""Optional gRPC bridge surfaces for desktop clients."""

from any2xsampa.grpc.product_bridge import (
    SCHEMA_VERSION,
    InferDraftVsqxRequest,
    InferDraftVsqxResponse,
    MikuV4CProductDraftBridge,
    grpc_runtime_available,
    require_grpc_runtime,
)

__all__ = [
    "SCHEMA_VERSION",
    "InferDraftVsqxRequest",
    "InferDraftVsqxResponse",
    "MikuV4CProductDraftBridge",
    "grpc_runtime_available",
    "require_grpc_runtime",
]

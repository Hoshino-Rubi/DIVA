from __future__ import annotations

import argparse
import json
from concurrent import futures
from pathlib import Path
import sys
from typing import Any

from any2xsampa.grpc.product_bridge import (
    MikuV4CProductDraftBridge,
    require_grpc_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the optional any2xsampa desktop gRPC bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Only verify that grpcio/generated stubs are available; do not start the server.",
    )
    return parser


def create_server(host: str = "127.0.0.1", port: int = 50051) -> Any:
    require_grpc_runtime()
    try:
        import grpc  # type: ignore
        from any2xsampa.grpc.generated import miku_v4c_product_pb2_grpc  # type: ignore
        from any2xsampa.grpc.servicer import MikuV4CProductDraftServicer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional generated deps
        raise RuntimeError(
            "gRPC runtime is present but generated protobuf stubs are missing. "
            "Generate stubs from proto/any2xsampa/v1/miku_v4c_product.proto before starting the network server."
        ) from exc

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    miku_v4c_product_pb2_grpc.add_MikuV4CProductDraftServiceServicer_to_server(
        MikuV4CProductDraftServicer(MikuV4CProductDraftBridge()),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    return server


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_runtime:
        try:
            require_grpc_runtime()
        except RuntimeError as exc:
            print(json.dumps({"ok": False, "error": str(exc), "error_kind": "grpc_runtime_missing"}, ensure_ascii=False), file=sys.stderr)
            return 1
        print(json.dumps({"ok": True, "runtime": "grpcio"}, ensure_ascii=False))
        return 0
    try:
        server = create_server(host=args.host, port=args.port)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_kind": "grpc_server_unavailable"}, ensure_ascii=False), file=sys.stderr)
        return 1
    server.start()
    print(f"any2xsampa gRPC listening on {args.host}:{args.port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:  # pragma: no cover
        server.stop(grace=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from any2xsampa.api.app import ApiApplication


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    app: ApiApplication | None = None,
    state_dir: Path | None = None,
) -> ThreadingHTTPServer:
    application = app or ApiApplication(storage_root=state_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._write_response("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._write_response("POST")

        def do_HEAD(self) -> None:  # noqa: N802
            self._write_response("GET", include_body=False)

        def do_DELETE(self) -> None:  # noqa: N802
            self._write_response("DELETE")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._write_response("OPTIONS")

        def do_PATCH(self) -> None:  # noqa: N802
            self._write_response("PATCH")

        def do_PUT(self) -> None:  # noqa: N802
            self._write_response("PUT")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_response(self, method: str, *, include_body: bool = True) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length else b""
            headers = {key: value for key, value in self.headers.items()}
            status_code, payload = application.handle(method, self.path, body, headers=headers)
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            if include_body:
                self.wfile.write(raw)

    return ThreadingHTTPServer((host, port), Handler)


@dataclass(slots=True)
class ApiServerHandle:
    server: ThreadingHTTPServer
    thread: threading.Thread

    @property
    def host(self) -> str:
        return str(self.server.server_address[0])

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def start_server_in_thread(
    host: str = "127.0.0.1",
    port: int = 0,
    app: ApiApplication | None = None,
    state_dir: Path | None = None,
) -> ApiServerHandle:
    server = create_server(host=host, port=port, app=app, state_dir=state_dir)
    thread = threading.Thread(target=server.serve_forever, name="any2xsampa-api", daemon=True)
    thread.start()
    return ApiServerHandle(server=server, thread=thread)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the any2xsampa local JSON API service.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8000, type=int, help="TCP port to bind.")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Optional local state directory for persisted upload/job metadata.",
    )
    args = parser.parse_args()
    state_dir = Path(args.state_dir) if args.state_dir else None
    server = create_server(host=args.host, port=args.port, state_dir=state_dir)
    try:
        print(f"any2xsampa API listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

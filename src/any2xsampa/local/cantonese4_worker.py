"""Isolated worker for the optional Cantonese adapter."""
from __future__ import annotations

import sys
from .common import sha


class Engine:
    def close(self):
        pass

    def execute(self, action, params, cancel, progress):
        if action != "cantonese4":
            raise ValueError("Cantonese worker accepts only cantonese4")
        from any2xsampa.adapters.cantonese4_runtime import run
        output = run(params, cancel, progress)
        return {"result_sha256": sha(output / "result.json")}


def main():
    from .server import Server, input_lines
    import json
    wire = sys.stdout
    sys.stdout = sys.stderr
    server = Server(Engine(), wire)
    server.emit("ready", data={"runtime_worker": True})
    try:
        for line in input_lines(sys.stdin):
            server.receive(json.loads(line))
    finally:
        server.close()


if __name__ == "__main__":
    main()

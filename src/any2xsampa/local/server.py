"""Versioned newline-JSON stdio server; serial jobs, responsive control channel."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import threading
from .common import VERSION, Cancelled

class Server:
    def __init__(self, engine, output):
        self.engine, self.output = engine, output
        self.output_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.pending = {}
        self.seen = set()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-task")

    def emit(self, kind, task_id=None, **fields):
        with self.output_lock:
            self.output.write(json.dumps({"protocol": VERSION, "event": kind, "id": task_id, **fields}, ensure_ascii=True, allow_nan=False)+"\n")
            self.output.flush()

    def receive(self, message):
        if not isinstance(message, dict) or message.get("protocol") != VERSION:
            raise ValueError("Unsupported local protocol version")
        task_id = message.get("id")
        if not isinstance(task_id, str) or not 1 <= len(task_id) <= 128:
            raise ValueError("Message id must be a nonempty string <=128 chars")
        if task_id in self.seen:
            raise ValueError("Duplicate request id")
        if len(self.seen) >= 10000:
            raise RuntimeError("Session request limit reached; release and reopen the worker")
        self.seen.add(task_id)
        action = message.get("command")
        if action == "cancel":
            with self.state_lock:
                event = self.pending.get(message.get("target"))
                if event is not None:
                    event.set()
            self.emit("result", task_id, data={"cancel_requested": event is not None, "target": message.get("target")})
            return
        if action == "capabilities":
            self.emit("result", task_id, data={"commands": ["recognize", "recognize-whole", "replay", "models", "doctor", "note_runtime", "model_setup_status", "install_models", "inspect_audio", "draft", "export", "export_vsqx", "build_score", "import_score", "export_score", "preview_xsampa", "cancel", "unload"],
                      "transport": "stdio", "max_pending": 8, "requires_server_port": False})
            return
        params = message.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        with self.state_lock:
            if len(self.pending) >= 8:
                raise RuntimeError("Task queue is full")
            cancel = self.pending[task_id] = threading.Event()
        self.emit("accepted", task_id)
        self.pool.submit(self._run, task_id, action, params, cancel)

    def _run(self, task_id, action, params, cancel):
        try:
            if cancel.is_set():
                raise Cancelled("Cancelled before execution")
            result = self.engine.execute(action, params, cancel, lambda data: self.emit("progress", task_id, data=data))
            if cancel.is_set():
                raise Cancelled("Cancelled at completion boundary")
            self.emit("result", task_id, data=result)
        except Cancelled as e:
            self.emit("cancelled", task_id, message=str(e))
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.emit("error", task_id, code=type(e).__name__, message=str(e), retry_performed=False)
        finally:
            with self.state_lock:
                self.pending.pop(task_id, None)

    def close(self):
        with self.state_lock:
            for event in self.pending.values():
                event.set()
        self.pool.shutdown(wait=True)
        self.engine.close()

def input_lines(stream):
    """Do not hold the Windows CRT stdin lock while importing native libraries.

    Blocking TextIO.readline on Windows can deadlock NumPy's native import in
    another thread. Peek the owned anonymous pipe, then read available bytes.
    This is also a responsive control loop, not a busy wait.
    """
    import time
    fd = stream.fileno()
    pending = b""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        kernel.PeekNamedPipe.restype = wintypes.BOOL
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
    while True:
        if os.name == "nt":
            available = wintypes.DWORD()
            if not kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
                error = ctypes.get_last_error()
                if error in {109, 232}:
                    break
                raise OSError(error, "Local worker stdin must be a pipe")
            if not available.value:
                time.sleep(.025)
                continue
            count = min(available.value, 65536)
        else:
            import select
            if not select.select([fd], [], [], .1)[0]:
                continue
            count = 65536
        block = os.read(fd, count)
        if not block:
            break
        pending += block
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if len(line) > 1024*1024:
                raise ValueError("Request exceeds 1 MiB")
            yield line.decode("utf-8")
        if len(pending) > 1024*1024:
            raise ValueError("Request exceeds 1 MiB")
    if pending.strip():
        raise ValueError("Truncated JSON message at EOF")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Keep a dedicated wire. Python/library print() can never corrupt stdout JSON.
    wire = sys.stdout
    sys.stdout = sys.stderr
    if args.runtime:
        from .runtime import RuntimeEngine
        engine = RuntimeEngine()
    else:
        from .engine import LocalEngine
        engine = LocalEngine()
    server = Server(engine, wire)
    server.emit("ready", data={"runtime_worker": args.runtime, "pid": os.getpid()})
    try:
        for line in input_lines(sys.stdin):
            message = None
            try:
                if len(line) > 1024*1024:
                    raise ValueError("Request exceeds 1 MiB")
                message = json.loads(line)
                server.receive(message)
            except Exception as e:
                server.emit("error", message.get("id") if isinstance(message, dict) else None,
                            code=type(e).__name__, message=str(e), retry_performed=False)
    finally:
        server.close()

if __name__ == "__main__":
    main()

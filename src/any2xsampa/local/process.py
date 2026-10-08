"""Own only the child we start; no shell, ports or process-name termination."""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
from .common import ROOT, VERSION, Cancelled

class OwnedJob:
    def __init__(self, process: subprocess.Popen):
        self.handle = None
        if os.name != "nt":
            return
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        class Basic(ctypes.Structure):
            _fields_ = [("per_process", ctypes.c_int64), ("per_job", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min", ctypes.c_size_t), ("max", ctypes.c_size_t),
                        ("active", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(x, ctypes.c_uint64) for x in ("read", "write", "other", "rb", "wb", "ob")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_limit", ctypes.c_size_t),
                        ("job_limit", ctypes.c_size_t), ("process_peak", ctypes.c_size_t), ("job_peak", ctypes.c_size_t)]
        h = k.CreateJobObjectW(None, None)
        if not h:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(h, 9, ctypes.byref(info), ctypes.sizeof(info)) or not k.AssignProcessToJobObject(h, wintypes.HANDLE(int(process._handle))):
            err = ctypes.get_last_error()
            k.CloseHandle(h)
            raise ctypes.WinError(err)
        self.kernel, self.handle = k, h

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

class WorkerClient:
    def __init__(self, python: str, log: Path, *, runtime: bool = True,
                 module: str = "any2xsampa.local.server"):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ANY2XSAMPA_HOME": str(ROOT),
               "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
               "NUMBA_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"}
        command = [python, "-X", "utf8", "-m", module] + (["--runtime"] if runtime else [])
        self.log = log
        log.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
                                        env=env, shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            self.job = OwnedJob(self.process)
        except BaseException:
            self.process.kill()
            self.process.wait()
            raise
        self.events = queue.Queue()
        self.send_lock = threading.Lock()
        self.out_thread = threading.Thread(target=self._read, daemon=True)
        self.err_thread = threading.Thread(target=self._logs, daemon=True)
        self.out_thread.start()
        self.err_thread.start()
        try:
            ready = self.events.get(timeout=30)
            if not isinstance(ready, dict) or ready.get("event") != "ready" or ready.get("protocol") != VERSION:
                raise RuntimeError("Invalid worker handshake")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            while line := self.process.stdout.readline(1024*1024+1):
                if len(line) > 1024*1024:
                    raise ValueError("Oversized worker message")
                self.events.put(json.loads(line))
        except Exception as exc:
            self.events.put({"event": "transport_error", "message": str(exc)})
        finally:
            self.events.put(None)

    def _logs(self):
        with self.log.open("a", encoding="utf-8") as f:
            while data := self.process.stderr.read(4096):
                f.write(data)
                f.flush()

    def send(self, command: dict):
        with self.send_lock:
            self.process.stdin.write(json.dumps({"protocol": VERSION, **command}, ensure_ascii=True, allow_nan=False) + "\n")
            self.process.stdin.flush()

    def call(self, action: str, params: dict, cancel: threading.Event, progress, *, timeout: float = 930):
        task_id = uuid.uuid4().hex
        self.send({"id": task_id, "command": action, "params": params})
        deadline = time.monotonic() + timeout
        cancel_at = None
        while True:
            if cancel.is_set() and cancel_at is None:
                self.send({"id": uuid.uuid4().hex, "command": "cancel", "target": task_id})
                cancel_at = time.monotonic()
            if cancel_at is not None and time.monotonic()-cancel_at >= 3:
                self.close()
                raise Cancelled("Owned worker stopped after cancellation grace period")
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError("Local worker timed out; not retried")
            try:
                event = self.events.get(timeout=.05)
            except queue.Empty:
                continue
            if event is None or event.get("event") == "transport_error":
                if cancel.is_set():
                    raise Cancelled("Worker stopped after cancellation")
                raise RuntimeError("Worker exited without a result; see " + str(self.log))
            if event.get("id") != task_id:
                continue
            kind = event.get("event")
            if kind == "progress":
                progress(event.get("data", {}))
            elif kind == "result":
                if cancel.is_set():
                    raise Cancelled("Task completed after cancellation; result not accepted")
                return event["data"]
            elif kind == "cancelled":
                raise Cancelled(event.get("message", "Cancelled"))
            elif kind == "error":
                raise RuntimeError(event.get("message", "Worker failed"))

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.job.close()
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=5)
        self.job.close()
        for stream in (self.process.stdin,):
            if stream and not stream.closed:
                stream.close()
        for thread in (getattr(self, "out_thread", None), getattr(self, "err_thread", None)):
            if thread:
                thread.join(timeout=2)

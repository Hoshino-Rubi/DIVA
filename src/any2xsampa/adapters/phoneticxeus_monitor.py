"""Fail-closed resource monitoring for this process, never other user jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable

from any2xsampa.training.limited_data_artifacts import write_json_new


GIB = 1024 ** 3


def query_total_gpu_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15, shell=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("resource monitor requires one unambiguous GPU")
    value = int(lines[0].strip()) * 1024 ** 2
    if value < 0:
        raise RuntimeError("invalid GPU memory measurement")
    return value


class ProcessResourceMonitor:
    def __init__(
        self, output_dir: Path, *, timeout_seconds: float,
        interval_seconds: float = 1.0, probe: Callable[[], int] = query_total_gpu_bytes,
    ) -> None:
        if timeout_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("resource monitor intervals must be positive")
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.interval_seconds = interval_seconds
        self.probe = probe
        self.stop = threading.Event()
        self.failure_lock = threading.Lock()
        self.samples: list[dict] = []
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self._watch, name="phoneticxeus-resource-monitor", daemon=True)
        self.timer = threading.Timer(timeout_seconds, self._abort, args=("wall_clock_limit",))
        self.timer.daemon = True

    def _abort(self, reason: str) -> None:
        with self.failure_lock:
            path = self.output_dir / "resource-failure.json"
            if not path.exists():
                try:
                    write_json_new(path, {
                        "schema_version": "phoneticxeus_resource_failure_v1",
                        "reason": reason, "pid": os.getpid(),
                        "at_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": time.monotonic() - self.start,
                        "latest_sample": self.samples[-1] if self.samples else None,
                        "only_current_process_terminated": True,
                    })
                finally:
                    os._exit(73)
            os._exit(73)

    def _watch(self) -> None:
        while not self.stop.is_set():
            try:
                used = self.probe()
                self.samples.append({"elapsed_seconds": time.monotonic() - self.start, "total_used_bytes": used})
                if used > 14 * GIB:
                    self._abort("total_gpu_memory_limit")
            except Exception as exc:
                self._abort(f"gpu_measurement_failed:{type(exc).__name__}")
            self.stop.wait(self.interval_seconds)

    def __enter__(self) -> ProcessResourceMonitor:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timer.start()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop.set()
        self.timer.cancel()
        self.thread.join(timeout=16)
        if self.thread.is_alive():
            self._abort("resource_monitor_did_not_stop")
        write_json_new(self.output_dir / "resource-monitor.json", {
            "schema_version": "phoneticxeus_resource_monitor_v1",
            "pid": os.getpid(), "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds, "total_gpu_limit_bytes": 14 * GIB,
            "elapsed_seconds": time.monotonic() - self.start,
            "completed_without_resource_abort": True,
            "operation_succeeded": exc_type is None,
            "maximum_sampled_total_used_bytes": max((row["total_used_bytes"] for row in self.samples), default=None),
            "samples": self.samples,
        })

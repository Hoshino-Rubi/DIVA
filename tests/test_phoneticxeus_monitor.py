from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor


ROOT = Path(__file__).resolve().parents[1]


def test_monitor_records_samples_and_stops_its_thread(tmp_path: Path) -> None:
    with ProcessResourceMonitor(tmp_path, timeout_seconds=5, interval_seconds=0.01, probe=lambda: 5 * 1024 ** 3) as monitor:
        assert monitor.thread.is_alive()
    assert not monitor.thread.is_alive()
    report = json.loads((tmp_path / "resource-monitor.json").read_text())
    assert report["completed_without_resource_abort"]
    assert report["operation_succeeded"]
    assert report["maximum_sampled_total_used_bytes"] == 5 * 1024 ** 3


def run_probe(tmp_path: Path, *, timeout: float, memory: int) -> subprocess.CompletedProcess:
    code = """
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
with ProcessResourceMonitor(Path(sys.argv[2]),timeout_seconds=float(sys.argv[3]),interval_seconds=0.01,probe=lambda:int(sys.argv[4])):
    time.sleep(10)
"""
    return subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT), str(tmp_path), str(timeout), str(memory)], capture_output=True, text=True, timeout=3)


def test_deadline_terminates_only_the_owned_child_process(tmp_path: Path) -> None:
    result = run_probe(tmp_path, timeout=0.15, memory=0)
    assert result.returncode == 73
    report = json.loads((tmp_path / "resource-failure.json").read_text())
    assert report["reason"] == "wall_clock_limit"
    assert report["only_current_process_terminated"]


def test_gpu_limit_terminates_only_the_owned_child_process(tmp_path: Path) -> None:
    result = run_probe(tmp_path, timeout=2, memory=15 * 1024 ** 3)
    assert result.returncode == 73
    report = json.loads((tmp_path / "resource-failure.json").read_text())
    assert report["reason"] == "total_gpu_memory_limit"

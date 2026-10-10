import json
import sys
import threading

from any2xsampa.local.common import sha
from any2xsampa.local.process import WorkerClient, WorkerError


def test_reused_worker_has_session_log_and_per_task_links(tmp_path):
    worker = WorkerClient(sys.executable, tmp_path / "session.log", runtime=False)
    try:
        rows = []
        for index in range(2):
            out = tmp_path / str(index)
            out.mkdir()
            worker.call("capabilities", {"_output": str(out)}, threading.Event(), lambda _: None)
            rows.append(json.loads((out / "worker-session.json").read_text()))
        assert rows[0]["session_id"] == rows[1]["session_id"]
        assert rows[0]["pid"] == rows[1]["pid"]
        assert rows[0]["task_id"] != rows[1]["task_id"]
        assert all(row["log_path"] == str((tmp_path / "session.log").resolve()) for row in rows)
        assert rows[0]["log_scope"] == "worker_session"
    finally:
        worker.close()


def test_remote_error_code_survives_transport(tmp_path):
    worker = WorkerClient(sys.executable, tmp_path / "error.log", runtime=False)
    try:
        try:
            worker.call("missing_command", {}, threading.Event(), lambda _: None)
        except WorkerError as error:
            assert error.code == "ValueError"
            assert "Unknown local command" in str(error)
        else:
            raise AssertionError("Worker error was hidden")
    finally:
        worker.close()

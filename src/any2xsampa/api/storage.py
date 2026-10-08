from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from any2xsampa.core.io import load_json, write_json


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _default_storage_root() -> Path:
    return Path(tempfile.gettempdir()) / "any2xsampa_api_state"


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload, ensure_ascii=False, indent=2, atomic=True)


def _load_json_records(records_dir: Path) -> tuple[list[dict[str, Any]], int]:
    if not records_dir.exists():
        return [], 0
    payloads: list[dict[str, Any]] = []
    load_errors = 0
    for path in sorted(records_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            load_errors += 1
            continue
        payloads.append(payload)
    return payloads, load_errors


@dataclass(slots=True)
class UploadRecord:
    upload_id: str
    original_filename: str
    stored_path: str
    size_bytes: int
    content_type: str
    created_at: str
    recovered_from_persistence: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any], recovered_from_persistence: bool = False) -> UploadRecord:
        return cls(
            upload_id=str(payload["upload_id"]),
            original_filename=str(payload["original_filename"]),
            stored_path=str(payload["stored_path"]),
            size_bytes=int(payload["size_bytes"]),
            content_type=str(payload["content_type"]),
            created_at=str(payload["created_at"]),
            recovered_from_persistence=recovered_from_persistence,
        )

    def to_dict(self) -> dict[str, Any]:
        file_exists = Path(self.stored_path).exists()
        payload = {
            "upload_id": self.upload_id,
            "original_filename": self.original_filename,
            "stored_path": self.stored_path,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "created_at": self.created_at,
            "file_exists": file_exists,
            "file_status": "present" if file_exists else "missing",
        }
        if self.recovered_from_persistence:
            payload["recovered_from_persistence"] = True
        return payload


class ApiUploadStore:
    def __init__(self, root_dir: Path | None = None) -> None:
        base_dir = root_dir or (_default_storage_root() / "uploads")
        self.root_dir = base_dir
        self.files_dir = self.root_dir / "files"
        self.records_dir = self.root_dir / "records"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._uploads: dict[str, UploadRecord] = {}
        self._lock = threading.Lock()
        self._load_errors = 0
        self._load_records()

    def save(self, filename: str, content: bytes, content_type: str) -> dict[str, Any]:
        safe_name = Path(filename).name.strip() or "upload.bin"
        upload_id = _new_identifier("upload")
        stored_path = self.files_dir / f"{upload_id}_{safe_name}"
        stored_path.write_bytes(content)
        record = UploadRecord(
            upload_id=upload_id,
            original_filename=safe_name,
            stored_path=str(stored_path),
            size_bytes=len(content),
            content_type=content_type,
            created_at=_timestamp(),
        )
        with self._lock:
            self._uploads[upload_id] = record
            self._persist_locked(record)
        return record.to_dict()

    def count(self) -> int:
        with self._lock:
            return len(self._uploads)

    def get(self, upload_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._uploads.get(upload_id)
            if record is None:
                raise KeyError(f"Unknown upload id: {upload_id}")
            return record.to_dict()

    def list(self) -> dict[str, Any]:
        with self._lock:
            uploads = [record.to_dict() for _, record in sorted(self._uploads.items(), key=lambda item: item[1].created_at)]
        return {"uploads": uploads}

    def cleanup(self, max_age_seconds: float, protected_paths: set[str] | None = None) -> dict[str, Any]:
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
        removed_upload_ids: list[str] = []
        skipped_protected_upload_ids: list[str] = []
        protected_paths = {_canonical_path(path) for path in (protected_paths or set())}
        with self._lock:
            for upload_id, record in list(self._uploads.items()):
                created_at = _parse_timestamp(record.created_at).timestamp()
                if created_at > cutoff:
                    continue
                if _canonical_path(record.stored_path) in protected_paths:
                    skipped_protected_upload_ids.append(upload_id)
                    continue
                Path(record.stored_path).unlink(missing_ok=True)
                self._record_path(upload_id).unlink(missing_ok=True)
                del self._uploads[upload_id]
                removed_upload_ids.append(upload_id)
        return {
            "kind": "uploads",
            "removed_count": len(removed_upload_ids),
            "removed_upload_ids": removed_upload_ids,
            "skipped_protected_upload_ids": skipped_protected_upload_ids,
            "max_age_seconds": max_age_seconds,
            "cleanup_policy": "old-upload-records-and-files-excluding-active-job-paths",
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            missing_ids = [
                record.upload_id for _, record in sorted(self._uploads.items(), key=lambda item: item[1].created_at)
                if not Path(record.stored_path).exists()
            ]
            return {
                "backend": "local_json",
                "root_dir": str(self.root_dir),
                "record_count": len(self._uploads),
                "existing_file_count": len(self._uploads) - len(missing_ids),
                "missing_file_count": len(missing_ids),
                "missing_upload_ids": missing_ids,
                "consistency_status": "degraded" if missing_ids or self._load_errors else "ok",
                "load_errors": self._load_errors,
            }

    def _load_records(self) -> None:
        payloads, load_errors = _load_json_records(self.records_dir)
        uploads: dict[str, UploadRecord] = {}
        for payload in payloads:
            try:
                record = UploadRecord.from_dict(payload, recovered_from_persistence=True)
            except (KeyError, TypeError, ValueError):
                load_errors += 1
                continue
            uploads[record.upload_id] = record
        self._uploads = uploads
        self._load_errors = load_errors

    def _persist_locked(self, record: UploadRecord) -> None:
        _write_json(self._record_path(record.upload_id), record.to_dict())

    def _record_path(self, upload_id: str) -> Path:
        return self.records_dir / f"{upload_id}.json"


@dataclass(slots=True)
class JobRecord:
    job_id: str
    kind: str
    mode: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    total_items: int = 1
    completed_items: int = 0
    success_count: int = 0
    failure_count: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    recovered_from_persistence: bool = False
    recovery_note: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], recovered_from_persistence: bool = False) -> JobRecord:
        raw_items = payload.get("items", [])
        items = [dict(item) for item in raw_items] if isinstance(raw_items, list) else []
        result = payload.get("result")
        normalized_result = dict(result) if isinstance(result, dict) else None
        error = payload.get("error")
        recovery_note = payload.get("recovery_note")
        return cls(
            job_id=str(payload["job_id"]),
            kind=str(payload["kind"]),
            mode=str(payload["mode"]),
            status=str(payload["status"]),
            created_at=str(payload["created_at"]),
            started_at=_optional_text(payload.get("started_at")),
            completed_at=_optional_text(payload.get("completed_at")),
            result=normalized_result,
            error=str(error) if error is not None else None,
            total_items=int(payload.get("total_items", 1)),
            completed_items=int(payload.get("completed_items", 0)),
            success_count=int(payload.get("success_count", 0)),
            failure_count=int(payload.get("failure_count", 0)),
            items=items,
            recovered_from_persistence=recovered_from_persistence or bool(payload.get("recovered_from_persistence")),
            recovery_note=str(recovery_note) if recovery_note is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "kind": self.kind,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.kind == "batch":
            payload["total_items"] = self.total_items
            payload["completed_items"] = self.completed_items
            payload["success_count"] = self.success_count
            payload["failure_count"] = self.failure_count
            payload["items"] = [dict(item) for item in self.items]
        elif self.result is not None:
            payload["result"] = dict(self.result)
        if self.recovered_from_persistence:
            payload["recovered_from_persistence"] = True
        if self.recovery_note is not None:
            payload["recovery_note"] = self.recovery_note
        return payload


class ApiJobStore:
    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or (_default_storage_root() / "jobs")
        self.records_dir = self.root_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._active_input_paths_by_job: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self._load_errors = 0
        self._recovered_interrupted_jobs = 0
        self._load_records()

    def submit(
        self,
        mode: str,
        request: Any,
        runner: Callable[[Any], dict[str, Any]],
    ) -> dict[str, Any]:
        record = JobRecord(
            job_id=_new_identifier("job"),
            kind="single",
            mode=mode,
            status="queued",
            created_at=_timestamp(),
        )
        with self._lock:
            self._jobs[record.job_id] = record
            self._active_input_paths_by_job[record.job_id] = _request_input_paths(request)
            self._persist_locked(record)
            submitted_payload = record.to_dict()
        thread = threading.Thread(
            target=self._run_single,
            args=(record.job_id, request, runner),
            name=f"any2xsampa-{record.job_id}",
            daemon=True,
        )
        thread.start()
        return submitted_payload

    def submit_batch(
        self,
        mode: str,
        requests: list[Any],
        runner: Callable[[Any], dict[str, Any]],
    ) -> dict[str, Any]:
        record = JobRecord(
            job_id=_new_identifier("batch"),
            kind="batch",
            mode=mode,
            status="queued",
            created_at=_timestamp(),
            total_items=len(requests),
            items=[{"index": index, "status": "queued"} for index, _ in enumerate(requests)],
        )
        with self._lock:
            self._jobs[record.job_id] = record
            self._active_input_paths_by_job[record.job_id] = _request_collection_input_paths(requests)
            self._persist_locked(record)
            submitted_payload = record.to_dict()
        thread = threading.Thread(
            target=self._run_batch,
            args=(record.job_id, requests, runner),
            name=f"any2xsampa-{record.job_id}",
            daemon=True,
        )
        thread.start()
        return submitted_payload

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown job id: {job_id}")
            return record.to_dict()

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def list(self) -> dict[str, Any]:
        with self._lock:
            jobs = [record.to_dict() for _, record in sorted(self._jobs.items(), key=lambda item: item[1].created_at)]
        return {"jobs": jobs}

    def active_input_paths(self) -> set[str]:
        with self._lock:
            active_paths: set[str] = set()
            for paths in self._active_input_paths_by_job.values():
                active_paths.update(paths)
            return active_paths

    def cleanup(self, max_age_seconds: float) -> dict[str, Any]:
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
        removed_job_ids: list[str] = []
        with self._lock:
            for job_id, record in list(self._jobs.items()):
                if record.status not in {"completed", "failed"}:
                    continue
                reference_time = record.completed_at or record.created_at
                completed_at = _parse_timestamp(reference_time).timestamp()
                if completed_at > cutoff:
                    continue
                self._record_path(job_id).unlink(missing_ok=True)
                del self._jobs[job_id]
                removed_job_ids.append(job_id)
        return {
            "kind": "jobs",
            "removed_count": len(removed_job_ids),
            "removed_job_ids": removed_job_ids,
            "max_age_seconds": max_age_seconds,
            "cleanup_policy": "completed-and-failed-only",
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            status_counts: dict[str, int] = {}
            kind_counts: dict[str, int] = {}
            for record in self._jobs.values():
                status_counts[record.status] = status_counts.get(record.status, 0) + 1
                kind_counts[record.kind] = kind_counts.get(record.kind, 0) + 1
            return {
                "backend": "local_json",
                "root_dir": str(self.root_dir),
                "record_count": len(self._jobs),
                "status_counts": status_counts,
                "kind_counts": kind_counts,
                "consistency_status": "degraded" if self._load_errors else "ok",
                "load_errors": self._load_errors,
                "recovered_interrupted_jobs": self._recovered_interrupted_jobs,
            }

    def _run_single(
        self,
        job_id: str,
        request: Any,
        runner: Callable[[Any], dict[str, Any]],
    ) -> None:
        self._mark_started(job_id)
        try:
            result = runner(request)
        except Exception as exc:
            self._mark_failed(job_id, str(exc))
            return
        self._mark_single_completed(job_id, result)

    def _run_batch(
        self,
        job_id: str,
        requests: list[Any],
        runner: Callable[[Any], dict[str, Any]],
    ) -> None:
        self._mark_started(job_id)
        try:
            for index, request in enumerate(requests):
                self._set_batch_item_status(job_id, index, "running")
                try:
                    result = runner(request)
                except Exception as exc:
                    self._set_batch_item_failed(job_id, index, str(exc))
                    continue
                self._set_batch_item_completed(job_id, index, result)
        except Exception as exc:
            self._mark_failed(job_id, str(exc))
            return
        self._mark_batch_completed(job_id)

    def _load_records(self) -> None:
        payloads, load_errors = _load_json_records(self.records_dir)
        jobs: dict[str, JobRecord] = {}
        recovered_interrupted_jobs = 0
        for payload in payloads:
            try:
                record = JobRecord.from_dict(payload, recovered_from_persistence=True)
            except (KeyError, TypeError, ValueError):
                load_errors += 1
                continue
            if self._recover_interrupted_record(record):
                recovered_interrupted_jobs += 1
                _write_json(self.records_dir / f"{record.job_id}.json", record.to_dict())
            jobs[record.job_id] = record
        self._jobs = jobs
        self._load_errors = load_errors
        self._recovered_interrupted_jobs = recovered_interrupted_jobs

    def _recover_interrupted_record(self, record: JobRecord) -> bool:
        if record.status not in {"queued", "running"}:
            return False
        note = "Job metadata was restored after API restart, but in-flight execution cannot resume automatically."
        record.status = "failed"
        record.recovered_from_persistence = True
        record.recovery_note = note
        record.error = note
        if record.kind == "batch":
            self._mark_interrupted_batch_items(record, note)
        if record.completed_at is None:
            record.completed_at = _timestamp()
        return True

    def _mark_interrupted_batch_items(self, record: JobRecord, note: str) -> None:
        normalized_items: list[dict[str, Any]] = []
        success_count = 0
        failure_count = 0
        completed_items = 0
        for index, item in enumerate(record.items):
            normalized_item = dict(item)
            normalized_item["index"] = int(normalized_item.get("index", index))
            status = str(normalized_item.get("status", "queued"))
            if status == "completed" and isinstance(normalized_item.get("result"), dict):
                success_count += 1
                completed_items += 1
                normalized_items.append(normalized_item)
                continue
            if status == "failed" and isinstance(normalized_item.get("error"), str):
                failure_count += 1
                completed_items += 1
                normalized_items.append(normalized_item)
                continue
            normalized_items.append(
                {
                    "index": int(normalized_item["index"]),
                    "status": "failed",
                    "error": note,
                }
            )
            failure_count += 1
            completed_items += 1
        record.items = normalized_items
        record.completed_items = completed_items
        record.success_count = success_count
        record.failure_count = failure_count
        record.total_items = max(record.total_items, len(normalized_items))

    def _mark_started(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = _timestamp()
            self._persist_locked(record)

    def _mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "failed"
            record.error = error
            record.completed_at = _timestamp()
            self._active_input_paths_by_job.pop(job_id, None)
            self._persist_locked(record)

    def _mark_single_completed(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "completed"
            record.result = dict(result)
            record.completed_at = _timestamp()
            record.completed_items = 1
            record.success_count = 1
            record.failure_count = 0
            self._active_input_paths_by_job.pop(job_id, None)
            self._persist_locked(record)

    def _set_batch_item_status(self, job_id: str, index: int, status: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.items[index]["status"] = status
            self._persist_locked(record)

    def _set_batch_item_failed(self, job_id: str, index: int, error: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.items[index] = {"index": index, "status": "failed", "error": error}
            record.completed_items += 1
            record.failure_count += 1
            self._persist_locked(record)

    def _set_batch_item_completed(self, job_id: str, index: int, result: dict[str, Any]) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.items[index] = {"index": index, "status": "completed", "result": dict(result)}
            record.completed_items += 1
            record.success_count += 1
            self._persist_locked(record)

    def _mark_batch_completed(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "completed"
            record.completed_at = _timestamp()
            self._active_input_paths_by_job.pop(job_id, None)
            self._persist_locked(record)

    def _persist_locked(self, record: JobRecord) -> None:
        _write_json(self._record_path(record.job_id), record.to_dict())

    def _record_path(self, job_id: str) -> Path:
        return self.records_dir / f"{job_id}.json"


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _request_input_paths(request: Any) -> set[str]:
    input_paths: set[str] = set()
    for field_name in (
        "audio_path",
        "recognizer_fixture",
        "recognizer_model",
        "recognizer_real_output",
        "engine_inventory",
        "note_context",
    ):
        value = getattr(request, field_name, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            input_paths.add(_canonical_path(text))
    return input_paths


def _canonical_path(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))


def _request_collection_input_paths(requests: list[Any]) -> set[str]:
    input_paths: set[str] = set()
    for request in requests:
        input_paths.update(_request_input_paths(request))
    return input_paths

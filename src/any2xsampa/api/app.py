from __future__ import annotations

import json
import math
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from any2xsampa.api.storage import ApiJobStore, ApiUploadStore
from any2xsampa.presentation import (
    augment_pipeline_payload,
    build_transport_error_payload,
    with_status_capabilities,
)
from any2xsampa.services.inference import InferenceRequest, InferenceService, RecognizerContractError


class ApiApplication:
    def __init__(
        self,
        service: InferenceService | None = None,
        upload_store: ApiUploadStore | None = None,
        job_store: ApiJobStore | None = None,
        storage_root: Path | None = None,
    ) -> None:
        upload_root = storage_root / "uploads" if storage_root is not None else None
        job_root = storage_root / "jobs" if storage_root is not None else None
        self.service = service or InferenceService()
        self.upload_store = upload_store or ApiUploadStore(root_dir=upload_root)
        self.job_store = job_store or ApiJobStore(root_dir=job_root)

    def handle(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        route = urlparse(path).path
        normalized_headers = {key.lower(): value for key, value in (headers or {}).items()}
        try:
            if method == "GET" and route == "/health":
                return (200, self.service.health())
            if method == "GET" and route == "/status":
                status = with_status_capabilities(
                    self.service.status(),
                    api_features=["sync_inference", "upload", "jobs", "batch", "cleanup"],
                )
                status["job_count"] = self.job_store.count()
                status["upload_count"] = self.upload_store.count()
                status["storage"] = {
                    "persistence_enabled": True,
                    "uploads": self.upload_store.summary(),
                    "jobs": self.job_store.summary(),
                }
                return (200, status)
            if method == "GET" and route == "/list-supported-export-formats":
                return (200, self.service.list_supported_export_formats())
            if method == "GET" and route == "/list-supported-note-context-formats":
                return (200, self.service.list_supported_note_context_formats())
            if method == "GET" and route == "/uploads":
                return (200, self.upload_store.list())
            if method == "GET" and route.startswith("/uploads/"):
                upload_id = route.removeprefix("/uploads/").strip()
                if not upload_id:
                    return (
                        404,
                        self._error_payload(
                            "Unknown endpoint: GET /uploads/",
                            status_code=404,
                            error_code="api.unknown_endpoint",
                            error_kind="not_found",
                        ),
                    )
                return (200, self.upload_store.get(upload_id))
            if method == "GET" and route == "/jobs":
                return (200, self.job_store.list())
            if method == "GET" and route.startswith("/jobs/"):
                job_id = route.removeprefix("/jobs/").strip()
                if not job_id:
                    return (
                        404,
                        self._error_payload(
                            "Unknown endpoint: GET /jobs/",
                            status_code=404,
                            error_code="api.unknown_endpoint",
                            error_kind="not_found",
                        ),
                    )
                return (200, self.job_store.get(job_id))
            if method == "POST" and route == "/uploads":
                return (201, self._save_upload(body=body, headers=normalized_headers))
            if method == "POST" and route == "/cleanup":
                return (200, self._cleanup(body=body))
            if method == "POST" and route == "/infer-open":
                request = InferenceRequest.from_mapping(self._json_body(body), mode="infer-open")
                return (200, self._run_presented(request))
            if method == "POST" and route == "/infer-with-lyrics":
                request = InferenceRequest.from_mapping(self._json_body(body), mode="infer-with-lyrics")
                return (200, self._run_presented(request))
            if method == "POST" and route == "/jobs/infer-open":
                request = InferenceRequest.from_mapping(self._json_body(body), mode="infer-open")
                return (202, self.job_store.submit("infer-open", request, self._run_presented))
            if method == "POST" and route == "/jobs/infer-with-lyrics":
                request = InferenceRequest.from_mapping(self._json_body(body), mode="infer-with-lyrics")
                return (202, self.job_store.submit("infer-with-lyrics", request, self._run_presented))
            if method == "POST" and route == "/batch/infer-open":
                return (202, self._submit_batch(mode="infer-open", body=body))
            if method == "POST" and route == "/batch/infer-with-lyrics":
                return (202, self._submit_batch(mode="infer-with-lyrics", body=body))
            return (404, self._error_payload(f"Unknown endpoint: {method} {route}", status_code=404, error_code="api.unknown_endpoint", error_kind="not_found"))
        except KeyError as exc:
            return (404, self._error_payload(str(exc), status_code=404, error_code="api.not_found", error_kind="not_found"))
        except RecognizerContractError as exc:
            return (
                400,
                self._error_payload(
                    str(exc),
                    status_code=400,
                    error_code=f"api.{exc.error_code_suffix}",
                    error_kind=exc.error_kind,
                ),
            )
        except json.JSONDecodeError as exc:
            return (
                400,
                self._error_payload(
                    f"Invalid JSON body: {exc.msg}",
                    status_code=400,
                    error_code="api.invalid_json",
                    error_kind="invalid_json",
                ),
            )
        except (FileNotFoundError, ValueError) as exc:
            if isinstance(exc, FileNotFoundError):
                return (400, self._error_payload(str(exc), status_code=400, error_code="api.file_not_found", error_kind="file_not_found"))
            return (400, self._error_payload(str(exc), status_code=400, error_code="api.validation_error", error_kind="validation_error"))
        except Exception as exc:  # pragma: no cover
            return (
                500,
                self._error_payload(
                    f"Internal server error: {exc}",
                    status_code=500,
                    error_code="api.internal_error",
                    error_kind="internal_error",
                ),
            )

    def _json_body(self, body: bytes | None) -> dict[str, Any]:
        if not body:
            return {}
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _submit_batch(self, mode: str, body: bytes | None) -> dict[str, Any]:
        payload = self._json_body(body)
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("Batch request must include a non-empty 'items' list")
        requests: list[InferenceRequest] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each batch item must be a JSON object")
            requests.append(InferenceRequest.from_mapping(item, mode=mode))
        return self.job_store.submit_batch(mode=mode, requests=requests, runner=self._run_presented)

    def _save_upload(self, body: bytes | None, headers: dict[str, str]) -> dict[str, Any]:
        if body is None:
            raise ValueError("Upload body is required")
        content_type = headers.get("content-type", "application/octet-stream")
        if content_type.lower().startswith("multipart/form-data"):
            filename, file_content_type, content = self._parse_multipart_upload(body=body, content_type=content_type)
            return self.upload_store.save(filename=filename, content=content, content_type=file_content_type)
        filename = headers.get("x-filename", "").strip()
        if not filename:
            raise ValueError("Raw uploads require an X-Filename header")
        return self.upload_store.save(filename=filename, content=body, content_type=content_type)

    def _parse_multipart_upload(self, body: bytes, content_type: str) -> tuple[str, str, bytes]:
        raw_message = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        message = BytesParser(policy=default).parsebytes(raw_message)
        if not message.is_multipart():
            raise ValueError("multipart/form-data upload could not be parsed")
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            filename = part.get_filename()
            if not filename:
                continue
            content = part.get_payload(decode=True) or b""
            return filename, part.get_content_type(), content
        raise ValueError("multipart upload must include a file part with a filename")

    def _cleanup(self, body: bytes | None) -> dict[str, Any]:
        payload = self._json_body(body)
        max_age_raw = payload.get("max_age_seconds", 86400)
        if isinstance(max_age_raw, bool):
            raise ValueError("max_age_seconds must be a non-negative number")
        try:
            max_age_seconds = float(max_age_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_age_seconds must be a non-negative number") from exc
        if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
            raise ValueError("max_age_seconds must be a non-negative number")
        jobs = self.job_store.cleanup(max_age_seconds=max_age_seconds)
        uploads = self.upload_store.cleanup(
            max_age_seconds=max_age_seconds,
            protected_paths=self.job_store.active_input_paths(),
        )
        return {
            "max_age_seconds": max_age_seconds,
            "jobs": jobs,
            "uploads": uploads,
        }

    def _run_presented(self, request: InferenceRequest) -> dict[str, Any]:
        return augment_pipeline_payload(self.service.run(request))

    def _error_payload(
        self,
        message: str,
        *,
        status_code: int,
        error_code: str,
        error_kind: str,
    ) -> dict[str, Any]:
        return build_transport_error_payload(
            message,
            transport="api",
            error_code=error_code,
            error_kind=error_kind,
            status_code=status_code,
        )

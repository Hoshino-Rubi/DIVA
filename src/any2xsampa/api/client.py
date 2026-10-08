from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


@dataclass(slots=True)
class ApiClientError(RuntimeError):
    message: str
    status_code: int | None = None
    error_code: str | None = None
    error_kind: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


class Any2XSampaApiClient:
    def __init__(self, base_url: str, timeout_sec: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def list_supported_export_formats(self) -> dict[str, Any]:
        return self._request("GET", "/list-supported-export-formats")

    def list_supported_note_context_formats(self) -> dict[str, Any]:
        return self._request("GET", "/list-supported-note-context-formats")

    def list_uploads(self) -> dict[str, Any]:
        return self._request("GET", "/uploads")

    def get_upload(self, upload_id: str) -> dict[str, Any]:
        return self._request("GET", f"/uploads/{upload_id}")

    def cleanup(self, max_age_seconds: float) -> dict[str, Any]:
        return self._request("POST", "/cleanup", payload={"max_age_seconds": max_age_seconds})

    def infer_open(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/infer-open", payload=payload)

    def infer_with_lyrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/infer-with-lyrics", payload=payload)

    def upload_file(self, file_path: str | Path, content_type: str = "application/octet-stream") -> dict[str, Any]:
        path = Path(file_path)
        boundary = f"----any2xsampa-{uuid.uuid4().hex}"
        body = self._multipart_file_body(
            field_name="file",
            filename=path.name,
            content=path.read_bytes(),
            content_type=content_type,
            boundary=boundary,
        )
        return self._request(
            "POST",
            "/uploads",
            raw_body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def create_infer_open_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/jobs/infer-open", payload=payload)

    def create_infer_with_lyrics_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/jobs/infer-with-lyrics", payload=payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")

    def list_jobs(self) -> dict[str, Any]:
        return self._request("GET", "/jobs")

    def submit_infer_open_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/batch/infer-open", payload={"items": items})

    def submit_infer_with_lyrics_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/batch/infer-with-lyrics", payload={"items": items})

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if payload is not None and raw_body is not None:
            raise ValueError("payload and raw_body are mutually exclusive")
        body = raw_body
        merged_headers = {"Accept": "application/json", **(headers or {})}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            merged_headers["Content-Type"] = "application/json"
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=merged_headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            raise self._error_from_raw(raw, fallback=f"HTTP {exc.code}", status_code=exc.code) from exc
        except error.URLError as exc:
            raise ApiClientError(f"API request failed: {exc.reason}") from exc
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiClientError(f"API returned invalid JSON: {exc.msg}") from exc
        if not isinstance(decoded, dict):
            raise ApiClientError("API returned a non-object JSON payload")
        return decoded

    def _error_from_raw(self, raw: str, fallback: str, status_code: int | None = None) -> ApiClientError:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return ApiClientError(fallback, status_code=status_code)
        if not isinstance(decoded, dict):
            return ApiClientError(fallback, status_code=status_code)
        message = decoded["error"] if isinstance(decoded.get("error"), str) else fallback
        details = decoded.get("details")
        normalized_details = dict(details) if isinstance(details, dict) else None
        payload_status = decoded.get("status_code")
        resolved_status = status_code
        if payload_status is not None:
            try:
                resolved_status = int(payload_status)
            except (TypeError, ValueError):
                resolved_status = status_code
        return ApiClientError(
            message,
            status_code=resolved_status,
            error_code=str(decoded.get("error_code")) if decoded.get("error_code") is not None else None,
            error_kind=str(decoded.get("error_kind")) if decoded.get("error_kind") is not None else None,
            details=normalized_details,
        )

    def _multipart_file_body(
        self,
        field_name: str,
        filename: str,
        content: bytes,
        content_type: str,
        boundary: str,
    ) -> bytes:
        parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        return b"".join(parts)

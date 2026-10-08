from __future__ import annotations

"""Functional reconstruction of the missing Python GUI application.

The original ``src/any2xsampa/gui/app.py`` and its bytecode were recovered as
NUL-filled/corrupt artifacts.  This module is therefore deliberately kept as
an isolated candidate.  It implements the contracts exercised by the readable
GUI tests and keeps fixture mode visible in the request form.
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Iterable, Mapping, Sequence

from any2xsampa.api.client import Any2XSampaApiClient
from any2xsampa.presentation import extract_status_capabilities
from any2xsampa.services.inference import InferenceRequest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _optional(value: object) -> str | None:
    text = _text(value).strip()
    return text or None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False, default=str)


def _safe_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _list_text(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item).strip()]


@dataclass(slots=True)
class GuiFileFingerprint:
    size_bytes: int
    modified_time_ns: int
    modified_at: str
    sha1_prefix: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> GuiFileFingerprint | None:
        data = _mapping(value)
        if not data:
            return None
        try:
            return cls(
                size_bytes=int(data.get("size_bytes", 0)),
                modified_time_ns=int(data.get("modified_time_ns", 0)),
                modified_at=_text(data.get("modified_at")),
                sha1_prefix=_text(data.get("sha1_prefix")),
            )
        except (TypeError, ValueError):
            return None


@dataclass(slots=True)
class GuiFormState:
    task_mode: str = "single"
    audio_path: str = ""
    batch_audio_paths: list[str] = field(default_factory=list)
    dialect: str = "cantonese"
    lyrics: str = ""
    export_format: str = "xsampa"
    engine_profile: str = ""
    engine_inventory: str = ""
    engine_subbank: str = ""
    engine_color: str = ""
    engine_style: str = ""
    engine_tags: list[str] = field(default_factory=list)
    note_context: str = ""
    note_context_format: str = "auto"
    recognizer_mode: str = "fixture"
    recognizer_fixture: str = ""
    recognizer_real_backend: str = ""
    recognizer_real_command: str = ""
    recognizer_real_output: str = "real_recognizer_output.json"
    recognizer_model: str = ""
    recognizer_device: str = "auto"
    disable_recognizer_fallback: bool = False

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        data["batch_audio_paths"] = list(self.batch_audio_paths)
        data["engine_tags"] = list(self.engine_tags)
        return data

    @classmethod
    def from_mapping(cls, value: object) -> GuiFormState:
        data = _mapping(value)
        known = {field_name for field_name in cls.__dataclass_fields__}
        kwargs: dict[str, Any] = {}
        for name in known:
            if name not in data:
                continue
            if name in {"batch_audio_paths", "engine_tags"}:
                kwargs[name] = _list_text(data[name])
            elif name == "disable_recognizer_fallback":
                kwargs[name] = bool(data[name])
            else:
                kwargs[name] = data[name]
        return cls(**kwargs)


@dataclass(slots=True)
class GuiUploadedFile:
    label: str
    local_path: str
    stored_path: str
    upload_id: str | None = None
    original_filename: str = ""
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    fingerprint: GuiFileFingerprint | None = None

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint.to_mapping() if self.fingerprint else None
        return data

    @classmethod
    def from_mapping(cls, value: object) -> GuiUploadedFile:
        data = _mapping(value)
        return cls(
            label=_text(data.get("label")),
            local_path=_text(data.get("local_path")),
            stored_path=_text(data.get("stored_path")),
            upload_id=_optional(data.get("upload_id")),
            original_filename=_text(data.get("original_filename")),
            content_type=_text(data.get("content_type"), "application/octet-stream"),
            size_bytes=int(data.get("size_bytes", 0) or 0),
            fingerprint=GuiFileFingerprint.from_mapping(data.get("fingerprint")),
        )


@dataclass(slots=True)
class GuiJobLaunch:
    mode: str
    payload: dict[str, Any]
    uploads: list[GuiUploadedFile]
    job_id: str
    job_status: str
    state: GuiFormState
    kind: str = "single"
    batch_items: list[dict[str, Any]] = field(default_factory=list)
    total_items: int = 1
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    current_stage: str = "queued"
    stage_updated_at: str | None = None
    retried_from_job_id: str | None = None
    reused_uploads: bool = False
    reused_labels: list[str] = field(default_factory=list)
    reuploaded_labels: list[str] = field(default_factory=list)
    reuse_reason: str | None = None
    recovery_warning: str | None = None


@dataclass(slots=True)
class GuiRunHistoryEntry:
    job_id: str
    mode: str
    dialect: str
    status: str
    created_at: str
    updated_at: str
    state: GuiFormState
    payload: dict[str, Any]
    uploads: list[GuiUploadedFile]
    kind: str = "single"
    result: dict[str, Any] | None = None
    error: str | None = None
    active: bool = False
    retried_from_job_id: str | None = None
    inputs_reused: bool = False
    reuse_reason: str | None = None
    recovery_warning: str | None = None
    result_summary: str | None = None
    error_summary: str | None = None
    current_stage: str = "idle"
    stage_updated_at: str | None = None
    last_error: str | None = None
    batch_items: list[dict[str, Any]] = field(default_factory=list)
    total_items: int = 1
    success_count: int = 0
    failure_count: int = 0
    restored_from_persistence: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "mode": self.mode,
            "dialect": self.dialect,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state.to_mapping(),
            "payload": self.payload,
            "uploads": [upload.to_mapping() for upload in self.uploads],
            "result": self.result,
            "error": self.error,
            "active": self.active,
            "retried_from_job_id": self.retried_from_job_id,
            "inputs_reused": self.inputs_reused,
            "reuse_reason": self.reuse_reason,
            "recovery_warning": self.recovery_warning,
            "result_summary": self.result_summary,
            "error_summary": self.error_summary,
            "current_stage": self.current_stage,
            "stage_updated_at": self.stage_updated_at,
            "last_error": self.last_error,
            "batch_items": self.batch_items,
            "total_items": self.total_items,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_mapping(cls, value: object, *, restored: bool = True) -> GuiRunHistoryEntry:
        data = _mapping(value)
        result = data.get("result")
        return cls(
            job_id=_text(data.get("job_id")),
            kind=_text(data.get("kind"), "single"),
            mode=_text(data.get("mode"), "infer-open"),
            dialect=_text(data.get("dialect"), "cantonese"),
            status=_text(data.get("status"), "unknown"),
            created_at=_text(data.get("created_at")),
            updated_at=_text(data.get("updated_at")),
            state=GuiFormState.from_mapping(data.get("state")),
            payload=dict(_mapping(data.get("payload"))),
            uploads=[GuiUploadedFile.from_mapping(item) for item in data.get("uploads", []) if isinstance(item, Mapping)],
            result=dict(result) if isinstance(result, Mapping) else None,
            error=_optional(data.get("error")),
            active=bool(data.get("active", False)),
            retried_from_job_id=_optional(data.get("retried_from_job_id")),
            inputs_reused=bool(data.get("inputs_reused", False)),
            reuse_reason=_optional(data.get("reuse_reason")),
            recovery_warning=_optional(data.get("recovery_warning")),
            result_summary=_optional(data.get("result_summary")),
            error_summary=_optional(data.get("error_summary")),
            current_stage=_text(data.get("current_stage"), "idle"),
            stage_updated_at=_optional(data.get("stage_updated_at")),
            last_error=_optional(data.get("last_error")),
            batch_items=[dict(item) for item in data.get("batch_items", []) if isinstance(item, Mapping)],
            total_items=int(data.get("total_items", 1) or 1),
            success_count=int(data.get("success_count", 0) or 0),
            failure_count=int(data.get("failure_count", 0) or 0),
            restored_from_persistence=restored,
        )


class GuiHistoryStore:
    """Versioned JSON persistence for recent GUI runs."""

    def __init__(self, path: str | Path, max_entries: int = 20) -> None:
        self.path = Path(path)
        self.max_entries = max(1, int(max_entries))

    def save(self, entries: Iterable[GuiRunHistoryEntry]) -> None:
        values = list(entries)[-self.max_entries :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "saved_at": _now_iso(), "entries": [entry.to_mapping() for entry in values]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load(self) -> list[GuiRunHistoryEntry]:
        if not self.path.exists():
            return []
        try:
            decoded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []
        entries = _mapping(decoded).get("entries", [])
        if not isinstance(entries, list):
            return []
        return [GuiRunHistoryEntry.from_mapping(item) for item in entries if isinstance(item, Mapping)]


def default_gui_history_path() -> Path:
    appdata = os.environ.get("LOCALAPPDATA")
    if appdata:
        return Path(appdata) / "any2xsampa" / "gui_recent_runs.json"
    return Path.home() / ".local" / "share" / "any2xsampa" / "gui_recent_runs.json"


def _fingerprint(path: str | Path) -> GuiFileFingerprint:
    candidate = Path(path)
    stat = candidate.stat()
    digest = hashlib.sha1(candidate.read_bytes()).hexdigest()[:12]
    modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    return GuiFileFingerprint(stat.st_size, stat.st_mtime_ns, modified_at, digest)


def _fingerprint_changed(previous: GuiFileFingerprint | None, current: GuiFileFingerprint) -> bool:
    if previous is None:
        return False
    return (
        previous.size_bytes != current.size_bytes
        or previous.modified_time_ns != current.modified_time_ns
        or previous.sha1_prefix != current.sha1_prefix
    )


def _content_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix in {".json", ".ustx"}:
        return "application/json"
    if suffix == ".ust":
        return "text/plain"
    return "application/octet-stream"


class Any2XSampaGuiController:
    """Thin controller that keeps Tk state independent from API transport."""

    def __init__(self, api_client: Any) -> None:
        self.api_client = api_client

    def _payload_from_state(self, state: GuiFormState, *, audio_path: str | None = None) -> dict[str, Any]:
        mode = "infer-with-lyrics" if state.lyrics.strip() else "infer-open"
        request = InferenceRequest(
            mode=mode,
            audio_path=audio_path if audio_path is not None else state.audio_path,
            dialect=state.dialect,
            lyrics=_optional(state.lyrics),
            export_format=state.export_format,
            engine_profile=_optional(state.engine_profile),
            engine_inventory=_optional(state.engine_inventory),
            engine_subbank=_optional(state.engine_subbank),
            engine_color=_optional(state.engine_color),
            engine_style=_optional(state.engine_style),
            engine_tags=list(state.engine_tags),
            note_context=_optional(state.note_context),
            note_context_format=state.note_context_format or "auto",
            recognizer_mode=state.recognizer_mode or "fixture",
            recognizer_fixture=_optional(state.recognizer_fixture),
            recognizer_real_backend=_optional(state.recognizer_real_backend),
            recognizer_real_command=_optional(state.recognizer_real_command),
            recognizer_real_output=state.recognizer_real_output or "real_recognizer_output.json",
            recognizer_model=_optional(state.recognizer_model),
            recognizer_device=state.recognizer_device or "auto",
            disable_recognizer_fallback=state.disable_recognizer_fallback,
        )
        return request.to_runtime_mapping()

    def submit(self, state: GuiFormState) -> dict[str, Any]:
        payload = self._payload_from_state(state)
        if state.lyrics.strip():
            return self.api_client.infer_with_lyrics(payload)
        return self.api_client.infer_open(payload)

    def _upload(self, label: str, path: str | Path) -> GuiUploadedFile:
        candidate = Path(path)
        if not candidate.exists():
            raise ValueError(f"{label} file not found: {candidate}")
        response = self.api_client.upload_file(candidate, content_type=_content_type(candidate))
        stored = _text(response.get("stored_path"))
        if not stored:
            raise ValueError(f"upload response did not include stored_path for {label}")
        return GuiUploadedFile(
            label=label,
            local_path=str(candidate),
            stored_path=stored,
            upload_id=_optional(response.get("upload_id")),
            original_filename=_text(response.get("original_filename"), candidate.name),
            content_type=_text(response.get("content_type"), _content_type(candidate)),
            size_bytes=int(response.get("size_bytes", candidate.stat().st_size) or 0),
            fingerprint=_fingerprint(candidate),
        )

    def start_job_flow(self, state: GuiFormState) -> GuiJobLaunch:
        if state.task_mode == "batch":
            return self.start_batch_job_flow(state)
        uploads: list[GuiUploadedFile] = []
        audio = self._upload("audio", state.audio_path)
        uploads.append(audio)
        inventory_path = _optional(state.engine_inventory)
        note_context_path = _optional(state.note_context)
        inventory = self._upload("engine_inventory", inventory_path) if inventory_path else None
        note_context = self._upload("note_context", note_context_path) if note_context_path else None
        if inventory:
            uploads.append(inventory)
        if note_context:
            uploads.append(note_context)
        upload_paths = {
            "audio_path": audio.stored_path,
            "engine_inventory": inventory.stored_path if inventory else state.engine_inventory or None,
            "note_context": note_context.stored_path if note_context else state.note_context or None,
        }
        payload = self._payload_from_state(state, audio_path=audio.stored_path)
        payload["engine_inventory"] = upload_paths["engine_inventory"]
        payload["note_context"] = upload_paths["note_context"]
        mode = "infer-with-lyrics" if state.lyrics.strip() else "infer-open"
        response = (
            self.api_client.create_infer_with_lyrics_job(payload)
            if mode == "infer-with-lyrics"
            else self.api_client.create_infer_open_job(payload)
        )
        created = _text(response.get("created_at"), _now_iso())
        return GuiJobLaunch(
            mode=mode,
            payload=payload,
            uploads=uploads,
            job_id=_text(response.get("job_id")),
            job_status=_text(response.get("status"), "queued"),
            state=state,
            created_at=created,
            current_stage=_text(response.get("current_stage"), _text(response.get("status"), "queued")),
            stage_updated_at=_optional(response.get("stage_updated_at")) or created,
        )

    def start_batch_job_flow(self, state: GuiFormState) -> GuiJobLaunch:
        paths = list(state.batch_audio_paths)
        if not paths and state.audio_path:
            paths = [state.audio_path]
        if not paths:
            raise ValueError("batch audio paths are required")
        uploads: list[GuiUploadedFile] = []
        items: list[dict[str, Any]] = []
        mode = "infer-with-lyrics" if state.lyrics.strip() else "infer-open"
        for index, local_path in enumerate(paths):
            upload = self._upload(f"audio[{index}]", local_path)
            uploads.append(upload)
            item = self._payload_from_state(state, audio_path=upload.stored_path)
            items.append(item)
        response = (
            self.api_client.submit_infer_with_lyrics_batch(items)
            if mode == "infer-with-lyrics"
            else self.api_client.submit_infer_open_batch(items)
        )
        created = _text(response.get("created_at"), _now_iso())
        batch_items = [
            {
                "index": index,
                "audio_filename": Path(path).name,
                "audio_local_path": str(path),
                "status": "queued",
                "summary": "queued",
            }
            for index, path in enumerate(paths)
        ]
        payload = {"items": items}
        return GuiJobLaunch(
            kind="batch",
            mode=mode,
            payload=payload,
            uploads=uploads,
            job_id=_text(response.get("job_id")),
            job_status=_text(response.get("status"), "queued"),
            state=state,
            batch_items=batch_items,
            total_items=len(items),
            created_at=created,
            current_stage=_text(response.get("current_stage"), _text(response.get("status"), "queued")),
            stage_updated_at=_optional(response.get("stage_updated_at")) or created,
        )

    def retry_job_flow(self, entry: GuiRunHistoryEntry) -> GuiJobLaunch:
        state = entry.state
        if entry.kind == "batch" or state.task_mode == "batch":
            # Batch retries use the same local list; annotate the launch after
            # the ordinary upload path has been reconstructed.
            launch = self.start_batch_job_flow(state)
            launch.retried_from_job_id = entry.job_id
            return launch

        upload_by_label = {upload.label: upload for upload in entry.uploads}
        uploads: list[GuiUploadedFile] = []
        reused: list[str] = []
        reuploaded: list[str] = []
        warnings: list[str] = []
        stored_paths: dict[str, str | None] = {}
        for label, source_path in (
            ("audio", entry.state.audio_path),
            ("engine_inventory", entry.state.engine_inventory),
            ("note_context", entry.state.note_context),
        ):
            old = upload_by_label.get(label)
            stored = old.stored_path if old else _text(entry.payload.get({"audio": "audio_path", "engine_inventory": "engine_inventory", "note_context": "note_context"}[label]))
            if stored and Path(stored).exists():
                stored_paths[label] = stored
                if old:
                    uploads.append(old)
                    reused.append(label)
                continue
            if source_path and Path(source_path).exists():
                current = _fingerprint(source_path)
                if old and _fingerprint_changed(old.fingerprint, current):
                    warnings.append(
                        f"Local {label} file still exists, but its fingerprint changed before retry re-upload "
                        f"(previous=size={old.fingerprint.size_bytes if old.fingerprint else 0}, "
                        f"current=size={current.size_bytes})."
                    )
                replacement = self._upload(label, source_path)
                uploads.append(replacement)
                reuploaded.append(label)
                stored_paths[label] = replacement.stored_path
                continue
            if label == "audio":
                raise ValueError("uploaded audio path is unavailable")
            stored_paths[label] = None

        payload = dict(entry.payload)
        payload["audio_path"] = stored_paths["audio"]
        if "engine_inventory" in payload:
            payload["engine_inventory"] = stored_paths["engine_inventory"]
        if "note_context" in payload:
            payload["note_context"] = stored_paths["note_context"]
        mode = entry.mode
        response = (
            self.api_client.create_infer_with_lyrics_job(payload)
            if mode == "infer-with-lyrics"
            else self.api_client.create_infer_open_job(payload)
        )
        created = _text(response.get("created_at"), _now_iso())
        reason: str | None = None
        if reused:
            reason = f"Reused uploaded inputs: {', '.join(reused)}."
        elif reuploaded:
            reason = f"Re-uploaded from original local paths: {', '.join(reuploaded)}."
        return GuiJobLaunch(
            mode=mode,
            payload=payload,
            uploads=uploads,
            job_id=_text(response.get("job_id")),
            job_status=_text(response.get("status"), "queued"),
            state=state,
            created_at=created,
            retried_from_job_id=entry.job_id,
            reused_uploads=bool(reused) and not reuploaded,
            reused_labels=reused,
            reuploaded_labels=reuploaded,
            reuse_reason=reason,
            recovery_warning=" ".join(warnings) if warnings else None,
            current_stage=_text(response.get("current_stage"), _text(response.get("status"), "queued")),
            stage_updated_at=_optional(response.get("stage_updated_at")) or created,
        )


def _provenance_view(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metadata.get("future_aligner_provenance_view")
    if isinstance(value, Mapping):
        return value
    return {}


def _evidence_items(view: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for key in ("accepted_evidence", "rejected_evidence"):
        entries = view.get(key, [])
        if isinstance(entries, list):
            values.extend(item for item in entries if isinstance(item, Mapping))
    return values


def _display(value: object, fallback: str = "<not provided>") -> str:
    if value is None or value == "":
        return fallback
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else fallback
    if isinstance(value, Mapping):
        return _json_dumps(value)
    return str(value)


def format_evidence_rows(segment: Mapping[str, Any]) -> list[tuple[str, str, str, str, str, str, str]]:
    metadata = _mapping(segment.get("metadata"))
    view = _provenance_view(metadata)
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for item in _evidence_items(view):
        source_index = _safe_int(item.get("source_segment_index"))
        source_note = _display(item.get("source_note_id"), "<none>")
        reason = _display(item.get("rejection_reason"), "<none>")
        rows.append(
            (
                _text(item.get("support_direction"), "window"),
                _text(item.get("evidence_type"), "unknown"),
                f"{_safe_float(item.get('support_score')):.3f}",
                str(source_index) if source_index is not None else "<none>",
                source_note,
                "accepted" if bool(item.get("accepted", False)) else "rejected",
                reason,
            )
        )
    return rows


def format_segment_row(segment: Mapping[str, Any], index: int) -> tuple[str, str, str, str, str, str]:
    metadata = _mapping(segment.get("metadata"))
    view = _provenance_view(metadata)
    selected_alias = _display(metadata.get("selected_alias"), "<none>")
    note_id = _display(view.get("aligned_note_id") or metadata.get("aligned_note_id"), "<none>")
    reason = _display(metadata.get("selection_reason"), "No provenance metadata available.")
    if not metadata:
        reason = "No provenance metadata available."
    return (
        str(index),
        _text(segment.get("normalized_ipa")),
        _text(segment.get("xsampa")),
        selected_alias,
        note_id,
        reason,
    )


def _format_provenance_lines(metadata: Mapping[str, Any]) -> list[str]:
    view = _provenance_view(metadata)
    aligned_note = view.get("aligned_note_id") or metadata.get("aligned_note_id")
    summary = view.get("summary") or metadata.get("future_aligner_provenance_summary")
    fallback_used = bool(view.get("fallback_used", False))
    accepted = view.get("accepted_evidence", []) if isinstance(view.get("accepted_evidence", []), list) else []
    rejected = view.get("rejected_evidence", []) if isinstance(view.get("rejected_evidence", []), list) else []
    left = view.get("left_evidence", []) if isinstance(view.get("left_evidence", []), list) else []
    right = view.get("right_evidence", []) if isinstance(view.get("right_evidence", []), list) else []
    reasons = view.get("rejection_reasons", []) if isinstance(view.get("rejection_reasons", []), list) else []
    lines = [
        "Provenance",
        f"aligned_note_id: {_display(aligned_note)}",
        f"provenance_summary: {_display(summary)}",
        f"evidence_counts: left={len(left)} right={len(right)} accepted={len(accepted)} rejected={len(rejected)}",
        f"rejection_reasons: {_display(reasons)}",
        f"provenance_fallback_used: {str(fallback_used).lower()}",
    ]
    for label, entries in (("accepted_evidence", accepted), ("rejected_evidence", rejected)):
        for item in entries:
            source_note = _display(item.get("source_note_id"), "<none>")
            lines.append(
                f"{label}: {_text(item.get('evidence_type'), 'unknown')}@"
                f"{_text(item.get('support_direction'), 'window')}:note={source_note}:"
                f"score={_safe_float(item.get('support_score')):.3f}"
            )
    return lines


def format_segment_detail(segment: Mapping[str, Any], index: int) -> str:
    metadata = _mapping(segment.get("metadata"))
    selected_alias = metadata.get("selected_alias")
    note_applied = bool(metadata.get("note_context_applied", False))
    note_source = metadata.get("note_context_source")
    candidates = metadata.get("candidate_aliases")
    ranking = metadata.get("ranking_features")
    selected_alias_metadata = metadata.get("selected_alias_metadata")
    ownership_basis = metadata.get("ownership_resolution_basis")
    ownership_candidates = metadata.get("ownership_conflict_candidates")
    competition_summary = metadata.get("alignment_competition_summary")
    fallback_used = bool(metadata.get("aligner_v2_fallback_used", False))
    boundary_conflict = bool(metadata.get("aligner_v2_boundary_conflict_likely", False))
    view = _provenance_view(metadata)
    view_fallback = bool(view.get("fallback_used", False))
    aligned_note = view.get("aligned_note_id") or metadata.get("aligned_note_id")
    lines = [
        f"Segment [{index}]",
        f"time: {_display(segment.get('start_sec'), '<not provided>')} - {_display(segment.get('end_sec'), '<not provided>')}",
        f"normalized_ipa: {_display(segment.get('normalized_ipa'))}",
        f"xsampa: {_display(segment.get('xsampa'))}",
        "",
        "Alias Decision",
        f"selected_alias: {_display(selected_alias, '<none>')}",
        f"selection_reason: {_display(metadata.get('selection_reason'))}",
        f"candidate_aliases: {_display(candidates)}",
        f"ranking_features: {_display(ranking)}",
        "selected_alias_metadata: "
        + (
            json.dumps(selected_alias_metadata, ensure_ascii=False, sort_keys=True)
            if isinstance(selected_alias_metadata, Mapping)
            else _display(selected_alias_metadata)
        ),
        f"alias_explanation: selected_alias={_display(selected_alias)}",
        f"note_context: applied={str(note_applied).lower()} source={_display(note_source)}",
        "",
        "Aligned Note",
        f"aligned_note_id: {_display(aligned_note)}",
        f"note_explanation: {'aligned_note_id=' + str(aligned_note) if aligned_note else 'No aligned note selected.'}",
        f"decision_summary: {_display(metadata.get('decision_summary') or metadata.get('selection_reason'))}",
        "",
        "Competition / Ownership",
        f"ownership_resolution_basis: {_display(ownership_basis)}",
        f"ownership_conflict_candidates: {_display(ownership_candidates)}",
        f"alignment_competition_summary: {_display(competition_summary)}",
        f"competition_explanation: basis={_display(ownership_basis)}"
        + (f" | candidates={_display(ownership_candidates)}" if ownership_candidates else ""),
        f"aligner_v2: {'aligner_v2_boundary_conflict_likely=' + str(boundary_conflict) + ', aligner_v2_fallback_used=' + str(fallback_used)}"
        if ("aligner_v2_fallback_used" in metadata or "aligner_v2_boundary_conflict_likely" in metadata)
        else "aligner_v2: <not provided>",
        "",
        "Fallback / Provenance",
        *_format_provenance_lines(metadata),
        f"fallback_explanation: {'Fallback was used.' if (fallback_used or view_fallback) else 'Fallback state not provided.'}",
    ]
    return "\n".join(lines)


def _segment_decision(segment: Mapping[str, Any]) -> str:
    metadata = _mapping(segment.get("metadata"))
    return _text(metadata.get("selection_reason"), "<not provided>")


def format_result_summary(result: Mapping[str, Any]) -> str:
    segments = result.get("segments") if isinstance(result.get("segments"), list) else []
    warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
    lines = [
        f"Mode: {_display(result.get('mode'))}",
        f"Dialect: {_display(result.get('dialect'))}",
        f"Audio: {_display(result.get('audio_path'))}",
        "",
        "Overview",
        f"segments={len(segments)} lyrics={'yes' if result.get('mode') == 'infer-with-lyrics' else 'no'} warnings={len(warnings)}",
        "",
        "Decision Snapshot",
    ]
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            continue
        metadata = _mapping(segment.get("metadata"))
        view = _provenance_view(metadata)
        summary = view.get("summary") or metadata.get("future_aligner_provenance_summary")
        note_id = view.get("aligned_note_id") or metadata.get("aligned_note_id")
        lines.append(
            f"[{index}] ipa={_display(segment.get('normalized_ipa'))} xsampa={_display(segment.get('xsampa'))} "
            f"selected_alias={_display(metadata.get('selected_alias'))} decision={_segment_decision(segment)}"
        )
        if summary:
            reasons = view.get("rejection_reasons", []) if isinstance(view.get("rejection_reasons", []), list) else []
            suffix = f"; rejections={','.join(str(item) for item in reasons)}" if reasons else ""
            lines.append(f"provenance=summary={summary}{suffix}")
        lines.append(f"why_note=aligned_note_id={_display(note_id)}")
        lines.append(
            f"fallback={'Fallback was used.' if view.get('fallback_used', False) else 'Fallback state not provided.'}"
        )
    if warnings:
        lines.extend(["", "Warnings", *[f"- {warning}" for warning in warnings]])
    return "\n".join(lines)


def _recovery_label(entry: GuiRunHistoryEntry) -> str:
    if entry.kind == "batch":
        return "batch run"
    if entry.recovery_warning:
        return "changed local file"
    if entry.reuse_reason and entry.reuse_reason.startswith("Re-uploaded"):
        return "reuploaded local"
    if entry.inputs_reused:
        return "reused upload"
    return "fresh run"


def format_history_row(entry: GuiRunHistoryEntry) -> tuple[str, ...]:
    label = "restored" if entry.restored_from_persistence else "current"
    summary = entry.error_summary or entry.error
    if entry.kind == "batch":
        summary = f"batch={entry.total_items} success={entry.success_count} failure={entry.failure_count}"
    elif not summary:
        summary = entry.result_summary or ""
    if summary and entry.error and "error=" not in summary:
        summary = f"error={entry.error}"
    return (
        label,
        entry.kind,
        entry.job_id,
        entry.mode,
        entry.dialect,
        entry.status,
        entry.current_stage,
        _recovery_label(entry),
        entry.updated_at,
        summary,
    )


def format_history_detail(entry: GuiRunHistoryEntry) -> str:
    lines = [
        f"Job ID: {entry.job_id}",
        f"Kind: {entry.kind}",
        f"Mode: {entry.mode}",
        f"Dialect: {entry.dialect}",
        f"Status: {entry.status}",
        f"Stage: {entry.current_stage}",
        f"Recovery Label: {_recovery_label(entry)}",
        f"Created: {entry.created_at}",
        f"Updated: {entry.updated_at}",
    ]
    if entry.retried_from_job_id:
        lines.append(f"Retried From: {entry.retried_from_job_id}")
    if entry.reuse_reason:
        lines.append(f"Reuse Reason: {entry.reuse_reason}")
    if entry.recovery_warning:
        lines.append(f"Recovery Warning: {entry.recovery_warning}")
    if entry.error:
        lines.append(f"Error: {entry.error}")
    if entry.kind == "batch":
        lines.extend([
            "",
            "Batch Items:",
            f"total={entry.total_items} success={entry.success_count} failure={entry.failure_count}",
        ])
        failed = next((item for item in entry.batch_items if _text(item.get("status")) == "failed"), None)
        if failed:
            lines.append("Latest Failed Batch Item:")
            lines.append(format_batch_item_detail(failed))
    lines.extend(["", "Browse Hint: Select a segment or batch item to inspect details."])
    return "\n".join(lines)


def format_batch_item_detail(item: Mapping[str, Any]) -> str:
    index = _text(item.get("index"), "<none>")
    lines = [
        f"Batch Item [{index}]",
        f"Audio: {_display(item.get('audio_filename'))}",
        f"Local Path: {_display(item.get('audio_local_path'))}",
        f"Status: {_display(item.get('status'))}",
    ]
    if item.get("error"):
        lines.append(f"Error: {item.get('error')}")
    if item.get("summary"):
        lines.append(f"Summary: {item.get('summary')}")
    result = item.get("result")
    if isinstance(result, Mapping):
        warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        lines.append("")
        lines.append(f"Warnings: {len(warnings)}")
        lines.append(format_result_summary(result))
        segments = result.get("segments") if isinstance(result.get("segments"), list) else []
        if segments and isinstance(segments[0], Mapping):
            lines.append(f"First Decision: {_segment_decision(segments[0])}")
    return "\n".join(lines)


class Any2XSampaGuiApp:
    """Tk presentation shell for sync, job, batch, and history workflows."""

    _STAGES = ["idle", "uploading audio", "uploading inventory", "uploading note context", "submitting job", "queued", "running", "fetching result", "completed", "failed"]

    def __init__(
        self,
        root: tk.Tk,
        controller: Any2XSampaGuiController,
        *,
        api_base_url: str = "http://127.0.0.1:8000",
        history_store: GuiHistoryStore | None = None,
    ) -> None:
        self.root = root
        self.controller = controller
        self.api_base_url = api_base_url
        self.history_store = history_store or GuiHistoryStore(default_gui_history_path())
        self.history_entries = self.history_store.load()
        self.selected_history_job_id: str | None = None
        self.current_batch_items: list[dict[str, Any]] = []
        self.current_result: dict[str, Any] | None = None
        self.current_job_id: str | None = None
        self.current_launch: GuiJobLaunch | None = None
        self._poll_after_id: str | None = None
        self._worker: threading.Thread | None = None
        self._build_variables()
        self._build_widgets()
        if self.history_entries:
            active = next((entry for entry in self.history_entries if entry.active), self.history_entries[0])
            self.selected_history_job_id = active.job_id
            self.history_status_var.set(f"Restored {len(self.history_entries)} Recent Runs")
        self._refresh_history_tree()

    def _build_variables(self) -> None:
        self.task_mode_var = tk.StringVar(self.root, value="single")
        self.audio_var = tk.StringVar(self.root, value="")
        self.lyrics_var = tk.StringVar(self.root, value="")
        self.dialect_var = tk.StringVar(self.root, value="cantonese")
        self.export_format_var = tk.StringVar(self.root, value="xsampa")
        self.engine_profile_var = tk.StringVar(self.root, value="")
        self.engine_inventory_var = tk.StringVar(self.root, value="")
        self.note_context_var = tk.StringVar(self.root, value="")
        self.note_context_format_var = tk.StringVar(self.root, value="auto")
        self.recognizer_mode_var = tk.StringVar(self.root, value="fixture")
        self.recognizer_fixture_var = tk.StringVar(self.root, value="")
        self.job_id_var = tk.StringVar(self.root, value="<none>")
        self.job_status_var = tk.StringVar(self.root, value="idle")
        self.stage_var = tk.StringVar(self.root, value="idle")
        self.stage_progress_var = tk.DoubleVar(self.root, value=0.0)
        self.workflow_var = tk.StringVar(self.root, value="Ready for a fixture-backed run.")
        self.error_var = tk.StringVar(self.root, value="<none>")
        self.recovery_note_var = tk.StringVar(self.root, value="")
        self.result_overview_var = tk.StringVar(self.root, value="No result loaded.")
        self.segment_overview_var = tk.StringVar(self.root, value="No segment selected.")
        self.segment_status_var = tk.StringVar(self.root, value="No segment evidence.")
        self.batch_summary_var = tk.StringVar(self.root, value="No batch selected.")
        self.history_status_var = tk.StringVar(self.root, value="No recent runs.")

    def _build_widgets(self) -> None:
        self.root.title("Any2Xsampa")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        form = ttk.LabelFrame(outer, text="Inference")
        form.pack(fill="x")
        ttk.Label(form, text="Audio").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.audio_var, width=52).grid(row=0, column=1, columnspan=2, sticky="ew")
        ttk.Button(form, text="Browse", command=self._browse_audio).grid(row=0, column=3)
        ttk.Label(form, text="Lyrics").grid(row=1, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.lyrics_var, width=52).grid(row=1, column=1, columnspan=3, sticky="ew")
        ttk.Label(form, text="Dialect").grid(row=2, column=0, sticky="w")
        self.dialect_box = ttk.Combobox(form, textvariable=self.dialect_var, state="readonly", width=18)
        self.dialect_box.grid(row=2, column=1, sticky="w")
        ttk.Label(form, text="Export").grid(row=2, column=2, sticky="e")
        self.export_box = ttk.Combobox(form, textvariable=self.export_format_var, state="readonly", width=18)
        self.export_box.grid(row=2, column=3, sticky="w")
        ttk.Label(form, text="Recognizer").grid(row=3, column=0, sticky="w")
        self.recognizer_box = ttk.Combobox(form, textvariable=self.recognizer_mode_var, state="readonly", width=18)
        self.recognizer_box.grid(row=3, column=1, sticky="w")
        ttk.Entry(form, textvariable=self.recognizer_fixture_var, width=38).grid(row=3, column=2, columnspan=2, sticky="ew")
        self.run_button = ttk.Button(form, text="Upload + Submit Job", command=self._submit_from_form)
        self.run_button.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        for column in range(4):
            form.columnconfigure(column, weight=1 if column in (1, 3) else 0)

        status = ttk.LabelFrame(outer, text="Job Status")
        status.pack(fill="x", pady=(8, 0))
        ttk.Label(status, textvariable=self.workflow_var).pack(anchor="w")
        ttk.Label(status, textvariable=self.stage_var).pack(anchor="w")
        self.stage_progress = ttk.Progressbar(status, variable=self.stage_progress_var, maximum=10)
        self.stage_progress.pack(fill="x")
        self.stage_listbox = tk.Listbox(status, height=4, exportselection=False)
        self.stage_listbox.pack(fill="x")
        ttk.Label(status, textvariable=self.job_id_var).pack(anchor="w")
        ttk.Label(status, textvariable=self.job_status_var).pack(anchor="w")
        ttk.Label(status, textvariable=self.error_var).pack(anchor="w")
        ttk.Label(status, textvariable=self.recovery_note_var).pack(anchor="w")

        panes = ttk.Panedwindow(outer, orient="vertical")
        panes.pack(fill="both", expand=True, pady=(8, 0))
        result_frame = ttk.Frame(panes)
        history_frame = ttk.Frame(panes)
        panes.add(result_frame, weight=3)
        panes.add(history_frame, weight=2)
        self.result_text = tk.Text(result_frame, height=8, wrap="word")
        self.result_text.pack(fill="both", expand=True)
        ttk.Label(result_frame, textvariable=self.result_overview_var).pack(anchor="w")
        self.result_json_text = tk.Text(result_frame, height=6, wrap="none")
        self.result_json_text.pack(fill="both", expand=True)
        self.request_json_text = tk.Text(result_frame, height=4, wrap="none")
        self.request_json_text.pack(fill="both", expand=True)
        self.detail_text = tk.Text(result_frame, height=10, wrap="word")
        self.detail_text.pack(fill="both", expand=True)
        self.segment_overview_var.set("No segment selected.")
        ttk.Label(result_frame, textvariable=self.segment_overview_var).pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.segment_status_var).pack(anchor="w")
        self.segment_tree = ttk.Treeview(result_frame, columns=("index", "ipa", "xsampa", "alias", "note", "decision"), show="headings", height=4)
        for name in ("index", "ipa", "xsampa", "alias", "note", "decision"):
            self.segment_tree.heading(name, text=name)
        self.segment_tree.pack(fill="x")
        self.segment_tree.bind("<<TreeviewSelect>>", self._on_segment_selected)
        self.evidence_tree = ttk.Treeview(result_frame, columns=("direction", "type", "score", "source", "note", "status", "reason"), show="headings", height=3)
        for name in ("direction", "type", "score", "source", "note", "status", "reason"):
            self.evidence_tree.heading(name, text=name)
        self.evidence_tree.pack(fill="x")
        self.rejection_list = tk.Listbox(result_frame, height=2, exportselection=False)
        self.rejection_list.pack(fill="x")

        self.history_tree = ttk.Treeview(history_frame, columns=("state", "kind", "job", "mode", "dialect", "status", "stage", "recovery", "updated", "summary"), show="headings", height=5)
        for name in ("state", "kind", "job", "mode", "dialect", "status", "stage", "recovery", "updated", "summary"):
            self.history_tree.heading(name, text=name)
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_selected)
        self.history_detail_text = tk.Text(history_frame, height=8, wrap="word")
        self.history_detail_text.pack(fill="both", expand=True)
        self.batch_summary_var.set("No batch selected.")
        ttk.Label(history_frame, textvariable=self.batch_summary_var).pack(anchor="w")
        self.batch_items_tree = ttk.Treeview(history_frame, columns=("index", "audio", "status", "summary"), show="headings", height=4)
        for name in ("index", "audio", "status", "summary"):
            self.batch_items_tree.heading(name, text=name)
        self.batch_items_tree.pack(fill="x")
        self.batch_items_tree.bind("<<TreeviewSelect>>", self._on_batch_item_selected)
        self.batch_detail_text = tk.Text(history_frame, height=8, wrap="word")
        self.batch_detail_text.pack(fill="both", expand=True)
        ttk.Label(history_frame, textvariable=self.history_status_var).pack(anchor="w")
        self._load_capabilities()
        self._set_text(self.request_json_text, "No submitted request yet.")

    def _load_capabilities(self) -> None:
        try:
            capabilities = extract_status_capabilities(self.controller.api_client.status())
        except Exception:
            capabilities = {
                "supported_dialects": ["cantonese", "taigi"],
                "supported_export_formats": ["xsampa", "openutau_like"],
                "supported_note_context_formats": ["auto", "json", "ustx", "ust"],
                "supported_recognizer_modes": ["fixture", "real"],
            }
        self.dialect_box["values"] = tuple(capabilities.get("supported_dialects", ["cantonese", "taigi"]))
        self.export_box["values"] = tuple(capabilities.get("supported_export_formats", ["xsampa", "openutau_like"]))
        self.recognizer_box["values"] = tuple(capabilities.get("supported_recognizer_modes", ["fixture", "real"]))
        if self.dialect_var.get() not in self.dialect_box["values"]:
            self.dialect_var.set(self.dialect_box["values"][0] if self.dialect_box["values"] else "cantonese")
        if self.export_format_var.get() not in self.export_box["values"]:
            self.export_format_var.set(self.export_box["values"][0] if self.export_box["values"] else "xsampa")

    def _browse_audio(self) -> None:
        selected = filedialog.askopenfilename(filetypes=(("Audio", "*.wav *.flac *.mp3"), ("All files", "*.*")))
        if selected:
            self.audio_var.set(selected)

    def _state_from_form(self) -> GuiFormState:
        return GuiFormState(
            task_mode="single",
            audio_path=self.audio_var.get(),
            dialect=self.dialect_var.get(),
            lyrics=self.lyrics_var.get(),
            export_format=self.export_format_var.get(),
            engine_profile=self.engine_profile_var.get(),
            engine_inventory=self.engine_inventory_var.get(),
            note_context=self.note_context_var.get(),
            note_context_format=self.note_context_format_var.get(),
            recognizer_mode=self.recognizer_mode_var.get() or "fixture",
            recognizer_fixture=self.recognizer_fixture_var.get(),
        )

    def _submit_from_form(self) -> None:
        try:
            self._set_busy(True)
            self._set_stage("uploading audio")
            launch = self.controller.start_job_flow(self._state_from_form())
            self._handle_job_launch_success(launch)
        except Exception as exc:
            self._handle_job_error(str(exc))

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)

    def _set_stage(self, stage: str, *, workflow_text: str | None = None) -> None:
        self.stage_var.set(stage)
        if workflow_text:
            self.workflow_var.set(workflow_text)
        if stage not in self._STAGES:
            self._STAGES.append(stage)
        index = self._STAGES.index(stage)
        self.stage_progress_var.set(index)
        self.stage_listbox.selection_clear(0, tk.END)
        self.stage_listbox.selection_set(index)
        self.stage_listbox.see(index)
        if self.stage_listbox.size() < len(self._STAGES):
            self.stage_listbox.delete(0, tk.END)
            for value in self._STAGES:
                self.stage_listbox.insert(tk.END, value)
            self.stage_listbox.selection_set(index)
            self.stage_listbox.see(index)

    def _handle_worker_stage(self, stage: str) -> None:
        self._set_stage(stage)

    def _set_busy(self, busy: bool) -> None:
        self.run_button.configure(state="disabled" if busy else "normal")

    def _handle_job_launch_success(self, launch: GuiJobLaunch) -> None:
        self.current_launch = launch
        self.current_job_id = launch.job_id
        self.job_id_var.set(launch.job_id)
        self.job_status_var.set(launch.job_status)
        self.error_var.set("<none>")
        self._set_stage(launch.current_stage or launch.job_status)
        self._set_text(self.request_json_text, _json_dumps(launch.payload))
        if launch.recovery_warning:
            self.recovery_note_var.set(launch.recovery_warning)
        elif launch.reuse_reason:
            self.recovery_note_var.set(launch.reuse_reason)
        self.current_batch_items = list(launch.batch_items)
        if launch.kind == "batch":
            self._render_batch_items(self.current_batch_items)
        entry = GuiRunHistoryEntry(
            job_id=launch.job_id,
            kind=launch.kind,
            mode=launch.mode,
            dialect=launch.state.dialect,
            status=launch.job_status,
            created_at=launch.created_at,
            updated_at=launch.created_at,
            state=launch.state,
            payload=launch.payload,
            uploads=launch.uploads,
            active=True,
            retried_from_job_id=launch.retried_from_job_id,
            inputs_reused=launch.reused_uploads,
            reuse_reason=launch.reuse_reason,
            recovery_warning=launch.recovery_warning,
            current_stage=launch.current_stage,
            stage_updated_at=launch.stage_updated_at,
            batch_items=list(launch.batch_items),
            total_items=launch.total_items,
            success_count=launch.success_count,
            failure_count=launch.failure_count,
        )
        self._store_history_entry(entry, select=True)
        self.workflow_var.set(f"Job {launch.job_id} was submitted; waiting for completion.")
        if launch.job_status not in {"completed", "failed"}:
            self._schedule_poll()

    def _handle_job_error(self, message: str) -> None:
        self._set_busy(False)
        self.job_status_var.set("failed")
        self.error_var.set(message)
        self._set_stage("failed")
        self._set_text(self.result_text, f"Job failed.\n{message}")
        try:
            messagebox.showerror("Any2Xsampa job failed", message)
        except tk.TclError:
            pass

    def _handle_job_update(self, update: Mapping[str, Any]) -> None:
        status = _text(update.get("status"), "unknown")
        if self.current_job_id and update.get("job_id") not in {None, self.current_job_id}:
            return
        self.job_status_var.set(status)
        self._set_stage(status)
        self.error_var.set(_text(update.get("error"), "<none>"))
        if isinstance(update.get("items"), list):
            self.current_batch_items = [dict(item) for item in update["items"] if isinstance(item, Mapping)]
            self._render_batch_items(self.current_batch_items)
        if status == "completed":
            result = update.get("result")
            if isinstance(result, Mapping):
                result_map = dict(result)
                self._render_result(result_map)
                self.workflow_var.set("Job completed and result was loaded.")
            elif self.current_batch_items:
                self.workflow_var.set("Batch job completed and results were loaded.")
            self._set_busy(False)
            self._update_current_history(status, result=update.get("result"), update=update)
        elif status == "failed":
            message = _text(update.get("error"), "Job failed")
            self._set_text(self.result_text, f"Job failed.\n{message}")
            self.workflow_var.set("Job failed.")
            self._set_busy(False)
            self._update_current_history(status, error=message, update=update)
            try:
                messagebox.showerror("Any2Xsampa job failed", message)
            except tk.TclError:
                pass
        elif status in {"queued", "running"}:
            self._set_text(self.result_text, "Job in progress.")
            self.workflow_var.set("Job in progress.")
            self._update_current_history(status, update=update)
            self._schedule_poll()

    def _update_current_history(self, status: str, *, result: object = None, error: str | None = None, update: Mapping[str, Any] | None = None) -> None:
        if not self.current_job_id:
            return
        entry = next((item for item in self.history_entries if item.job_id == self.current_job_id), None)
        if entry is None:
            return
        entry.status = status
        entry.current_stage = status
        entry.updated_at = _text((update or {}).get("completed_at"), _now_iso())
        entry.stage_updated_at = entry.updated_at
        if result is not None and isinstance(result, Mapping):
            entry.result = dict(result)
            entry.result_summary = _result_summary_short(entry.result)
        if error:
            entry.error = error
            entry.error_summary = error
            entry.last_error = error
        if entry.kind == "batch" and isinstance(update, Mapping) and isinstance(update.get("items"), list):
            entry.batch_items = [dict(item) for item in update["items"] if isinstance(item, Mapping)]
            entry.total_items = int(update.get("total_items", entry.total_items) or entry.total_items)
            entry.success_count = int(update.get("success_count", entry.success_count) or 0)
            entry.failure_count = int(update.get("failure_count", entry.failure_count) or 0)
        self._store_history_entry(entry, select=True)

    def _schedule_poll(self, delay_ms: int = 500) -> None:
        if not self.current_job_id:
            return
        if self._poll_after_id:
            try:
                self.root.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
        self._poll_after_id = self.root.after(delay_ms, self._poll_job)

    def _poll_job(self) -> None:
        self._poll_after_id = None
        if not self.current_job_id:
            return
        try:
            update = self.controller.api_client.get_job(self.current_job_id)
            self._handle_job_update(update)
        except Exception as exc:
            self._handle_job_error(str(exc))

    def _store_history_entry(self, entry: GuiRunHistoryEntry, *, select: bool = False, persist_only: bool = False) -> None:
        self.history_entries = [item for item in self.history_entries if item.job_id != entry.job_id]
        if select:
            for item in self.history_entries:
                item.active = False
            entry.active = True
            self.selected_history_job_id = entry.job_id
        self.history_entries.insert(0, entry)
        self.history_entries = self.history_entries[: self.history_store.max_entries]
        self.history_store.save(self.history_entries)
        if not persist_only:
            self._refresh_history_tree()

    def _refresh_history_tree(self) -> None:
        if not hasattr(self, "history_tree"):
            return
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for entry in self.history_entries:
            self.history_tree.insert("", "end", iid=entry.job_id, values=format_history_row(entry))
        if self.selected_history_job_id and self.history_tree.exists(self.selected_history_job_id):
            self.history_tree.selection_set(self.selected_history_job_id)

    def _on_history_selected(self, _event: object | None = None) -> None:
        selection = self.history_tree.selection()
        if not selection:
            return
        self.selected_history_job_id = selection[0]
        self._load_selected_history()

    def _selected_entry(self) -> GuiRunHistoryEntry | None:
        if not self.selected_history_job_id:
            return None
        return next((item for item in self.history_entries if item.job_id == self.selected_history_job_id), None)

    def _load_selected_history(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        self._set_text(self.history_detail_text, format_history_detail(entry))
        self._set_text(self.request_json_text, _json_dumps(entry.payload))
        if entry.result:
            self._render_result(entry.result)
        elif entry.error:
            self._set_text(self.result_text, f"Job failed.\n{entry.error}")
        if entry.kind == "batch":
            self.current_batch_items = list(entry.batch_items)
            self._render_batch_items(self.current_batch_items)
            self._set_text(self.batch_detail_text, format_batch_item_detail(self.current_batch_items[0]) if self.current_batch_items else "No batch item selected.")
        self.history_status_var.set(f"Loaded {entry.job_id}")

    def _retry_selected_history(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        self.audio_var.set(entry.state.audio_path)
        self.lyrics_var.set(entry.state.lyrics)
        self.dialect_var.set(entry.state.dialect)
        self.export_format_var.set(entry.state.export_format)
        self.engine_profile_var.set(entry.state.engine_profile)
        self.engine_inventory_var.set(entry.state.engine_inventory)
        self.note_context_var.set(entry.state.note_context)
        self.note_context_format_var.set(entry.state.note_context_format)
        self._begin_retry_flow(entry)

    def _begin_retry_flow(self, entry: GuiRunHistoryEntry) -> None:
        try:
            self._set_busy(True)
            launch = self.controller.retry_job_flow(entry)
            self._handle_job_launch_success(launch)
        except Exception as exc:
            self._handle_job_error(str(exc))

    def _open_history_file(self) -> None:
        path = str(self.history_store.path)
        opener = getattr(os, "startfile", None)
        if opener:
            opener(path)
        else:
            self.history_status_var.set(f"History file: {path}")

    def _clear_history(self) -> None:
        count = len(self.history_entries)
        self.history_entries = []
        self.selected_history_job_id = None
        self.history_store.save([])
        self._refresh_history_tree()
        self._set_text(self.history_detail_text, f"Cleared {count} Recent Runs (retention cap of {self.history_store.max_entries}).")
        self.history_status_var.set(f"Cleared {count} Recent Runs (retention cap of {self.history_store.max_entries})")

    def _render_result(self, result: Mapping[str, Any]) -> None:
        result_map = dict(result)
        self.current_result = result_map
        self._set_text(self.result_text, format_result_summary(result_map))
        self._set_text(self.result_json_text, _json_dumps(result_map))
        self.result_overview_var.set(
            f"mode={_text(result_map.get('mode'))} segments={len(result_map.get('segments', []) if isinstance(result_map.get('segments'), list) else [])} warnings={len(result_map.get('warnings', []) if isinstance(result_map.get('warnings'), list) else [])}"
        )
        self._render_segments(result_map.get("segments", []))

    def _render_segments(self, segments: object) -> None:
        for item in self.segment_tree.get_children():
            self.segment_tree.delete(item)
        for item in self.evidence_tree.get_children():
            self.evidence_tree.delete(item)
        self.rejection_list.delete(0, tk.END)
        values = segments if isinstance(segments, list) else []
        for index, segment in enumerate(values):
            if not isinstance(segment, Mapping):
                continue
            self.segment_tree.insert("", "end", iid=str(index), values=format_segment_row(segment, index))
            for evidence_index, row in enumerate(format_evidence_rows(segment)):
                self.evidence_tree.insert("", "end", iid=f"evidence-{index}" if evidence_index == 0 else f"evidence-{index}-{evidence_index}", values=row)
            view = _provenance_view(_mapping(segment.get("metadata")))
            reasons = view.get("rejection_reasons", []) if isinstance(view.get("rejection_reasons", []), list) else []
            for reason in reasons:
                self.rejection_list.insert(tk.END, str(reason))
        if values:
            self.segment_tree.selection_set("0")
            self._on_segment_selected()

    def _on_segment_selected(self, _event: object | None = None) -> None:
        selection = self.segment_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        entry = self._selected_entry()
        result = entry.result if entry and entry.result else self.current_result
        segments = result.get("segments", []) if isinstance(result, Mapping) else []
        if index >= len(segments) or not isinstance(segments[index], Mapping):
            return
        segment = segments[index]
        self._set_text(self.detail_text, format_segment_detail(segment, index))
        metadata = _mapping(segment.get("metadata"))
        alias = _text(metadata.get("selected_alias"), "<none>")
        self.segment_overview_var.set(
            f"Segment {index}: alias {alias} | {_text(segment.get('normalized_ipa'))} -> {_text(segment.get('xsampa'))}"
        )
        self.segment_status_var.set("Rejection reasons: " + ", ".join(format_evidence_rows(segment)[0][-1] for _ in []))

    def _render_batch_items(self, items: Iterable[Mapping[str, Any]]) -> None:
        for item_id in self.batch_items_tree.get_children():
            self.batch_items_tree.delete(item_id)
        values = [dict(item) for item in items]
        self.current_batch_items = values
        success = sum(1 for item in values if _text(item.get("status")) == "completed")
        failure = sum(1 for item in values if _text(item.get("status")) == "failed")
        self.batch_summary_var.set(f"batch={len(values)} success={success} failed={failure}")
        for index, item in enumerate(values):
            item_index = _safe_int(item.get("index"))
            iid = f"batch-item-{item_index if item_index is not None else index}"
            self.batch_items_tree.insert("", "end", iid=iid, values=(item_index if item_index is not None else index, _text(item.get("audio_filename")), _text(item.get("status")), _text(item.get("summary"))))

    def _on_batch_item_selected(self, _event: object | None = None) -> None:
        selection = self.batch_items_tree.selection()
        if not selection:
            return
        iid = selection[0]
        try:
            index = int(iid.rsplit("-", 1)[1])
        except (ValueError, IndexError):
            index = 0
        item = next((value for value in self.current_batch_items if _safe_int(value.get("index")) == index), None)
        if item is None and index < len(self.current_batch_items):
            item = self.current_batch_items[index]
        if item is None:
            return
        self._set_text(self.batch_detail_text, format_batch_item_detail(item))
        result = item.get("result")
        if isinstance(result, Mapping):
            self._render_result(result)
        elif item.get("error"):
            self._set_text(self.result_text, f"Batch item failed.\n{item.get('error')}")

    def _jump_to_latest_failure(self) -> None:
        for entry in self.history_entries:
            if entry.kind == "batch":
                failed = next((item for item in entry.batch_items if _text(item.get("status")) == "failed"), None)
                if failed is not None:
                    self.selected_history_job_id = entry.job_id
                    self.history_tree.selection_set(entry.job_id)
                    self._render_batch_items(entry.batch_items)
                    item_index = _safe_int(failed.get("index")) or 0
                    iid = f"batch-item-{item_index}"
                    self.batch_items_tree.selection_set(iid)
                    self._on_batch_item_selected()
                    self.history_status_var.set("Opened latest failure")
                    return

    def close(self) -> None:
        if self._poll_after_id:
            try:
                self.root.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def _result_summary_short(result: Mapping[str, Any]) -> str:
    segments = result.get("segments") if isinstance(result.get("segments"), list) else []
    warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
    return f"segments={len(segments)} lyrics={'yes' if result.get('mode') == 'infer-with-lyrics' else 'no'} warnings={len(warnings)}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Any2Xsampa Tk GUI")
    parser.add_argument("--api-base-url", default=os.environ.get("ANY2XSAMPA_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--history", default=None, help="Path to persisted recent-run JSON")
    args = parser.parse_args(argv)
    root = tk.Tk()
    client = Any2XSampaApiClient(args.api_base_url)
    controller = Any2XSampaGuiController(client)
    store = GuiHistoryStore(args.history) if args.history else GuiHistoryStore(default_gui_history_path())
    Any2XSampaGuiApp(root, controller, api_base_url=args.api_base_url, history_store=store)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tkinter as tk

import pytest

from any2xsampa.gui.app import (
    Any2XSampaGuiApp,
    Any2XSampaGuiController,
    GuiFileFingerprint,
    GuiFormState,
    GuiHistoryStore,
    GuiJobLaunch,
    GuiRunHistoryEntry,
    GuiUploadedFile,
    default_gui_history_path,
    format_batch_item_detail,
    format_evidence_rows,
    format_history_detail,
    format_history_row,
    format_result_summary,
    format_segment_detail,
    format_segment_row,
)
from any2xsampa.presentation import extract_status_capabilities
from any2xsampa.services.inference import InferenceRequest


def _lyrics_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "lyrics_constrained_fixture.json"


def _sequence_hint_inventory_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "openutau_bank_inventory_sequence_hint.json"


def _note_context_project_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "note_context_openutau_project.ustx"


def _note_context_ust_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "note_context_openutau_project.ust"


def _history_store(tmp_path: Path) -> GuiHistoryStore:
    return GuiHistoryStore(tmp_path / "gui_recent_runs.json")


def _fingerprint(path: Path) -> GuiFileFingerprint:
    return GuiFileFingerprint(
        size_bytes=path.stat().st_size,
        modified_time_ns=path.stat().st_mtime_ns,
        modified_at="2026-03-11T09:00:00+00:00",
        sha1_prefix=hashlib.sha1(path.read_bytes()).hexdigest()[:12],
    )


class _FakeApiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.upload_calls: list[tuple[str, str]] = []
        self.polled_job_ids: list[str] = []

    def status(self) -> dict[str, object]:
        return {
            "supported_dialects": ["cantonese", "taigi"],
            "supported_export_formats": ["xsampa", "openutau_like"],
            "supported_note_context_formats": ["auto", "json", "ustx", "ust"],
            "supported_recognizer_modes": ["stub", "fixture", "real"],
            "capabilities": {
                "supported_dialects": ["cantonese", "taigi"],
                "supported_export_formats": ["xsampa", "openutau_like"],
                "supported_note_context_formats": ["auto", "json", "ustx", "ust"],
                "supported_recognizer_modes": ["stub", "fixture", "real"],
                "api_features": ["sync_inference", "upload", "jobs"],
            },
        }

    def infer_open(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("infer_open", payload))
        return {
            "mode": "infer-open",
            "dialect": payload["dialect"],
            "audio_path": payload["audio_path"],
            "warnings": [],
            "segments": [
                {
                    "normalized_ipa": "a",
                    "xsampa": "a",
                    "metadata": {
                        "selected_alias": "a",
                        "future_aligner_provenance_view": {
                            "has_provenance": True,
                            "fallback_used": False,
                            "aligned_note_id": "note-001",
                            "summary": "accepted=2,rejected=1,left=1,right=1,types=same_note_continuity+order_continuity",
                            "message": "Accepted 2 evidence items; rejected 1 competing or insufficient items.",
                            "left_evidence": [
                                {
                                    "evidence_type": "same_note_continuity",
                                    "source_segment_index": 0,
                                    "source_note_id": "note-001",
                                    "support_direction": "left",
                                    "support_score": 0.88,
                                    "accepted": True,
                                    "rejection_reason": None,
                                }
                            ],
                            "right_evidence": [
                                {
                                    "evidence_type": "order_continuity",
                                    "source_segment_index": 2,
                                    "source_note_id": "note-001",
                                    "support_direction": "right",
                                    "support_score": 0.08,
                                    "accepted": True,
                                    "rejection_reason": None,
                                }
                            ],
                            "accepted_evidence": [
                                {
                                    "evidence_type": "same_note_continuity",
                                    "source_segment_index": 0,
                                    "source_note_id": "note-001",
                                    "support_direction": "left",
                                    "support_score": 0.88,
                                    "accepted": True,
                                    "rejection_reason": None,
                                },
                                {
                                    "evidence_type": "order_continuity",
                                    "source_segment_index": 2,
                                    "source_note_id": "note-001",
                                    "support_direction": "right",
                                    "support_score": 0.08,
                                    "accepted": True,
                                    "rejection_reason": None,
                                },
                            ],
                            "rejected_evidence": [
                                {
                                    "evidence_type": "fallback",
                                    "source_segment_index": None,
                                    "source_note_id": None,
                                    "support_direction": "window",
                                    "support_score": 0.0,
                                    "accepted": False,
                                    "rejection_reason": "insufficient_same_note_support",
                                }
                            ],
                            "rejection_reasons": ["insufficient_same_note_support"],
                        },
                    },
                }
            ],
        }

    def infer_with_lyrics(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("infer_with_lyrics", payload))
        return {
            "mode": "infer-with-lyrics",
            "dialect": payload["dialect"],
            "audio_path": payload["audio_path"],
            "warnings": [],
            "segments": [
                {"normalized_ipa": "l", "xsampa": "l", "metadata": {"selected_alias": "l a"}},
                {
                    "normalized_ipa": "a",
                    "xsampa": "a",
                    "metadata": {
                        "selected_alias": "a ng",
                        "selection_reason": "note-aware alias selected after lyrics-guided alignment",
                        "future_aligner_provenance_view": {
                            "aligned_note_id": "note-lyrics-001",
                            "summary": "accepted=1,rejected=0,left=1,right=0,types=same_note_continuity",
                            "message": "Accepted 1 evidence item for note stabilization.",
                            "left_evidence": [
                                {
                                    "evidence_type": "same_note_continuity",
                                    "source_segment_index": 0,
                                    "source_note_id": "note-lyrics-001",
                                    "support_direction": "left",
                                    "support_score": 0.91,
                                    "accepted": True,
                                    "rejection_reason": None,
                                }
                            ],
                            "right_evidence": [],
                            "accepted_evidence": [
                                {
                                    "evidence_type": "same_note_continuity",
                                    "source_segment_index": 0,
                                    "source_note_id": "note-lyrics-001",
                                    "support_direction": "left",
                                    "support_score": 0.91,
                                    "accepted": True,
                                    "rejection_reason": None,
                                }
                            ],
                            "rejected_evidence": [],
                            "rejection_reasons": [],
                        },
                    },
                },
            ],
        }

    def upload_file(self, file_path: str | Path, content_type: str = "application/octet-stream") -> dict[str, object]:
        path = Path(file_path)
        self.upload_calls.append((str(path), content_type))
        return {
            "upload_id": f"upload-{len(self.upload_calls)}",
            "original_filename": path.name,
            "stored_path": f"/uploaded/{path.name}",
            "size_bytes": len(path.read_bytes()),
            "content_type": content_type,
        }

    def create_infer_open_job(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("create_infer_open_job", payload))
        return {"job_id": "job-open-001", "status": "queued", "created_at": "2026-03-11T09:00:00+00:00"}

    def create_infer_with_lyrics_job(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("create_infer_with_lyrics_job", payload))
        return {"job_id": "job-lyrics-001", "status": "queued", "created_at": "2026-03-11T09:05:00+00:00"}

    def submit_infer_open_batch(self, items: list[dict[str, object]]) -> dict[str, object]:
        self.calls.append(("submit_infer_open_batch", {"items": items}))
        return {"job_id": "batch-open-001", "status": "queued", "created_at": "2026-03-11T09:10:00+00:00"}

    def submit_infer_with_lyrics_batch(self, items: list[dict[str, object]]) -> dict[str, object]:
        self.calls.append(("submit_infer_with_lyrics_batch", {"items": items}))
        return {"job_id": "batch-lyrics-001", "status": "queued", "created_at": "2026-03-11T09:10:00+00:00"}

    def get_job(self, job_id: str) -> dict[str, object]:
        self.polled_job_ids.append(job_id)
        if job_id.startswith("batch"):
            success_result = self.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})
            return {
                "job_id": job_id,
                "kind": "batch",
                "status": "completed",
                "total_items": 2,
                "completed_items": 2,
                "success_count": 1,
                "failure_count": 1,
                "items": [
                    {
                        "index": 0,
                        "status": "completed",
                        "result": success_result,
                    },
                    {
                        "index": 1,
                        "status": "failed",
                        "error": "Audio file not found",
                    },
                ],
            }
        return {
            "job_id": job_id,
            "kind": "single",
            "status": "completed",
            "result": self.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"}),
        }


class _FlatOnlyStatusApiClient(_FakeApiClient):
    def status(self) -> dict[str, object]:
        return {
            "supported_dialects": ["cantonese", "taigi"],
            "supported_export_formats": ["xsampa", "openutau_like"],
            "supported_note_context_formats": ["auto", "json", "ustx", "ust"],
            "supported_recognizer_modes": ["stub", "fixture", "real"],
            "api_features": ["sync_inference", "upload", "jobs"],
        }


def test_gui_controller_routes_to_api_client() -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]

    controller.submit(
        GuiFormState(
            audio_path="samples/demo.wav",
            dialect="cantonese",
            export_format="xsampa",
        )
    )
    controller.submit(
        GuiFormState(
            audio_path="samples/demo.wav",
            dialect="cantonese",
            lyrics="laa ng",
            export_format="openutau_like",
        )
    )

    assert client.calls[0][0] == "infer_open"
    assert client.calls[1][0] == "infer_with_lyrics"


def test_gui_controller_start_job_flow_uploads_audio_inventory_and_note_context(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    audio_path = tmp_path / "demo.wav"
    inventory_path = tmp_path / "inventory.json"
    note_context_path = tmp_path / "score.ustx"
    audio_path.write_bytes(b"RIFFdemo")
    inventory_path.write_text('{"aliases":["a"]}', encoding="utf-8")
    note_context_path.write_text("{}", encoding="utf-8")

    launch = controller.start_job_flow(
        GuiFormState(
            audio_path=str(audio_path),
            dialect="cantonese",
            export_format="openutau_like",
            engine_profile="generic_openutau",
            engine_inventory=str(inventory_path),
            note_context=str(note_context_path),
            note_context_format="ustx",
        )
    )

    assert launch.job_id == "job-open-001"
    assert launch.job_status == "queued"
    assert [upload.label for upload in launch.uploads] == ["audio", "engine_inventory", "note_context"]
    assert client.calls[-1][0] == "create_infer_open_job"
    job_payload = client.calls[-1][1]
    assert job_payload["audio_path"] == "/uploaded/demo.wav"
    assert job_payload["engine_inventory"] == "/uploaded/inventory.json"
    assert job_payload["note_context"] == "/uploaded/score.ustx"
    assert len(client.upload_calls) == 3


def test_gui_controller_payload_matches_inference_request_contract() -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]

    state = GuiFormState(
        audio_path="samples/demo.wav",
        dialect="cantonese",
        lyrics="laa ng",
        export_format="openutau_like",
        engine_profile="generic_openutau",
        engine_inventory="tests/fixtures/openutau_bank_inventory_behavior.json",
        note_context="tests/fixtures/note_context_openutau_project.ustx",
        note_context_format="ustx",
    )

    payload = controller._payload_from_state(state, audio_path=state.audio_path)
    request = InferenceRequest(
        mode="infer-with-lyrics",
        audio_path=state.audio_path,
        dialect=state.dialect,
        lyrics=state.lyrics,
        export_format=state.export_format,
        engine_profile=state.engine_profile,
        engine_inventory=state.engine_inventory,
        note_context=state.note_context,
        note_context_format=state.note_context_format,
    )

    assert payload == request.to_runtime_mapping()


def test_gui_controller_start_batch_job_flow_smoke(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    audio_a.write_bytes(b"A")
    audio_b.write_bytes(b"B")

    launch = controller.start_batch_job_flow(
        GuiFormState(
            task_mode="batch",
            batch_audio_paths=[str(audio_a), str(audio_b)],
            dialect="cantonese",
            export_format="xsampa",
        )
    )

    assert launch.kind == "batch"
    assert launch.job_id == "batch-open-001"
    assert launch.total_items == 2
    assert client.calls[-1][0] == "submit_infer_open_batch"
    submitted_items = client.calls[-1][1]["items"]
    assert submitted_items[0]["audio_path"] == "/uploaded/a.wav"
    assert submitted_items[1]["audio_path"] == "/uploaded/b.wav"


def test_gui_capability_view_accepts_nested_status_payload() -> None:
    capabilities = extract_status_capabilities(_FakeApiClient().status())

    assert capabilities["supported_dialects"] == ["cantonese", "taigi"]
    assert capabilities["supported_export_formats"] == ["xsampa", "openutau_like"]
    assert capabilities["supported_note_context_formats"] == ["auto", "json", "ustx", "ust"]
    assert capabilities["supported_recognizer_modes"] == ["stub", "fixture", "real"]


def test_gui_capability_view_accepts_flat_status_payload() -> None:
    capabilities = extract_status_capabilities(_FlatOnlyStatusApiClient().status())

    assert capabilities["supported_dialects"] == ["cantonese", "taigi"]
    assert capabilities["supported_export_formats"] == ["xsampa", "openutau_like"]
    assert capabilities["supported_note_context_formats"] == ["auto", "json", "ustx", "ust"]
    assert capabilities["supported_recognizer_modes"] == ["stub", "fixture", "real"]
    assert capabilities["api_features"] == ["sync_inference", "upload", "jobs"]


def test_gui_controller_retry_job_flow_reuses_previous_uploaded_inputs(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    stored_audio = tmp_path / "stored_demo.wav"
    stored_inventory = tmp_path / "stored_inventory.json"
    stored_audio.write_bytes(b"")
    stored_inventory.write_text("{}", encoding="utf-8")
    entry = GuiRunHistoryEntry(
        job_id="job-open-000",
        mode="infer-open",
        dialect="cantonese",
        status="completed",
        created_at="2026-03-11T09:00:00+00:00",
        updated_at="2026-03-11T09:01:00+00:00",
        state=GuiFormState(
            audio_path="C:/local/demo.wav",
            dialect="cantonese",
            export_format="openutau_like",
            engine_inventory="C:/local/inventory.json",
        ),
        payload={
            "audio_path": str(stored_audio),
            "dialect": "cantonese",
            "export_format": "openutau_like",
            "engine_inventory": str(stored_inventory),
            "note_context": None,
            "note_context_format": "auto",
            "engine_profile": None,
        },
        uploads=[
            GuiUploadedFile(
                label="audio",
                local_path="C:/local/demo.wav",
                stored_path=str(stored_audio),
                upload_id="upload-1",
                original_filename="demo.wav",
                content_type="audio/wav",
                size_bytes=0,
                fingerprint=_fingerprint(stored_audio),
            ),
            GuiUploadedFile(
                label="engine_inventory",
                local_path="C:/local/inventory.json",
                stored_path=str(stored_inventory),
                upload_id="upload-2",
                original_filename="inventory.json",
                content_type="application/json",
                size_bytes=2,
                fingerprint=_fingerprint(stored_inventory),
            ),
        ],
    )

    launch = controller.retry_job_flow(entry)

    assert launch.reused_uploads is True
    assert launch.retried_from_job_id == "job-open-000"
    assert launch.reused_labels == ["audio", "engine_inventory"]
    assert launch.reuploaded_labels == []
    assert launch.reuse_reason == "Reused uploaded inputs: audio, engine_inventory."
    assert client.calls[-1][0] == "create_infer_open_job"
    assert client.upload_calls == []
    assert launch.payload["audio_path"] == str(stored_audio)


def test_gui_controller_retry_job_flow_reuploads_from_local_path_when_stored_upload_is_missing(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    missing_audio = tmp_path / "missing_demo.wav"
    local_audio = tmp_path / "demo.wav"
    local_audio.write_bytes(b"RIFFdemo")
    entry = GuiRunHistoryEntry(
        job_id="job-open-000",
        mode="infer-with-lyrics",
        dialect="cantonese",
        status="completed",
        created_at="2026-03-11T09:00:00+00:00",
        updated_at="2026-03-11T09:01:00+00:00",
        state=GuiFormState(audio_path=str(local_audio), dialect="cantonese", lyrics="laa ng"),
        payload={
            "audio_path": str(missing_audio),
            "dialect": "cantonese",
            "export_format": "xsampa",
            "lyrics": "laa ng",
            "engine_inventory": None,
            "note_context": None,
            "note_context_format": "auto",
            "engine_profile": None,
        },
        uploads=[
            GuiUploadedFile(
                label="audio",
                local_path=str(local_audio),
                stored_path=str(missing_audio),
                upload_id="upload-1",
                original_filename="demo.wav",
                content_type="audio/wav",
                size_bytes=8,
                fingerprint=_fingerprint(local_audio),
            )
        ],
    )

    launch = controller.retry_job_flow(entry)

    assert launch.reused_uploads is False
    assert launch.reused_labels == []
    assert launch.reuploaded_labels == ["audio"]
    assert launch.reuse_reason == "Re-uploaded from original local paths: audio."
    assert launch.recovery_warning is None
    assert client.calls[-1][0] == "create_infer_with_lyrics_job"
    assert client.upload_calls[0][0] == str(local_audio)
    assert client.upload_calls[0][1].startswith("audio/")
    assert launch.payload["audio_path"] == "/uploaded/demo.wav"


def test_gui_controller_retry_job_flow_warns_when_local_file_fingerprint_changed(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    missing_audio = tmp_path / "missing_demo.wav"
    local_audio = tmp_path / "demo.wav"
    local_audio.write_bytes(b"before")
    original_fingerprint = _fingerprint(local_audio)
    local_audio.write_bytes(b"after-after")
    entry = GuiRunHistoryEntry(
        job_id="job-open-000",
        mode="infer-open",
        dialect="cantonese",
        status="completed",
        created_at="2026-03-11T09:00:00+00:00",
        updated_at="2026-03-11T09:01:00+00:00",
        state=GuiFormState(audio_path=str(local_audio), dialect="cantonese"),
        payload={
            "audio_path": str(missing_audio),
            "dialect": "cantonese",
            "export_format": "xsampa",
            "engine_inventory": None,
            "note_context": None,
            "note_context_format": "auto",
            "engine_profile": None,
        },
        uploads=[
            GuiUploadedFile(
                label="audio",
                local_path=str(local_audio),
                stored_path=str(missing_audio),
                upload_id="upload-1",
                original_filename="demo.wav",
                content_type="audio/wav",
                size_bytes=6,
                fingerprint=original_fingerprint,
            )
        ],
    )

    launch = controller.retry_job_flow(entry)

    assert launch.reuploaded_labels == ["audio"]
    assert launch.recovery_warning is not None
    assert "fingerprint changed" in launch.recovery_warning
    assert "previous=size=6" in launch.recovery_warning


def test_gui_controller_retry_job_flow_fails_when_upload_and_local_path_are_missing(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    missing_audio = tmp_path / "missing_demo.wav"
    missing_local_audio = tmp_path / "missing_local_demo.wav"
    entry = GuiRunHistoryEntry(
        job_id="job-open-000",
        mode="infer-open",
        dialect="cantonese",
        status="failed",
        created_at="2026-03-11T09:00:00+00:00",
        updated_at="2026-03-11T09:01:00+00:00",
        state=GuiFormState(audio_path=str(missing_local_audio), dialect="cantonese"),
        payload={
            "audio_path": str(missing_audio),
            "dialect": "cantonese",
            "export_format": "xsampa",
            "engine_inventory": None,
            "note_context": None,
            "note_context_format": "auto",
            "engine_profile": None,
        },
        uploads=[
            GuiUploadedFile(
                label="audio",
                local_path=str(missing_local_audio),
                stored_path=str(missing_audio),
                upload_id="upload-1",
                original_filename="demo.wav",
                content_type="audio/wav",
                size_bytes=0,
                fingerprint=GuiFileFingerprint(
                    size_bytes=0,
                    modified_time_ns=0,
                    modified_at="2026-03-11T09:00:00+00:00",
                    sha1_prefix="da39a3ee5e6b",
                ),
            )
        ],
        error="Audio file not found",
    )

    with pytest.raises(ValueError, match="uploaded audio path is unavailable"):
        controller.retry_job_flow(entry)


def test_gui_result_summary_formats_structured_output() -> None:
    summary = format_result_summary(
        {
            "mode": "infer-open",
            "dialect": "cantonese",
            "audio_path": "samples/demo.wav",
            "warnings": [],
            "segments": [{"normalized_ipa": "a", "xsampa": "a", "metadata": {"selected_alias": "a ng"}}],
        }
    )

    assert "Mode: infer-open" in summary
    assert "selected_alias=a ng" in summary
    assert "Overview" in summary
    assert "Decision Snapshot" in summary


def test_gui_result_summary_formats_warning_and_provenance_snapshot() -> None:
    summary = format_result_summary(
        {
            "mode": "infer-with-lyrics",
            "dialect": "cantonese",
            "audio_path": "/uploaded/demo.wav",
            "warnings": ["inventory fallback used"],
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.2,
                    "normalized_ipa": "a",
                    "xsampa": "a",
                    "metadata": {
                        "selected_alias": "a ng",
                        "selection_reason": "exact alias hit",
                        "future_aligner_provenance_view": {
                            "aligned_note_id": "note-001",
                            "summary": "accepted=1,rejected=1",
                            "message": "Accepted 1 evidence item; rejected 1 competing item.",
                            "accepted_evidence": [],
                            "rejected_evidence": [],
                            "left_evidence": [],
                            "right_evidence": [],
                            "rejection_reasons": ["insufficient_order_support"],
                        },
                    },
                }
            ],
        }
    )

    assert "Warnings" in summary
    assert "- inventory fallback used" in summary
    assert "provenance=summary=accepted=1,rejected=1; rejections=insufficient_order_support" in summary
    assert "why_note=aligned_note_id=note-001" in summary
    assert "fallback=Fallback state not provided." in summary


def test_gui_segment_detail_formats_accepted_and_rejected_evidence() -> None:
    detail = format_segment_detail(
        {
            "start_sec": 0.0,
            "end_sec": 0.42,
            "normalized_ipa": "a",
            "xsampa": "a",
            "metadata": {
                "selected_alias": "a ng",
                "selection_reason": "exact inventory alias won after future_aligner tie-break",
                "candidate_aliases": ["a ng", "a"],
                "ranking_features": ["alias_type=vc", "note_context_applied=true"],
                "note_context_applied": True,
                "note_context_source": "score_hint",
                "selected_alias_metadata": {"color": "power"},
                "future_aligner_provenance_view": {
                    "aligned_note_id": "note-001",
                    "summary": "accepted=2,rejected=1,left=1,right=1,types=same_note_continuity+order_continuity",
                    "message": "Accepted 2 evidence items; rejected 1 competing or insufficient items.",
                    "left_evidence": [
                        {
                            "evidence_type": "same_note_continuity",
                            "source_segment_index": 0,
                            "source_note_id": "note-001",
                            "support_direction": "left",
                            "support_score": 0.88,
                            "accepted": True,
                            "rejection_reason": None,
                        }
                    ],
                    "right_evidence": [],
                    "accepted_evidence": [
                        {
                            "evidence_type": "same_note_continuity",
                            "source_segment_index": 0,
                            "source_note_id": "note-001",
                            "support_direction": "left",
                            "support_score": 0.88,
                            "accepted": True,
                            "rejection_reason": None,
                        }
                    ],
                    "rejected_evidence": [
                        {
                            "evidence_type": "fallback",
                            "source_segment_index": None,
                            "source_note_id": None,
                            "support_direction": "window",
                            "support_score": 0.0,
                            "accepted": False,
                            "rejection_reason": "insufficient_order_support",
                        }
                    ],
                    "rejection_reasons": ["insufficient_order_support"],
                }
            },
        },
        0,
    )

    assert "aligned_note_id: note-001" in detail
    assert "selection_reason: exact inventory alias won after future_aligner tie-break" in detail
    assert "evidence_counts: left=1 right=0 accepted=1 rejected=1" in detail
    assert "rejection_reasons: insufficient_order_support" in detail
    assert 'selected_alias_metadata: {"color": "power"}' in detail
    assert "accepted_evidence: same_note_continuity@left:note=note-001:score=0.880" in detail
    assert "Provenance" in detail
    assert "Alias Decision" in detail
    assert "Aligned Note" in detail
    assert "Competition / Ownership" in detail
    assert "Fallback / Provenance" in detail
    assert "alias_explanation: selected_alias=a ng" in detail
    assert "note_explanation: aligned_note_id=note-001" in detail
    assert "competition_explanation: basis=<not provided>" in detail
    assert "fallback_explanation: Fallback state not provided." in detail


def test_gui_segment_detail_displays_optional_ownership_and_aligner_v2_fields() -> None:
    detail = format_segment_detail(
        {
            "normalized_ipa": "a",
            "xsampa": "a",
            "metadata": {
                "selected_alias": "a ng",
                "ownership_resolution_basis": "part-aware note span",
                "ownership_conflict_candidates": ["note-001", "note-002"],
                "alignment_competition_summary": "note-001 beat note-002 on continuity",
                "aligner_v2_fallback_used": True,
                "aligner_v2_boundary_conflict_likely": True,
                "future_aligner_provenance_view": {
                    "aligned_note_id": "note-001",
                    "summary": "accepted=1,rejected=0",
                    "message": "Fallback kept note ownership conservative.",
                    "fallback_used": True,
                    "accepted_evidence": [],
                    "rejected_evidence": [],
                    "left_evidence": [],
                    "right_evidence": [],
                    "rejection_reasons": [],
                },
            },
        },
        0,
    )

    assert "ownership_resolution_basis: part-aware note span" in detail
    assert "ownership_conflict_candidates: note-001, note-002" in detail
    assert "alignment_competition_summary: note-001 beat note-002 on continuity" in detail
    assert "provenance_fallback_used: true" in detail
    assert "aligner_v2: aligner_v2_boundary_conflict_likely=True, aligner_v2_fallback_used=True" in detail
    assert "competition_explanation: basis=part-aware note span | candidates=note-001, note-002" in detail
    assert "fallback_explanation: Fallback was used." in detail


def test_gui_segment_detail_tolerates_missing_or_non_mapping_metadata() -> None:
    detail = format_segment_detail(
        {
            "normalized_ipa": "a",
            "xsampa": "a",
            "metadata": "unexpected-string",
        },
        0,
    )
    row = format_segment_row({"normalized_ipa": "a", "xsampa": "a", "metadata": "unexpected-string"}, 0)

    assert "selected_alias: <none>" in detail
    assert "note_context: applied=false" in detail
    assert "ownership_resolution_basis: <not provided>" in detail
    assert "ownership_conflict_candidates: <not provided>" in detail
    assert "alignment_competition_summary: <not provided>" in detail
    assert "aligner_v2: <not provided>" in detail
    assert "fallback_explanation: Fallback state not provided." in detail
    assert row == ("0", "a", "a", "<none>", "<none>", "No provenance metadata available.")


def test_gui_segment_detail_explains_fallback_without_aligned_note() -> None:
    detail = format_segment_detail(
        {
            "normalized_ipa": "a",
            "xsampa": "a",
            "metadata": {
                "selected_alias": "a",
                "aligner_v2_fallback_used": True,
                "aligner_v2_boundary_conflict_likely": True,
                "future_aligner_provenance_view": {
                    "aligned_note_id": None,
                    "summary": None,
                    "message": "Conservative fallback kept ownership unresolved.",
                    "fallback_used": True,
                    "accepted_evidence": [],
                    "rejected_evidence": [],
                    "left_evidence": [],
                    "right_evidence": [],
                    "rejection_reasons": ["boundary_conflict"],
                },
            },
        },
        0,
    )

    assert "aligned_note_id: <not provided>" in detail
    assert "note_explanation: No aligned note selected." in detail
    assert "fallback_explanation: Fallback was used." in detail
    assert "rejection_reasons: boundary_conflict" in detail


def test_history_row_and_detail_format_result_and_error_state() -> None:
    entry = GuiRunHistoryEntry(
        job_id="job-open-001",
        mode="infer-open",
        dialect="cantonese",
        status="failed",
        created_at="2026-03-11T09:00:00+00:00",
        updated_at="2026-03-11T09:03:00+00:00",
        state=GuiFormState(audio_path="demo.wav", dialect="cantonese", lyrics="laa ng"),
        payload={"audio_path": "/uploaded/demo.wav"},
        uploads=[],
        error="Audio file not found",
    )

    row = format_history_row(entry)
    detail = format_history_detail(entry)

    assert row[1:6] == ("single", "job-open-001", "infer-open", "cantonese", "failed")
    assert row[0] == "current"
    assert row[6] == "idle"
    assert row[7] == "fresh run"
    assert "error=Audio file not found" in row[9]
    assert "Error: Audio file not found" in detail
    assert "Recovery Label: fresh run" in detail


def test_gui_segment_row_and_evidence_rows_support_table_views() -> None:
    segment = {
        "normalized_ipa": "a",
        "xsampa": "a",
        "metadata": {
            "selected_alias": "a ng",
            "selection_reason": "exact inventory alias won after future_aligner tie-break",
            "future_aligner_provenance_view": {
                "aligned_note_id": "note-001",
                "summary": "accepted=2,rejected=1",
                "message": "Accepted 2 evidence items; rejected 1 competing or insufficient items.",
                "accepted_evidence": [
                    {
                        "evidence_type": "same_note_continuity",
                        "source_segment_index": 0,
                        "source_note_id": "note-001",
                        "support_direction": "left",
                        "support_score": 0.88,
                        "accepted": True,
                        "rejection_reason": None,
                    }
                ],
                "rejected_evidence": [
                    {
                        "evidence_type": "fallback",
                        "source_segment_index": None,
                        "source_note_id": None,
                        "support_direction": "window",
                        "support_score": 0.0,
                        "accepted": False,
                        "rejection_reason": "insufficient_order_support",
                    }
                ],
                "left_evidence": [],
                "right_evidence": [],
                "rejection_reasons": ["insufficient_order_support"],
            },
        },
    }

    row = format_segment_row(segment, 3)
    evidence_rows = format_evidence_rows(segment)

    assert row[:5] == ("3", "a", "a", "a ng", "note-001")
    assert "exact inventory alias won" in row[5]
    assert evidence_rows[0] == ("left", "same_note_continuity", "0.880", "0", "note-001", "accepted", "<none>")
    assert evidence_rows[1][-1] == "insufficient_order_support"


def test_gui_batch_item_detail_includes_completed_result_snapshot() -> None:
    detail = format_batch_item_detail(
        {
            "index": 0,
            "audio_filename": "a.wav",
            "audio_local_path": "C:/audio/a.wav",
            "status": "completed",
            "summary": "segments=1",
            "result": {
                "mode": "infer-open",
                "dialect": "cantonese",
                "warnings": ["metadata fallback used"],
                "segments": [
                    {
                        "normalized_ipa": "a",
                        "xsampa": "a",
                        "metadata": {"selected_alias": "a ng", "selection_reason": "exact alias hit"},
                    }
                ],
            },
        }
    )

    assert "Mode: infer-open" in detail
    assert "Warnings: 1" in detail
    assert "First Decision: exact alias hit" in detail


def test_gui_app_builds_minimal_shell(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        assert tuple(app.dialect_box["values"]) == ("cantonese", "taigi")
        assert "openutau_like" in app.export_box["values"]
        assert app.run_button.cget("text") == "Upload + Submit Job"
        assert app.history_tree.get_children() == ()
        assert app.stage_var.get() == "idle"
        app._render_result(client.infer_open({"audio_path": "samples/demo.wav", "dialect": "cantonese"}))
        detail = app.detail_text.get("1.0", tk.END)
        assert "selection_reason:" in detail
        assert "mode=infer-open" in app.result_overview_var.get()
        assert '"mode": "infer-open"' in app.result_json_text.get("1.0", tk.END)
        assert "No submitted request yet." in app.request_json_text.get("1.0", tk.END)
        assert app.segment_tree.item("0", "values")[3] == "a"
        evidence_values = app.evidence_tree.item("evidence-0", "values")
        assert evidence_values[0] == "left"
        assert app.rejection_list.get(0) == "insufficient_same_note_support"
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_result_panel_tolerates_missing_optional_algorithm_fields(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        app._render_result(
            {
                "mode": "infer-open",
                "dialect": "cantonese",
                "audio_path": "/uploaded/demo.wav",
                "warnings": ["legacy provenance only"],
                "segments": [
                    {
                        "normalized_ipa": "a",
                        "xsampa": "a",
                        "metadata": {
                            "selected_alias": "a ng",
                            "aligned_note_id": "note-legacy-001",
                            "future_aligner_provenance_summary": "accepted=0,rejected=0",
                        },
                    }
                ],
            }
        )

        assert "warnings=1" in app.result_overview_var.get()
        assert "selected_alias: a ng" in app.detail_text.get("1.0", tk.END)
        assert "aligned_note_id: note-legacy-001" in app.detail_text.get("1.0", tk.END)
        assert "provenance_summary: accepted=0,rejected=0" in app.detail_text.get("1.0", tk.END)
        assert "ownership_resolution_basis: <not provided>" in app.detail_text.get("1.0", tk.END)
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_stage_status_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )

        app._handle_worker_stage("uploading audio")
        assert app.stage_var.get() == "uploading audio"
        assert app.stage_listbox.get(app.stage_listbox.curselection()[0]) == "uploading audio"

        app._handle_worker_stage("uploading inventory")
        assert app.stage_var.get() == "uploading inventory"

        app._handle_worker_stage("uploading note context")
        assert app.stage_var.get() == "uploading note context"

        app._handle_worker_stage("submitting job")
        assert app.stage_var.get() == "submitting job"
        assert int(float(app.stage_progress["value"])) >= 4
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_status_display_covers_upload_submit_poll_and_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        monkeypatch.setattr(app, "_schedule_poll", lambda delay_ms=500: None)

        app._handle_worker_stage("uploading audio")
        assert app.stage_var.get() == "uploading audio"
        app._handle_worker_stage("submitting job")
        assert app.stage_var.get() == "submitting job"

        launch = GuiJobLaunch(
            mode="infer-open",
            payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
            uploads=[],
            job_id="job-open-001",
            job_status="queued",
            state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
            created_at="2026-03-11T09:00:00+00:00",
            current_stage="queued",
            stage_updated_at="2026-03-11T09:00:00+00:00",
        )
        app._handle_job_launch_success(launch)
        assert app.stage_var.get() == "queued"
        assert '"audio_path": "/uploaded/demo.wav"' in app.request_json_text.get("1.0", tk.END)

        app._handle_job_update({"job_id": "job-open-001", "status": "running", "started_at": "now"})
        assert app.stage_var.get() == "running"

        app._set_stage("fetching result", workflow_text="Fetching latest job status for job-open-001.")
        assert app.stage_var.get() == "fetching result"

        result = client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})
        app._handle_job_update({"job_id": "job-open-001", "status": "completed", "result": result})
        assert app.stage_var.get() == "completed"
        assert app.history_entries[0].current_stage == "completed"
        assert app.history_entries[0].stage_updated_at is not None
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_history_load_restores_submitted_request_payload(tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        entry = GuiRunHistoryEntry(
            job_id="job-open-req-001",
            mode="infer-open",
            dialect="cantonese",
            status="completed",
            created_at="2026-03-11T09:00:00+00:00",
            updated_at="2026-03-11T09:01:00+00:00",
            state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
            payload={
                "audio_path": "/uploaded/demo.wav",
                "dialect": "cantonese",
                "export_format": "xsampa",
            },
            uploads=[],
            result=client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"}),
        )
        app._store_history_entry(entry, select=True)
        app._load_selected_history()

        request_text = app.request_json_text.get("1.0", tk.END)
        assert '"audio_path": "/uploaded/demo.wav"' in request_text
        assert '"dialect": "cantonese"' in request_text
        assert '"export_format": "xsampa"' in request_text
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_batch_status_polling_and_list_display(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        monkeypatch.setattr(app, "_schedule_poll", lambda delay_ms=500: None)
        launch = GuiJobLaunch(
            kind="batch",
            mode="infer-open",
            payload={"items": []},
            uploads=[],
            job_id="batch-open-001",
            job_status="queued",
            state=GuiFormState(task_mode="batch", batch_audio_paths=["a.wav", "b.wav"], dialect="cantonese"),
            batch_items=[
                {"index": 0, "audio_filename": "a.wav", "audio_local_path": "a.wav", "status": "queued", "summary": "queued"},
                {"index": 1, "audio_filename": "b.wav", "audio_local_path": "b.wav", "status": "queued", "summary": "queued"},
            ],
            total_items=2,
            created_at="2026-03-11T09:10:00+00:00",
            current_stage="queued",
            stage_updated_at="2026-03-11T09:10:00+00:00",
        )

        app._handle_job_launch_success(launch)
        assert app.stage_var.get() == "queued"
        assert len(app.batch_items_tree.get_children()) == 2

        app._handle_job_update(
            {
                "job_id": "batch-open-001",
                "kind": "batch",
                "status": "running",
                "total_items": 2,
                "completed_items": 1,
                "success_count": 1,
                "failure_count": 0,
                "items": [
                    {"index": 0, "status": "completed", "result": client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})},
                    {"index": 1, "status": "running"},
                ],
            }
        )
        assert app.stage_var.get() == "running"
        assert app.batch_items_tree.item("batch-item-0", "values")[2] == "completed"

        app._handle_job_update(client.get_job("batch-open-001"))
        assert app.stage_var.get() == "completed"
        assert app.batch_items_tree.item("batch-item-1", "values")[2] == "failed"
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_selecting_batch_item_shows_result_detail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        result = client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})
        app.current_batch_items = [
            {"index": 0, "audio_filename": "a.wav", "audio_local_path": "a.wav", "status": "completed", "result": result, "summary": "segments=1"},
            {"index": 1, "audio_filename": "b.wav", "audio_local_path": "b.wav", "status": "failed", "error": "Audio file not found", "summary": "error=Audio file not found"},
        ]
        app._render_batch_items(app.current_batch_items)

        app.batch_items_tree.selection_set("batch-item-0")
        app._on_batch_item_selected()

        assert "Mode: infer-open" in app.result_text.get("1.0", tk.END)
        assert app.segment_tree.item("0", "values")[1] == "a"
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_batch_partial_success_and_failure_are_browseable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        job_payload = client.get_job("batch-open-001")
        launch = GuiJobLaunch(
            kind="batch",
            mode="infer-open",
            payload={"items": []},
            uploads=[],
            job_id="batch-open-001",
            job_status="queued",
            state=GuiFormState(task_mode="batch", batch_audio_paths=["a.wav", "b.wav"], dialect="cantonese"),
            batch_items=[
                {"index": 0, "audio_filename": "a.wav", "audio_local_path": "a.wav", "status": "queued", "summary": "queued"},
                {"index": 1, "audio_filename": "b.wav", "audio_local_path": "b.wav", "status": "queued", "summary": "queued"},
            ],
            total_items=2,
        )
        app._handle_job_launch_success(launch)
        app._handle_job_update(job_payload)

        assert app.history_entries[0].kind == "batch"
        assert app.history_entries[0].success_count == 1
        assert app.history_entries[0].failure_count == 1
        history_row = app.history_tree.item("batch-open-001", "values")
        assert history_row[1] == "batch"
        assert history_row[7] == "batch run"
        assert "batch=2 success=1 failure=1" in history_row[9]

        app.batch_items_tree.selection_set("batch-item-1")
        app._on_batch_item_selected()
        assert "Audio file not found" in app.result_text.get("1.0", tk.END)
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_completed_job_is_recorded_in_recent_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        monkeypatch.setattr(app, "_schedule_poll", lambda delay_ms=500: None)
        app._handle_job_launch_success(
            GuiJobLaunch(
                mode="infer-open",
                payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
                uploads=[],
                job_id="job-open-001",
                job_status="queued",
                state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
                created_at="2026-03-11T09:00:00+00:00",
            )
        )
        result = client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})
        app._handle_job_update(
            {
                "job_id": "job-open-001",
                "status": "completed",
                "created_at": "2026-03-11T09:00:00+00:00",
                "completed_at": "2026-03-11T09:01:00+00:00",
                "result": result,
            }
        )

        row = app.history_tree.item("job-open-001", "values")
        assert row[0] == "current"
        assert row[1] == "single"
        assert row[2] == "job-open-001"
        assert row[5] == "completed"
        assert row[6] == "completed"
        assert row[7] == "fresh run"
        assert "segments=1" in row[9]
        assert len(app.history_entries) == 1
        assert app.history_entries[0].result is not None
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_failed_job_is_recorded_in_recent_runs_with_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    errors: list[str] = []
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda _title, message: errors.append(message))
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        monkeypatch.setattr(app, "_schedule_poll", lambda delay_ms=500: None)
        app._handle_job_launch_success(
            GuiJobLaunch(
                mode="infer-open",
                payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
                uploads=[],
                job_id="job-open-001",
                job_status="queued",
                state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
                created_at="2026-03-11T09:00:00+00:00",
            )
        )
        app._handle_job_update(
            {
                "job_id": "job-open-001",
                "status": "failed",
                "created_at": "2026-03-11T09:00:00+00:00",
                "completed_at": "2026-03-11T09:02:00+00:00",
                "error": "Audio file not found",
            }
        )

        row = app.history_tree.item("job-open-001", "values")
        assert row[5] == "failed"
        assert row[6] == "failed"
        assert row[7] == "fresh run"
        assert "error=Audio file not found" in row[9]
        assert app.history_entries[0].error == "Audio file not found"
        assert app.stage_var.get() == "failed"
        assert app.error_var.get() == "Audio file not found"
        assert errors == ["Audio file not found"]
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_retry_selected_history_reuses_previous_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        calls: list[GuiRunHistoryEntry] = []
        monkeypatch.setattr(app, "_begin_retry_flow", lambda entry: calls.append(entry))
        history_entry = GuiRunHistoryEntry(
            job_id="job-open-001",
            mode="infer-with-lyrics",
            dialect="cantonese",
            status="completed",
            created_at="2026-03-11T09:00:00+00:00",
            updated_at="2026-03-11T09:03:00+00:00",
            state=GuiFormState(
                audio_path="C:/local/demo.wav",
                dialect="cantonese",
                lyrics="laa ng",
                export_format="openutau_like",
                engine_profile="generic_openutau",
                engine_inventory="C:/local/inventory.json",
                note_context="C:/local/score.ustx",
                note_context_format="ustx",
            ),
            payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
            uploads=[],
            result=client.infer_with_lyrics({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"}),
        )
        app._store_history_entry(history_entry, select=True)

        app._retry_selected_history()

        assert calls and calls[0].job_id == "job-open-001"
        assert app.audio_var.get() == "C:/local/demo.wav"
        assert app.lyrics_var.get() == "laa ng"
        assert app.note_context_var.get() == "C:/local/score.ustx"
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_retry_launch_shows_recovery_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        monkeypatch.setattr(app, "_schedule_poll", lambda delay_ms=500: None)
        launch = GuiJobLaunch(
            mode="infer-open",
            payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
            uploads=[],
            job_id="job-open-002",
            job_status="queued",
            state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
            created_at="2026-03-11T09:10:00+00:00",
            retried_from_job_id="job-open-001",
            reuse_reason="Re-uploaded from original local paths: audio.",
            recovery_warning="Local audio file still exists, but its fingerprint changed before retry re-upload.",
            current_stage="queued",
            stage_updated_at="2026-03-11T09:10:00+00:00",
        )

        app._handle_job_launch_success(launch)

        assert "fingerprint changed" in app.recovery_note_var.get()
        assert app.stage_var.get() == "queued"
        assert app.history_entries[0].recovery_warning is not None
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_loading_selected_history_rehydrates_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        result = client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})
        history_entry = GuiRunHistoryEntry(
            job_id="job-open-001",
            mode="infer-open",
            dialect="cantonese",
            status="completed",
            created_at="2026-03-11T09:00:00+00:00",
            updated_at="2026-03-11T09:03:00+00:00",
            state=GuiFormState(audio_path="C:/local/demo.wav", dialect="cantonese"),
            payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
            uploads=[],
            result=result,
        )
        app._store_history_entry(history_entry, select=True)

        app._load_selected_history()

        assert "Mode: infer-open" in app.result_text.get("1.0", tk.END)
        assert "Browse Hint:" in app.history_detail_text.get("1.0", tk.END)
        assert '"mode": "infer-open"' in app.result_json_text.get("1.0", tk.END)
        assert app.segment_tree.item("0", "values")[1] == "a"
        assert "Loaded job-open-001" in app.history_status_var.get()
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_running_job_status_updates_without_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    scheduled: list[int] = []
    try:
        root.withdraw()
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        monkeypatch.setattr(app, "_schedule_poll", lambda delay_ms=500: scheduled.append(delay_ms))
        launch = GuiJobLaunch(
            mode="infer-open",
            payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
            uploads=[],
            job_id="job-open-001",
            job_status="queued",
            state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
            created_at="2026-03-11T09:00:00+00:00",
        )
        app._handle_job_launch_success(launch)
        app._handle_job_update({"job_id": "job-open-001", "status": "running", "started_at": "now"})

        assert app.job_id_var.get() == "job-open-001"
        assert app.job_status_var.get() == "running"
        assert scheduled[-1] == 500
        assert "Job in progress." in app.result_text.get("1.0", tk.END)
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_completed_job_renders_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        monkeypatch.setattr(app, "_schedule_poll", lambda delay_ms=500: None)
        app._set_busy(True)
        app._handle_job_launch_success(
            GuiJobLaunch(
                mode="infer-open",
                payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
                uploads=[],
                job_id="job-open-001",
                job_status="queued",
                state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
                created_at="2026-03-11T09:00:00+00:00",
            )
        )
        result = client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})
        app._handle_job_update({"job_id": "job-open-001", "status": "completed", "result": result})

        assert app.job_status_var.get() == "completed"
        assert app.stage_var.get() == "completed"
        assert app.error_var.get() == "<none>"
        assert "Mode: infer-open" in app.result_text.get("1.0", tk.END)
        assert '"segments": [' in app.result_json_text.get("1.0", tk.END)
        assert "alias a" in app.segment_overview_var.get()
        assert "Rejection reasons" in app.segment_status_var.get()
        assert app.segment_tree.item("0", "values")[3] == "a"
        assert "completed and result was loaded" in app.workflow_var.get()
        assert "disabled" not in app.run_button.state()
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_batch_item_preview_is_updated_when_browsing_batch_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        result = client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})
        app.current_batch_items = [
            {"index": 0, "audio_filename": "a.wav", "audio_local_path": "a.wav", "status": "completed", "result": result, "summary": "segments=1"},
            {"index": 1, "audio_filename": "b.wav", "audio_local_path": "b.wav", "status": "failed", "error": "Audio file not found", "summary": "error=Audio file not found"},
        ]
        app._render_batch_items(app.current_batch_items)

        app.batch_items_tree.selection_set("batch-item-1")
        app._on_batch_item_selected()

        assert "Batch Item [1]" in app.batch_detail_text.get("1.0", tk.END)
        assert "Audio file not found" in app.batch_detail_text.get("1.0", tk.END)
        assert "failed=1" in app.batch_summary_var.get()
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_batch_history_browsing_restores_batch_items_and_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(root, controller, api_base_url="http://127.0.0.1:9999", history_store=_history_store(tmp_path))
        batch_entry = GuiRunHistoryEntry(
            job_id="batch-open-001",
            kind="batch",
            mode="infer-open",
            dialect="cantonese",
            status="completed",
            created_at="2026-03-11T09:10:00+00:00",
            updated_at="2026-03-11T09:12:00+00:00",
            state=GuiFormState(task_mode="batch", batch_audio_paths=["a.wav", "b.wav"], dialect="cantonese"),
            payload={"items": []},
            uploads=[],
            batch_items=[
                {"index": 0, "audio_filename": "a.wav", "audio_local_path": "a.wav", "status": "completed", "result": client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})},
                {"index": 1, "audio_filename": "b.wav", "audio_local_path": "b.wav", "status": "failed", "error": "Audio file not found"},
            ],
            total_items=2,
            success_count=1,
            failure_count=1,
        )
        app._store_history_entry(batch_entry, select=True)

        app._load_selected_history()

        assert len(app.batch_items_tree.get_children()) == 2
        assert "Batch Items:" in app.history_detail_text.get("1.0", tk.END)
        assert "Latest Failed Batch Item:" in app.history_detail_text.get("1.0", tk.END)
        assert "Batch Item [0]" in app.batch_detail_text.get("1.0", tk.END)
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_jump_to_latest_failure_opens_failed_batch_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(root, controller, api_base_url="http://127.0.0.1:9999", history_store=_history_store(tmp_path))
        app._store_history_entry(
            GuiRunHistoryEntry(
                job_id="job-open-001",
                mode="infer-open",
                dialect="cantonese",
                status="completed",
                created_at="2026-03-11T09:00:00+00:00",
                updated_at="2026-03-11T09:01:00+00:00",
                state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
                payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
                uploads=[],
                result=client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"}),
            )
        )
        app._store_history_entry(
            GuiRunHistoryEntry(
                job_id="batch-open-001",
                kind="batch",
                mode="infer-open",
                dialect="cantonese",
                status="completed",
                created_at="2026-03-11T09:10:00+00:00",
                updated_at="2026-03-11T09:12:00+00:00",
                state=GuiFormState(task_mode="batch", batch_audio_paths=["a.wav", "b.wav"], dialect="cantonese"),
                payload={"items": []},
                uploads=[],
                batch_items=[
                    {"index": 0, "audio_filename": "a.wav", "audio_local_path": "a.wav", "status": "completed", "result": client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})},
                    {"index": 1, "audio_filename": "b.wav", "audio_local_path": "b.wav", "status": "failed", "error": "Audio file not found"},
                ],
                total_items=2,
                success_count=1,
                failure_count=1,
            ),
            select=True,
        )

        app._jump_to_latest_failure()

        assert app.selected_history_job_id == "batch-open-001"
        assert app.batch_items_tree.selection() == ("batch-item-1",)
        assert "Audio file not found" in app.batch_detail_text.get("1.0", tk.END)
        assert "Opened latest failure" in app.history_status_var.get()
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_history_management_can_open_and_clear_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    store = GuiHistoryStore(tmp_path / "gui_recent_runs.json", max_entries=3)
    opened: list[str] = []
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.os.startfile", lambda path: opened.append(path), raising=False)
        app = Any2XSampaGuiApp(root, controller, api_base_url="http://127.0.0.1:9999", history_store=store)
        app._store_history_entry(
            GuiRunHistoryEntry(
                job_id="job-open-001",
                mode="infer-open",
                dialect="cantonese",
                status="completed",
                created_at="2026-03-11T09:00:00+00:00",
                updated_at="2026-03-11T09:01:00+00:00",
                state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
                payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
                uploads=[],
                result=client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"}),
            )
        )

        app._open_history_file()
        app._clear_history()

        assert opened and opened[0].endswith("gui_recent_runs.json")
        assert app.history_tree.get_children() == ()
        assert "retention cap of 3" in app.history_detail_text.get("1.0", tk.END)
        assert "Cleared 1 Recent Runs" in app.history_status_var.get()
        saved = json.loads(store.path.read_text(encoding="utf-8"))
        assert saved["entries"] == []
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_history_row_shows_recovery_labels_for_reused_reuploaded_and_changed_files() -> None:
    base_entry = dict(
        job_id="job-open-001",
        mode="infer-open",
        dialect="cantonese",
        status="completed",
        created_at="2026-03-11T09:00:00+00:00",
        updated_at="2026-03-11T09:01:00+00:00",
        state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
        payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
        uploads=[],
        result_summary="segments=1 lyrics=no warnings=0",
    )

    reused = format_history_row(GuiRunHistoryEntry(**{**base_entry, "inputs_reused": True}))
    reuploaded = format_history_row(
        GuiRunHistoryEntry(**{**base_entry, "job_id": "job-open-002", "reuse_reason": "Re-uploaded from original local paths: audio."})
    )
    changed = format_history_row(
        GuiRunHistoryEntry(
            **{
                **base_entry,
                "job_id": "job-open-003",
                "reuse_reason": "Re-uploaded from original local paths: audio.",
                "recovery_warning": "Local audio file still exists, but its fingerprint changed before retry re-upload.",
            }
        )
    )

    assert reused[7] == "reused upload"
    assert reuploaded[7] == "reuploaded local"
    assert changed[7] == "changed local file"


def test_gui_app_history_selection_updates_result_and_preview_linkage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(root, controller, api_base_url="http://127.0.0.1:9999", history_store=_history_store(tmp_path))
        completed = GuiRunHistoryEntry(
            job_id="job-open-001",
            mode="infer-open",
            dialect="cantonese",
            status="completed",
            created_at="2026-03-11T09:00:00+00:00",
            updated_at="2026-03-11T09:01:00+00:00",
            state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
            payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
            uploads=[],
            result=client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"}),
        )
        failed = GuiRunHistoryEntry(
            job_id="job-open-002",
            mode="infer-open",
            dialect="cantonese",
            status="failed",
            created_at="2026-03-11T09:02:00+00:00",
            updated_at="2026-03-11T09:03:00+00:00",
            state=GuiFormState(audio_path="broken.wav", dialect="cantonese"),
            payload={"audio_path": "/uploaded/broken.wav", "dialect": "cantonese"},
            uploads=[],
            error="Audio file not found",
        )
        app._store_history_entry(completed)
        app._store_history_entry(failed)

        app.history_tree.selection_set("job-open-001")
        app._on_history_selected()
        assert "Mode: infer-open" in app.result_text.get("1.0", tk.END)
        assert "Job ID: job-open-001" in app.history_detail_text.get("1.0", tk.END)

        app.history_tree.selection_set("job-open-002")
        app._on_history_selected()
        assert "Audio file not found" in app.result_text.get("1.0", tk.END)
        assert "Job ID: job-open-002" in app.history_detail_text.get("1.0", tk.END)
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_app_failed_job_shows_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    errors: list[str] = []
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda _title, msg: errors.append(msg))
        app = Any2XSampaGuiApp(
            root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=_history_store(tmp_path),
        )
        monkeypatch.setattr(app, "_schedule_poll", lambda delay_ms=500: None)
        app._set_busy(True)
        app._handle_job_launch_success(
            GuiJobLaunch(
                mode="infer-open",
                payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
                uploads=[],
                job_id="job-open-001",
                job_status="queued",
                state=GuiFormState(audio_path="demo.wav", dialect="cantonese"),
                created_at="2026-03-11T09:00:00+00:00",
            )
        )
        app._handle_job_update({"job_id": "job-open-001", "status": "failed", "error": "Audio file not found"})

        assert app.job_status_var.get() == "failed"
        assert app.stage_var.get() == "failed"
        assert app.error_var.get() == "Audio file not found"
        assert "Job failed." in app.result_text.get("1.0", tk.END)
        assert errors == ["Audio file not found"]
        assert "disabled" not in app.run_button.state()
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_history_store_writes_json_with_retry_metadata(tmp_path: Path) -> None:
    store_path = tmp_path / "gui_recent_runs.json"
    store = GuiHistoryStore(store_path)
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"RIFFdemo")
    entry = GuiRunHistoryEntry(
        job_id="job-open-002",
        mode="infer-open",
        dialect="cantonese",
        status="completed",
        created_at="2026-03-11T09:00:00+00:00",
        updated_at="2026-03-11T09:03:00+00:00",
        state=GuiFormState(audio_path="C:/local/demo.wav", dialect="cantonese"),
        payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
        uploads=[
            GuiUploadedFile(
                label="audio",
                local_path="C:/local/demo.wav",
                stored_path="/uploaded/demo.wav",
                upload_id="upload-1",
                original_filename="demo.wav",
                content_type="audio/wav",
                size_bytes=8,
                fingerprint=_fingerprint(audio_path),
            )
        ],
        retried_from_job_id="job-open-001",
        inputs_reused=True,
        reuse_reason="Reused uploaded inputs: audio.",
        result_summary="segments=1 lyrics=no warnings=0",
        error_summary=None,
        current_stage="completed",
        stage_updated_at="2026-03-11T09:03:00+00:00",
    )

    store.save([entry])
    saved = json.loads(store_path.read_text(encoding="utf-8"))

    assert saved["version"] == 1
    assert saved["entries"][0]["retried_from_job_id"] == "job-open-001"
    assert saved["entries"][0]["reuse_reason"] == "Reused uploaded inputs: audio."
    assert saved["entries"][0]["uploads"][0]["local_path"] == "C:/local/demo.wav"
    assert saved["entries"][0]["uploads"][0]["fingerprint"]["size_bytes"] == 8
    assert saved["entries"][0]["current_stage"] == "completed"


def test_gui_app_restart_restores_persisted_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    store = GuiHistoryStore(tmp_path / "gui_recent_runs.json")
    app: Any2XSampaGuiApp | None = None
    reopened_app: Any2XSampaGuiApp | None = None
    root: tk.Tk | None = None
    reopened_root: tk.Tk | None = None
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable in this environment: {exc}")
    try:
        root.withdraw()
        monkeypatch.setattr("any2xsampa.gui.app.messagebox.showerror", lambda *args, **kwargs: None)
        app = Any2XSampaGuiApp(root, controller, api_base_url="http://127.0.0.1:9999", history_store=store)
        result = client.infer_open({"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"})
        app._store_history_entry(
            GuiRunHistoryEntry(
                job_id="job-open-001",
                mode="infer-open",
                dialect="cantonese",
                status="completed",
                created_at="2026-03-11T09:00:00+00:00",
                updated_at="2026-03-11T09:03:00+00:00",
                state=GuiFormState(audio_path="C:/local/demo.wav", dialect="cantonese"),
                payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
                uploads=[],
                result=result,
            ),
            select=True,
        )
        app.close()
        app = None
        root = None

        reopened_root = tk.Tk()
        reopened_root.withdraw()
        reopened_app = Any2XSampaGuiApp(
            reopened_root,
            controller,
            api_base_url="http://127.0.0.1:9999",
            history_store=store,
        )

        assert len(reopened_app.history_entries) == 1
        assert reopened_app.history_entries[0].restored_from_persistence is True
        row = reopened_app.history_tree.item("job-open-001", "values")
        assert row[0] == "restored"
        assert row[6] == "idle" or row[6] == "completed"
        reopened_app._load_selected_history()
        assert "Mode: infer-open" in reopened_app.result_text.get("1.0", tk.END)
        assert "Restored 1 Recent Runs" in reopened_app.history_status_var.get() or "Loaded job-open-001" in reopened_app.history_status_var.get()
    finally:
        if reopened_app is not None:
            reopened_app.close()
        elif reopened_root is not None:
            reopened_root.destroy()
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()


def test_gui_history_store_restores_retry_metadata(tmp_path: Path) -> None:
    store_path = tmp_path / "gui_recent_runs.json"
    store = GuiHistoryStore(store_path)
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"RIFFdemo")
    store.save(
        [
            GuiRunHistoryEntry(
                job_id="job-open-002",
                mode="infer-open",
                dialect="cantonese",
                status="completed",
                created_at="2026-03-11T09:00:00+00:00",
                updated_at="2026-03-11T09:03:00+00:00",
                state=GuiFormState(audio_path="C:/local/demo.wav", dialect="cantonese"),
                payload={"audio_path": "/uploaded/demo.wav", "dialect": "cantonese"},
                uploads=[
                    GuiUploadedFile(
                        label="audio",
                        local_path="C:/local/demo.wav",
                        stored_path="/uploaded/demo.wav",
                        upload_id="upload-1",
                        original_filename="demo.wav",
                        content_type="audio/wav",
                        size_bytes=8,
                        fingerprint=_fingerprint(audio_path),
                    )
                ],
                retried_from_job_id="job-open-001",
                inputs_reused=False,
                reuse_reason="Re-uploaded from original local paths: audio.",
                result_summary="segments=1 lyrics=no warnings=0",
                current_stage="completed",
                stage_updated_at="2026-03-11T09:03:00+00:00",
            )
        ]
    )

    loaded = store.load()

    assert loaded[0].retried_from_job_id == "job-open-001"
    assert loaded[0].reuse_reason == "Re-uploaded from original local paths: audio."
    assert loaded[0].restored_from_persistence is True
    assert loaded[0].current_stage == "completed"
    assert loaded[0].uploads[0].fingerprint is not None
    assert loaded[0].uploads[0].fingerprint.size_bytes == 8


def test_default_gui_history_path_uses_local_appdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", "C:/Temp/LocalAppData")
    assert default_gui_history_path() == Path("C:/Temp/LocalAppData/any2xsampa/gui_recent_runs.json")


@pytest.mark.parametrize(
    ("note_context_path", "note_context_format"),
    [
        (_note_context_project_fixture(), "ustx"),
        (_note_context_ust_fixture(), "ust"),
    ],
)
def test_gui_controller_display_works_with_infer_with_lyrics_project_context(
    tmp_path: Path,
    note_context_path: Path,
    note_context_format: str,
) -> None:
    client = _FakeApiClient()
    controller = Any2XSampaGuiController(client)  # type: ignore[arg-type]
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    result = controller.submit(
        GuiFormState(
            audio_path=str(audio_path),
            dialect="cantonese",
            lyrics="laa ng",
            export_format="openutau_like",
            engine_profile="generic_openutau",
            engine_inventory=str(_sequence_hint_inventory_fixture()),
            note_context=str(note_context_path),
            note_context_format=note_context_format,
        )
    )
    detail = format_segment_detail(result["segments"][1], 1)

    assert client.calls[-1][0] == "infer_with_lyrics"
    assert client.calls[-1][1]["note_context"] == str(note_context_path)
    assert client.calls[-1][1]["note_context_format"] == note_context_format
    assert "aligned_note_id:" in detail
    assert "provenance_summary:" in detail
    assert "decision_summary:" in detail

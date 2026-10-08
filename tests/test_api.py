from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from any2xsampa.api.app import ApiApplication
from any2xsampa.api.client import Any2XSampaApiClient, ApiClientError
from any2xsampa.api.storage import ApiJobStore, ApiUploadStore
from any2xsampa.api.server import start_server_in_thread


def _lyrics_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "lyrics_constrained_fixture.json"


def _behavior_inventory_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "openutau_bank_inventory_behavior.json"


def _sequence_hint_inventory_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "openutau_bank_inventory_sequence_hint.json"


def _note_context_long_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "note_context_open_long.json"


def _note_context_short_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "note_context_open_short.json"


@pytest.fixture()
def api_client() -> Any2XSampaApiClient:
    server = start_server_in_thread()
    client = Any2XSampaApiClient(server.base_url)
    try:
        yield client
    finally:
        server.close()


def test_api_health_endpoint(api_client: Any2XSampaApiClient) -> None:
    payload = api_client.health()

    assert payload["status"] == "ok"
    assert payload["service"] == "any2xsampa"


def test_api_status_advertises_only_runnable_recognizer_modes(
    api_client: Any2XSampaApiClient,
) -> None:
    payload = api_client.status()
    assert payload["supported_recognizer_modes"] == ["fixture", "real"]


def test_api_infer_open_endpoint(api_client: Any2XSampaApiClient, tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    payload = api_client.infer_open(
        {
            "audio_path": str(audio_path),
            "dialect": "cantonese",
            "export_format": "xsampa",
        }
    )

    assert payload["mode"] == "infer-open"
    assert payload["dialect"] == "cantonese"
    assert len(payload["segments"]) >= 1


def test_api_infer_with_lyrics_endpoint(api_client: Any2XSampaApiClient, tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    payload = api_client.infer_with_lyrics(
        {
            "audio_path": str(audio_path),
            "dialect": "cantonese",
            "lyrics": "laa ng",
            "export_format": "openutau_like",
            "recognizer_fixture": str(_lyrics_fixture()),
        }
    )

    assert payload["mode"] == "infer-with-lyrics"
    assert payload["lyrics"] == "laa ng"
    assert payload["segments"][0]["metadata"]["lyric_constraint_applied"] is True


def test_api_request_validation_returns_structured_error(
    api_client: Any2XSampaApiClient,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    with pytest.raises(ApiClientError, match="Unsupported note_context_format") as invalid_format:
        api_client.infer_open(
            {
                "audio_path": str(audio_path),
                "dialect": "cantonese",
                "note_context_format": "midi",
            }
        )
    assert invalid_format.value.status_code == 400
    assert invalid_format.value.error_code == "api.validation_error"
    assert invalid_format.value.error_kind == "validation_error"
    assert invalid_format.value.details == {"transport": "api"}

    with pytest.raises(ApiClientError, match="lyrics is required for infer-with-lyrics"):
        api_client.infer_with_lyrics(
            {
                "audio_path": str(audio_path),
                "dialect": "cantonese",
            }
        )


def test_api_validation_error_payload_includes_structured_error_fields(tmp_path: Path) -> None:
    app = ApiApplication()
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    status_code, payload = app.handle(
        "POST",
        "/infer-open",
        body=json.dumps(
            {
                "audio_path": str(audio_path),
                "dialect": "cantonese",
                "note_context_format": "midi",
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    assert status_code == 400
    assert payload["error"] == "Unsupported note_context_format 'midi'. Available formats: auto, json, ustx, ust"
    assert payload["error_code"] == "api.validation_error"
    assert payload["error_kind"] == "validation_error"
    assert payload["status_code"] == 400


def test_api_upload_endpoint_returns_stored_path(api_client: Any2XSampaApiClient, tmp_path: Path) -> None:
    audio_path = tmp_path / "upload-demo.wav"
    audio_path.write_bytes(b"RIFFdemo")

    payload = api_client.upload_file(audio_path, content_type="audio/wav")

    assert payload["original_filename"] == "upload-demo.wav"
    assert payload["content_type"] == "audio/wav"
    assert Path(payload["stored_path"]).exists()
    assert Path(payload["stored_path"]).read_bytes() == b"RIFFdemo"


def test_api_upload_get_and_list_show_file_status(tmp_path: Path) -> None:
    audio_path = tmp_path / "upload-list.wav"
    audio_path.write_bytes(b"RIFFlist")
    server = start_server_in_thread(state_dir=tmp_path / "api-state")
    client = Any2XSampaApiClient(server.base_url)
    try:
        created = client.upload_file(audio_path, content_type="audio/wav")
        listed = client.list_uploads()
        fetched = client.get_upload(created["upload_id"])

        assert listed["uploads"][0]["upload_id"] == created["upload_id"]
        assert fetched["file_exists"] is True
        assert fetched["file_status"] == "present"

        Path(created["stored_path"]).unlink()

        missing = client.get_upload(created["upload_id"])
        status = client.status()
        assert missing["file_exists"] is False
        assert missing["file_status"] == "missing"
        assert created["upload_id"] in status["storage"]["uploads"]["missing_upload_ids"]
        assert status["storage"]["uploads"]["consistency_status"] == "degraded"
    finally:
        server.close()


def test_api_status_reports_local_persistence(tmp_path: Path) -> None:
    server = start_server_in_thread(state_dir=tmp_path / "api-state")
    client = Any2XSampaApiClient(server.base_url)
    try:
        payload = client.status()

        assert payload["storage"]["persistence_enabled"] is True
        assert "cleanup" in payload["api_features"]
        assert "cleanup" in payload["capabilities"]["api_features"]
        assert payload["supported_export_formats"] == payload["capabilities"]["supported_export_formats"]
        assert payload["supported_note_context_formats"] == payload["capabilities"]["supported_note_context_formats"]
        assert payload["supported_recognizer_modes"] == payload["capabilities"]["supported_recognizer_modes"]
        assert payload["storage"]["uploads"]["backend"] == "local_json"
        assert payload["storage"]["jobs"]["backend"] == "local_json"
        assert payload["storage"]["uploads"]["consistency_status"] == "ok"
        assert payload["storage"]["jobs"]["consistency_status"] == "ok"
    finally:
        server.close()


def test_api_status_keeps_flat_and_nested_capabilities_aligned() -> None:
    app = ApiApplication()

    status_code, payload = app.handle("GET", "/status")

    assert status_code == 200
    assert payload["supported_dialects"] == payload["capabilities"]["supported_dialects"]
    assert payload["supported_export_formats"] == payload["capabilities"]["supported_export_formats"]
    assert payload["supported_note_context_formats"] == payload["capabilities"]["supported_note_context_formats"]
    assert payload["supported_recognizer_modes"] == payload["capabilities"]["supported_recognizer_modes"]
    assert payload["api_features"] == payload["capabilities"]["api_features"]


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "OPTIONS"])
def test_api_server_returns_json_for_unsupported_methods(method: str) -> None:
    server = start_server_in_thread()
    try:
        request = Request(server.base_url + "/status", method=method)
        with pytest.raises(HTTPError) as captured:
            urlopen(request, timeout=2.0)
        payload = json.loads(captured.value.read().decode("utf-8"))
        assert captured.value.code == 404
        assert captured.value.headers.get_content_type() == "application/json"
        assert payload["error_code"] == "api.unknown_endpoint"
    finally:
        server.close()


def test_api_head_suppresses_body_but_preserves_json_length() -> None:
    server = start_server_in_thread()
    try:
        with urlopen(Request(server.base_url + "/status", method="HEAD"), timeout=2.0) as response:
            assert response.status == 200
            assert response.read() == b""
            assert int(response.headers["Content-Length"]) > 0
            assert response.headers.get_content_type() == "application/json"
    finally:
        server.close()


@pytest.mark.parametrize("value", [True, "NaN", "Infinity", "-Infinity"])
def test_api_cleanup_rejects_boolean_and_nonfinite_ages(value: object) -> None:
    app = ApiApplication()
    status, payload = app.handle(
        "POST",
        "/cleanup",
        json.dumps({"max_age_seconds": value}).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    assert status == 400
    assert payload["error_code"] == "api.validation_error"


def test_api_accepts_case_insensitive_multipart_media_type(tmp_path: Path) -> None:
    app = ApiApplication(storage_root=tmp_path / "state")
    boundary = "Any2XsampaBoundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("ascii") + b"RIFFfixture\r\n" + f"--{boundary}--\r\n".encode("ascii")
    status, payload = app.handle(
        "POST",
        "/uploads",
        body,
        headers={"content-type": f"Multipart/Form-Data; boundary={boundary}"},
    )
    assert status == 201
    assert payload["original_filename"] == "sample.wav"
    assert Path(payload["stored_path"]).read_bytes() == b"RIFFfixture"


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        ("/uploads/", "Unknown endpoint: GET /uploads/"),
        ("/jobs/", "Unknown endpoint: GET /jobs/"),
    ],
)
def test_api_blank_detail_routes_emit_structured_not_found_errors(
    path: str,
    expected_error: str,
) -> None:
    app = ApiApplication()

    status_code, payload = app.handle("GET", path)

    assert status_code == 404
    assert payload["error"] == expected_error
    assert payload["error_code"] == "api.unknown_endpoint"
    assert payload["error_kind"] == "not_found"
    assert payload["status_code"] == 404
    assert payload["details"] == {"transport": "api"}


def test_api_job_status_endpoint_tracks_async_inference(
    api_client: Any2XSampaApiClient,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "job-demo.wav"
    audio_path.write_bytes(b"")
    upload = api_client.upload_file(audio_path, content_type="audio/wav")

    submitted = api_client.create_infer_open_job(
        {
            "audio_path": upload["stored_path"],
            "dialect": "cantonese",
            "export_format": "xsampa",
        }
    )
    job = _wait_for_job(api_client, submitted["job_id"])

    assert submitted["status"] in {"queued", "running"}
    assert job["status"] == "completed"
    assert job["kind"] == "single"
    assert job["result"]["mode"] == "infer-open"
    assert job["result"]["audio_path"] == upload["stored_path"]


def test_api_batch_endpoint_reports_per_item_success_and_failure(
    api_client: Any2XSampaApiClient,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "batch-demo.wav"
    audio_path.write_bytes(b"")
    upload = api_client.upload_file(audio_path, content_type="audio/wav")

    submitted = api_client.submit_infer_open_batch(
        [
            {
                "audio_path": upload["stored_path"],
                "dialect": "cantonese",
                "export_format": "xsampa",
            },
            {
                "audio_path": str(tmp_path / "missing.wav"),
                "dialect": "cantonese",
                "export_format": "xsampa",
            },
        ]
    )
    job = _wait_for_job(api_client, submitted["job_id"])

    assert job["kind"] == "batch"
    assert job["status"] == "completed"
    assert job["total_items"] == 2
    assert job["success_count"] == 1
    assert job["failure_count"] == 1
    assert job["items"][0]["status"] == "completed"
    assert job["items"][0]["result"]["mode"] == "infer-open"
    assert job["items"][1]["status"] == "failed"
    assert "Audio file not found" in job["items"][1]["error"]


def test_api_returns_provenance_view_in_infer_open(api_client: Any2XSampaApiClient, tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    payload = api_client.infer_open(
        {
            "audio_path": str(audio_path),
            "dialect": "cantonese",
            "export_format": "openutau_like",
            "engine_profile": "generic_openutau",
            "engine_inventory": str(_sequence_hint_inventory_fixture()),
            "engine_subbank": "main",
            "engine_color": "clear",
            "engine_style": "neutral",
            "engine_tags": ["lead"],
            "note_context": str(_note_context_long_fixture()),
        }
    )

    metadata = payload["segments"][1]["metadata"]
    view = metadata["future_aligner_provenance_view"]

    assert isinstance(metadata["future_aligner_provenance_items"], list)
    assert view["has_provenance"] is True
    assert view["aligned_note_id"] is not None
    assert isinstance(view["accepted_evidence"], list)
    assert isinstance(view["left_evidence"], list)
    assert isinstance(view["right_evidence"], list)
    assert view["summary"]


def test_api_returns_rejected_fallback_provenance_view(api_client: Any2XSampaApiClient, tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    payload = api_client.infer_open(
        {
            "audio_path": str(audio_path),
            "dialect": "cantonese",
            "export_format": "openutau_like",
            "engine_profile": "generic_openutau",
            "engine_inventory": str(_behavior_inventory_fixture()),
            "note_context": str(_note_context_short_fixture()),
        }
    )

    metadata = payload["segments"][3]["metadata"]
    view = metadata["future_aligner_provenance_view"]

    assert view["has_provenance"] is True
    assert view["fallback_used"] is True
    assert view["accepted_evidence"] == []
    assert len(view["rejected_evidence"]) >= 1
    assert "insufficient_same_note_support" in view["rejection_reasons"]
    assert "insufficient_order_support" in view["rejection_reasons"]


def test_api_upload_metadata_survives_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    source_path = tmp_path / "persisted-upload.wav"
    source_path.write_bytes(b"RIFFpersisted")

    first_server = start_server_in_thread(state_dir=state_dir)
    first_client = Any2XSampaApiClient(first_server.base_url)
    upload = first_client.upload_file(source_path, content_type="audio/wav")
    first_server.close()

    second_server = start_server_in_thread(state_dir=state_dir)
    second_client = Any2XSampaApiClient(second_server.base_url)
    try:
        status = second_client.status()
        assert status["upload_count"] == 1
        assert status["storage"]["uploads"]["record_count"] == 1
        assert Path(upload["stored_path"]).exists()
        assert Path(upload["stored_path"]).read_bytes() == b"RIFFpersisted"
    finally:
        second_server.close()


def test_api_completed_job_survives_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    audio_path = tmp_path / "job-persist.wav"
    audio_path.write_bytes(b"")

    first_server = start_server_in_thread(state_dir=state_dir)
    first_client = Any2XSampaApiClient(first_server.base_url)
    upload = first_client.upload_file(audio_path, content_type="audio/wav")
    submitted = first_client.create_infer_open_job(
        {
            "audio_path": upload["stored_path"],
            "dialect": "cantonese",
            "export_format": "xsampa",
        }
    )
    completed = _wait_for_job(first_client, submitted["job_id"])
    first_server.close()

    second_server = start_server_in_thread(state_dir=state_dir)
    second_client = Any2XSampaApiClient(second_server.base_url)
    try:
        recovered = second_client.get_job(submitted["job_id"])
        status = second_client.status()
        assert completed["status"] == "completed"
        assert recovered["status"] == "completed"
        assert recovered["result"]["audio_path"] == upload["stored_path"]
        assert recovered["recovered_from_persistence"] is True
        assert status["job_count"] == 1
        assert status["storage"]["jobs"]["record_count"] == 1
    finally:
        second_server.close()


def test_api_job_list_includes_completed_and_batch_records(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    audio_path = tmp_path / "job-list.wav"
    audio_path.write_bytes(b"")

    server = start_server_in_thread(state_dir=state_dir)
    client = Any2XSampaApiClient(server.base_url)
    try:
        upload = client.upload_file(audio_path, content_type="audio/wav")
        single = client.create_infer_open_job(
            {
                "audio_path": upload["stored_path"],
                "dialect": "cantonese",
                "export_format": "xsampa",
            }
        )
        batch = client.submit_infer_open_batch(
            [
                {
                    "audio_path": upload["stored_path"],
                    "dialect": "cantonese",
                    "export_format": "xsampa",
                }
            ]
        )
        _wait_for_job(client, single["job_id"])
        _wait_for_job(client, batch["job_id"])

        payload = client.list_jobs()
        job_ids = {item["job_id"] for item in payload["jobs"]}
        kinds = {item["job_id"]: item["kind"] for item in payload["jobs"]}

        assert single["job_id"] in job_ids
        assert batch["job_id"] in job_ids
        assert kinds[single["job_id"]] == "single"
        assert kinds[batch["job_id"]] == "batch"
    finally:
        server.close()


def test_api_failed_job_survives_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    missing_audio = tmp_path / "missing-job.wav"

    first_server = start_server_in_thread(state_dir=state_dir)
    first_client = Any2XSampaApiClient(first_server.base_url)
    submitted = first_client.create_infer_open_job(
        {
            "audio_path": str(missing_audio),
            "dialect": "cantonese",
            "export_format": "xsampa",
        }
    )
    failed = _wait_for_job(first_client, submitted["job_id"])
    first_server.close()

    second_server = start_server_in_thread(state_dir=state_dir)
    second_client = Any2XSampaApiClient(second_server.base_url)
    try:
        recovered = second_client.get_job(submitted["job_id"])
        assert failed["status"] == "failed"
        assert "Audio file not found" in failed["error"]
        assert recovered["status"] == "failed"
        assert "Audio file not found" in recovered["error"]
        assert recovered["recovered_from_persistence"] is True
    finally:
        second_server.close()


def test_api_completed_batch_survives_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    audio_path = tmp_path / "batch-persist.wav"
    audio_path.write_bytes(b"")

    first_server = start_server_in_thread(state_dir=state_dir)
    first_client = Any2XSampaApiClient(first_server.base_url)
    upload = first_client.upload_file(audio_path, content_type="audio/wav")
    submitted = first_client.submit_infer_open_batch(
        [
            {
                "audio_path": upload["stored_path"],
                "dialect": "cantonese",
                "export_format": "xsampa",
            },
            {
                "audio_path": str(tmp_path / "batch-missing.wav"),
                "dialect": "cantonese",
                "export_format": "xsampa",
            },
        ]
    )
    completed = _wait_for_job(first_client, submitted["job_id"])
    first_server.close()

    second_server = start_server_in_thread(state_dir=state_dir)
    second_client = Any2XSampaApiClient(second_server.base_url)
    try:
        recovered = second_client.get_job(submitted["job_id"])
        assert completed["status"] == "completed"
        assert completed["kind"] == "batch"
        assert recovered["status"] == "completed"
        assert recovered["kind"] == "batch"
        assert recovered["success_count"] == 1
        assert recovered["failure_count"] == 1
        assert recovered["items"][0]["status"] == "completed"
        assert recovered["items"][1]["status"] == "failed"
        assert "Audio file not found" in recovered["items"][1]["error"]
        assert recovered["recovered_from_persistence"] is True
    finally:
        second_server.close()


def test_api_upload_list_survives_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    source_path = tmp_path / "persisted-upload-list.wav"
    source_path.write_bytes(b"RIFFpersisted-list")

    first_server = start_server_in_thread(state_dir=state_dir)
    first_client = Any2XSampaApiClient(first_server.base_url)
    created = first_client.upload_file(source_path, content_type="audio/wav")
    first_server.close()

    second_server = start_server_in_thread(state_dir=state_dir)
    second_client = Any2XSampaApiClient(second_server.base_url)
    try:
        uploads = second_client.list_uploads()["uploads"]
        fetched = second_client.get_upload(created["upload_id"])
        assert uploads[0]["upload_id"] == created["upload_id"]
        assert uploads[0]["recovered_from_persistence"] is True
        assert fetched["recovered_from_persistence"] is True
        assert fetched["file_status"] == "present"
    finally:
        second_server.close()


def test_api_restart_marks_interrupted_batch_as_failed(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    jobs_dir = state_dir / "jobs" / "records"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = "batch_recover_me"
    jobs_dir.joinpath(f"{job_id}.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "kind": "batch",
                "mode": "infer-open",
                "status": "running",
                "created_at": "2026-03-11T00:00:00+00:00",
                "started_at": "2026-03-11T00:00:01+00:00",
                "completed_at": None,
                "total_items": 2,
                "completed_items": 1,
                "success_count": 1,
                "failure_count": 0,
                "items": [
                    {
                        "index": 0,
                        "status": "completed",
                        "result": {"mode": "infer-open", "audio_path": "persisted.wav"},
                    },
                    {"index": 1, "status": "running"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    server = start_server_in_thread(state_dir=state_dir)
    client = Any2XSampaApiClient(server.base_url)
    try:
        job = client.get_job(job_id)
        status = client.status()
        assert job["status"] == "failed"
        assert job["recovered_from_persistence"] is True
        assert "cannot resume automatically" in job["error"]
        assert job["success_count"] == 1
        assert job["failure_count"] == 1
        assert job["completed_items"] == 2
        assert job["items"][0]["status"] == "completed"
        assert job["items"][1]["status"] == "failed"
        assert "cannot resume automatically" in job["items"][1]["error"]
        assert status["storage"]["jobs"]["recovered_interrupted_jobs"] == 1
    finally:
        server.close()


def test_api_cleanup_removes_old_completed_jobs_and_uploads(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    audio_path = tmp_path / "cleanup-demo.wav"
    audio_path.write_bytes(b"")

    server = start_server_in_thread(state_dir=state_dir)
    client = Any2XSampaApiClient(server.base_url)
    try:
        upload = client.upload_file(audio_path, content_type="audio/wav")
        single = client.create_infer_open_job(
            {
                "audio_path": upload["stored_path"],
                "dialect": "cantonese",
                "export_format": "xsampa",
            }
        )
        batch = client.submit_infer_open_batch(
            [
                {
                    "audio_path": upload["stored_path"],
                    "dialect": "cantonese",
                    "export_format": "xsampa",
                },
                {
                    "audio_path": str(tmp_path / "cleanup-missing.wav"),
                    "dialect": "cantonese",
                    "export_format": "xsampa",
                },
            ]
        )
        _wait_for_job(client, single["job_id"])
        _wait_for_job(client, batch["job_id"])

        cleanup = client.cleanup(0)
        jobs_after = client.list_jobs()["jobs"]
        uploads_after = client.list_uploads()["uploads"]

        assert cleanup["jobs"]["removed_count"] == 2
        assert set(cleanup["jobs"]["removed_job_ids"]) == {single["job_id"], batch["job_id"]}
        assert cleanup["uploads"]["removed_count"] == 1
        assert cleanup["uploads"]["removed_upload_ids"] == [upload["upload_id"]]
        assert jobs_after == []
        assert uploads_after == []
        assert not Path(upload["stored_path"]).exists()
    finally:
        server.close()


def test_api_cleanup_does_not_remove_active_job_or_protected_upload(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    upload_store = ApiUploadStore(root_dir=state_dir / "uploads")
    job_store = ApiJobStore(root_dir=state_dir / "jobs")
    app = ApiApplication(upload_store=upload_store, job_store=job_store)
    server = start_server_in_thread(app=app)
    client = Any2XSampaApiClient(server.base_url)
    blocker = threading.Event()
    try:
        audio_path = tmp_path / "protected.wav"
        audio_path.write_bytes(b"")
        upload = client.upload_file(audio_path, content_type="audio/wav")

        def blocking_runner(_: object) -> dict[str, object]:
            blocker.wait(timeout=5.0)
            return {"mode": "infer-open", "audio_path": upload["stored_path"]}

        submitted = job_store.submit(
            "infer-open",
            SimpleNamespace(
                audio_path=upload["stored_path"],
                recognizer_fixture=None,
                engine_inventory=None,
                note_context=None,
            ),
            blocking_runner,
        )
        running_job = _wait_for_job_status(client, submitted["job_id"], expected={"running"})

        cleanup = client.cleanup(0)
        protected_upload = client.get_upload(upload["upload_id"])
        protected_job = client.get_job(submitted["job_id"])

        assert running_job["status"] == "running"
        assert cleanup["jobs"]["removed_count"] == 0
        assert cleanup["uploads"]["removed_count"] == 0
        assert cleanup["uploads"]["skipped_protected_upload_ids"] == [upload["upload_id"]]
        assert protected_job["status"] == "running"
        assert protected_upload["file_status"] == "present"
        assert Path(upload["stored_path"]).exists()
    finally:
        blocker.set()
        _wait_for_job(client, submitted["job_id"])
        server.close()


def test_api_submission_snapshot_is_queued_and_model_upload_is_protected(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    upload_store = ApiUploadStore(root_dir=state_dir / "uploads")
    job_store = ApiJobStore(root_dir=state_dir / "jobs")
    model_upload = upload_store.save("model.onnx", b"model", "application/octet-stream")
    blocker = threading.Event()

    def blocking_runner(_: object) -> dict[str, object]:
        blocker.wait(timeout=5.0)
        return {"mode": "infer-open"}

    request = SimpleNamespace(
        audio_path=None,
        recognizer_fixture=None,
        recognizer_model=model_upload["stored_path"],
        engine_inventory=None,
        note_context=None,
    )
    submitted = job_store.submit("infer-open", request, blocking_runner)
    try:
        assert submitted["status"] == "queued"
        cleanup = upload_store.cleanup(0, protected_paths=job_store.active_input_paths())
        assert cleanup["removed_count"] == 0
        assert cleanup["skipped_protected_upload_ids"] == [model_upload["upload_id"]]
    finally:
        blocker.set()


def test_api_cleanup_canonicalizes_active_real_output_paths(tmp_path: Path) -> None:
    state_dir = tmp_path / "api-state"
    upload_store = ApiUploadStore(root_dir=state_dir / "uploads")
    job_store = ApiJobStore(root_dir=state_dir / "jobs")
    upload = upload_store.save("posterior.json", b"{}", "application/json")
    stored = Path(upload["stored_path"])
    equivalent = stored.parent / "nested" / ".." / stored.name
    blocker = threading.Event()

    def blocking_runner(_: object) -> dict[str, object]:
        blocker.wait(timeout=5.0)
        return {"mode": "infer-open"}

    request = SimpleNamespace(
        audio_path=None,
        recognizer_fixture=None,
        recognizer_model=None,
        recognizer_real_output=str(equivalent),
        engine_inventory=None,
        note_context=None,
    )
    job_store.submit("infer-open", request, blocking_runner)
    try:
        cleanup = upload_store.cleanup(0, protected_paths=job_store.active_input_paths())
        assert cleanup["removed_count"] == 0
        assert cleanup["skipped_protected_upload_ids"] == [upload["upload_id"]]
        assert stored.is_file()
    finally:
        blocker.set()


def _wait_for_job(
    api_client: Any2XSampaApiClient,
    job_id: str,
    timeout_sec: float = 5.0,
) -> dict[str, object]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        payload = api_client.get_job(job_id)
        if payload["status"] not in {"queued", "running"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job {job_id}")


def _wait_for_job_status(
    api_client: Any2XSampaApiClient,
    job_id: str,
    expected: set[str],
    timeout_sec: float = 5.0,
) -> dict[str, object]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        payload = api_client.get_job(job_id)
        if payload["status"] in expected:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job {job_id} to reach one of {sorted(expected)}")

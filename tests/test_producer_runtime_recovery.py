from __future__ import annotations

import json
from pathlib import Path

from any2xsampa.adapters.recognizer_producer_runtime import (
    BUNDLE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    RecognizerProducerRuntimeAdapter,
    available_producer_backend_names,
    build_onnx_espeak_runtime_diagnostics_snapshot,
    create_posterior_json_producer,
    prepare_onnx_espeak_runtime_bundle,
    write_onnx_espeak_bundle_state,
)
from any2xsampa.adapters.recognizer_real import ExternalPosteriorJSONUniversalPhoneRecognizer
from any2xsampa.core.models import PhoneSegment


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_AUDIO = REPO_ROOT / "samples" / "demo.wav"


def test_recovered_runtime_facade_exposes_original_route_symbols() -> None:
    assert SCHEMA_VERSION == "external_posterior_json_v1"
    assert BUNDLE_SCHEMA_VERSION == "onnx_espeak_runtime_bundle_v1"
    assert {"onnx-espeak", "prototype-logit", "heuristic-dsp"}.issubset(
        set(available_producer_backend_names())
    )
    diagnostics = build_onnx_espeak_runtime_diagnostics_snapshot(
        {"runtime_status": "unavailable", "ready": False}
    )
    assert diagnostics["schema_version"] == "onnx_espeak_runtime_diagnostics_v1"


def test_recovered_diagnostic_producer_round_trips_through_phone_segments(tmp_path: Path) -> None:
    output_path = tmp_path / "posterior.json"
    payload = create_posterior_json_producer("prototype-logit").produce(
        DEMO_AUDIO,
        output_path=output_path,
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["producer"]["diagnostic_baseline"] is True
    assert len(payload["producer"]["input_audio"]["sha256"]) == 64
    assert output_path.is_file()
    segments = ExternalPosteriorJSONUniversalPhoneRecognizer(
        output_path,
        backend_name="posterior-json",
    ).recognize(DEMO_AUDIO)
    assert segments
    assert all(isinstance(segment, PhoneSegment) for segment in segments)
    assert segments[0].metadata["audio_binding"]["status"] == "verified"
    json.dumps([segment.to_dict() for segment in segments], ensure_ascii=False)


def test_recovered_bundle_adapter_writes_manifest_and_state(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    layout = prepare_onnx_espeak_runtime_bundle(runtime_root, bundle_name="recovered-test")
    state = write_onnx_espeak_bundle_state(
        runtime_root,
        runtime_diagnostics={
            "runtime_status": "unavailable",
            "ready": False,
            "model_resources": {"status": "missing"},
        },
    )

    manifest = json.loads(Path(layout["manifest_path"]).read_text(encoding="utf-8"))
    persisted = json.loads(Path(layout["status_file"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert manifest["bundle_name"] == "recovered-test"
    assert persisted["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert persisted["runtime_status"] == "unavailable"
    assert state["missing_components"] == persisted["missing_components"]


def test_runtime_adapter_accepts_simple_injected_diagnostics_callable() -> None:
    adapter = RecognizerProducerRuntimeAdapter(
        runtime_describer=lambda: {"runtime_status": "injected", "ready": False}
    )
    assert adapter.describe_runtime(prefetch=True)["runtime_status"] == "injected"


def test_producer_factory_forwards_cpu_thread_limit() -> None:
    producer = create_posterior_json_producer("prototype-logit", cpu_thread_limit=4)
    assert producer.cpu_thread_limit == 4


def test_producer_factory_forwards_chunking_options() -> None:
    producer = create_posterior_json_producer(
        "prototype-logit",
        chunk_duration_sec=10.0,
        chunk_overlap_sec=0.5,
    )
    assert producer.chunk_duration_sec == 10.0
    assert producer.chunk_overlap_sec == 0.5


def test_producer_note_context_is_explicitly_post_recognition_metadata(tmp_path: Path) -> None:
    note_context = tmp_path / "notes.json"
    note_context.write_text("{}", encoding="utf-8")
    payload = create_posterior_json_producer("prototype-logit").produce(
        DEMO_AUDIO,
        output_path=tmp_path / "posterior.json",
        note_context_path=note_context,
    )
    context = payload["producer"]
    assert context["note_context_role"] == "metadata_only_not_acoustic_input"
    assert context["note_context_applied_to_acoustic_decoder"] is False
    assert context["note_context_pending_consumer"] == "pipeline_note_context_applier"

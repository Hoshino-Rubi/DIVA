from pathlib import Path

import pytest

from any2xsampa.adapters.recognizer_acoustic_evidence import (
    apply_acoustic_evidence_metadata,
    build_segment_evidence_views,
    build_segment_summary_payload,
    candidate_entropy,
    candidate_logit_map,
    normalize_candidate_evidence,
)
from any2xsampa.adapters.recognizer_producer_projection import (
    frame_evidence_summary_from_posterior_payload,
    prediction_indices_for_reference_window,
    producer_runtime_contract_from_posterior_payload,
)
from any2xsampa.adapters.recognizer import (
    frame_evidence_summary_from_posterior_payload as recognizer_frame_evidence_summary_from_posterior_payload,
    producer_runtime_contract_from_posterior_payload as recognizer_producer_runtime_contract_from_posterior_payload,
)
from any2xsampa.adapters.recognizer_json_source import load_segments_from_json_source
from any2xsampa.adapters.recognizer_segment_builder import (
    build_segment_from_payload,
    build_segments_from_payload,
)
from any2xsampa.core.models import AcousticEvidence, AcousticEvidenceCandidate, PhoneCandidate


def _default_features(symbol: str) -> dict[str, str]:
    return {"symbol": symbol, "kind": "test"}


def _annotate(
    metadata: dict[str, object],
    candidates: list[PhoneCandidate],
    source_mode: str,
    start_sec: float,
    end_sec: float,
) -> dict[str, object]:
    payload = dict(metadata)
    payload["annotated"] = True
    payload["annotated_source_mode"] = source_mode
    payload["annotated_duration_sec"] = round(end_sec - start_sec, 3)
    payload["candidate_count"] = len(candidates)
    return payload


def _infer(metadata: dict[str, object]) -> list[PhoneCandidate]:
    raw_candidates = metadata.get("raw_candidates")
    if isinstance(raw_candidates, list) and raw_candidates:
        return [PhoneCandidate(symbol=str(raw_candidates[0]), confidence=0.5)]
    return []


def _probability_to_logit(value: float) -> float:
    clamped = max(1e-4, min(value, 1.0 - 1e-4))
    return __import__("math").log(clamped / (1.0 - clamped))


def test_build_segment_from_payload_populates_defaults_and_annotation() -> None:
    segment = build_segment_from_payload(
        {
            "start_sec": 0.0,
            "end_sec": 0.2,
            "candidate_phones": [{"symbol": "aa", "confidence": 0.8}],
            "metadata": {},
        },
        source_mode="fixture",
        index=0,
        infer_candidates_from_metadata=_infer,
        default_articulatory_features=_default_features,
        annotate_acoustic_evidence=_annotate,
    )

    assert segment.candidate_phones[0].symbol == "aa"
    assert segment.metadata["source_mode"] == "fixture"
    assert segment.metadata["raw_candidates"] == ["aa"]
    assert segment.metadata["articulatory_features"] == {"symbol": "aa", "kind": "test"}
    assert segment.metadata["annotated"] is True
    assert segment.metadata["annotated_duration_sec"] == 0.2


def test_build_segments_from_payload_can_infer_candidates_from_metadata() -> None:
    segments = build_segments_from_payload(
        {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.1,
                    "metadata": {"raw_candidates": ["l"]},
                }
            ]
        },
        source_mode="real",
        infer_candidates_from_metadata=_infer,
        default_articulatory_features=_default_features,
        annotate_acoustic_evidence=_annotate,
    )

    assert len(segments) == 1
    assert segments[0].candidate_phones[0].symbol == "l"
    assert segments[0].metadata["annotated_source_mode"] == "real"


def test_candidate_logit_map_aggregates_frame_logits() -> None:
    candidates = [PhoneCandidate(symbol="aa", confidence=0.9), PhoneCandidate(symbol="ax", confidence=0.5)]

    def _derive_logit_frames(
        _candidates: list[PhoneCandidate],
        _metadata: dict[str, object],
    ) -> tuple[list[dict[str, float]], str, float | None]:
        return [{"aa": 1.2, "ax": 0.4}, {"aa": 0.8, "ax": 0.6}], "logit_trace", 0.02

    payload = candidate_logit_map(
        candidates,
        {"unused": True},
        derive_logit_frames=_derive_logit_frames,
        probability_to_logit=_probability_to_logit,
    )

    assert payload == {"aa": 1.0, "ax": 0.5}


def test_normalize_candidate_evidence_populates_rank_margin_and_logit() -> None:
    payload = normalize_candidate_evidence(
        [PhoneCandidate(symbol="aa", confidence=0.9), PhoneCandidate(symbol="ax", confidence=0.6)],
        candidate_logits={"aa": 1.3, "ax": 0.2},
        default_articulatory_features=_default_features,
        probability_to_logit=_probability_to_logit,
    )

    assert payload[0].rank == 1
    assert payload[1].rank == 2
    assert payload[1].margin_from_top == pytest.approx(0.3)
    assert payload[0].articulatory_features == {"symbol": "aa", "kind": "test"}
    assert payload[0].logit == 1.3
    assert payload[1].logit == 0.2


def test_candidate_entropy_returns_normalized_value() -> None:
    high_confidence = candidate_entropy([PhoneCandidate("aa", 0.95), PhoneCandidate("ax", 0.05)])
    balanced = candidate_entropy([PhoneCandidate("aa", 0.5), PhoneCandidate("ax", 0.5)])

    assert 0.0 <= high_confidence <= 1.0
    assert 0.0 <= balanced <= 1.0
    assert high_confidence < balanced


def test_load_segments_from_json_source_applies_postprocess_and_source_override(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    fixture_path = tmp_path / "fixture.json"
    audio_path.write_bytes(b"")
    fixture_path.write_text(
        '{"segments":[{"start_sec":0.0,"end_sec":0.1,"candidate_phones":[{"symbol":"a","confidence":0.9}],"metadata":{}}]}',
        encoding="utf-8",
    )

    def _normalize(payload: dict[str, object], **_: object) -> dict[str, object]:
        return payload

    def _build(payload: dict[str, object], *, source_mode: str) -> list[object]:
        return build_segments_from_payload(
            payload,
            source_mode=source_mode,
            infer_candidates_from_metadata=_infer,
            default_articulatory_features=_default_features,
            annotate_acoustic_evidence=_annotate,
        )

    def _postprocess(segment) -> None:
        segment.metadata["postprocessed"] = True

    segments = load_segments_from_json_source(
        audio_path=audio_path,
        json_path=fixture_path,
        source_mode="fixture",
        backend_name="fixture_json",
        source_label=str(fixture_path),
        missing_path_label="Recognizer fixture file",
        lyrics="la",
        normalize_payload=_normalize,
        build_segments=_build,
        source_override="override_source",
        postprocess_segment=_postprocess,
    )

    assert len(segments) == 1
    assert segments[0].metadata["source"] == "override_source"
    assert segments[0].metadata["lyrics_guided"] is True
    assert segments[0].metadata["postprocessed"] is True


def test_load_segments_from_json_source_surfaces_missing_json_label(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="Recognizer fixture file not found"):
        load_segments_from_json_source(
            audio_path=audio_path,
            json_path=tmp_path / "missing.json",
            source_mode="fixture",
            backend_name="fixture_json",
            source_label="missing.json",
            missing_path_label="Recognizer fixture file",
            lyrics=None,
            normalize_payload=lambda payload, **kwargs: payload,
            build_segments=lambda payload, *, source_mode: [],
        )


def test_apply_acoustic_evidence_metadata_populates_expected_flat_views() -> None:
    metadata: dict[str, object] = {}
    evidence = AcousticEvidence(
        schema_version="acoustic_evidence_v1",
        source_mode="fixture",
        top_phone="aa",
        acoustic_confidence=0.91,
        timing_reliability=0.83,
        duration_sec=0.2,
        candidate_margin=0.08,
        candidate_count=2,
        candidate_entropy=0.42,
        frame_consistency=0.79,
        evidence_quality=0.81,
        segment_likelihood=0.88,
        dominant_candidate_stability=0.77,
        boundary_transition_strength=0.36,
        boundary_softness=0.22,
        ownership_ambiguity=0.15,
        candidate_phones=[
            AcousticEvidenceCandidate(symbol="aa", confidence=0.9, rank=1),
            AcousticEvidenceCandidate(symbol="ax", confidence=0.82, rank=2),
        ],
        boundary_uncertainty={"uncertain": False, "window_ms": 14.0},
        note_alignment={"same_note_continuation_bias": 0.9},
        ownership={"ownership_conflict_risk": 0.12},
        candidate_distribution={"runner_up_phone": "ax"},
        continuation_profile={"continuation_friendliness": 0.88},
        boundary_profile={"boundary_conflict_likelihood": 0.18},
        adapter_diagnostics={"backend_kind": "fixture_json"},
        posterior_summary={"available": True, "source": "posterior"},
        logit_summary={"available": True, "source": "posterior:derived_logits"},
        notes=["fixture evidence"],
    )
    candidates = [
        PhoneCandidate(symbol="aa", confidence=0.9),
        PhoneCandidate(symbol="ax", confidence=0.82),
    ]

    enriched = apply_acoustic_evidence_metadata(
        metadata,
        evidence=evidence,
        candidates=candidates,
    )

    assert enriched["acoustic_evidence_applied"] is True
    assert enriched["acoustic_evidence"]["schema_version"] == "acoustic_evidence_v1"
    assert enriched["acoustic_confidence"] == 0.91
    assert enriched["timing_reliability"] == 0.83
    assert enriched["candidate_margin"] == 0.08
    assert enriched["boundary_uncertainty"] == {"uncertain": False, "window_ms": 14.0}
    assert enriched["note_alignment_friendly"] == {"same_note_continuation_bias": 0.9}
    assert enriched["ownership_signal"] == {"ownership_conflict_risk": 0.12}
    assert enriched["posterior_summary"]["source"] == "posterior"
    assert enriched["logit_summary"]["source"] == "posterior:derived_logits"
    assert enriched["posterior"] == [
        {"symbol": "aa", "confidence": 0.9},
        {"symbol": "ax", "confidence": 0.82},
    ]


def test_segment_evidence_view_helpers_build_stable_payloads() -> None:
    metadata = {
        "posterior_trace": {"frames": [1]},
        "logit_trace": {"frames": [2]},
        "acoustic_confidence": 0.8,
        "timing_reliability": 0.7,
        "frame_consistency": 0.6,
        "evidence_quality": 0.5,
        "segment_likelihood": 0.4,
        "boundary_uncertainty": {"uncertain": True},
        "posterior_summary": {"source": "posterior"},
        "logit_summary": {"source": "logit"},
        "continuation_profile": {"continuation_friendliness": 0.9},
        "boundary_profile": {"boundary_conflict_likelihood": 0.2},
        "ownership_signal": {"ownership_conflict_risk": 0.3},
        "adapter_diagnostics": {"backend_kind": "fixture"},
    }

    summary = build_segment_summary_payload(metadata)
    views = build_segment_evidence_views(metadata)

    assert summary["acoustic_confidence"] == 0.8
    assert summary["posterior_summary"]["source"] == "posterior"
    assert views["posterior_trace"]["frames"] == [1]
    assert views["logit_trace"]["frames"] == [2]
    assert views["segment_summary"]["logit_summary"]["source"] == "logit"
    assert views["continuation_profile"]["continuation_friendliness"] == 0.9
    assert views["ownership"]["ownership_conflict_risk"] == 0.3
    assert views["adapter_diagnostics"]["backend_kind"] == "fixture"


def test_prediction_indices_for_reference_window_supports_center_inside_window_fallback() -> None:
    predictions = [
        {"start_sec": 0.00, "end_sec": 0.10},
        {"start_sec": 0.10, "end_sec": 0.20},
        {"start_sec": 0.20, "end_sec": 0.30},
    ]

    centered = prediction_indices_for_reference_window(
        predictions,
        start_sec=0.05,
        end_sec=0.25,
        selection_mode="center_inside_window",
    )
    fallback = prediction_indices_for_reference_window(
        predictions,
        start_sec=0.31,
        end_sec=0.33,
        selection_mode="center_inside_window",
    )

    assert centered == [0, 1, 2]
    assert fallback == [2]


def test_frame_evidence_summary_from_posterior_payload_reports_near_tie_and_runs() -> None:
    payload = {
        "producer": {"backend_kind": "onnx-espeak"},
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 0.2,
                "posterior_trace": {
                    "frames": [
                        {
                            "candidates": [
                                {"symbol": "l", "confidence": 0.240},
                                {"symbol": "a", "confidence": 0.239},
                            ]
                        },
                        {
                            "candidates": [
                                {"symbol": "a", "confidence": 0.320},
                                {"symbol": "s", "confidence": 0.110},
                            ]
                        },
                        {
                            "candidates": [
                                {"symbol": "s", "confidence": 0.289},
                                {"symbol": "a", "confidence": 0.288},
                            ]
                        },
                    ]
                },
                "metadata": {},
            }
        ],
    }

    summary = frame_evidence_summary_from_posterior_payload(
        payload,
        anchor_start_sec=0.05,
        anchor_end_sec=0.15,
    )

    assert summary["backend_kind"] == "onnx-espeak"
    assert summary["full_top_symbol_sequence"] == ["l", "a", "s"]
    assert summary["full_run_symbols"] == ["l", "a", "s"]
    assert summary["anchor_window_frame_count"] == 3
    assert summary["target_neighborhood_frame_count"] == 3
    assert summary["near_tie_audit"]["full"]["near_tie_present"] is True
    assert summary["near_tie_audit"]["full"]["structural_instability_candidate_count"] >= 2


def test_producer_runtime_contract_from_posterior_payload_tracks_projection_and_runtime_lineage() -> None:
    payload = {
        "producer": {
            "backend_kind": "onnx-espeak",
            "requested_backend_kind": "onnx-espeak",
            "audio_path": "tmp/demo.wav",
            "note_context_path": "tmp/demo.json",
            "dispatch": {
                "requested_backend_kind": "onnx-espeak",
                "resolved_backend_kind": "onnx-espeak",
                "auto_fallback_used": False,
            },
            "raw_projection_from_enriched_runtime_payload": True,
            "reference_segment_contract_selection_mode": "center_inside_window",
            "onnx_espeak_runtime": {
                "bundle_status": "bundle_missing_runtime",
                "runtime_status": "external_worker_fallback",
                "ready": True,
                "runtime_path": {
                    "repo_root": "C:/Users/Example User/Desktop/any2xsampa-runtime",
                    "selected_runtime_root": "C:/Users/Example User/Desktop/any2xsampa-runtime/onnx-espeak",
                    "selected_worker_entrypoint": "C:/Users/Example User/Desktop/any2xsampa-runtime/onnx-espeak/worker/posterior_json_producer.py",
                    "selected_command": ["py", "-3.11"],
                    "primary_bundle_status": "bundle_missing_runtime",
                    "primary_bundle_layout": {
                        "runtime_root": "C:/Users/Example User/Desktop/any2xsampa-runtime/onnx-espeak",
                        "worker_entrypoint": "C:/Users/Example User/Desktop/any2xsampa-runtime/onnx-espeak/worker/posterior_json_producer.py",
                        "manifest_path": "C:/Users/Example User/Desktop/any2xsampa-runtime/onnx-espeak/manifest.json",
                    },
                },
                "external_worker": {
                    "selected_source": "external_worker_fallback",
                    "selected_command": ["py", "-3.11"],
                    "selected_runtime_root": "C:/Users/Example User/Desktop/any2xsampa-runtime/onnx-espeak",
                    "selected_worker_entrypoint": "C:/Users/Example User/Desktop/any2xsampa-runtime/onnx-espeak/worker/posterior_json_producer.py",
                    "python_version": "3.11.9",
                    "dependencies": {
                        "numpy": {"available": True, "version": "1.26.4"},
                        "onnxruntime": {"available": True, "version": "1.24.3"},
                    },
                    "candidates": [
                        {
                            "ready": True,
                            "command": ["py", "-3.11"],
                            "python_version": "3.11.9",
                            "dependencies": {
                                "numpy": {"available": True, "version": "1.26.4"},
                                "onnxruntime": {"available": True, "version": "1.24.3"},
                            },
                        }
                    ],
                },
                "model_resources": {
                    "status": "ready",
                    "checked_via": "external_worker",
                    "checked_with_command": ["py", "-3.11"],
                    "local_files": [
                        {
                            "resource": "model",
                            "local_path": "C:/cache/models--demo/snapshots/abc123/model.onnx",
                        }
                    ],
                },
            },
        },
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 0.1,
                "posterior_trace": {"frames": [{"candidates": [{"symbol": "a", "confidence": 0.9}]}]},
                "metadata": {
                    "score_hint": {"normalized_note_id": "note-1"},
                    "producer_context": {"matched_note_id": "note-1"},
                    "producer_name": "onnx-espeak",
                    "producer_backend_kind": "onnx-espeak",
                    "producer_contract_layers": {"payload": "raw"},
                    "producer_generated_fields": ["posterior_trace"],
                    "producer_raw_model_output": {
                        "raw_model_output_kind": "trained_onnx_ctc_model",
                        "model_backend": "onnxruntime",
                        "model_repo": "demo/model",
                        "model_file": "onnx/model.onnx",
                        "vocab_repo": "demo/vocab",
                    },
                },
            }
        ],
    }

    contract = producer_runtime_contract_from_posterior_payload(payload)
    signature = contract["payload_signature"]

    assert contract["backend_kind"] == "onnx-espeak"
    assert contract["dispatch"]["resolved_backend_kind"] == "onnx-espeak"
    assert contract["onnx_runtime"]["runtime_status"] == "external_worker_fallback"
    assert contract["onnx_runtime"]["selected_runtime_root_within_repo"] is False
    assert contract["model_identity"]["model_backend"] == "onnxruntime"
    assert signature["serialized_payload_origin_kind"] == "projected_raw_from_enriched_runtime_payload"
    assert signature["serialized_generation_path_kind"] == "projected_from_enriched_runtime_generation_path"
    assert signature["serialized_runtime_lineage_kind"] == "repo_external_sibling_runtime_lineage"
    assert signature["serialized_environment_signature_kind"] == "external_worker_dependency_probe_environment_signature"
    assert signature["serialized_environment_dependency_probe_completeness_kind"] == "versioned_dependency_probe"
    assert signature["serialized_environment_resource_snapshot_ids"] == {"model": "abc123"}
    assert signature["serialized_score_hint_note_ids"] == ["note-1"]
    assert signature["serialized_producer_context_match_note_ids"] == ["note-1"]


def test_recognizer_public_projection_surfaces_match_helper_contracts() -> None:
    payload = {
        "producer": {
            "backend_kind": "onnx-espeak",
            "onnx_espeak_runtime": {
                "runtime_status": "external_worker_fallback",
                "runtime_path": {
                    "repo_root": "C:/Users/Example User/Desktop/any2xsampa/runtime",
                    "selected_runtime_root": "C:/Users/Example User/Desktop/any2xsampa/runtime/onnx-espeak",
                    "selected_worker_entrypoint": "C:/Users/Example User/Desktop/any2xsampa/runtime/onnx-espeak/worker/posterior_json_producer.py",
                },
                "external_worker": {
                    "selected_source": "external_worker_fallback",
                    "selected_command": ["py", "-3.11"],
                },
            },
        },
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 0.2,
                "posterior_trace": {
                    "frames": [
                        {"candidates": [{"symbol": "l", "confidence": 0.24}, {"symbol": "a", "confidence": 0.23}]},
                        {"candidates": [{"symbol": "a", "confidence": 0.31}, {"symbol": "s", "confidence": 0.11}]},
                    ]
                },
                "metadata": {
                    "score_hint": {"normalized_note_id": "note-1"},
                    "producer_context": {"matched_note_id": "note-1"},
                },
            }
        ],
    }

    helper_contract = producer_runtime_contract_from_posterior_payload(payload)
    recognizer_contract = recognizer_producer_runtime_contract_from_posterior_payload(payload)
    helper_frame = frame_evidence_summary_from_posterior_payload(payload, anchor_start_sec=0.0, anchor_end_sec=0.2)
    recognizer_frame = recognizer_frame_evidence_summary_from_posterior_payload(
        payload,
        anchor_start_sec=0.0,
        anchor_end_sec=0.2,
    )

    assert recognizer_contract["onnx_runtime"] == helper_contract["onnx_runtime"]
    assert recognizer_contract["payload_signature"] == helper_contract["payload_signature"]
    assert recognizer_frame["full_top_symbol_sequence"] == helper_frame["full_top_symbol_sequence"]
    assert recognizer_frame["anchor_window_run_symbols"] == helper_frame["anchor_window_run_symbols"]
    assert recognizer_frame["near_tie_audit"] == helper_frame["near_tie_audit"]

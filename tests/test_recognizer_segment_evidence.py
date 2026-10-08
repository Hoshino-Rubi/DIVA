from __future__ import annotations

import any2xsampa.adapters.recognizer as recognizer_module
from any2xsampa.adapters.recognizer_segment_evidence import (
    build_adapter_diagnostics,
    derive_logit_frames,
    derive_posterior_frames,
    frame_trace,
    infer_candidates_from_metadata,
    logit_summary,
    posterior_summary,
    resolve_evidence_kind,
)
from any2xsampa.core.models import PhoneCandidate


def _candidates() -> list[PhoneCandidate]:
    return [
        PhoneCandidate(symbol="a", confidence=0.81),
        PhoneCandidate(symbol="i", confidence=0.16),
    ]


def _metadata() -> dict[str, object]:
    return {
        "posterior_trace": {
            "source": "producer:frame_posteriors",
            "frame_hop_sec": 0.02,
            "frames": [
                {
                    "candidates": [
                        {"symbol": "a", "confidence": 0.82},
                        {"symbol": "i", "confidence": 0.12},
                    ]
                },
                {
                    "candidates": [
                        {"symbol": "a", "confidence": 0.76},
                        {"symbol": "i", "confidence": 0.18},
                    ]
                },
            ],
        },
        "logit_trace": {
            "source": "producer:frame_logits",
            "frame_hop_sec": 0.02,
            "frames": [
                {
                    "candidates": [
                        {"symbol": "a", "logit": 1.7},
                        {"symbol": "i", "logit": 0.4},
                    ]
                },
                {
                    "candidates": [
                        {"symbol": "a", "logit": 1.5},
                        {"symbol": "i", "logit": 0.6},
                    ]
                },
            ],
        },
    }


def test_segment_evidence_helper_trace_and_summary_contracts_match_recognizer_wrappers() -> None:
    metadata = _metadata()
    candidates = _candidates()

    assert frame_trace(metadata, "posterior_trace", "confidence") == recognizer_module._frame_trace(
        metadata,
        "posterior_trace",
        "confidence",
    )
    assert derive_posterior_frames(candidates, metadata) == recognizer_module._derive_posterior_frames(
        candidates,
        metadata,
    )
    assert derive_logit_frames(candidates, metadata) == recognizer_module._derive_logit_frames(
        candidates,
        metadata,
    )
    assert posterior_summary(candidates, metadata, "a") == recognizer_module._posterior_summary(
        candidates,
        metadata,
        "a",
    )
    assert logit_summary(candidates, metadata, "a") == recognizer_module._logit_summary(
        candidates,
        metadata,
        "a",
    )


def test_segment_evidence_helper_infers_candidates_from_logit_only_metadata() -> None:
    metadata = {
        "logit_trace": {
            "frames": [
                {"candidates": [{"symbol": "u", "logit": 1.8}, {"symbol": "o", "logit": 0.3}]},
                {"candidates": [{"symbol": "u", "logit": 1.6}, {"symbol": "o", "logit": 0.4}]},
            ]
        }
    }

    helper_candidates = infer_candidates_from_metadata(metadata)
    recognizer_candidates = recognizer_module._infer_candidates_from_metadata(metadata)

    assert [candidate.symbol for candidate in helper_candidates] == ["u", "o"]
    assert [(candidate.symbol, candidate.confidence) for candidate in helper_candidates] == [
        (candidate.symbol, candidate.confidence) for candidate in recognizer_candidates
    ]


def test_segment_evidence_helper_adapter_diagnostics_match_recognizer_wrapper() -> None:
    payload = {
        "adapter_kind": "real_backend",
        "backend_kind": "onnx-espeak",
        "source": "producer:onnx_mapped_frame_posteriors",
        "aggregation": "sum_mapped_posterior_mass",
        "candidate_source": "producer:onnx_mapped_frame_posteriors",
        "candidate_aggregation": "sum_mapped_posterior_mass",
        "posterior_like_available": True,
        "logit_like_available": True,
        "approximation_only": False,
        "posterior_source": "producer:onnx_mapped_frame_posteriors",
        "logit_source": "producer:onnx_mapped_frame_logits",
        "frame_count": 3,
        "frame_hop_sec": 0.02,
    }

    assert resolve_evidence_kind(True, True) == "posterior_and_logit_adapter_like"
    assert build_adapter_diagnostics(**payload) == recognizer_module._build_adapter_diagnostics(**payload)

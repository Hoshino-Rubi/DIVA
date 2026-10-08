from __future__ import annotations

import pytest

from any2xsampa.adapters.recognizer_evidence_formatter import (
    build_segment_evidence_views,
    build_segment_summary_payload,
    segment_evidence_views,
    segment_summary_payload,
)
from any2xsampa.adapters.recognizer_producer_projection import (
    frame_evidence_summary_from_posterior_payload,
    prediction_candidate_payload,
    prediction_mapped_frame_candidates,
    prediction_raw_label_candidates,
    prediction_records_from_posterior_payload,
    prediction_symbol_scores,
    run_summary_from_predictions,
)
from any2xsampa.core.models import PhoneCandidate


def test_prediction_candidate_payload_prefers_posterior_candidates_over_other_inputs() -> None:
    payload = prediction_candidate_payload(
        {
            "posterior_candidates": [{"symbol": "a", "confidence": 0.8}],
            "candidates": [PhoneCandidate(symbol="b", confidence=0.6)],
            "logit_candidates": [{"symbol": "c", "logit": 1.2}],
        }
    )

    assert payload == [{"symbol": "a", "confidence": 0.8}]


def test_prediction_candidate_payload_normalizes_phonecandidate_and_dict_inputs() -> None:
    payload = prediction_candidate_payload(
        {
            "candidates": [
                PhoneCandidate(symbol="aa", confidence=0.9),
                {"symbol": "ax", "confidence": 0.4},
                {"symbol": "", "confidence": 0.2},
            ]
        }
    )

    assert payload == [
        {"symbol": "aa", "confidence": 0.9},
        {"symbol": "ax", "confidence": 0.4},
    ]


def test_prediction_symbol_scores_softmaxes_logit_candidates() -> None:
    scores = prediction_symbol_scores(
        {
            "logit_candidates": [
                {"symbol": "aa", "logit": 2.0},
                {"symbol": "ax", "logit": 1.0},
            ]
        }
    )

    assert set(scores) == {"aa", "ax"}
    assert scores["aa"] == pytest.approx(0.731, abs=1e-3)
    assert scores["ax"] == pytest.approx(0.269, abs=1e-3)


def test_prediction_raw_and_mapped_frame_candidates_round_values_and_drop_invalid_rows() -> None:
    raw_candidates = prediction_raw_label_candidates(
        {
            "raw_top_labels": [
                {"label": "AA", "posterior": 0.123456, "logit": 1.23456, "mapped_symbol": "a"},
                {"label": "", "posterior": 0.2},
                {"posterior": 0.3},
            ]
        }
    )
    mapped_candidates = prediction_mapped_frame_candidates(
        {
            "mapped_frame_candidates": [
                {
                    "symbol": "a",
                    "posterior": 0.87654321,
                    "raw_posterior_mass": 0.9123456,
                    "logit": 1.987654,
                    "source_labels": [
                        {"label": "AA", "posterior": 0.7777777, "logit": 1.2222222},
                        {"label": ""},
                    ],
                },
                {"symbol": ""},
                {"posterior": 0.5},
            ]
        }
    )

    assert raw_candidates == [{"label": "AA", "posterior": 0.1235, "logit": 1.2346, "mapped_symbol": "a"}]
    assert mapped_candidates == [
        {
            "symbol": "a",
            "posterior": 0.876543,
            "raw_posterior_mass": 0.912346,
            "logit": 1.987654,
            "source_labels": [{"label": "AA", "posterior": 0.777778, "logit": 1.222222}],
        }
    ]


def test_prediction_records_from_posterior_payload_falls_back_to_metadata_and_producer_raw_traces() -> None:
    predictions, backend_name = prediction_records_from_posterior_payload(
        {
            "producer": {"backend_kind": "onnx-espeak"},
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.2,
                    "posterior_trace": {
                        "frames": [
                            {"candidates": [{"symbol": "fallback", "confidence": 0.9}]},
                            {"candidates": [{"symbol": "fallback", "confidence": 0.8}]},
                        ]
                    },
                    "metadata": {
                        "raw_label_trace": {
                            "frames": [
                                {"candidates": [{"label": "AA", "posterior": 0.6, "mapped_symbol": "a"}]},
                                {"candidates": [{"label": "SS", "posterior": 0.55, "mapped_symbol": "s"}]},
                            ]
                        },
                        "producer_raw_model_output": {
                            "mapped_frame_trace": {
                                "frames": [
                                    {
                                        "candidates": [
                                            {
                                                "symbol": "a",
                                                "posterior": 0.61,
                                                "source_labels": [{"label": "AA", "posterior": 0.61}],
                                            }
                                        ]
                                    },
                                    {
                                        "candidates": [
                                            {
                                                "symbol": "s",
                                                "posterior": 0.56,
                                                "source_labels": [{"label": "SS", "posterior": 0.56}],
                                            }
                                        ]
                                    },
                                ]
                            }
                        },
                    },
                    "candidate_phones": [{"symbol": "fallback", "confidence": 0.9}],
                }
            ],
        }
    )

    assert backend_name == "onnx-espeak"
    assert [prediction["source_frame_index"] for prediction in predictions] == [0, 1]
    assert predictions[0]["raw_top_labels"] == [{"label": "AA", "posterior": 0.6, "mapped_symbol": "a"}]
    assert predictions[1]["mapped_frame_candidates"] == [
        {"symbol": "s", "posterior": 0.56, "source_labels": [{"label": "SS", "posterior": 0.56}]}
    ]
    assert predictions[0]["posterior_candidates"][0]["symbol"] == "fallback"


def test_run_summary_from_predictions_groups_repeated_symbols_and_keeps_duration_shape() -> None:
    summary = run_summary_from_predictions(
        [
            {"start_sec": 0.0, "end_sec": 0.1},
            {"start_sec": 0.1, "end_sec": 0.2},
            {"start_sec": 0.2, "end_sec": 0.3},
        ],
        ["a", "a", "b"],
    )

    assert summary == [
        {
            "symbol": "a",
            "frame_count": 2,
            "start_index": 0,
            "end_index": 1,
            "start_sec": 0.0,
            "end_sec": 0.2,
            "duration_sec": 0.2,
        },
        {
            "symbol": "b",
            "frame_count": 1,
            "start_index": 2,
            "end_index": 2,
            "start_sec": 0.2,
            "end_sec": 0.3,
            "duration_sec": 0.1,
        },
    ]


def test_frame_evidence_summary_prefers_mapped_frame_trace_candidate_source_when_available() -> None:
    summary = frame_evidence_summary_from_posterior_payload(
        {
            "producer": {"backend_kind": "onnx-espeak"},
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.2,
                    "posterior_trace": {
                        "frames": [
                            {"candidates": [{"symbol": "a", "confidence": 0.51}, {"symbol": "s", "confidence": 0.49}]},
                            {"candidates": [{"symbol": "a", "confidence": 0.52}, {"symbol": "s", "confidence": 0.48}]},
                        ]
                    },
                    "mapped_frame_trace": {
                        "frames": [
                            {
                                "candidates": [
                                    {"symbol": "s", "posterior": 0.6, "logit": 1.4},
                                    {"symbol": "a", "posterior": 0.4, "logit": 1.0},
                                ]
                            },
                            {
                                "candidates": [
                                    {"symbol": "a", "posterior": 0.7, "logit": 1.8},
                                    {"symbol": "s", "posterior": 0.3, "logit": 0.9},
                                ]
                            },
                        ]
                    },
                    "metadata": {},
                }
            ],
        },
        anchor_start_sec=0.0,
        anchor_end_sec=0.2,
    )

    assert summary["full_frames"][0]["candidate_source_kind"] == "mapped_frame_trace"
    assert summary["full_frames"][0]["top_symbol"] == "s"
    assert summary["full_frames"][1]["top_symbol"] == "a"
    assert summary["near_tie_audit"]["full"]["candidate_source_kinds"] == ["mapped_frame_trace"]


def test_evidence_formatter_aliases_match_underlying_helpers() -> None:
    metadata = {
        "posterior_trace": {"frames": [1]},
        "logit_trace": {"frames": [2]},
        "acoustic_confidence": 0.82,
        "segment_likelihood": 0.74,
        "posterior_summary": {"source": "posterior"},
        "ownership_signal": {"ownership_conflict_risk": 0.2},
    }

    assert segment_summary_payload(metadata) == build_segment_summary_payload(metadata)
    assert segment_evidence_views(metadata) == build_segment_evidence_views(metadata)
    assert segment_evidence_views(metadata)["segment_summary"]["posterior_summary"]["source"] == "posterior"

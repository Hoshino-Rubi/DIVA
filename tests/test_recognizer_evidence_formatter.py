from __future__ import annotations

from pathlib import Path

from any2xsampa.adapters.recognizer import ConfigurableUniversalPhoneRecognizer, RecognizerConfig
from any2xsampa.adapters.recognizer_evidence_formatter import (
    apply_acoustic_evidence_metadata,
    build_acoustic_evidence_from_components,
    build_acoustic_evidence_from_metadata,
    resolved_candidate_evidence,
    segment_evidence_views,
)
from any2xsampa.adapters.recognizer_acoustic_evidence_builder import (
    build_acoustic_evidence_from_components as build_acoustic_evidence_from_components_builder,
    build_acoustic_evidence_from_metadata as build_acoustic_evidence_from_metadata_builder,
)
from any2xsampa.core.models import PhoneCandidate


def _fixture_segment(tmp_path: Path):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "recognizer_evidence_fixture.json"
    recognizer = ConfigurableUniversalPhoneRecognizer(
        RecognizerConfig(mode="fixture", fixture_path=fixture_path, allow_stub_fallback=False)
    )
    return recognizer.recognize(audio_path)[1]


def test_resolved_candidate_evidence_falls_back_to_posterior_payload() -> None:
    metadata = {
        "posterior": [
            {"symbol": "aa", "confidence": 0.61},
            {"symbol": "ax", "confidence": 0.53},
        ]
    }
    candidates = [
        PhoneCandidate(symbol="aa", confidence=0.61),
        PhoneCandidate(symbol="ax", confidence=0.53),
    ]

    evidence = resolved_candidate_evidence(
        candidates=candidates,
        metadata=metadata,
        candidate_logits={"aa": 0.33, "ax": 0.11},
    )

    assert [item.symbol for item in evidence] == ["aa", "ax"]
    assert evidence[0].articulatory_features["type"] == "vowel"
    assert evidence[0].logit == 0.33
    assert evidence[1].logit == 0.11


def test_build_acoustic_evidence_from_components_preserves_flat_metadata_views(tmp_path: Path) -> None:
    segment = _fixture_segment(tmp_path)

    rebuilt = build_acoustic_evidence_from_components(
        candidates=segment.candidate_phones,
        metadata=segment.metadata,
        source_mode="fixture",
        duration_sec=segment.time_range.end_sec - segment.time_range.start_sec,
        posterior_summary=segment.metadata["posterior_summary"],
        logit_summary=segment.metadata["logit_summary"],
        adapter_diagnostics=segment.metadata["adapter_diagnostics"],
    )
    rebuilt_metadata: dict[str, object] = {}
    apply_acoustic_evidence_metadata(
        rebuilt_metadata,
        evidence=rebuilt,
        candidates=segment.candidate_phones,
    )

    assert rebuilt_metadata["acoustic_evidence_applied"] is True
    assert rebuilt_metadata["acoustic_evidence_summary"].startswith("top=aa")
    assert rebuilt_metadata["candidate_count"] == 2
    assert rebuilt_metadata["boundary_uncertainty"]["start_ms"] == 42.0
    assert rebuilt_metadata["boundary_uncertainty"]["end_ms"] == 46.0
    assert rebuilt_metadata["note_alignment_friendly"]["same_note_continuation_bias"] == 0.9
    assert rebuilt_metadata["ownership_signal"]["ownership_conflict_risk"] == 0.83
    assert rebuilt_metadata["posterior_summary"]["source"] == segment.metadata["posterior_summary"]["source"]
    assert rebuilt_metadata["logit_summary"]["source"] == segment.metadata["logit_summary"]["source"]

    views = segment_evidence_views(rebuilt_metadata)
    assert views["segment_summary"]["acoustic_confidence"] == rebuilt_metadata["acoustic_confidence"]
    assert views["segment_summary"]["boundary_uncertainty"]["window_ms"] == 88.0
    assert views["continuation_profile"]["same_note_continuation_bias"] == 0.9
    assert views["boundary_profile"]["boundary_conflict_likelihood"] == rebuilt_metadata["boundary_profile"][
        "boundary_conflict_likelihood"
    ]
    assert views["boundary_profile"]["boundary_conflict_likelihood"] > 0.0


def test_build_acoustic_evidence_from_components_infers_adapter_diagnostics_when_omitted() -> None:
    metadata = {
        "frame_count": 2,
        "real_backend": "heuristic-dsp",
        "recognizer_backend_family": "real",
        "candidate_source": "posterior_trace",
        "posterior": [
            {"symbol": "aa", "confidence": 0.84},
            {"symbol": "ax", "confidence": 0.58},
        ],
        "posterior_summary": {
            "available": True,
            "source": "heuristic-dsp:frame_posteriors",
            "frame_count": 2,
            "frame_vote_ratio": 0.8,
            "temporal_consistency": 0.76,
            "mean_margin": 0.26,
            "interior_mean_margin": 0.24,
            "interior_vote_ratio": 0.73,
            "entry_margin": 0.21,
            "exit_margin": 0.19,
            "top_phone_mean_posterior": 0.81,
        },
        "logit_summary": {
            "available": True,
            "source": "heuristic-dsp:frame_logits",
            "frame_count": 2,
            "mean_margin": 0.33,
            "interior_mean_margin": 0.27,
        },
    }
    candidates = [
        PhoneCandidate(symbol="aa", confidence=0.84),
        PhoneCandidate(symbol="ax", confidence=0.58),
    ]

    evidence = build_acoustic_evidence_from_components(
        candidates=candidates,
        metadata=metadata,
        source_mode="real",
        duration_sec=0.22,
        posterior_summary=metadata["posterior_summary"],
        logit_summary=metadata["logit_summary"],
        candidate_logits={"aa": 1.2, "ax": 0.4},
    )

    payload = evidence.to_metadata()
    assert payload["adapter_diagnostics"]["backend_kind"] == "heuristic-dsp"
    assert payload["adapter_diagnostics"]["posterior_like_available"] is True
    assert payload["adapter_diagnostics"]["logit_like_available"] is True
    assert payload["adapter_diagnostics"]["evidence_kind"] == "posterior_and_logit_adapter_like"
    assert payload["candidate_phones"][0]["logit"] is not None


def test_build_acoustic_evidence_from_metadata_matches_component_builder_views(tmp_path: Path) -> None:
    segment = _fixture_segment(tmp_path)

    direct = build_acoustic_evidence_from_metadata(
        candidates=segment.candidate_phones,
        metadata=segment.metadata,
        source_mode="fixture",
        duration_sec=segment.time_range.end_sec - segment.time_range.start_sec,
    )
    rebuilt = build_acoustic_evidence_from_components(
        candidates=segment.candidate_phones,
        metadata=segment.metadata,
        source_mode="fixture",
        duration_sec=segment.time_range.end_sec - segment.time_range.start_sec,
        posterior_summary=segment.metadata["posterior_summary"],
        logit_summary=segment.metadata["logit_summary"],
        adapter_diagnostics=segment.metadata["adapter_diagnostics"],
    )

    direct_payload = direct.to_metadata()
    rebuilt_payload = rebuilt.to_metadata()

    assert direct_payload["top_phone"] == rebuilt_payload["top_phone"]
    assert direct_payload["candidate_distribution"] == rebuilt_payload["candidate_distribution"]
    assert direct_payload["boundary_profile"] == rebuilt_payload["boundary_profile"]
    assert direct_payload["adapter_diagnostics"] == rebuilt_payload["adapter_diagnostics"]


def test_builder_module_component_entry_matches_formatter_contract(tmp_path: Path) -> None:
    segment = _fixture_segment(tmp_path)

    via_formatter = build_acoustic_evidence_from_components(
        candidates=segment.candidate_phones,
        metadata=segment.metadata,
        source_mode="fixture",
        duration_sec=segment.time_range.end_sec - segment.time_range.start_sec,
        posterior_summary=segment.metadata["posterior_summary"],
        logit_summary=segment.metadata["logit_summary"],
        adapter_diagnostics=segment.metadata["adapter_diagnostics"],
    )
    via_builder = build_acoustic_evidence_from_components_builder(
        candidates=segment.candidate_phones,
        metadata=segment.metadata,
        source_mode="fixture",
        duration_sec=segment.time_range.end_sec - segment.time_range.start_sec,
        posterior_summary=segment.metadata["posterior_summary"],
        logit_summary=segment.metadata["logit_summary"],
        adapter_diagnostics=segment.metadata["adapter_diagnostics"],
    )

    assert via_builder.to_metadata() == via_formatter.to_metadata()


def test_builder_module_metadata_entry_matches_formatter_contract(tmp_path: Path) -> None:
    segment = _fixture_segment(tmp_path)

    via_formatter = build_acoustic_evidence_from_metadata(
        candidates=segment.candidate_phones,
        metadata=segment.metadata,
        source_mode="fixture",
        duration_sec=segment.time_range.end_sec - segment.time_range.start_sec,
    )
    via_builder = build_acoustic_evidence_from_metadata_builder(
        candidates=segment.candidate_phones,
        metadata=segment.metadata,
        source_mode="fixture",
        duration_sec=segment.time_range.end_sec - segment.time_range.start_sec,
    )

    assert via_builder.to_metadata() == via_formatter.to_metadata()

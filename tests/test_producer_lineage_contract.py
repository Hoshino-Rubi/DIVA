from __future__ import annotations

from pathlib import Path

from any2xsampa.adapters.recognizer_real import ExternalPosteriorJSONUniversalPhoneRecognizer


ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "samples" / "demo.wav"


def test_diagnostic_producer_lineage_stays_approximate() -> None:
    artifact = ROOT / "tests" / "fixtures" / "live_producer_captured_phrase_boundary_dual_touch_prototype-logit.json"
    segment = ExternalPosteriorJSONUniversalPhoneRecognizer(artifact).recognize(AUDIO)[0]
    metadata = segment.metadata

    assert metadata["producer_backend_kind"] == "prototype-logit"
    assert metadata["producer_approximation_only"] is True
    assert metadata["producer_diagnostic_baseline"] is True
    assert metadata["approximation_only"] is True
    assert metadata["real_model_output"] is False
    assert metadata["adapter_diagnostics"]["approximation_only"] is True


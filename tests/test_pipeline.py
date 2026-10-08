from pathlib import Path

from any2xsampa.pipeline.runner import SingingToPhonesPipeline


def test_pipeline_runs_end_to_end_with_structured_output(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    result = SingingToPhonesPipeline().run(
        audio_path=str(audio_path),
        dialect="cantonese",
        mode="infer-open",
    )

    payload = result.to_dict()

    assert payload["mode"] == "infer-open"
    assert payload["dialect"] == "cantonese"
    assert payload["segments"]
    assert payload["segments"][0]["normalized_ipa"]
    assert payload["segments"][0]["xsampa"]
    assert payload["segments"][0]["dialect_result"]["dialect"] == "cantonese"

from pathlib import Path

import pytest

from any2xsampa.adapters.recognizer import ConfigurableUniversalPhoneRecognizer, RecognizerConfig
from any2xsampa.adapters.recognizer_dispatch import (
    build_recognizer_dispatch_plan,
    render_real_command,
    render_real_command_args,
)
from any2xsampa.pipeline.runner import SingingToPhonesPipeline


def test_fixture_mode_loads_segments_from_json(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "recognizer_fixture.json"

    recognizer = ConfigurableUniversalPhoneRecognizer(
        RecognizerConfig(mode="fixture", fixture_path=fixture_path, allow_stub_fallback=False)
    )
    segments = recognizer.recognize(audio_path)

    assert len(segments) == 3
    assert segments[0].candidate_phones[0].symbol == "l"
    assert segments[0].metadata["source_mode"] == "fixture"
    assert segments[0].metadata["raw_candidates"] == ["l", "n"]


def test_real_mode_falls_back_to_stub_when_enabled(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")

    recognizer = ConfigurableUniversalPhoneRecognizer(
        RecognizerConfig(
            mode="real",
            real_command="cmd /c exit 1",
            real_output_path=tmp_path / "missing.json",
            allow_stub_fallback=True,
        )
    )
    segments = recognizer.recognize(audio_path)

    assert segments[0].metadata["source_mode"] == "stub"
    assert segments[0].metadata["fallback_from_mode"] == "real"
    assert segments[0].metadata["recognizer_fallback"]["requested_mode"] == "real"
    assert segments[0].metadata["recognizer_fallback"]["primary_kind"] == "real_command"
    assert segments[0].metadata["recognizer_fallback"]["error_type"] == "RuntimeError"


def test_pipeline_fixture_mode_produces_stable_xsampa(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "recognizer_fixture.json"

    pipeline = SingingToPhonesPipeline(
        recognizer_config=RecognizerConfig(
            mode="fixture",
            fixture_path=fixture_path,
            allow_stub_fallback=False,
            real_output_path=tmp_path / "unused.json",
        )
    )
    result = pipeline.run(audio_path=str(audio_path), dialect="cantonese", mode="infer-open")

    assert [segment.xsampa for segment in result.segments] == ["l", "a", "N"]
    assert result.segments[1].metadata["normalization_notes"]


def test_fixture_mode_preserves_explicit_acoustic_evidence_schema(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "recognizer_evidence_fixture.json"

    recognizer = ConfigurableUniversalPhoneRecognizer(
        RecognizerConfig(mode="fixture", fixture_path=fixture_path, allow_stub_fallback=False)
    )
    segments = recognizer.recognize(audio_path)

    evidence = segments[1].metadata["acoustic_evidence"]

    assert segments[1].metadata["acoustic_evidence_applied"] is True
    assert evidence["schema_version"] == "acoustic_evidence_v1"
    assert evidence["duration_sec"] == 0.32
    assert evidence["candidate_margin"] == 0.08
    assert evidence["candidate_count"] == 2
    assert evidence["candidate_entropy"] > 0.0
    assert evidence["frame_consistency"] > 0.0
    assert evidence["evidence_quality"] > 0.0
    assert evidence["segment_likelihood"] > 0.0
    assert evidence["dominant_candidate_stability"] > 0.0
    assert evidence["boundary_transition_strength"] > 0.0
    assert evidence["boundary_softness"] == 0.82
    assert evidence["ownership_ambiguity"] > 0.0
    assert evidence["candidate_phones"][0]["rank"] == 1
    assert evidence["candidate_phones"][1]["rank"] == 2
    assert evidence["candidate_phones"][1]["margin_from_top"] == 0.08
    assert evidence["boundary_uncertainty"]["uncertain"] is True
    assert evidence["boundary_uncertainty"]["window_ms"] == 88.0
    assert evidence["boundary_uncertainty"]["severity"] > 0.0
    assert evidence["note_alignment"]["same_note_continuation_bias"] == 0.9
    assert evidence["note_alignment"]["boundary_anchor_strength"] > 0.0
    assert evidence["note_alignment"]["supports_note_span_extension"] is True
    assert evidence["note_alignment"]["long_note_hold_bias"] > 0.0
    assert evidence["note_alignment"]["same_note_stability"] == 0.91
    assert evidence["note_alignment"]["weak_overlap_recovery"] == 0.87
    assert evidence["ownership"]["ownership_conflict_risk"] == 0.83
    assert evidence["ownership"]["boundary_reassignment_risk"] == 0.8
    assert evidence["ownership"]["part_note_span_reliability"] == 0.76
    assert evidence["candidate_distribution"]["runner_up_phone"] == "ax"
    assert evidence["candidate_distribution"]["candidate_lead"] > 0.0
    assert evidence["continuation_profile"]["continuation_friendliness"] > 0.0
    assert evidence["continuation_profile"]["weak_overlap_friendliness"] > 0.0
    assert evidence["boundary_profile"]["boundary_conflict_likelihood"] > 0.0
    assert evidence["adapter_diagnostics"]["posterior_like_available"] is True
    assert evidence["adapter_diagnostics"]["candidate_source"] == "posterior"
    assert evidence["posterior_summary"]["available"] is True
    assert evidence["posterior_summary"]["source"] == "posterior"
    assert evidence["logit_summary"]["available"] is True
    assert evidence["candidate_phones"][0]["logit"] is not None
    assert segments[1].metadata["acoustic_evidence_summary"].startswith("top=aa")


def test_recognizer_dispatch_plan_keeps_fixture_and_real_routing_explicit(tmp_path: Path) -> None:
    fixture_plan = build_recognizer_dispatch_plan(
        RecognizerConfig(mode="fixture", fixture_path=tmp_path / "fixture.json")
    )
    real_backend_plan = build_recognizer_dispatch_plan(
        RecognizerConfig(mode="real", real_backend="posterior-json", real_output_path=tmp_path / "output.json")
    )
    real_command_plan = build_recognizer_dispatch_plan(
        RecognizerConfig(
            mode="real",
            real_command="tool --audio {audio} --output {output}",
            real_output_path=tmp_path / "output.json",
        )
    )

    assert fixture_plan.primary_kind == "fixture"
    assert fixture_plan.fixture_path == tmp_path / "fixture.json"
    assert real_backend_plan.primary_kind == "real_backend"
    assert real_backend_plan.real_backend == "posterior-json"
    assert real_backend_plan.real_output_path == tmp_path / "output.json"
    assert real_command_plan.primary_kind == "real_command"
    assert real_command_plan.real_command == "tool --audio {audio} --output {output}"


def test_recognizer_dispatch_plan_normalizes_text_inputs(tmp_path: Path) -> None:
    plan = build_recognizer_dispatch_plan(
        RecognizerConfig(
            mode=" REAL ",
            real_backend=" posterior-json ",
            real_output_path=f" {tmp_path / 'output.json'} ",
        )
    )

    assert plan.mode == "real"
    assert plan.primary_kind == "real_backend"
    assert plan.real_backend == "posterior-json"
    assert plan.real_output_path == tmp_path / "output.json"


@pytest.mark.parametrize(
    ("config", "expected_message"),
    [
        (
            RecognizerConfig(mode="fixture", fixture_path=None, allow_stub_fallback=False),
            "Fixture mode requires a fixture_path.",
        ),
        (
            RecognizerConfig(mode="real", real_command=None, real_output_path=None, allow_stub_fallback=False),
            "Real mode requires real_backend or real_command with real_output_path.",
        ),
        (
            RecognizerConfig(mode="mystery", allow_stub_fallback=False),
            "Unknown recognizer mode: mystery",
        ),
    ],
)
def test_recognizer_dispatch_plan_rejects_incomplete_or_unknown_modes(
    config: RecognizerConfig,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        build_recognizer_dispatch_plan(config)


def test_render_real_command_rejects_unknown_placeholders(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown placeholder\\(s\\): \\{model\\}"):
        render_real_command(
            "tool --audio {audio} --output {output} --model {model}",
            audio_path=tmp_path / "demo.wav",
            output_path=tmp_path / "result.json",
        )


def test_render_real_command_keeps_literal_braces_available(tmp_path: Path) -> None:
    command = render_real_command(
        "tool --audio {audio} --output {output} --meta '{{\"mode\":\"fixture\"}}'",
        audio_path=tmp_path / "demo.wav",
        output_path=tmp_path / "result.json",
        lyrics="la",
    )

    assert str(tmp_path / "demo.wav") in command
    assert str(tmp_path / "result.json") in command
    assert '{"mode":"fixture"}' in command


def test_real_command_args_preserve_spaced_paths_without_a_shell(tmp_path: Path) -> None:
    audio = tmp_path / "audio with spaces.wav"
    output = tmp_path / "result with spaces.json"
    args = render_real_command_args(
        'producer --audio "{audio}" --output "{output}"',
        audio_path=audio,
        output_path=output,
    )

    assert args == ["producer", "--audio", str(audio), "--output", str(output)]

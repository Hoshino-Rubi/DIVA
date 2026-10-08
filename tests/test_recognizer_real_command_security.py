from pathlib import Path

import pytest

from any2xsampa.adapters import recognizer_real


def _write_output(path: Path) -> None:
    path.write_text(
        '{"schema_version":"external_posterior_json_v1",'
        '"segments":[{"start_sec":0.0,"end_sec":0.1,'
        '"candidate_phones":[{"symbol":"a","confidence":1.0}]}]}',
        encoding="utf-8",
    )


def test_real_command_uses_explicit_argv_without_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "audio with spaces.wav"
    output = tmp_path / "result with spaces.json"
    audio.write_bytes(b"RIFF")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        _write_output(output)
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(recognizer_real.subprocess, "run", fake_run)
    segments = recognizer_real.RealCommandUniversalPhoneRecognizer(
        'producer --audio "{audio}" --output "{output}" --label "A & B"',
        output,
    ).recognize(audio)

    assert len(segments) == 1
    assert calls == [
        (
            ["producer", "--audio", str(audio), "--output", str(output), "--label", "A & B"],
            {
                "shell": False,
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": 120.0,
            },
        )
    ]


def test_real_command_keeps_shell_metacharacters_in_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "audio.wav"
    output = tmp_path / "result.json"
    audio.write_bytes(b"RIFF")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        _write_output(output)
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(recognizer_real.subprocess, "run", fake_run)
    recognizer_real.RealCommandUniversalPhoneRecognizer(
        'producer --audio "{audio}" --output "{output}" --marker=before&echo',
        output,
    ).recognize(audio)

    assert calls == [[
        "producer",
        "--audio",
        str(audio),
        "--output",
        str(output),
        "--marker=before&echo",
    ]]


def test_real_command_surfaces_stderr_and_rejects_missing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    output = tmp_path / "missing.json"
    audio.write_bytes(b"RIFF")

    def failed_run(command: list[str], **kwargs: object) -> object:
        return type("Completed", (), {"returncode": 7, "stdout": "", "stderr": "producer failed"})()

    monkeypatch.setattr(recognizer_real.subprocess, "run", failed_run)
    with pytest.raises(RuntimeError, match="exit code 7: producer failed"):
        recognizer_real.RealCommandUniversalPhoneRecognizer("producer {audio} {output}", output).recognize(audio)

    def successful_without_output(command: list[str], **kwargs: object) -> object:
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(recognizer_real.subprocess, "run", successful_without_output)
    with pytest.raises(RuntimeError, match="did not produce output"):
        recognizer_real.RealCommandUniversalPhoneRecognizer("producer {audio} {output}", output).recognize(audio)


def test_real_command_does_not_reuse_stale_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "audio.wav"
    output = tmp_path / "stale.json"
    audio.write_bytes(b"RIFF")
    _write_output(output)

    def successful_without_replacement(command: list[str], **kwargs: object) -> object:
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(recognizer_real.subprocess, "run", successful_without_replacement)
    with pytest.raises(RuntimeError, match="did not produce output"):
        recognizer_real.RealCommandUniversalPhoneRecognizer("producer {audio} {output}", output).recognize(audio)
    assert not output.exists()

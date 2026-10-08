from __future__ import annotations

from pathlib import Path

import pytest

from any2xsampa.adapters.auxiliary_audio import read_stereo_context, vocal_energy, write_stereo_float
from any2xsampa.adapters.melband_runtime import demix_bounded, safe_model_config
from any2xsampa.adapters.qwen_timing_runtime import RawTimestampCapture
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file


def test_stereo_crop_hash_and_original_clock(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    source = tmp_path / "source.wav"
    original = np.stack([np.arange(48000, dtype=np.float32) / 48000, np.zeros(48000, dtype=np.float32)], axis=1)
    # External sources retain the project's existing PCM-WAV contract;
    # float separation output has a separate derived-audio record.
    sf.write(source, original, 48000, subtype="PCM_24")
    digest = sha256_file(source)
    audio, clock = read_stereo_context(source, digest, .25, .75)
    assert audio.shape == (2, 22050)
    assert clock["source_start_sample"] == 12000 and clock["effective_original_start_sec"] == .25
    assert np.all(audio[1] == 0)
    record = write_stereo_float(tmp_path / "derived.wav", audio, 44100)
    assert record["samples"] == 22050 and record["subtype"] == "FLOAT"
    with pytest.raises(ValueError, match="hash"):
        read_stereo_context(source, "a" * 64, .25, .75)
    assert sha256_file(source) == digest
    with pytest.raises(FileExistsError):
        write_stereo_float(tmp_path / "derived.wav", audio, 44100)


@pytest.mark.parametrize("length", [1, 9, 31, 50, 99, 100, 101, 199, 300])
def test_overlap_add_preserves_every_sample_for_identity_fixture(length: int) -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    mixture = torch.arange(length, dtype=torch.float32).repeat(2, 1)
    result = demix_bounded(torch, lambda x: x, mixture, chunk_size=100, overlap=2, device="cpu", mixed_precision=False)
    assert result.shape == mixture.shape
    assert torch.allclose(result, mixture, atol=1e-4, rtol=1e-6)


def test_separator_nan_is_not_silently_changed_to_silence() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="non-finite"):
        demix_bounded(torch, lambda x: x * float("nan"), torch.ones(2, 30), chunk_size=100, overlap=2, device="cpu", mixed_precision=False)


def test_energy_keeps_stereo_power_and_silent_case_explicit() -> None:
    np = pytest.importorskip("numpy")
    policy = read_json(Path("src/any2xsampa/config/quality8h_v1.json"))["slicing"]
    report = vocal_energy(np.zeros((2, 4410)), 44100, policy)
    assert not report["has_signal"] and report["frames"]
    # Opposite-phase stereo must not cancel to an invented silence.
    stereo = np.stack([np.ones(4410), -np.ones(4410)])
    assert vocal_energy(stereo, 44100, policy)["peak_rms"] == 1


def test_yaml_does_not_execute_arbitrary_python_tags(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")
    with pytest.raises(Exception):
        safe_model_config(path)


def test_upstream_timestamp_repair_is_captured_but_never_used_for_proposals() -> None:
    np = pytest.importorskip("numpy")
    class Processor:
        def fix_timestamp(self, values):
            return sorted(values.tolist())
    original = np.array([100, 500, 200, 800])
    hook = RawTimestampCapture(Processor())
    assert hook(original) == [100, 500, 200, 800]
    assert hook.calls[0]["native_repaired_ms_diagnostic_only"] == [100, 200, 500, 800]
    assert hook.calls[0]["native_changed_values"]
    assert original.tolist() == [100, 500, 200, 800]

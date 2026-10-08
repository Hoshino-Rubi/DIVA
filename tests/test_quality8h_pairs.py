from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from any2xsampa.adapters.audio_provenance import describe_audio_file


def module():
    spec = importlib.util.spec_from_file_location("pair_builder_test", Path("tools/build_quality8h_pairs.py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pair_uses_one_gain_and_remains_valid_pcm_without_clipping(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    mixed = np.full((44100 * 8, 2), .5, dtype=np.float32)
    vocal = np.full_like(mixed, 2.0)
    records, transform = module().paired_pcm(mixed, vocal, 0, len(mixed), tmp_path / "pair")
    assert .49 < transform["shared_gain"] < .5 and not transform["clipping_performed"]
    assert describe_audio_file(records["mixed"]["path"])["sample_width_bytes"] == 4
    for role, expected in (("mixed", .25), ("vocal", 1.0)):
        values, _ = sf.read(records[role]["path"], dtype="float32")
        assert np.allclose(values, expected, atol=2e-7)
    assert np.all(vocal == 2) and np.all(mixed == .5)


def test_pair_must_not_drop_the_hard_tail(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    values = np.zeros((44100 * 8, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="identical stereo"):
        module().paired_pcm(values, values[:-1], 0, len(values), tmp_path / "pair")


def test_audio_request_projection_cannot_forward_source_phone_context() -> None:
    spec = importlib.util.spec_from_file_location("paired_worker_test", Path("tools/run_quality8h_phonetic_pairs.py"))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    context = {"context_id": "freedom-flower-original-01-context", "example_id": "freedom-flower-original-01", "song_id": "freedom-flower", "split": "train", "audio_path": "source.wav", "audio_sha256": "a" * 64,
               "start_sec": 0., "end_sec": 30., "old_start_sec": 6., "old_end_sec": 24., "contains_labels_or_score_context": False}
    audio = {"path": "pair.wav", "sha256": "b" * 64, "samples": 18 * 44100, "sample_rate": 44100, "channels": 2}
    inputs = {"schema_version": "quality8h_paired_inputs_v1", "reference_labels_or_score_context_included": False, "test_songs_loaded": False,
              "pairs": [{"context": context, "audio": {"mixed": audio, "vocal": audio}}]}
    requests = worker.audio_requests(inputs, [context])
    assert {r["condition"] for r in requests} == {"mixed", "vocal"}
    assert set(requests[0]["audio_request"]) == {"example_id", "song_id", "split", "audio_role", "audio_path", "audio_sha256", "start_sec", "end_sec"}
    with pytest.raises(ValueError, match="reference context"):
        worker.audio_requests({**inputs, "lyrics": "must not enter acoustic decoding"}, [context])
    context["lyrics"] = "forbidden"
    with pytest.raises(ValueError, match="context"):
        worker.audio_requests(inputs, [context])

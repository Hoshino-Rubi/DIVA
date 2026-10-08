from __future__ import annotations

import os
from pathlib import Path

import pytest

from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
from any2xsampa.adapters.phoneticxeus_runtime import read_audio_crop
from any2xsampa.training.limited_data_artifacts import sha256_file

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows write-sharing cache")


def source(path: Path) -> str:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    t = np.arange(48000 * 10, dtype=np.float32) / 48000
    samples = np.stack([np.sin(t * 320), np.cos(t * 480)], axis=1) * .4
    sf.write(path, samples, 48000, subtype="PCM_24")
    return sha256_file(path)


def test_locked_audio_exactly_matches_legacy_and_returns_independent_arrays(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    path = tmp_path / "source.wav"
    digest = source(path)
    expected = read_audio_crop(path, digest, .13, 8.13)
    with LockedAudioCache() as cache:
        actual = cache.read(path, digest, .13, 8.13)
        assert np.array_equal(actual[0], expected[0]) and actual[1:] == expected[1:]
        actual[0][0] += 1
        again = cache.read(path, digest, .13, 8.13)
        assert np.array_equal(again[0], expected[0])
        assert cache.stats()["source_hashes"] == 1 and cache.stats()["waveform_hits"] == 1
    assert cache.stats()["open_sources"] == 0 and cache.stats()["cached_bytes"] == 0


def test_writes_and_replacement_are_denied_until_source_handle_closes(tmp_path: Path) -> None:
    path, replacement = tmp_path / "source.wav", tmp_path / "replacement.wav"
    digest = source(path)
    replacement.write_bytes(b"replacement")
    with LockedAudioCache() as cache:
        cache.read(path, digest, 0, 8)
        with pytest.raises(PermissionError):
            path.write_bytes(b"must not truncate")
        with pytest.raises(PermissionError):
            os.replace(replacement, path)
        assert sha256_file(path) == digest
    os.replace(replacement, path)
    with LockedAudioCache() as fresh:
        with pytest.raises(ValueError, match="SHA-256"):
            fresh.read(path, digest, 0, 8)
        assert fresh.stats()["open_sources"] == 0


def test_lru_is_bounded_and_oversize_audio_is_not_cached(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    digest = source(path)
    with LockedAudioCache(max_bytes=16000 * 4 * 4) as cache:
        cache.read(path, digest, 0, 4)
        cache.read(path, digest, 4, 8)
        assert cache.stats()["waveform_evictions"] == 1
        cache.read(path, digest, 0, 8)
        assert cache.stats()["oversize_not_cached"] == 1
        assert cache.stats()["peak_cached_bytes"] <= 16000 * 4 * 4


def test_evicted_source_must_be_rehashed_before_reusing_its_audio(tmp_path: Path) -> None:
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    ah, bh = source(a), source(b)
    with LockedAudioCache(max_sources=1) as cache:
        cache.read(a, ah, 0, 8)
        cache.read(b, bh, 0, 8)
        a.write_bytes(b"changed after its handle was evicted")
        with pytest.raises(ValueError, match="SHA-256"):
            cache.read(a, ah, 0, 8)
        assert cache.stats()["source_evictions"] == 1


def test_wrong_digest_and_closed_cache_fail_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    digest = source(path)
    cache = LockedAudioCache()
    cache.read(path, digest, 0, 8)
    with pytest.raises(ValueError, match="changed source digest"):
        cache.read(path, "b" * 64, 0, 8)
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.read(path, digest, 0, 8)

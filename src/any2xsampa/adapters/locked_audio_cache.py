"""Opt-in Windows read-sharing locks and a bounded preprocessed-audio LRU.

No neural features or logits are cached. A source is hashed while its handle
denies writes and deletion; path identity is checked again on every access.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, BinaryIO
import wave

from any2xsampa.training.limited_data_artifacts import finite_number, require_digest


def _open_locked(path: Path) -> BinaryIO:
    if os.name != "nt":
        raise RuntimeError("write-excluding audio cache is Windows-only; use the uncached adapter elsewhere")
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ, FILE_SHARE_READ (no write/delete sharing), OPEN_EXISTING.
    handle = create(str(path), 0x80000000, 0x1, None, 3, 0x08000080, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "rb")


def _stamp(status: os.stat_result) -> tuple[int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


@dataclass
class _Source:
    path: Path
    stream: BinaryIO
    expected_sha256: str
    stamp: tuple[int, int, int, int]
    info: dict


class LockedAudioCache:
    def __init__(self, *, max_bytes: int = 8 * 1024 ** 2, max_sources: int = 8) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (max_bytes, max_sources)):
            raise ValueError("audio cache limits must be positive integers")
        self.max_bytes, self.max_sources = max_bytes, max_sources
        self.sources: OrderedDict[str, _Source] = OrderedDict()
        self.waveforms: OrderedDict[tuple, Any] = OrderedDict()
        self.cached_bytes = 0
        self.closed = False
        self.lock = threading.RLock()
        self.counters = {"source_hashes": 0, "source_hash_bytes": 0, "source_hash_seconds": 0.0,
                         "waveform_hits": 0, "waveform_misses": 0, "waveform_evictions": 0,
                         "source_evictions": 0, "oversize_not_cached": 0, "peak_cached_bytes": 0}

    def _describe(self, path: Path, stream: BinaryIO, expected: str) -> dict:
        before = time.perf_counter()
        digest, count = hashlib.sha256(), 0
        stream.seek(0)
        while block := stream.read(1024 ** 2):
            count += len(block)
            digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError("locked source SHA-256 mismatch")
        stream.seek(0)
        with wave.open(stream, "rb") as reader:
            rate, channels, width, frames = reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getnframes()
        if min(rate, channels, width) < 1 or frames < 0:
            raise ValueError("invalid locked PCM WAV metadata")
        self.counters["source_hashes"] += 1
        self.counters["source_hash_bytes"] += count
        self.counters["source_hash_seconds"] += time.perf_counter() - before
        return {"sha256": actual, "sample_rate": rate, "channels": channels,
                "sample_width_bytes": width, "frame_count": frames, "duration_sec": frames / rate}

    def _source(self, path: Path, expected: str) -> _Source:
        key = str(path).casefold()
        if key in self.sources:
            source = self.sources[key]
            if source.expected_sha256 != expected:
                raise ValueError("changed source digest requires a new cache session")
            self._verify_identity(source)
            self.sources.move_to_end(key)
            return source
        stream = _open_locked(path)
        try:
            stamp = _stamp(os.fstat(stream.fileno()))
            info = self._describe(path, stream, expected)
            source = _Source(path, stream, expected, stamp, info)
            self._verify_identity(source)
        except BaseException:
            stream.close()
            raise
        while len(self.sources) >= self.max_sources:
            _, old = self.sources.popitem(last=False)
            old.stream.close()
            self.counters["source_evictions"] += 1
        self.sources[key] = source
        return source

    @staticmethod
    def _verify_identity(source: _Source) -> None:
        if _stamp(source.path.stat()) != source.stamp or _stamp(os.fstat(source.stream.fileno())) != source.stamp:
            raise ValueError("locked audio source identity or metadata changed")

    def _waveform(self, source: _Source, first: int, last: int) -> Any:
        import numpy as np
        from scipy.signal import resample_poly
        import soundfile as sf

        key = (str(source.path).casefold(), source.expected_sha256, first, last,
               "mono_float32_scipy_resample_poly_16000_v1")
        if key in self.waveforms:
            self.waveforms.move_to_end(key)
            self.counters["waveform_hits"] += 1
            return self.waveforms[key].copy()
        self.counters["waveform_misses"] += 1
        # A Python file object selects libsndfile virtual IO on Windows, avoiding
        # CRT file-descriptor incompatibility while retaining the original lock.
        with os.fdopen(os.dup(source.stream.fileno()), "rb") as duplicate:
            duplicate.seek(0)
            with sf.SoundFile(duplicate) as audio:
                audio.seek(first)
                waveform = audio.read(last - first, dtype="float32", always_2d=True)
        if waveform.shape[0] != last - first or not np.isfinite(waveform).all():
            raise ValueError("invalid or truncated locked waveform crop")
        waveform = waveform.mean(axis=1, dtype=np.float32)
        rate = source.info["sample_rate"]
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
        if not waveform.size or not np.isfinite(waveform).all():
            raise ValueError("invalid locked resampled waveform")
        self._verify_identity(source)
        if waveform.nbytes <= self.max_bytes:
            while self.waveforms and self.cached_bytes + waveform.nbytes > self.max_bytes:
                _, removed = self.waveforms.popitem(last=False)
                self.cached_bytes -= removed.nbytes
                self.counters["waveform_evictions"] += 1
            stored = waveform.copy()
            stored.flags.writeable = False
            self.waveforms[key] = stored
            self.cached_bytes += stored.nbytes
            self.counters["peak_cached_bytes"] = max(self.counters["peak_cached_bytes"], self.cached_bytes)
        else:
            self.counters["oversize_not_cached"] += 1
        return waveform

    def read(self, path: Path, expected_sha256: str, start_sec: float, end_sec: float) -> tuple[Any, dict, dict]:
        start = finite_number(start_sec, "crop start", minimum=0)
        end = finite_number(end_sec, "crop end", minimum=0)
        require_digest(expected_sha256, "audio_sha256")
        if not 0 < end - start <= 20.000001:
            raise ValueError("audio crop must be >0 and <=20 seconds")
        with self.lock:
            if self.closed:
                raise RuntimeError("audio cache is closed")
            source = self._source(path.resolve(), expected_sha256)
            info = source.info.copy()
            if end > info["duration_sec"] + 1e-6:
                raise ValueError("audio crop extends beyond its source")
            rate = info["sample_rate"]
            first, last = round(start * rate), round(end * rate)
            waveform = self._waveform(source, first, last)
            crop = {"requested_start_sec": start, "requested_end_sec": end,
                "source_start_sample": first, "source_end_sample_exclusive": last,
                "effective_start_sec": first / rate, "effective_end_sec": last / rate,
                "model_sample_rate": 16000, "model_samples": int(waveform.size),
                "channel_policy": "arithmetic_mean", "resample_policy": "scipy_resample_poly_v1",
                "waveform_normalization": "none_before_model_frontend"}
            return waveform, info, crop

    def stats(self) -> dict:
        with self.lock:
            return {**self.counters, "cached_bytes": self.cached_bytes, "open_sources": len(self.sources),
                    "max_bytes": self.max_bytes, "max_sources": self.max_sources,
                    "source_write_and_delete_sharing_denied": True, "neural_features_cached": False}

    def close(self) -> None:
        with self.lock:
            for source in self.sources.values():
                source.stream.close()
            self.sources.clear()
            self.waveforms.clear()
            self.cached_bytes = 0
            self.closed = True

    def __enter__(self) -> LockedAudioCache:
        if self.closed:
            raise RuntimeError("cannot reopen a closed audio cache")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

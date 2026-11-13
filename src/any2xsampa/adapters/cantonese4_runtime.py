"""Replaceable offline MelBand/Whisper adapter for candidate 4.

Only model windows have times. Canonical IPA is a lexical projection, never a
claim about acoustic phone boundaries. The normal recognizer contract stays intact.
"""
from __future__ import annotations

import copy
import gc
import importlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
import types
from typing import Any, Callable

from any2xsampa.local.common import check_cancel, read, sha, write
from any2xsampa.local.cantonese4 import ASSET_LOCK, save_predictions

GIB = 1024 ** 3


def offline(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      TORCH_FORCE_WEIGHTS_ONLY_LOAD="1", HF_HOME=str(cache / "hf"),
                      NUMBA_CACHE_DIR=str(cache / "numba"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "4"

    def deny(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network is disabled during Cantonese inference")

    for key in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        setattr(socket, key, deny)
    for key in ("connect", "connect_ex", "sendto"):
        setattr(socket.socket, key, deny)


def resource_job() -> Any:
    """Bound only this isolated worker; its parent owns cancellation/termination."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                    ("active", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

    class Limits(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = Limits()
    limits.basic.flags = 0x100 | 0x8 | 0x2
    limits.basic.active, limits.basic.process_time = 1, 1800 * 10000000
    limits.process_memory = 8 * GIB
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel.CloseHandle(handle)
        raise ctypes.WinError(error)
    return kernel, handle


def verify_bundle(bundle: Path, check: Callable) -> dict:
    lock = read(ASSET_LOCK)
    for entry in lock["files"]:
        check()
        path = (bundle / entry["path"]).resolve()
        if not path.is_relative_to(bundle) or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("Cantonese model binding changed: " + entry["path"])
    return lock


def separator_config(path: Path) -> dict:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
    if config["model"]["sample_rate"] != 44100 or config["inference"] != {"num_overlap": 2, "chunk_size": 352800}:
        raise ValueError("Separator audio geometry changed")
    return config


def demix(torch: Any, model: Any, mixture: Any, check: Callable) -> Any:
    """Unchanged candidate overlap/add geometry, including edge windows."""
    if mixture.ndim != 2 or mixture.shape[0] != 2 or mixture.shape[1] < 1 or not torch.isfinite(mixture).all():
        raise ValueError("Separator input must be finite [2, samples]")
    chunk, step, fade = 352800, 176400, 35280
    length = mixture.shape[1]
    reflected = length > 352800
    if reflected:
        mixture = torch.nn.functional.pad(mixture, (step, step), mode="reflect")
    mixture = mixture.to("cuda:0")
    result, counter = torch.zeros_like(mixture, dtype=torch.float32), torch.zeros_like(mixture, dtype=torch.float32)
    window = torch.ones(chunk, device="cuda:0")
    window[:fade] *= torch.linspace(0, 1, fade, device="cuda:0")
    window[-fade:] *= torch.linspace(1, 0, fade, device="cuda:0")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for first in range(0, mixture.shape[1], step):
            check()
            part = mixture[:, first:first + chunk]
            actual = part.shape[-1]
            if actual < chunk:
                part = torch.nn.functional.pad(part, (0, chunk - actual), mode="reflect" if actual > chunk // 2 + 1 else "constant")
            predicted = model(part.unsqueeze(0))[0]
            if tuple(predicted.shape) != (2, chunk) or not torch.isfinite(predicted).all():
                raise RuntimeError("Separator returned invalid samples")
            weight = window.clone()
            if first == 0:
                weight[:fade] = 1
            elif first + chunk >= mixture.shape[1]:
                weight[-fade:] = 1
            result[:, first:first + actual] += predicted[:, :actual] * weight[:actual]
            counter[:, first:first + actual] += weight[:actual]
    if not torch.all(counter > 0):
        raise RuntimeError("Separator left uncovered samples")
    output = result / counter
    if reflected:
        output = output[:, step:-step]
    if output.shape != (2, length) or not torch.isfinite(output).all():
        raise RuntimeError("Separator changed the audio clock")
    return output.float().cpu().contiguous()


def separate(torch: Any, bundle: Path, audio: Any, check: Callable) -> Any:
    code = bundle / "separator/code"
    if "models" in sys.modules:
        raise RuntimeError("Ambiguous separator models namespace")
    namespace = types.ModuleType("models")
    namespace.__path__ = [str(code / "models")]
    sys.modules["models"] = namespace
    module = importlib.import_module("models.mel_band_roformer")
    config = separator_config(code / "configs/config_vocals_mel_band_roformer.yaml")
    model = module.MelBandRoformer(**config["model"])
    weights = torch.load(str(bundle / "separator/MelBandRoformer.ckpt"), map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not all(isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("Expected a tensor-only separator checkpoint")
    model.load_state_dict(weights, strict=True)
    del weights
    model = model.eval().requires_grad_(False).to("cuda:0")
    try:
        check()
        output = demix(torch, model, torch.from_numpy(audio), check)
        torch.cuda.synchronize()
        return output.numpy()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def generation_settings(model: Any) -> Any:
    config = copy.deepcopy(model.generation_config)
    config.num_beams, config.do_sample, config.max_new_tokens = 1, False, 400
    if (getattr(config, "num_return_sequences", 1) != 1 or getattr(config, "return_timestamps", False)
            or getattr(config, "return_token_timestamps", False) or config.eos_token_id is None):
        raise ValueError("Unexpected pinned short-form generation settings")
    return config


def infer(torch: Any, runtime: tuple, samples: Any, event: Callable) -> dict:
    processor, model = runtime
    raw = {"status": "failed", "generate_started": False, "generate_completed": False}
    try:
        config = generation_settings(model)
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        if set(inputs) - {"input_features", "attention_mask"} or "input_features" not in inputs:
            raise ValueError("Unexpected non-audio processor fields")
        event("generate_started")
        raw["generate_started"] = True
        with torch.inference_mode():
            generated = model.generate(**{key: value.to("cpu") for key, value in inputs.items()},
                                       generation_config=config, temperature=0.0, return_dict_in_generate=True)
        raw["generate_completed"] = True
        event("generate_completed")
        sequence = generated["sequences"] if isinstance(generated, dict) else generated.sequences
        tokens = sequence.detach().cpu().tolist()
        if len(tokens) != 1 or not tokens[0] or len(tokens[0]) > model.config.max_target_positions:
            raise ValueError("Unexpected generated sequence geometry")
        if any(type(token) is not int or token < 0 for token in tokens[0]):
            raise ValueError("Non-integer or negative token ID")
        eos_ids = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
        eos = any(token in eos_ids for token in tokens[0])
        forced = getattr(config, "forced_eos_token_id", None)
        raw.update(token_ids=tokens[0], returned_token_count=len(tokens[0]), eos_token_ids=eos_ids, eos_observed=eos,
                   truncated_or_incomplete=None if eos and forced is not None else not eos,
                   truncation_status="incomplete_without_eos" if not eos else "unknown_if_forced_eos_at_limit" if forced is not None else "complete_with_eos")
        for name, skip in (("raw_decoded", False), ("decoded", True)):
            text = processor.batch_decode(tokens, skip_special_tokens=skip, clean_up_tokenization_spaces=False)[0]
            if not isinstance(text, str):
                raise TypeError("Decoder must return text")
            raw[name] = text
        raw["status"] = "ok"
    except Exception as exc:
        raw["error"] = {"type": type(exc).__name__, "message": str(exc)[:1024]}
        event("prediction_failed", error=raw["error"])
    return raw


def run(params: dict, cancel: threading.Event, progress: Callable) -> Path:
    output, bundle = Path(params["_output"]), Path(params["bundle_dir"]).resolve()
    job = resource_job()
    started = time.monotonic()
    ledger = (output / "events.jsonl").open("x", encoding="utf-8", buffering=1)

    def event(name: str, **fields: Any) -> None:
        ledger.write(json.dumps({"event": name, "seconds": time.monotonic() - started, **fields}, ensure_ascii=True) + "\n")
        ledger.flush()

    def check() -> None:
        check_cancel(cancel)
        if time.monotonic() - started > 1750:
            raise TimeoutError("Cantonese inference time limit; no retry")
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            torch = sys.modules["torch"]
            if max(torch.cuda.memory_reserved(), torch.cuda.memory_allocated()) >= 7 * GIB:
                raise RuntimeError("Cantonese GPU memory limit reached")

    try:
        event("validation_started")
        lock = verify_bundle(bundle, check)
        offline(output / "cache")
        import numpy as np
        import soundfile as sf
        import librosa
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("Cantonese vocal separation requires CUDA; CPU separation is not enabled")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(min(1.0, 7 * GIB / torch.cuda.get_device_properties(0).total_memory), 0)
        original = Path(params["audio"])
        digest = sha(original)
        with sf.SoundFile(original) as source:
            rate, source_frames = source.samplerate, source.frames
            first, last = (0, source_frames) if params["scope"] == "whole_file" else (
                math.floor(params["start"] * rate), min(source_frames, math.floor((params["start"] + params["duration"]) * rate)))
            source.seek(first)
            audio = source.read(last - first, dtype="float32", always_2d=True)
        if len(audio) != last - first or audio.shape[1] not in (1, 2) or not np.isfinite(audio).all() or sha(original) != digest:
            raise ValueError("Input WAV changed or returned invalid samples")
        audio = audio.T
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        if rate != 44100:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=44100, res_type="soxr_hq", fix=True, scale=False, axis=-1)
        check()
        progress({"stage": "separating"})
        event("separation_started")
        vocals = separate(torch, bundle, np.ascontiguousarray(audio), check)
        event("separation_completed", frames=vocals.shape[1])
        del audio
        sf.write(output / "vocal.wav", vocals.T, 44100, subtype="FLOAT")
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from any2xsampa.adapters.cantonese4_lexicon import completion_gate
        progress({"stage": "recognizing"})
        runtimes: dict[str, tuple] = {}

        def runtime(name: str) -> tuple:
            if name not in runtimes:
                event("model_load_started", model=name)
                root = str(bundle / name)
                processor = WhisperProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
                model = WhisperForConditionalGeneration.from_pretrained(root, local_files_only=True, trust_remote_code=False, weights_only=True).to("cpu").eval()
                runtimes[name] = processor, model
                write(output / f"generation-{name}.json", {"pinned": model.generation_config.to_dict(),
                      "effective": generation_settings(model).to_dict(), "cpu_threads": 4,
                      "max_new_tokens": 400, "num_beams": 1, "do_sample": False})
                event("model_loaded", model=name)
            return runtimes[name]

        windows = []
        count = math.ceil(vocals.shape[1] / (18 * 44100))
        for index in range(count):
            check()
            start, end = index * 18 * 44100, min((index + 1) * 18 * 44100, vocals.shape[1])
            mono = vocals[:, start:end].mean(axis=0, dtype=np.float32)
            samples = librosa.resample(mono, orig_sr=44100, target_sr=16000, res_type="soxr_hq", fix=True, scale=False)
            if len(samples) != math.ceil((end - start) * 16000 / 44100) or not np.isfinite(samples).all():
                raise ValueError("Resampling changed the declared crop clock")
            window = {"start_sec": first / rate + start / 44100, "end_sec": first / rate + end / 44100,
                      "small": None, "base": None}
            for name in ("small", "base"):
                check()
                window[name] = infer(torch, runtime(name), np.ascontiguousarray(samples, dtype="<f4"),
                                     lambda name_event, **fields: event(name_event, window=index, model=name, **fields))
                write(output / "raw" / f"{index:04d}-{name}.json", window[name])
                if name == "small" and completion_gate(window[name]):
                    break
            windows.append(window)
            write(output / "windows" / f"{index:04d}.json", window)
            progress({"stage": "recognizing", "completed": index + 1, "total": count})
        write(output / "RAW_FROZEN.json", {"schema_version": "cantonese4_raw_v1", "windows": windows,
                                           "parsing_started": False})
        progress({"stage": "saving"})
        save_predictions(output, windows, source={"path": str(original), "sha256": digest,
            "start_sec": first / rate, "end_sec": last / rate, "sample_rate": rate,
            "models": lock["models"], "separator_gain": 1.0, "latency_compensation_samples": 0,
            "separator_latency": "unknown"}, mode="real", cancel=cancel)
        event("completed", windows=count)
        return output
    except BaseException as exc:
        event("failed", type=type(exc).__name__, message=str(exc)[:1500])
        raise
    finally:
        ledger.close()
        if job:
            kernel, handle = job
            kernel.CloseHandle(handle)

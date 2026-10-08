"""Optional isolated PyTorch runtime. Import only inside the dedicated worker."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

from any2xsampa.adapters.audio_provenance import describe_audio_file
from any2xsampa.adapters.phoneticxeus import convolution_geometry, emission_payload, verify_snapshot
from any2xsampa.training.limited_data_artifacts import finite_number, read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.limited_data_reference import load_vocabulary


GIB = 1024 ** 3


def enforce_offline(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "tmp").mkdir(exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_MODULES_CACHE": str(cache_dir / "huggingface/modules"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface/hub"),
        "TORCH_HOME": str(cache_dir / "torch"),
        "CUDA_CACHE_PATH": str(cache_dir / "cuda"),
        "TMP": str(cache_dir / "tmp"), "TEMP": str(cache_dir / "tmp"),
    })

    def deny(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network access forbidden inside the PhoneticXeus worker")

    socket.create_connection = deny
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.socket.sendto = deny
    socket.getaddrinfo = deny
    socket.gethostbyname = deny
    socket.gethostbyname_ex = deny
    socket.gethostbyaddr = deny


def gpu_total_used_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15, shell=False,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("this experiment requires an unambiguous single GPU")
    return int(lines[0].strip()) * 1024 ** 2


def check_gpu_budget(torch: Any) -> dict[str, int]:
    reserved = int(torch.cuda.memory_reserved())
    allocated = int(torch.cuda.memory_allocated())
    used = gpu_total_used_bytes()
    if reserved > 7 * GIB or used > 14 * GIB:
        raise RuntimeError(f"GPU memory budget exceeded: process_reserved={reserved}, total_used={used}")
    return {"reserved_bytes": reserved, "allocated_bytes": allocated, "total_used_bytes": used}


def configure_cuda() -> Any:
    import torch

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("PhoneticXeus requires its isolated Python 3.12 runtime")
    if torch.__version__ != "2.8.0+cu126" or torch.version.cuda != "12.6":
        raise RuntimeError(f"unexpected isolated Torch/CUDA: {torch.__version__}, {torch.version.cuda}")
    if importlib.metadata.version("torchaudio") != "2.8.0+cu126":
        raise RuntimeError("torchaudio does not match the isolated CUDA Torch pair")
    if not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA unavailable; no CPU fallback")
    torch.set_num_threads(4)
    # Serial checkpoint loads reuse this process's already configured pool.
    # A different pool size after parallel work starts must still fail.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 7 * GIB / total), 0)
    torch.empty(1, device="cuda")
    check_gpu_budget(torch)
    return torch


def read_audio_crop(path: Path, expected_sha256: str, start_sec: float, end_sec: float) -> tuple[Any, dict, dict]:
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf

    start = finite_number(start_sec, "crop start", minimum=0)
    end = finite_number(end_sec, "crop end", minimum=0)
    if not 0 < end - start <= 20.000001:
        raise ValueError("audio crop must be >0 and <=20 seconds")
    info = describe_audio_file(path)
    if info["sha256"] != expected_sha256:
        raise ValueError("audio source hash mismatch before feature extraction")
    if end > info["duration_sec"] + 1e-6:
        raise ValueError("audio crop extends beyond its source")
    rate = info["sample_rate"]
    first, last = round(start * rate), round(end * rate)
    with sf.SoundFile(path) as stream:
        stream.seek(first)
        waveform = stream.read(last - first, dtype="float32", always_2d=True)
    if waveform.shape[0] != last - first or not np.isfinite(waveform).all():
        raise ValueError("invalid or truncated waveform crop")
    waveform = waveform.mean(axis=1, dtype=np.float32)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("invalid resampled waveform")
    crop = {
        "requested_start_sec": start, "requested_end_sec": end,
        "source_start_sample": first, "source_end_sample_exclusive": last,
        "effective_start_sec": first / rate, "effective_end_sec": last / rate,
        "model_sample_rate": 16000, "model_samples": int(waveform.size),
        "channel_policy": "arithmetic_mean", "resample_policy": "scipy_resample_poly_v1",
        "waveform_normalization": "none_before_model_frontend",
    }
    return waveform, info, crop


class PhoneticXeusRuntime:
    def __init__(self, snapshot: Path, *, cache_dir: Path | None = None, audio_cache: Any = None) -> None:
        if not sys.flags.utf8_mode:
            raise RuntimeError("launch the isolated worker with -X utf8; upstream vocabulary reads use the process default encoding")
        self.snapshot = snapshot.resolve()
        self.audio_cache = audio_cache
        self.identity = verify_snapshot(self.snapshot)
        enforce_offline(cache_dir if cache_dir is not None else self.snapshot.parent / "runtime-cache")
        self.torch = configure_cuda()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(str(self.snapshot), trust_remote_code=True, local_files_only=True)
        config._name_or_path = str(self.snapshot)
        model, loading = AutoModel.from_pretrained(
            str(self.snapshot), config=config, trust_remote_code=True,
            local_files_only=True, use_safetensors=True, torch_dtype=self.torch.float32,
            low_cpu_mem_usage=False, output_loading_info=True,
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if loading.get(key):
                raise RuntimeError(f"model architecture/weights mismatch: {key}={loading[key]}")
        self.model = model.eval().requires_grad_(False).to("cuda")
        if any(parameter.device.type != "cuda" for parameter in self.model.parameters()):
            raise RuntimeError("model parameters are not entirely on CUDA")
        self.vocabulary = load_vocabulary(self.snapshot / "src/model/xeusphoneme/resources/ipa_vocab.json")
        if len(self.vocabulary) != 428 or self.model.model.ctc.ctc_lo.in_features != 1024:
            raise RuntimeError("unexpected pretrained CTC head dimensions")
        if self.model.model.ctc.ctc_lo.out_features != 428:
            raise RuntimeError("pretrained phone output inventory differs from the lock")
        self.shapes = [(int(layer.kernel_size), int(layer.stride)) for layer in self.model.model.frontend.layers]
        self.torch.cuda.reset_peak_memory_stats()
        self.runtime = {
            "torch_version": self.torch.__version__, "cuda_version": self.torch.version.cuda,
            "device": "cuda:0", "cuda_device_name": self.torch.cuda.get_device_name(0),
            "cpu_threads": 4, "offline": True, "parameter_device_verified": True,
            "phone_head_pretrained": True, "encoder_frozen": True,
            "loading_info": loading, "initial_memory": check_gpu_budget(self.torch),
        }

    def encode(self, row: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        reader = self.audio_cache.read if self.audio_cache is not None else read_audio_crop
        waveform, audio, crop = reader(
            Path(row["audio_path"]), row["audio_sha256"], row["start_sec"], row["end_sec"]
        )
        torch = self.torch
        check_gpu_budget(torch)
        before = time.monotonic()
        speech = torch.from_numpy(waveform).unsqueeze(0).to("cuda")
        lengths = torch.tensor([waveform.size], dtype=torch.long, device="cuda")
        begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.inference_mode():
            features, feature_lengths = self.model.model.encode(speech, lengths)
            if isinstance(features, tuple):
                features = features[0]
            valid = int(feature_lengths[0].item())
            features = features[0, :valid]
            logits = self.model.model.ctc.ctc_lo(features)
        finish.record()
        torch.cuda.synchronize()
        if features.device.type != "cuda" or logits.device.type != "cuda":
            raise RuntimeError("model output did not originate on CUDA")
        if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise RuntimeError("non-finite acoustic evidence")
        geometry = convolution_geometry(self.shapes, int(waveform.size))
        if valid != geometry["frame_count"]:
            raise RuntimeError("actual encoder timebase differs from frontend geometry")
        measured = {
            **self.runtime, "elapsed_seconds": time.monotonic() - before,
            "cuda_event_ms": begin.elapsed_time(finish),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "memory_after": check_gpu_budget(torch),
        }
        payload = emission_payload(
            logits.float().cpu().numpy(), self.vocabulary,
            input_audio=audio, crop=crop, model_metadata=self.identity,
            geometry=geometry, runtime=measured,
        )
        payload["example_id"] = row["example_id"]
        payload["song_id"] = row["song_id"]
        payload["split"] = row["split"]
        payload["audio_role"] = row["audio_role"]
        return features.float().cpu().contiguous(), logits.float().cpu().contiguous(), payload

    def save_example(self, row: Mapping[str, Any], output_dir: Path, *, sealed: bool = False) -> dict[str, Any]:
        from safetensors.torch import save_file

        output_dir.mkdir(parents=True, exist_ok=True)
        tensors_path, posterior_path = output_dir / "evidence.safetensors", output_dir / "posterior.json"
        if tensors_path.exists() or posterior_path.exists():
            raise FileExistsError("refusing to overwrite model evidence")
        features, logits, payload = self.encode(row)
        save_file({"encoder_features": features, "frame_logits": logits}, str(tensors_path))
        tensor_hash = sha256_file(tensors_path)
        vocabulary_path = output_dir / "vocabulary.json"
        with vocabulary_path.open("xb") as stream:
            stream.write((self.snapshot / "src/model/xeusphoneme/resources/ipa_vocab.json").read_bytes())
        verify_file(vocabulary_path, self.identity["vocabulary_sha256"])
        payload["raw_evidence"] = {
            "file": tensors_path.name, "sha256": tensor_hash,
            "feature_shape": list(features.shape), "logit_shape": list(logits.shape),
            "dtype": "float32", "vocabulary_sha256": self.identity["vocabulary_sha256"],
            "vocabulary_file": vocabulary_path.name,
        }
        payload["test_outputs_sealed"] = sealed
        posterior_hash = write_json_new(posterior_path, payload)
        return {
            "example_id": row["example_id"], "song_id": row["song_id"], "split": row["split"],
            "audio_role": row["audio_role"], "audio_sha256": row["audio_sha256"],
            "start_sec": row["start_sec"], "end_sec": row["end_sec"],
            "evidence_path": str(tensors_path.resolve()), "evidence_sha256": tensor_hash,
            "posterior_path": str(posterior_path.resolve()), "posterior_sha256": posterior_hash,
            "frames": logits.shape[0], "feature_width": features.shape[1],
            "vocabulary_sha256": self.identity["vocabulary_sha256"],
            "model_snapshot_lock_sha256": self.identity["snapshot_lock_sha256"],
            "test_outputs_sealed": sealed,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

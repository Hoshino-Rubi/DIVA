"""Strict local contracts; no HTTP, lyrics, score or network conditions."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import wave

VERSION = "any2xsampa.local.v1"
ROOT = Path(os.environ.get("ANY2XSAMPA_HOME", Path(__file__).resolve().parents[3])).resolve()

class Cancelled(RuntimeError):
    pass

def check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled("Task cancelled; no automatic retry")

def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value

def write(path: Path, value: dict, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if not replace:
        with path.open("x", encoding="utf-8") as f:
            f.write(data)
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)

def finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric")
    return float(value)

def audio_info(path: Path) -> dict:
    # Use the existing WAV contract: PCM inputs only. Never silently convert an unsupported file.
    with wave.open(str(path), "rb") as f:
        return {"path": str(path.resolve()), "sample_rate": f.getframerate(),
                "channels": f.getnchannels(), "frames": f.getnframes(),
                "sample_width": f.getsampwidth(), "duration": f.getnframes() / f.getframerate()}

def crop(audio: Path, start: float, duration: float, destination: Path) -> dict:
    start, duration = finite(start, "start"), finite(duration, "duration")
    info = audio_info(audio)
    if start < 0 or not .1 <= duration <= 20 or start + duration > info["duration"] + 1e-6:
        raise ValueError("Select a valid 0.1–20 second WAV interval")
    first, last = round(start * info["sample_rate"]), round((start + duration) * info["sample_rate"])
    digest = sha(audio)
    with wave.open(str(audio), "rb") as src, wave.open(str(destination), "wb") as dst:
        src.setpos(first)
        frames = src.readframes(last - first)
        if len(frames) != (last-first)*info["channels"]*info["sample_width"]:
            raise ValueError("Truncated audio")
        dst.setparams(src.getparams())
        dst.writeframes(frames)
    if sha(audio) != digest:
        raise ValueError("Audio changed while reading")
    return {**info, "sha256": digest, "start_sec": first/info["sample_rate"],
            "end_sec": last/info["sample_rate"], "source_start_sample": first,
            "source_end_sample_exclusive": last, "crop_sha256": sha(destination)}

def request(value: dict) -> dict:
    allowed = {"audio", "start", "duration", "model", "device", "output_dir"}
    if set(value) - allowed or not {"audio", "model"} <= set(value):
        raise ValueError("recognize accepts only audio, model, device, start, duration, output_dir")
    if not isinstance(value["audio"], str) or not isinstance(value["model"], str):
        raise ValueError("audio and model must be strings")
    start = finite(value.get("start", 0), "start")
    duration = finite(value.get("duration", 8), "duration")
    if start < 0 or not .1 <= duration <= 20:
        raise ValueError("Select a valid 0.1–20 second interval")
    device = value.get("device", "cuda")
    if device not in {"cuda", "cpu"}:
        raise ValueError("Choose explicit cuda or cpu; silent auto fallback is forbidden")
    return {**value, "audio": str(Path(value["audio"]).resolve()), "start": start,
            "duration": duration, "device": device}

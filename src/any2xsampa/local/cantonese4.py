"""Optional Cantonese candidate: untimed IPA evidence, separate from phone tasks."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid
from typing import Any, Callable

from .common import ROOT, audio_info, check_cancel, finite, read, sha, write
from .storage import state_dir, model_cache

ASSET_LOCK = Path(__file__).resolve().parents[1] / "config/cantonese4_assets.json"


def bundle_path(value: str | None = None) -> Path:
    return Path(value).expanduser().resolve() if value else ROOT / "models/cantonese4"


def status(params: dict[str, Any]) -> dict[str, Any]:
    if set(params) - {"bundle_dir"}:
        raise ValueError("cantonese4_status accepts only bundle_dir")
    bundle = bundle_path(params.get("bundle_dir"))
    result: dict[str, Any] = {"bundle_dir": str(bundle), "model_loaded": False, "available": False}
    if not ASSET_LOCK.is_file():
        return {**result, "reason": "asset_lock_missing"}
    missing = [row["path"] for row in read(ASSET_LOCK)["files"]
               if not (bundle / row["path"]).is_file()
               or (bundle / row["path"]).stat().st_size != row["bytes"]]
    python = ROOT / "runtime/phonetic/python.exe"
    reason = ("separator_code_missing" if missing and all(name.startswith("separator/code/") for name in missing)
              else "models_missing" if missing else "runtime_missing" if not python.is_file() else "ready")
    return {**result, "available": not missing and python.is_file(), "reason": reason,
            "missing": missing, "python": str(python)}


def prepare(params: dict[str, Any], cancel: threading.Event, progress: Callable, *, opener=None) -> dict[str, Any]:
    """Explicit user action only: obtain four pinned upstream separator files."""
    import urllib.request
    from urllib.parse import urlsplit
    from .model_install import _download, _copy_verified, inside, install_lock, matches

    if set(params) - {"bundle_dir", "proxy"}:
        raise ValueError("cantonese4_prepare accepts bundle_dir and proxy")
    ready = status({"bundle_dir": params.get("bundle_dir")})
    if ready["reason"] == "models_missing":
        raise FileNotFoundError("Select the extracted Cantonese model bundle first")
    proxy = (params.get("proxy") or "").strip()
    if proxy and (urlsplit(proxy).scheme not in {"http", "https"} or not urlsplit(proxy).hostname):
        raise ValueError("Proxy must be an HTTP(S) URL")
    bundle = Path(ready["bundle_dir"])
    cache = model_cache(ROOT) / "cantonese4-code"
    cache.mkdir(parents=True, exist_ok=True)
    with install_lock(ROOT):
        for row in read(ASSET_LOCK)["files"]:
            if not row["path"].startswith("separator/code/"):
                continue
            check_cancel(cancel)
            target = inside(bundle, row["path"])
            if matches(target, row):
                continue
            if opener is None:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy})
                                                     if proxy else urllib.request.ProxyHandler())
            blob = _download({**row, "offline_name": Path(row["path"]).name}, cache, cancel, progress, opener)
            with blob.open("rb") as stream:
                _copy_verified(stream, target, row, cancel, progress)
    return status({"bundle_dir": str(bundle)})


def validate(params: dict[str, Any]) -> dict[str, Any]:
    if set(params) - {"audio", "bundle_dir", "scope", "start", "duration", "output_dir"}:
        raise ValueError("cantonese4 accepts only audio, bundle_dir, scope, start, duration, output_dir")
    audio = Path(params["audio"]).resolve()
    info = audio_info(audio)
    if info["channels"] not in (1, 2) or not 0 < info["duration"] <= 420:
        raise ValueError("Cantonese enhancement requires a mono/stereo PCM WAV up to 420 seconds")
    scope = params.get("scope", "whole_file")
    if scope not in {"whole_file", "clip"}:
        raise ValueError("Choose whole_file or clip")
    start = 0.0 if scope == "whole_file" else finite(params.get("start", 0), "start")
    duration = info["duration"] if scope == "whole_file" else finite(params.get("duration", 8), "duration")
    if start < 0 or duration <= 0 or start + duration > info["duration"] + 1e-6:
        raise ValueError("Selected audio interval is outside the WAV")
    if scope == "clip" and not .1 <= duration <= 20:
        raise ValueError("Select a 0.1–20 second clip")
    return {**params, "audio": str(audio), "scope": scope, "start": start, "duration": duration}


def summarize(output: Path, result: dict[str, Any]) -> dict[str, Any]:
    return {"result_path": str(output / "result.json"), "output_dir": str(output),
            "jyutping": result["jyutping"], "ipa": result["ipa"],
            "window_count": len(result["windows"]), "unknown_count": result["unknown_count"]}


def save_predictions(output: Path, windows: list[dict[str, Any]], *, source: dict[str, Any],
                     mode: str, cancel: threading.Event) -> dict[str, Any]:
    """Keep complete raw rows; project selected syllables without inventing times."""
    from any2xsampa.adapters.cantonese4_lexicon import choose_prediction, project
    from any2xsampa.exporters.ipa_xsampa import IpaXSampaConverter

    rows, texts, phones, unknown_count = [], [], [], 0
    previous_end = -1.0
    for index, window in enumerate(windows):
        check_cancel(cancel)
        start, end = (finite(window[k], k) for k in ("start_sec", "end_sec"))
        if not 0 <= start < end or start < previous_end or end - start > 18.000001:
            raise ValueError("Invalid or overlapping candidate window clock")
        previous_end = end
        small, base = window.get("small"), window.get("base")
        choice = choose_prediction(small, base)
        selected = small if choice["selected_model"] == "small" else base
        if choice["selected_model"] is None:
            text = "<UNK:completion>"
            converted = {"phones": [text], "unknown_tokens": [text], "syllables": [], "scored_tokens": [text]}
        else:
            text = selected["decoded"]
            if not isinstance(text, str):
                raise ValueError("Selected prediction must retain decoded text")
            converted = project(text)
        rows.append({"index": index, "start_sec": start, "end_sec": end,
                     "time_scope": "model_window_only", "selection": choice,
                     "raw": {"small": small, "base": base}, "jyutping": text, "projection": converted})
        texts.append(text)
        phones.extend(converted["phones"])
        unknown_count += len(converted["unknown_tokens"])
    if not rows:
        raise ValueError("Candidate result must contain every requested window")
    projection = IpaXSampaConverter().project(phones)
    result = {"schema_version": "cantonese4_result_v1", "mode": mode, "source": source,
              "windows": rows, "jyutping": "\n".join(texts), "ipa": " ".join(phones),
              "unknown_count": unknown_count, "canonical_phones": phones,
              "phone_times": None, "tone": "unknown", "timings_are_phone_durations": False,
              "default_model_changed": False, "reference_used": False,
              "xsampa_export": projection, "repetition_cleanup": False,
              "accuracy_scope": "toneless Jyutping on two development songs; not acoustic IPA or note accuracy"}
    write(output / "result.json", result)
    for name, text in (("jyutping.txt", result["jyutping"]), ("phones.ipa.txt", result["ipa"]),
                       ("phones.xsampa.txt", " ".join(value or "[UNMAPPED:" + phone + "]"
                        for value, phone in zip(projection["xsampa"], phones, strict=True)))):
        (output / name).write_text(text + "\n", encoding="utf-8")
    return result


def run(params: dict[str, Any], cancel: threading.Event, progress: Callable) -> dict[str, Any]:
    from .process import WorkerClient

    params = validate(params)
    check_cancel(cancel)
    ready = status({"bundle_dir": params.get("bundle_dir")})
    if not ready["available"]:
        raise FileNotFoundError("Cantonese models are not ready; select the extracted Cantonese model folder")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(params.get("output_dir") or state_dir(ROOT) / "runs" / ("cantonese4-" + stamp + "-" + uuid.uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "request.json", {"params": params, "bundle_dir": ready["bundle_dir"], "retry": False})
    worker = None
    try:
        progress({"stage": "validating"})
        worker = WorkerClient(ready["python"], output / "worker.log", runtime=False,
                              module="any2xsampa.local.cantonese4_worker")
        response = worker.call("cantonese4", {**params, "bundle_dir": ready["bundle_dir"],
                               "_output": str(output)}, cancel, progress, timeout=1800)
        result = read(output / "result.json")
        if response["result_sha256"] != sha(output / "result.json") or result["schema_version"] != "cantonese4_result_v1":
            raise ValueError("Cantonese result identity mismatch")
        check_cancel(cancel)
        write(output / "EXIT.json", {"status": "completed", "retried": False})
        return summarize(output, result)
    except BaseException as exc:
        write(output / "EXIT.json", {"status": "cancelled" if cancel.is_set() else "failed",
                                     "error": str(exc), "retried": False})
        raise
    finally:
        if worker:
            worker.close()


def replay(params: dict[str, Any], cancel: threading.Event, progress: Callable) -> dict[str, Any]:
    """Replay external raw outcomes explicitly; never infer from a text fixture."""
    if set(params) - {"fixture", "output_dir"}:
        raise ValueError("cantonese4_replay accepts fixture and output_dir")
    fixture = Path(params["fixture"]).resolve()
    data = read(fixture)
    if data.get("schema_version") != "cantonese4_raw_v1":
        raise ValueError("Expected cantonese4_raw_v1")
    output = Path(params["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    check_cancel(cancel)
    progress({"stage": "saving"})
    result = save_predictions(output, data["windows"], source={"fixture_sha256": sha(fixture),
                              "model_forwards": 0}, mode="replay", cancel=cancel)
    return summarize(output, result)

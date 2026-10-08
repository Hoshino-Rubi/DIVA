"""GAME score jobs and saved-note exports for the shared local engine."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

from .common import ROOT, Cancelled, audio_info, check_cancel, read, sha, write
from .generated_vsqx import export_generated_score, validate_grid
from .note_score import load_prediction, score_from_notes
from .note_windows import note_windows, stitch_note_windows
from .process import WorkerClient


def runtime_config(snapshot: str | None = None) -> tuple[Path, str]:
    config_file = Path(os.environ.get("ANY2XSAMPA_NOTES", ROOT / "local-notes.json"))
    config = read(config_file) if config_file.is_file() else {}
    if config and config.get("schema_version") != "local_note_runtime_v1":
        raise ValueError("Unsupported local note runtime config")
    directory = snapshot or os.environ.get("ANY2XSAMPA_GAME_SNAPSHOT") or config.get("snapshot")
    directory = Path(directory) if directory else ROOT / "models/game"
    if not directory.is_absolute():
        directory = (config_file.parent if config else ROOT) / directory
    if not (directory / "snapshot-lock.json").is_file():
        raise FileNotFoundError("GAME is not configured. Set --snapshot, ANY2XSAMPA_GAME_SNAPSHOT or local-notes.json; install it with models install or configure a local snapshot; no inference fallback.")
    from any2xsampa.music.game_notes import load_snapshot
    load_snapshot(directory)
    python = Path(config.get("python", sys.executable))
    if not python.is_absolute():
        python = config_file.parent / python
    if not python.is_file():
        raise FileNotFoundError("Configured GAME Python runtime is missing")
    return directory.resolve(), str(python.resolve())


def _transcribe(result: dict, output: Path, snapshot: str | None, cancel, progress) -> dict:
    directory, python = runtime_config(snapshot)
    info = result["input"]
    audio = Path(info["path"])
    rate = audio_info(audio)["sample_rate"]
    windows = note_windows(info["start_sec"], info["end_sec"], rate)
    predictions, records = [], []
    write(output / "NOTE_PLAN.json", {"windows": windows, "maximum_encoder_calls": len(windows),
              "maximum_graph_calls": 11 * len(windows), "device": "cpu", "automatic_retry": False})
    worker = WorkerClient(python, output / "game-worker.log", runtime=False,
                          module="any2xsampa.local.game_worker")
    try:
        for window in windows:
            check_cancel(cancel)
            progress({"stage": "note_transcription", "chunk_index": window["index"]+1,
                      "chunk_total": len(windows), "processed_seconds": window["keep_start_sec"]-info["start_sec"],
                      "total_seconds": info["end_sec"]-info["start_sec"], "device": "cpu"})
            destination = output / "windows" / f"{window['index']:05d}"
            response = worker.call("transcribe_notes", {"audio_path": str(audio),
                  "audio_sha256": info["sha256"], "start_sec": window["start_sec"],
                  "end_sec": window["end_sec"], "snapshot": str(directory),
                  "output_dir": str(destination)}, cancel, progress, timeout=180)
            file = destination / "prediction.json"
            if Path(response["prediction_path"]).resolve() != file.resolve() or sha(file) != response["prediction_sha256"]:
                raise ValueError("GAME worker returned mismatched output")
            prediction, digest = load_prediction(file)
            if (prediction["input"]["audio_sha256"] != info["sha256"]
                    or abs(prediction["source_start_sec"] - window["start_sec"]) > 1/rate
                    or abs(prediction["source_end_sec"] - window["end_sec"]) > 1/rate + 1/44100):
                raise ValueError("GAME worker returned a different source interval")
            predictions.append(prediction)
            records.append({"path": file.relative_to(output).as_posix(), "sha256": digest, "window": window})
    finally:
        worker.close()
    check_cancel(cancel)
    if sha(audio) != info["sha256"]:
        raise ValueError("Audio changed during note inference")
    combined = {"schema_version": "note_transcription_v1", "mode": "real", "time_basis": "original_audio",
        "model_id": predictions[0]["model_id"], "source_start_sec": info["start_sec"],
        "source_end_sec": info["end_sec"], "input": {"audio_path": str(audio),
        "audio_sha256": info["sha256"], "source_sample_rate": rate},
        "notes": stitch_note_windows(predictions, windows), "chunks": records,
        "seam_policy": "midpoint_time_ownership_join_only_continuing_same_pitch_v1",
        "known_reference_boundaries_used": False, "automatic_pitch_transpose": False,
        "quality_validated": False, "model_forwards": len(windows), "graph_calls": 11*len(windows)}
    write(output / "note-prediction.json", combined)
    return load_prediction(output / "note-prediction.json")[0]


def build_game_score(params: dict, cancel, progress) -> dict:
    from .score_tasks import _bound_result, _output

    if set(params) - {"result_path", "result_sha256", "tempo", "output_dir", "snapshot", "notes_path"}:
        raise ValueError("Unexpected GAME score parameters")
    if params.get("snapshot") and params.get("notes_path"):
        raise ValueError("Choose saved notes or live GAME snapshot, not both")
    file, digest, result = _bound_result(params)
    tempo = params.get("tempo", 120)
    validate_grid(tempo, 480)
    check_cancel(cancel)
    # Configuration errors precede all model work.
    if not params.get("notes_path"):
        runtime_config(params.get("snapshot"))
    output = _output(params)
    started = time.monotonic()
    write(output / "REQUEST.json", {"command": "build_game_score", "params": params, "result_sha256": digest})
    try:
        if params.get("notes_path"):
            prediction_file = Path(params["notes_path"]).resolve()
            prediction, prediction_hash = load_prediction(prediction_file)
            forwards, calls = 0, 0
        else:
            prediction = _transcribe(result, output, params.get("snapshot"), cancel, progress)
            prediction_file = output / "note-prediction.json"
            prediction_hash = sha(prediction_file)
            forwards, calls = prediction["model_forwards"], prediction["graph_calls"]
        check_cancel(cancel)
        if sha(file) != digest:
            raise ValueError("Phone result changed during note transcription")
        score = score_from_notes(prediction, result, tempo)
        score.update(result_path=str(file), result_sha256=digest,
                     note_prediction_path=str(prediction_file), note_prediction_sha256=prediction_hash)
        write(output / "score.json", score)
        progress({"stage": "exporting_vsqx", "note_count": len(score["notes"])})
        exported = export_generated_score(output / "score.json", output / "transcription.vsqx")
        response = {"score_path": str(output / "score.json"), "score_sha256": sha(output / "score.json"),
            "vsqx_path": exported["output"], "vsqx_sha256": sha(Path(exported["output"])),
            "source": "audio_game", "note_count": len(score["notes"]),
            "placeholder_note_count": score["placeholder_note_count"],
            "model_forwards": forwards, "graph_calls": calls}
        write(output / "DELIVERY.json", response)
        write(output / "EXIT.json", {"status": "completed", **response, "retried": False,
                                    "elapsed_seconds": time.monotonic()-started})
        return response
    except BaseException as exc:
        write(output / "EXIT.json", {"status": "cancelled" if isinstance(exc, Cancelled) else "failed",
            "error": str(exc), "retried": False, "elapsed_seconds": time.monotonic()-started,
            "per_window_accounting": "windows/*/START.json, FORWARD_STARTED.json, EXIT.json"})
        raise


def export_saved_notes(params: dict) -> dict:
    """Export note JSON / generated score / result plus notes with zero forwards."""
    allowed = {"result_path", "output_path", "notes_path", "phones_path", "tempo"}
    if set(params) - allowed:
        raise ValueError("Unexpected saved-note export parameters")
    source_file = Path(params["result_path"]).resolve()
    source = read(source_file)
    output = Path(params["output_path"]).resolve()
    if source.get("schema_version") == "local_note_score_v1":
        if params.get("notes_path") or params.get("phones_path") or "tempo" in params:
            raise ValueError("A saved score already defines its notes, phones and tempo")
        return export_generated_score(source_file, output)
    if source.get("schema_version") == "note_transcription_v1":
        if params.get("notes_path"):
            raise ValueError("Do not specify two note predictions")
        note_file = source_file
        phone_file = Path(params["phones_path"]).resolve() if params.get("phones_path") else None
    elif source.get("schema_version") == "local_task_result_v1" and params.get("notes_path"):
        if params.get("phones_path"):
            raise ValueError("Phone result was already supplied")
        note_file, phone_file = Path(params["notes_path"]).resolve(), source_file
    else:
        raise ValueError("Supply a score.json, note prediction.json, or result.json with --notes; no model is run by export")
    prediction, digest = load_prediction(note_file)
    phone_digest = sha(phone_file) if phone_file else None
    score = score_from_notes(prediction, read(phone_file) if phone_file else None, params.get("tempo", 120))
    score.update(note_prediction_path=str(note_file), note_prediction_sha256=digest,
                 result_path=str(phone_file) if phone_file else None, result_sha256=phone_digest)
    score_path = output.with_name(output.name + ".score.json")
    if any(p.exists() for p in (output, score_path, output.with_name(output.name + ".provenance.json"))):
        raise FileExistsError("Refusing to overwrite VSQX or its source records")
    if sha(note_file) != digest or (phone_file and sha(phone_file) != phone_digest):
        raise ValueError("Source evidence changed during export")
    write(score_path, score)
    return export_generated_score(score_path, output)

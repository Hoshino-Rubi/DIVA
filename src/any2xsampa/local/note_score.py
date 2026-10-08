"""Convert saved audio-derived note evidence to a local score, without inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import finite, read, sha
from .score_export import make_score, note_number


def validate_notes(prediction: dict[str, Any]) -> None:
    if prediction.get("schema_version") != "note_transcription_v1":
        raise ValueError("Expected note_transcription_v1")
    if prediction.get("time_basis") != "original_audio":
        raise ValueError("Notes must use their source audio clock")
    start = finite(prediction["source_start_sec"], "source_start_sec")
    end = finite(prediction["source_end_sec"], "source_end_sec")
    if start < 0 or end <= start:
        raise ValueError("Invalid note source interval")
    previous = start
    for row in prediction["notes"]:
        begin = finite(row["start_sec"], "note start")
        finish = finite(row["end_sec"], "note end")
        duration = finite(row["duration_sec"], "note duration")
        note_number(row["pitch_midi"])
        if row.get("presence") is not True or finish <= begin:
            raise ValueError("Expected a present, positive-duration note")
        if begin < previous - 1e-8 or finish > end + 1e-8:
            raise ValueError("Overlapping, unordered or out-of-range notes")
        if abs((finish - begin) - duration) > 1e-8:
            raise ValueError("Note duration disagrees with its onset/offset")
        previous = finish


def load_prediction(file: Path) -> tuple[dict[str, Any], str]:
    digest = sha(file)
    prediction = read(file)
    validate_notes(prediction)
    if "raw_evidence" not in prediction and "chunks" not in prediction and prediction.get("mode") != "fixture":
        raise ValueError("Real note predictions require raw evidence or bound chunks; use explicit fixture mode for external notes")
    source = prediction["input"]
    audio = Path(source["audio_path"])
    if sha(audio) != source["audio_sha256"]:
        raise ValueError("Note source audio hash mismatch")
    if "raw_evidence" in prediction:
        import numpy as np
        from any2xsampa.music.game_notes import decode_regions

        binding = prediction["raw_evidence"]
        raw = (file.parent / binding["file"]).resolve()
        if not raw.is_relative_to(file.parent.resolve()) or sha(raw) != binding["sha256"]:
            raise ValueError("Raw note evidence path/hash mismatch")
        with np.load(raw, allow_pickle=False) as arrays:
            decoded = decode_regions(dict(arrays), start_sec=prediction["source_start_sec"],
                                     duration_sec=source["model_duration_sec"],
                                     timestep=prediction["timestep_sec"])
        if decoded["notes"] != prediction["notes"] or decoded["regions"] != prediction["regions"]:
            raise ValueError("Saved notes differ from raw GAME evidence")
    if "effective_start_sec" in source:
        rate = finite(source["source_sample_rate"], "sample rate")
        if rate <= 0 or abs(prediction["source_start_sec"] - source["effective_start_sec"]) > 1e-8:
            raise ValueError("Note clock disagrees with source crop")
        if abs(prediction["source_end_sec"] - source["effective_end_sec"]) > 1/rate + 1/44100:
            raise ValueError("Note clock does not cover source crop")
    if "chunks" in prediction:
        from .note_windows import note_windows, stitch_note_windows
        expected = note_windows(prediction["source_start_sec"], prediction["source_end_sec"],
                                source["source_sample_rate"])
        records = prediction["chunks"]
        if (prediction.get("seam_policy") != "midpoint_time_ownership_join_only_continuing_same_pitch_v1"
                or [c["window"] for c in records] != expected):
            raise ValueError("Whole-note coverage/window policy changed")
        children = []
        identity = None
        for record, window in zip(records, expected, strict=True):
            child = (file.parent / record["path"]).resolve()
            if not child.is_relative_to(file.parent.resolve()) or child == file.resolve():
                raise ValueError("Note chunk must be a child artifact")
            payload = read(child)
            if "chunks" in payload or sha(child) != record["sha256"]:
                raise ValueError("Invalid or changed note chunk")
            payload, _ = load_prediction(child)
            if (payload["input"]["audio_sha256"] != source["audio_sha256"]
                    or abs(payload["source_start_sec"]-window["start_sec"]) > 1e-8
                    or abs(payload["source_end_sec"]-window["end_sec"]) > 1e-4):
                raise ValueError("Note chunk source interval mismatch")
            current = tuple(payload.get(k) for k in ("model_id", "snapshot_lock_sha256", "policy_sha256", "language"))
            if identity is not None and identity != current:
                raise ValueError("Note model changed between windows")
            identity = current
            children.append(payload)
        if stitch_note_windows(children, expected) != prediction["notes"]:
            raise ValueError("Combined notes differ from independently replayed chunks")
    if sha(file) != digest:
        raise ValueError("Note prediction changed while reading")
    return prediction, digest


def score_from_notes(prediction: dict[str, Any], phones: dict[str, Any] | None,
                     tempo: float = 120) -> dict[str, Any]:
    """Phone evidence may label notes only; it never determines pitch or timing."""
    validate_notes(prediction)
    source = prediction["input"]
    start, end = prediction["source_start_sec"], prediction["source_end_sec"]
    info = {"path": source["audio_path"], "sha256": source["audio_sha256"],
            "start_sec": start, "end_sec": end}
    if phones is None:
        result = {"input": info, "segments": []}
    else:
        if phones.get("schema_version") != "local_task_result_v1":
            raise ValueError("Expected local_task_result_v1 phone evidence")
        if phones.get("time_basis") != "original_audio":
            raise ValueError("Phone evidence must use original audio time")
        p = phones["input"]
        if p["sha256"] != info["sha256"] or Path(p["path"]).resolve() != Path(info["path"]).resolve():
            raise ValueError("Phone and note evidence belong to different audio")
        if p["start_sec"] > start + 1e-6 or p["end_sec"] < end - 1e-6:
            raise ValueError("Phone evidence does not cover the note interval")
        segments = []
        for segment in phones["segments"]:
            a = finite(segment["start_sec"], "phone start")
            b = finite(segment["end_sec"], "phone end")
            if a < p["start_sec"] or b > p["end_sec"] or b <= a:
                raise ValueError("Invalid phone emission interval")
            if not isinstance(segment["normalized_ipa"], str) or not segment["normalized_ipa"].strip():
                raise ValueError("Empty phone label")
            if start <= (a + b) / 2 < end:
                segments.append(segment)
        result = {**phones, "input": info, "segments": segments}
    pitch = {"note_candidates": [
        {"start_sec": n["start_sec"], "duration_sec": n["duration_sec"], "midi": n["pitch_midi"]}
        for n in prediction["notes"]
    ]}
    score = make_score(result, pitch, tempo)
    for row, original in zip(score["notes"], prediction["notes"], strict=True):
        row.update(pitch_source="GAME", note_evidence=dict(original))
    score.update(source="audio_game", time_basis="original_audio",
                 note_model_id=prediction.get("model_id"),
                 note_evidence_mode=prediction.get("mode", "unspecified"),
                 quality_validated=False, manual_review_required=False,
                 alignment_status="automatic_emission_to_note_assignment",
                 phone_evidence_supplied=phones is not None,
                 seam_policy=prediction.get("seam_policy"),
                 model_forwards=0)
    return score

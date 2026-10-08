"""Offline, prediction-blind original-note review and strict saved-output evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.signal import stft
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PAIR_RUN = ROOT / ".state/runs/game-paired-input-20261002T223405653Z"
CLOCK_RUN = ROOT / ".state/runs/note-correspondence-20261002T125911Z"
GUI = ROOT / "tools/note_joint_review_gui"
SONGS = (
    ("freedom-flower", "自由花", "batch-r05"),
    ("queen-road-east", "皇后大道東", "batch-n01"),
    ("half-moon-night-song", "月半小夜曲", "batch-n05"),
    ("promise", "约定", "batch-r11"),
    ("like-you", "喜欢你", "batch-c05"),
)
LOCKS = {
    PAIR_RUN / "RESULTS.json": "1d090f69d2c4e28751e829cc0677718f856b061ef4eff4ed0c0f47b79264517d",
    PAIR_RUN / "LAUNCH_REVIEW.json": "0f76a86083a48e08a9616a57dee1bc92f1ffdf11474e04aa53e3f8bdf652795e",
    PAIR_RUN / "REPORT.md": "140d03ee7731fc7644d4919dac91f2c8b7823be14008e733e938ddbdd6e66529",
    PAIR_RUN / "regression.xml": "821fcdc463e1b5c94e11dc0c6532e63f7b9985b6b1b3e73e00ccb80edaee93a0",
    CLOCK_RUN / "SOURCE_CLOCKS.json": "19b11945605dadfbca0743cce5409f45aa62fb68665e11fecbb794be1a8b7607",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checked(path: Path, expected: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}: {actual} != {expected}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _integer(value: object, name: str, *, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) > 4000:
        raise ValueError(f"{name} must be text <= 4000 characters")
    return value


def _checked_sources() -> tuple[list[dict], dict]:
    for path, expected in LOCKS.items():
        checked(path, expected)
    clocks = read_json(CLOCK_RUN / "SOURCE_CLOCKS.json")
    if clocks["formal_windows"] != 0:
        raise ValueError("Frozen source clock unexpectedly changed")
    results = read_json(PAIR_RUN / "RESULTS.json")
    jobs = read_json(PAIR_RUN / "JOBS.json")
    precheck = read_json(PAIR_RUN / "PRECHECK.json")
    launch = read_json(PAIR_RUN / "LAUNCH_COMPLETED.json")
    if (launch["completed_jobs"], launch["graph_calls"], launch["training_updates"]) != (11, 121, 0):
        raise ValueError("Old GAME job accounting changed")
    wanted = {(song, example, role) for song, _, example in SONGS for role in ("mixed", "vocal")}
    rows = {(row["song_id"], row["example_id"], row["role"]): row for row in results["rows"]}
    if set(rows) != wanted or len(jobs["jobs"]) != 11:
        raise ValueError("Expected exactly five training-song pairs and one smoke")
    pairs = {row["song_id"]: row for row in precheck["pairs"]}
    if set(pairs) != {song for song, _, _ in SONGS}:
        raise ValueError("Paired audio does not cover the five frozen training songs")
    verified = []
    for song, title, example in SONGS:
        pre = pairs[song]
        if pre["example_id"] != example:
            raise ValueError("Audio pair example mismatch")
        source = None
        paths = {}
        for role in ("mixed", "vocal"):
            row = rows[song, example, role]
            job_dir = PAIR_RUN / "jobs" / f"{example}-{role}"
            exit_data = read_json(job_dir / "EXIT.json")
            audio_data = read_json(job_dir / "AUDIO_READY.json")
            prediction = job_dir / "prediction.json"
            raw = job_dir / "raw-notes.npz"
            if (exit_data["status"], exit_data["encoder_calls"], exit_data["graph_calls"],
                exit_data["optimizer_updates"]) != ("completed", 1, 11, 0):
                raise ValueError(f"Incomplete or unexpected model job: {job_dir}")
            checked(prediction, row["prediction_sha256"])
            checked(raw, row["raw_sha256"])
            if exit_data["prediction_sha256"] != row["prediction_sha256"]:
                raise ValueError("EXIT/prediction disagreement")
            if len(read_json(prediction)["notes"]) != row["notes"]:
                raise ValueError("Frozen note count disagrees with prediction")
            audio_path = Path(audio_data["audio_path"])
            checked(audio_path, pre["hashes"][f"{role}_wav"])
            if audio_data["audio_sha256"] != pre["hashes"][f"{role}_wav"]:
                raise ValueError("AUDIO_READY SHA mismatch")
            if (audio_data["source_sample_rate"], audio_data["source_start_sample"],
                audio_data["source_end_sample_exclusive"]) != (44100, 0, 441000):
                raise ValueError("Model audio window changed")
            pair_path = audio_path.parent.parent / "pair.json"
            checked(pair_path, pre["hashes"]["pair_json"])
            pair = read_json(pair_path)
            if pair["song_id"] != song or pair["example_id"] != example or pair["split"] != "train":
                raise ValueError("Song leakage or wrong pair")
            if Path(pair["audio"][role]["path"]) != audio_path:
                raise ValueError("Job audio does not match pair record")
            if source is not None and source != pair:
                raise ValueError("Mixed and vocal jobs use different pair records")
            source = pair
            paths[role] = {"audio": audio_path, "prediction": prediction, "raw": raw}
        assert source is not None
        checked(Path(source["source_audio_path"]), pre["hashes"]["original_source"])
        if source["source_audio_sha256"] != pre["hashes"]["original_source"]:
            raise ValueError("Original audio binding changed")
        start = source["source_start_sample"]
        end = source["source_end_sample_exclusive"]
        rate = source["source_rate"]
        if (end - start) != 10 * rate:
            raise ValueError("Original audio interval is not exactly ten seconds")
        if abs(start / rate - pre["source_interval_sec"][0]) > 1 / rate:
            raise ValueError("Original audio sample clock does not match precheck")
        verified.append({"song_id": song, "title": title, "example_id": example,
                         "source": source, "paths": paths, "precheck": pre})
    return verified, results


def _waveform(samples: np.ndarray, width: int = 1200) -> list[list[float]]:
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    edges = np.linspace(0, mono.size, width + 1, dtype=int)
    return [[round(float(mono[edges[i]:edges[i + 1]].min()), 4),
             round(float(mono[edges[i]:edges[i + 1]].max()), 4)] for i in range(width)]


def _spectrogram(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    freqs, _, spectrum = stft(mono, sample_rate, nperseg=4096, noverlap=3840,
                              boundary=None, padded=False)
    # Fixed 36..84 MIDI log-frequency display; this is visualization, not F0 inference.
    midi = 84 - (np.arange(588) + .5) / 12
    target = 440 * 2 ** ((midi - 69) / 12)
    strength = np.abs(spectrum)
    bins = np.array([np.interp(target, freqs, strength[:, col])
                     for col in range(strength.shape[1])], dtype=np.float32).T
    db = 20 * np.log10(np.maximum(bins, 1e-8))
    level = np.clip((db + 78) / 66, 0, 1)
    rgb = np.stack((12 + 205 * level, 28 + 181 * level, 33 + 101 * level), axis=-1)
    Image.fromarray(np.uint8(rgb), "RGB").resize((1200, 588), Image.Resampling.BILINEAR).save(path)


def build(output: Path) -> dict:
    verified, results = _checked_sources()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    clips, provenance = [], []
    for entry in verified:
        song = entry["song_id"]
        source = entry["source"]
        preview = {}
        waveforms = {}
        for role in ("mixed", "vocal"):
            audio, rate = sf.read(entry["paths"][role]["audio"], dtype="float32", always_2d=True)
            if rate != 44100 or audio.shape != (441000, 2) or not np.isfinite(audio).all():
                raise ValueError("Invalid reviewed-audio source")
            name = f"{song}-{role}.wav"
            sf.write(media / name, audio, rate, subtype="PCM_16")
            preview[role] = {"file": f"media/{name}", "sha256": digest(media / name),
                             "source_sha256": entry["precheck"]["hashes"][f"{role}_wav"]}
            waveforms[role] = _waveform(audio)
            spec_name = f"{song}-{role}.png"
            _spectrogram(audio, rate, media / spec_name)
            preview[role]["spectrogram_file"] = f"media/{spec_name}"
            preview[role]["spectrogram_sha256"] = digest(media / spec_name)
        clips.append({"song_id": song, "title": entry["title"], "example_id": entry["example_id"],
                      "source_audio_sha256": source["source_audio_sha256"],
                      "source_start_sample": source["source_start_sample"],
                      "source_end_sample_exclusive": source["source_end_sample_exclusive"],
                      "source_sample_rate": source["source_rate"],
                      "duration_sec": 10, "preview": preview, "waveforms": waveforms})
        provenance.append({"song_id": song, "example_id": entry["example_id"],
                           "pair_json": str(entry["paths"]["mixed"]["audio"].parent.parent / "pair.json"),
                           "pair_sha256": entry["precheck"]["hashes"]["pair_json"],
                           "predictions": {role: {"path": str(entry["paths"][role]["prediction"]),
                                                   "sha256": digest(entry["paths"][role]["prediction"]),
                                                   "raw_sha256": digest(entry["paths"][role]["raw"])}
                                           for role in ("mixed", "vocal")}})
    packet = {"schema_version": "original_note_blind_packet_v1", "clips": clips,
              "preview_only": True, "game_predictions_in_packet": False,
              "formal_original_note_truth_created": False}
    write_json(output / "packet.json", packet)
    packet_hash = digest(output / "packet.json")
    template = (GUI / "index.html").read_text(encoding="utf-8")
    for marker, filename in (("/* INLINE_CSS */", "style.css"),
                             ("/* INLINE_STATE */", "state.js"),
                             ("/* INLINE_APP */", "app.js")):
        if template.count(marker) != 1:
            raise ValueError(f"GUI template marker mismatch: {marker}")
        template = template.replace(marker, (GUI / filename).read_text(encoding="utf-8"))
    packet_text = (output / "packet.json").read_text(encoding="utf-8").replace("</", "<\\/")
    template = template.replace("/* INLINE_PACKET */", packet_text)
    template = template.replace("__PACKET_SHA256__", packet_hash)
    (output / "index.html").write_text(template, encoding="utf-8")
    code_paths = [Path(__file__), *(GUI / p for p in ("index.html", "style.css", "state.js", "app.js"))]
    manifest = {"schema_version": "original_note_review_build_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "packet_sha256": packet_hash, "html_sha256": digest(output / "index.html"),
                "source_code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in code_paths},
                "locked_inputs": {str(p.relative_to(ROOT)): hash_value for p, hash_value in LOCKS.items()},
                "provenance": provenance, "review_mode": "blinded_original_audio_only",
                "model_forwards": 0, "training_updates": 0, "downloads": 0,
                "historical_reference_note_count": results["gates"]["vocal"]["reference_notes"],
                "status": "awaiting_single_concentrated_human_review"}
    write_json(output / "BUILD.json", manifest)
    return manifest


def validate_answers(packet: dict, packet_sha256: str, answer: dict) -> dict:
    if answer.get("schema_version") != "original_note_review_v1" or answer.get("packet_sha256") != packet_sha256:
        raise ValueError("Review schema or packet SHA mismatch")
    _string(answer.get("reviewer"), "reviewer")
    _string(answer.get("raw_statement"), "raw_statement")
    if not isinstance(answer.get("clips"), list) or len(answer["clips"]) != len(packet["clips"]):
        raise ValueError("All five clips must be present")
    qualified, counts = [], []
    for clip, response in zip(packet["clips"], answer["clips"], strict=True):
        if not isinstance(response, dict):
            raise ValueError("Invalid clip response")
        for key in ("song_id", "example_id", "source_audio_sha256", "source_start_sample", "source_sample_rate"):
            if response.get(key) != clip[key]:
                raise ValueError(f"Clip source binding mismatch: {clip['song_id']}/{key}")
        state = response.get("review_state")
        if state not in ("pending", "uncertain", "complete"):
            raise ValueError("Invalid review state")
        if type(response.get("all_notes_reviewed")) is not bool:
            raise ValueError("all_notes_reviewed must be explicit")
        if type(response.get("silent_confirmed")) is not bool:
            raise ValueError("silent_confirmed must be explicit")
        _string(response.get("raw_statement"), "clip raw_statement")
        notes = response.get("notes")
        intervals = response.get("uncertain_intervals")
        if not isinstance(notes, list) or not isinstance(intervals, list):
            raise ValueError("Notes and uncertain intervals must be arrays")
        lower, upper = clip["source_start_sample"], clip["source_end_sample_exclusive"]
        seen, complete_notes = set(), []
        for event in notes:
            if not isinstance(event, dict):
                raise ValueError("Invalid note event")
            event_id = _string(event.get("id"), "note id")
            if not event_id or event_id in seen:
                raise ValueError("Duplicate or empty note ID")
            seen.add(event_id)
            start = _integer(event.get("start_sample_abs"), "note onset", lower=lower, upper=upper - 1)
            end = _integer(event.get("end_sample_abs"), "note offset", lower=start + 1, upper=upper)
            _integer(event.get("pitch_midi"), "pitch MIDI", lower=0, upper=127)
            _string(event.get("raw_statement"), "event raw_statement")
            _string(event.get("overlap_reason"), "overlap_reason")
            if event.get("review_state") not in ("draft", "confirmed", "uncertain"):
                raise ValueError("Invalid note review state")
            edge = event.get("edge_status")
            if edge not in ("fully_audible", "left_censored", "right_censored", "both_censored", "uncertain"):
                raise ValueError("Invalid note edge status")
            if edge in ("left_censored", "both_censored") and start != lower:
                raise ValueError("Left censor must touch window start")
            if edge in ("right_censored", "both_censored") and end != upper:
                raise ValueError("Right censor must touch window end")
            if event["review_state"] == "confirmed" and edge == "fully_audible":
                complete_notes.append(event)
        ordered = sorted(notes, key=lambda item: (item["start_sample_abs"], item["end_sample_abs"]))
        for left, right in zip(ordered, ordered[1:]):
            if right["start_sample_abs"] < left["end_sample_abs"] and not (left["overlap_reason"] or right["overlap_reason"]):
                raise ValueError("Overlapping notes require a stated reason")
        for interval in intervals:
            if not isinstance(interval, dict):
                raise ValueError("Invalid uncertain interval")
            start = _integer(interval.get("start_sample_abs"), "uncertain start", lower=lower, upper=upper - 1)
            _integer(interval.get("end_sample_abs"), "uncertain end", lower=start + 1, upper=upper)
            if not _string(interval.get("reason"), "uncertainty reason").strip():
                raise ValueError("Uncertain intervals require a reason")
        if response["silent_confirmed"] and notes:
            raise ValueError("Confirmed silence cannot contain notes")
        formal = bool(state == "complete" and response["all_notes_reviewed"]
                      and (notes or response["silent_confirmed"]) and not intervals
                      and all(event["review_state"] == "confirmed"
                              and event["edge_status"] == "fully_audible" for event in notes)
                      and answer["reviewer"].strip())
        qualified.append(formal)
        counts.append({"song_id": clip["song_id"], "review_state": state,
                       "notes": len(notes), "confirmed_complete_notes": len(complete_notes),
                       "censored_or_uncertain_notes": len(notes) - len(complete_notes),
                       "uncertain_intervals": len(intervals), "formal_window": formal})
    return {"formal_windows": sum(qualified), "all_five_formal": all(qualified), "coverage": counts}


def validate_file(packet_dir: Path, answer_path: Path) -> dict:
    packet_path = packet_dir / "packet.json"
    build_data = read_json(packet_dir / "BUILD.json")
    checked(packet_path, build_data["packet_sha256"])
    packet = read_json(packet_path)
    for clip in packet["clips"]:
        for preview in clip["preview"].values():
            checked(packet_dir / preview["file"], preview["sha256"])
            checked(packet_dir / preview["spectrogram_file"], preview["spectrogram_sha256"])
    return validate_answers(packet, build_data["packet_sha256"], read_json(answer_path))


def _lcs_pitch(reference: list, prediction: list) -> int:
    previous = [0] * (len(prediction) + 1)
    for ref in reference:
        current = [0]
        for index, pred in enumerate(prediction):
            current.append(previous[index] + 1 if ref.pitch_midi == pred.pitch_midi
                           else max(current[-1], previous[index + 1]))
        previous = current
    return previous[-1]


def _scored_predictions(path: Path, source: dict) -> list:
    from any2xsampa.music.note_metrics import NoteEvent

    prediction = read_json(path)
    if prediction["schema_version"] != "note_transcription_v1" or prediction["time_basis"] != "original_audio":
        raise ValueError("Prediction schema or clock changed")
    source_start = source["source_start_sample"] / source["source_rate"]
    result = []
    for note in prediction["notes"]:
        start, end = note["start_sec"], note["end_sec"]
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end > 10 + 1e-8):
            raise ValueError("Prediction outside original window")
        result.append(NoteEvent(source_start + start, source_start + end, note["pitch_midi"]))
    return result


def evaluate(packet_dir: Path, answer_path: Path, output: Path) -> dict:
    from any2xsampa.music.note_metrics import NoteEvent, evaluate_notes, match_notes

    qualification = validate_file(packet_dir, answer_path)
    if not qualification["all_five_formal"]:
        raise ValueError("Five complete original-audio note references required; no partial formal score")
    verified, old_results = _checked_sources()
    packet_path = packet_dir / "packet.json"
    packet = read_json(packet_path)
    answer = read_json(answer_path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frozen = {"schema_version": "original_note_truth_frozen_v1", "packet_sha256": digest(packet_path),
              "answers_sha256": digest(answer_path), "answers": answer,
              "original_source_sha256": {entry["song_id"]: entry["source"]["source_audio_sha256"]
                                         for entry in verified}}
    write_json(output / "FROZEN_REFERENCE.json", frozen)
    frozen_sha = digest(output / "FROZEN_REFERENCE.json")
    rows = []
    for source_clip, human_clip, entry in zip(packet["clips"], answer["clips"], verified, strict=True):
        source = entry["source"]
        reference = [NoteEvent(note["start_sample_abs"] / source["source_rate"],
                               note["end_sample_abs"] / source["source_rate"], note["pitch_midi"])
                     for note in human_clip["notes"]]
        reference.sort(key=lambda note: (note.start_sec, note.end_sec, note.pitch_midi))
        conditions = {}
        for role in ("mixed", "vocal"):
            prediction = _scored_predictions(entry["paths"][role]["prediction"], source)
            metrics = evaluate_notes(reference, prediction, reference_clock_confirmed=True)
            pitch_lcs = _lcs_pitch(reference, prediction)
            onset_only = match_notes(reference, prediction, offset_tolerance=None, require_pitch=False)
            conditions[role] = {"prediction_sha256": digest(entry["paths"][role]["prediction"]),
                                "pitch_sequence_lcs_diagnostic": pitch_lcs,
                                "pitch_only_sequence_error_diagnostic": len(reference) + len(prediction) - 2 * pitch_lcs,
                                "onset_only_matches": len(onset_only), "metrics": metrics}
        rows.append({"song_id": source_clip["song_id"], "reference_notes": len(reference),
                     "historical_reference_notes": next(row["reference_notes"] for row in
                                                        read_json(CLOCK_RUN / "COUNT_CEILING.json")["rows"]
                                                        if row["song_id"] == source_clip["song_id"]),
                     "conditions": conditions})
    totals = {}
    for role in ("mixed", "vocal"):
        ref_total = sum(row["reference_notes"] for row in rows)
        pred_total = sum(row["conditions"][role]["metrics"]["pitch_onset_offset"]["predicted_notes"] for row in rows)
        matches = sum(row["conditions"][role]["metrics"]["pitch_onset_offset"]["matches"] for row in rows)
        totals[role] = {"reference_notes": ref_total, "predicted_notes": pred_total, "joint_matches": matches,
                        "false_positives": pred_total - matches, "false_negatives": ref_total - matches,
                        "precision": matches / pred_total if pred_total else 0,
                        "recall": matches / ref_total if ref_total else 0}
    result = {"schema_version": "paired_game_original_note_joint_evaluation_v1", "reference_sha256": frozen_sha,
              "historical_reference_notes": old_results["gates"]["vocal"]["reference_notes"],
              "historical_vocal_count_ceiling": "105/113; not observed accuracy",
              "song_level_only_matching": True, "automatic_shift_or_warp": False,
              "no_model_forwards": True, "rows": rows, "totals": totals}
    write_json(output / "RESULTS.json", result)
    return result


def diagnose_saved_pairs(output: Path) -> dict:
    """Compare two frozen outputs on their shared clock, without calling either truth."""
    from any2xsampa.music.note_metrics import match_notes

    verified, frozen = _checked_sources()
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for entry in verified:
        mixed = _scored_predictions(entry["paths"]["mixed"]["prediction"], entry["source"])
        vocal = _scored_predictions(entry["paths"]["vocal"]["prediction"], entry["source"])
        joint = match_notes(mixed, vocal)
        pitch_onset = match_notes(mixed, vocal, offset_tolerance=None)
        onset = match_notes(mixed, vocal, offset_tolerance=None, require_pitch=False)
        boundary = match_notes(mixed, vocal, require_pitch=False)
        joint_mixed = {i for i, _ in joint}
        joint_vocal = {j for _, j in joint}
        rows.append({"song_id": entry["song_id"], "example_id": entry["example_id"],
                     "original_source_sha256": entry["source"]["source_audio_sha256"],
                     "original_start_sample": entry["source"]["source_start_sample"],
                     "source_sample_rate": entry["source"]["source_rate"],
                     "mixed_prediction_sha256": digest(entry["paths"]["mixed"]["prediction"]),
                     "vocal_prediction_sha256": digest(entry["paths"]["vocal"]["prediction"]),
                     "mixed_notes": len(mixed), "vocal_notes": len(vocal),
                     "pitch_sequence_lcs": _lcs_pitch(mixed, vocal),
                     "onset_only_agreements_50ms": len(onset),
                     "boundary_only_agreements_50ms": len(boundary),
                     "pitch_onset_agreements_50ms": len(pitch_onset),
                     "joint_agreements_pitch_onset_offset_50ms": len(joint),
                     "mixed_without_joint_agreement": len(mixed) - len(joint),
                     "vocal_without_joint_agreement": len(vocal) - len(joint),
                     "joint_mixed_indices": sorted(joint_mixed),
                     "joint_vocal_indices": sorted(joint_vocal)})
    totals = {name: sum(row[name] for row in rows) for name in (
        "mixed_notes", "vocal_notes", "pitch_sequence_lcs", "onset_only_agreements_50ms",
        "boundary_only_agreements_50ms", "pitch_onset_agreements_50ms",
        "joint_agreements_pitch_onset_offset_50ms", "mixed_without_joint_agreement",
        "vocal_without_joint_agreement")}
    result = {"schema_version": "saved_game_pair_agreement_diagnostic_v1",
              "evaluation_kind": "cross_input_consistency_not_accuracy",
              "original_note_truth_available": False, "formal_joint_accuracy_measured": False,
              "historical_reference_notes": frozen["gates"]["vocal"]["reference_notes"],
              "historical_vocal_count_precision_upper_bound": frozen["gates"]["vocal"]["precision_upper_bound"],
              "user_accepted_continuation_despite_count_gate": True,
              "old_gate_not_rewritten": True, "automatic_time_shift_or_transpose": False,
              "model_forwards": 0, "training_updates": 0,
              "input_hashes": {str(path.relative_to(ROOT)): expected for path, expected in LOCKS.items()},
              "execution_code_sha256": digest(Path(__file__)), "metrics_code_sha256": digest(
                  ROOT / "src/any2xsampa/music/note_metrics.py"),
              "rows": rows, "totals": totals}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_cmd = sub.add_parser("build")
    build_cmd.add_argument("--output", type=Path, required=True)
    check_cmd = sub.add_parser("validate")
    check_cmd.add_argument("--packet-dir", type=Path, required=True)
    check_cmd.add_argument("--answers", type=Path, required=True)
    eval_cmd = sub.add_parser("evaluate")
    eval_cmd.add_argument("--packet-dir", type=Path, required=True)
    eval_cmd.add_argument("--answers", type=Path, required=True)
    eval_cmd.add_argument("--output", type=Path, required=True)
    diagnose_cmd = sub.add_parser("diagnose")
    diagnose_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.output)
    elif args.command == "evaluate":
        result = evaluate(args.packet_dir, args.answers, args.output)
    elif args.command == "diagnose":
        result = diagnose_saved_pairs(args.output)
    else:
        result = validate_file(args.packet_dir, args.answers)
    print(json.dumps(result if args.command != "build" else
                     {"packet_sha256": result["packet_sha256"], "status": result["status"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

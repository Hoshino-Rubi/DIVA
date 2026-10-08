"""Serial, offline separation / ASR / raw-timestamp worker for frozen contexts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.auxiliary_slicing import validate_context
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def load_contexts(run: Path) -> tuple[dict, list[dict]]:
    start = read_json(run / "execution-start.json")
    verify_file(run / "policy.json", start["policy_sha256"])
    verify_file(run / "contexts.json", start["contexts_sha256"])
    contexts = read_json(run / "contexts.json")["contexts"]
    if len(contexts) != 18 or len({c["context_id"] for c in contexts}) != 18:
        raise ValueError("auxiliary run must retain the eighteen frozen contexts")
    for row in contexts:
        validate_context(row)
    return start, contexts


def load_index(path: Path, digest: str, run: Path, stage: str, contexts: list[dict]) -> dict:
    if not path.resolve().is_relative_to(run.resolve()):
        raise ValueError("auxiliary input index must belong to this run")
    verify_file(path, digest)
    index = read_json(path)
    if index.get("schema_version") != "quality8h_auxiliary_index_v1" or index.get("stage") != stage:
        raise ValueError("auxiliary input stage/schema mismatch")
    start = read_json(run / "execution-start.json")
    if index.get("policy_sha256") != start["policy_sha256"] or index.get("test_songs_loaded") is not False:
        raise ValueError("auxiliary input policy/split mismatch")
    selection = index.get("selection")
    if selection not in {"contexts", "smoke8", "smoke20"}:
        raise ValueError("unknown auxiliary selection")
    expected = {row["context_id"]: row for row in contexts}
    active = set(expected) if selection == "contexts" else {contexts[0]["context_id"]}
    required = {(key, role) for key in active for role in (("mixed", "vocal") if stage in {"asr", "timestamps"} else (None,))}
    observed = set()
    for entry in index["entries"]:
        context = entry["context"]
        validate_context(context)
        if context != expected[context["context_id"]]:
            raise ValueError("derived evidence differs from frozen context")
        identity = (context["context_id"], entry.get("audio_role"))
        if identity in observed or identity not in required:
            raise ValueError("duplicate or unexpected auxiliary evidence")
        observed.add(identity)
        record_path = Path(entry["record_path"])
        if not record_path.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("derived record escapes its stage directory")
        verify_file(record_path, entry["record_sha256"])
        record = read_json(record_path)
        if record.get("context") != context or record.get("audio_role") != entry.get("audio_role"):
            raise ValueError("auxiliary record identity differs from its index")
        if record["clock"]["source"]["sha256"] != context["audio_sha256"]:
            raise ValueError("derived clock is bound to another original source")
    if observed != required:
        raise ValueError("auxiliary evidence must cover the entire declared subset")
    return index


def separated_audio(entry: dict, role: str, run: Path) -> tuple[Path, dict]:
    record = read_json(Path(entry["record_path"]))
    audio = record[role]
    path = Path(audio["path"])
    if not path.resolve().is_relative_to(run.resolve()):
        raise ValueError("derived waveform escapes the run")
    verify_file(path, audio["sha256"])
    if audio["sample_rate"] != 44100 or audio["samples"] > 44100 * 30 + 2:
        raise ValueError("derived waveform exceeds the auxiliary limit")
    return path, record


def separate_stage(args: argparse.Namespace, contexts: list[dict]) -> tuple[list[dict], dict]:
    from any2xsampa.adapters.auxiliary_audio import read_stereo_context, vocal_energy, write_stereo_float
    from any2xsampa.adapters.melband_runtime import MelBandRuntime

    runtime = MelBandRuntime(args.run_root, args.output_dir / "runtime-cache")
    rows = contexts if args.selection == "contexts" else contexts[:1]
    entries = []
    for row in rows:
        begin, end = row["start_sec"], row["end_sec"]
        if args.selection != "contexts":
            begin = row["old_start_sec"]
            end = begin + (8 if args.selection == "smoke8" else 20)
        audio, clock = read_stereo_context(Path(row["audio_path"]), row["audio_sha256"], begin, end)
        vocals, measured = runtime.separate(audio)
        folder = args.output_dir / row["context_id"]
        mixed_record = write_stereo_float(folder / "mixed-context.wav", audio, 44100)
        vocal_record = write_stereo_float(folder / "vocal-context.wav", vocals, 44100)
        energy_path = folder / "vocal-energy.json"
        energy_hash = write_json_new(energy_path, vocal_energy(vocals, 44100, runtime.policy["slicing"]))
        record = {"schema_version": "vocal_separation_evidence_v1", "context": row, "selection": args.selection,
                  "clock": clock, "mixed": mixed_record, "vocal": vocal_record, "runtime": measured,
                  "model": runtime.identity, "energy": {"path": str(energy_path), "sha256": energy_hash},
                  "labels_or_score_used": False, "human_approved": False, "original_file_modified": False}
        record_path = folder / "separation.json"
        digest = write_json_new(record_path, record)
        entries.append({"context": row, "record_path": str(record_path), "record_sha256": digest})
        print(f"separated={row['context_id']}; samples={vocals.shape[1]}; peak_reserved={measured['peak_reserved_bytes']}", flush=True)
    return entries, runtime.identity


def asr_stage(args: argparse.Namespace, index: dict) -> tuple[list[dict], dict]:
    from any2xsampa.adapters.qwen_timing_runtime import QwenAuxiliaryRuntime

    runtime = QwenAuxiliaryRuntime(args.run_root, args.output_dir / "runtime-cache", "asr")
    entries = []
    for entry in index["entries"]:
        for role in ("mixed", "vocal"):
            path, separation = separated_audio(entry, role, args.run_root)
            result, transcript = runtime.transcribe(path)
            folder = args.output_dir / entry["context"]["context_id"] / role
            record = {"schema_version": "quality8h_asr_evidence_v1", **result, "context": entry["context"],
                "audio_role": role, "audio": separation[role], "clock": separation["clock"],
                "separation_record_path": entry["record_path"], "separation_record_sha256": entry["record_sha256"],
                "score_or_lyrics_context_used": False, "human_approved": False}
            if transcript:
                transcript_path = folder / "transcript.json"
                record["transcript_path"] = str(transcript_path)
                record["transcript_sha256"] = write_json_new(transcript_path, transcript)
            record_path = folder / "asr.json"
            digest = write_json_new(record_path, record)
            entries.append({"context": entry["context"], "audio_role": role, "record_path": str(record_path), "record_sha256": digest})
            print(f"asr={entry['context']['context_id']}; role={role}; status={result['status']}", flush=True)
    return entries, runtime.identity


def timestamps_stage(args: argparse.Namespace, index: dict) -> tuple[list[dict], dict]:
    from any2xsampa.adapters.qwen_timing_runtime import QwenAuxiliaryRuntime

    runtime = QwenAuxiliaryRuntime(args.run_root, args.output_dir / "runtime-cache", "timestamps")
    entries = []
    for entry in index["entries"]:
        asr = read_json(Path(entry["record_path"]))
        record = {"schema_version": "quality8h_timing_evidence_v1", "context": entry["context"], "audio_role": entry["audio_role"],
                  "asr_record_path": entry["record_path"], "asr_record_sha256": entry["record_sha256"],
                  "audio": asr["audio"], "clock": asr["clock"], "human_approved": False}
        if asr["status"] != "transcribed":
            record.update(status="unavailable_empty_asr", timing=None)
        else:
            transcript_path = Path(asr["transcript_path"])
            if not transcript_path.resolve().is_relative_to(args.run_root):
                raise ValueError("ASR transcript escapes the run")
            verify_file(transcript_path, asr["transcript_sha256"])
            transcript = read_json(transcript_path)
            if transcript["provenance"]["context_supplied"] is not False or transcript["provenance"]["audio_sha256"] != asr["audio"]["sha256"]:
                raise ValueError("timestamp text was not produced from the same audio without context")
            audio = Path(asr["audio"]["path"])
            if not audio.resolve().is_relative_to(args.run_root):
                raise ValueError("timestamp audio escapes the run")
            verify_file(audio, asr["audio"]["sha256"])
            record.update(status="raw_timestamps_produced", timing=runtime.align(audio, transcript["text"], asr["audio"]["samples"] / asr["audio"]["sample_rate"]))
        path = args.output_dir / entry["context"]["context_id"] / entry["audio_role"] / "timing.json"
        digest = write_json_new(path, record)
        entries.append({"context": entry["context"], "audio_role": entry["audio_role"], "record_path": str(path), "record_sha256": digest})
        print(f"timestamps={entry['context']['context_id']}; role={entry['audio_role']}; status={record['status']}", flush=True)
    return entries, runtime.identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["separate", "asr", "timestamps"])
    parser.add_argument("--selection", choices=["smoke8", "smoke20", "contexts"], default="contexts")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.run_root, args.output_dir = args.run_root.resolve(), args.output_dir.resolve()
    if not args.run_root.is_relative_to(ROOT / ".state/runs") or not args.output_dir.is_relative_to(args.run_root):
        parser.error("worker outputs must remain inside the current run")
    if args.output_dir.exists():
        parser.error("refusing to overwrite auxiliary evidence")
    if not sys.flags.utf8_mode:
        parser.error("isolated worker must use -X utf8")
    start, contexts = load_contexts(args.run_root)
    index = None
    if args.stage != "separate":
        if args.inputs is None or args.inputs_sha256 is None:
            parser.error("ASR/timestamps require a hash-bound previous stage")
        index = load_index(args.inputs, args.inputs_sha256, args.run_root, "separate" if args.stage == "asr" else "asr", contexts)
        if index["selection"] != args.selection:
            parser.error("selection must match the frozen previous stage")
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    before, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    with ProcessResourceMonitor(args.output_dir, timeout_seconds=1800):
        if args.stage == "separate":
            entries, model = separate_stage(args, contexts)
        elif args.stage == "asr":
            entries, model = asr_stage(args, index)
        else:
            entries, model = timestamps_stage(args, index)
        report = {"schema_version": "quality8h_auxiliary_index_v1", "stage": args.stage, "selection": args.selection,
                  "policy_sha256": start["policy_sha256"], "contexts_sha256": start["contexts_sha256"],
                  "parent_index_sha256": args.inputs_sha256, "entries": entries, "model": model,
                  "test_songs_loaded": False, "human_approval_inferred": False, "offline": True,
                  "started_at_utc": started, "elapsed_seconds": time.monotonic() - before}
        print(f"index_sha256={write_json_new(args.output_dir / 'index.json', report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

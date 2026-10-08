"""Isolated CPU pYIN analysis; output pitches are never constant placeholders."""
from __future__ import annotations
import dataclasses
import math
from pathlib import Path
import sys
import time
from .common import ROOT, check_cancel, read, sha, write
from .server import Server, input_lines

class PitchEngine:
    def close(self):
        pass

    def execute(self, action, params, cancel, progress):
        if action != "analyze_pitch" or set(params) != {"result_path", "output_dir"}:
            raise ValueError("Pitch worker only accepts a bound result and output directory")
        file = Path(params["result_path"])
        source = read(file)
        if source.get("schema_version") != "local_task_result_v1":
            raise ValueError("Expected local recognition result")
        audio = Path(source["input"]["path"])
        if sha(audio) != source["input"]["sha256"]:
            raise ValueError("Pitch source audio hash mismatch")
        policy_path = ROOT / "src/any2xsampa/config/local_score_v1.json"
        policy = read(policy_path)["pitch"]
        started = time.monotonic()
        from any2xsampa.music.pitch_note_analysis import (
            LibrosaPitchNoteAnalyzer, PitchNoteAnalysisConfig, build_pitch_note_analysis,
        )
        deps = LibrosaPitchNoteAnalyzer()._load_dependencies()
        librosa, np = deps.librosa, deps.numpy
        cfg = PitchNoteAnalysisConfig(**{key:policy[key] for key in (
            "sample_rate", "hop_length", "frame_length", "fmin_hz", "fmax_hz", "min_note_duration_sec")})
        begin, end = float(source["input"]["start_sec"]), float(source["input"]["end_sec"])
        context = float(policy["context_seconds"])
        core_seconds = float(policy["chunk_seconds"]) - 2*context
        count = math.ceil((end-begin)/core_seconds)
        times, frequencies, flags, confidences = [], [], [], []
        for index in range(count):
            check_cancel(cancel)
            core_start, core_end = begin+index*core_seconds, min(end,begin+(index+1)*core_seconds)
            # Align all crops to one global F0 frame grid to avoid overlapping
            # note durations at chunk seams after the fixed frame-interval step.
            first_frame=max(0,math.ceil((core_start-context-begin)*cfg.sample_rate/cfg.hop_length))
            crop_start=begin+first_frame*cfg.hop_length/cfg.sample_rate
            crop_end=min(end,core_end+context)
            progress({"stage":"pitch_analysis", "chunk_index":index+1, "chunk_total":count,
                      "processed_seconds":core_start-begin, "total_seconds":end-begin})
            y, sr = librosa.load(str(audio), sr=cfg.sample_rate, mono=True,
                                 offset=crop_start, duration=crop_end-crop_start)
            f0, voiced, probability = librosa.pyin(y, fmin=cfg.fmin_hz, fmax=cfg.fmax_hz, sr=sr,
                hop_length=cfg.hop_length, frame_length=cfg.frame_length)
            check_cancel(cancel)
            for frame in range(len(f0)):
                time_sec = crop_start + frame*cfg.hop_length/sr
                if core_start <= time_sec < core_end:
                    times.append(float(time_sec))
                    frequencies.append(float(f0[frame]))
                    flags.append(bool(voiced[frame]))
                    confidences.append(float(probability[frame]))
        if sha(audio) != source["input"]["sha256"]:
            raise ValueError("Audio changed during pitch analysis")
        payload = build_pitch_note_analysis(audio_path=audio, f0_hz=frequencies,
            frame_times_sec=times, voiced_probabilities=confidences, voiced_flags=flags,
            frame_hz=cfg.sample_rate/cfg.hop_length, config=cfg,
            provenance_extra={"analyzer":"librosa_pyin_v1", "dependency_versions":deps.versions,
                              "window_policy":"bounded_context_then_core_frame_ownership",
                              "audio_role":"original_mix", "reference_conditions_used":False})
        payload.update(input=source["input"], result_sha256=sha(file), result_path=str(file.resolve()),
                       policy_sha256=sha(policy_path), analysis_config=dataclasses.asdict(cfg),
                       elapsed_seconds=time.monotonic()-started, model_forwards=0,
                       pitch_transpose_semitones=0, timing_source="audio_f0_frames_not_ctc_emissions")
        output = Path(params["output_dir"])/"pitch-analysis.json"
        check_cancel(cancel)
        write(output,payload)
        progress({"stage":"pitch_done", "processed_seconds":end-begin,"total_seconds":end-begin,
                  "note_count":len(payload["note_candidates"])})
        return {"pitch_path":str(output),"pitch_sha256":sha(output),"note_count":len(payload["note_candidates"]),
                "model_forwards":0,"elapsed_seconds":payload["elapsed_seconds"]}

def main():
    wire=sys.stdout
    sys.stdout=sys.stderr
    server=Server(PitchEngine(),wire)
    server.emit("ready",data={"worker":"cpu_pitch_analysis"})
    try:
        for line in input_lines(sys.stdin):
            import json
            server.receive(json.loads(line))
    finally:
        server.close()

if __name__ == "__main__":
    main()

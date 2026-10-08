"""Audio-only PhoneticXeus evidence for matched resampled-mix/vocal pairs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.auxiliary_slicing import validate_context
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def audio_requests(inputs: dict, frozen: list[dict]) -> list[dict]:
    allowed = {"schema_version", "policy_sha256", "pair_policy_sha256", "separation_index_sha256", "pairs",
               "test_songs_loaded", "reference_labels_or_score_context_included", "historical_raw_baseline_replaced"}
    if set(inputs) - allowed:
        raise ValueError("unexpected paired input fields; reference context is forbidden")
    if inputs.get("schema_version") != "quality8h_paired_inputs_v1" or inputs.get("reference_labels_or_score_context_included") is not False or inputs.get("test_songs_loaded") is not False:
        raise ValueError("paired inference requires an audio-only non-test manifest")
    expected = {r["context_id"]: r for r in frozen}
    if len(inputs["pairs"]) != len(expected):
        raise ValueError("paired inference must retain every frozen context")
    seen, requests = set(), []
    for pair in inputs["pairs"]:
        context = pair["context"]
        validate_context(context)
        if context != expected.get(context["context_id"]) or context["context_id"] in seen:
            raise ValueError("duplicate or altered paired source context")
        seen.add(context["context_id"])
        if set(pair["audio"]) != {"mixed", "vocal"}:
            raise ValueError("each pair needs both mix and vocal controls")
        for condition, audio in pair["audio"].items():
            if set(audio) - {"path", "sha256", "sample_rate", "samples", "channels", "subtype", "roundtrip_max_abs_error"}:
                raise ValueError("unexpected paired waveform fields; reference context is forbidden")
            if audio["sample_rate"] != 44100 or audio["samples"] != 18 * 44100 or audio["channels"] != 2:
                raise ValueError("paired inference requires intact eighteen-second stereo inputs")
            requests.append({"original_example_id": context["example_id"], "condition": condition,
                "audio_request": {"example_id": context["example_id"] + "-paired-" + condition,
                    "song_id": context["song_id"], "split": context["split"],
                    "audio_role": "original_resampled_mix" if condition == "mixed" else "original_separated_vocal",
                    "audio_path": audio["path"], "audio_sha256": audio["sha256"], "start_sec": 0.0, "end_sec": 18.0}})
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paired-inputs", type=Path, required=True)
    parser.add_argument("--paired-inputs-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run) or output.exists():
        parser.error("paired inference requires a new output directory within the run")
    if not args.paired_inputs.resolve().is_relative_to(run):
        parser.error("paired source manifest must belong to this run")
    verify_file(args.paired_inputs, args.paired_inputs_sha256)
    inputs = read_json(args.paired_inputs)
    start = read_json(run / "execution-start.json")
    verify_file(run / "contexts.json", start["contexts_sha256"])
    if inputs["policy_sha256"] != start["policy_sha256"]:
        raise ValueError("paired input experiment policy mismatch")
    requests = audio_requests(inputs, read_json(run / "contexts.json")["contexts"])
    for row in requests:
        path = Path(row["audio_request"]["audio_path"]).resolve()
        if not path.is_relative_to(run):
            raise ValueError("paired waveform escapes the run")
        verify_file(path, row["audio_request"]["audio_sha256"])
    write_json_new(output / "audio-requests.json", {"schema_version": "quality8h_paired_audio_requests_v1", "requests": requests,
        "paired_inputs_sha256": args.paired_inputs_sha256, "contains_reference_context": False})
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime
    from safetensors.torch import save_file

    before = time.monotonic()
    with ProcessResourceMonitor(output, timeout_seconds=1800):
        runtime = PhoneticXeusRuntime(ROOT / ".state/runs/limited-data-20260906T050251/model", cache_dir=output / "runtime-cache")
        head = runtime.model.model.ctc.ctc_lo
        head_path = output / "pretrained-head.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(head_path))
        entries = []
        for row in requests:
            evidence = runtime.save_example(row["audio_request"], output / "evidence" / row["audio_request"]["example_id"], sealed=False)
            entries.append({**evidence, "original_example_id": row["original_example_id"], "condition": row["condition"]})
            print(f"phonetic_pair={row['original_example_id']}; condition={row['condition']}; frames={evidence['frames']}", flush=True)
        report = {"schema_version": "quality8h_paired_phonetic_index_v1", "purpose": "paired_original_conditionals_not_training",
            "policy_sha256": start["policy_sha256"], "paired_inputs_sha256": args.paired_inputs_sha256,
            "paired_inputs_path": str(args.paired_inputs.resolve()), "audio_requests_sha256": sha256_file(output / "audio-requests.json"),
            "model_metadata": runtime.identity, "head": {"path": str(head_path), "sha256": sha256_file(head_path)},
            "entries": entries, "runtime": runtime.runtime, "elapsed_seconds": time.monotonic() - before,
            "labels_or_score_received": False, "test_songs_loaded": False, "source_clock": "relative_to_each_derived_18_second_wav"}
        print(f"paired_phonetic_index_sha256={write_json_new(output / 'index.json', report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

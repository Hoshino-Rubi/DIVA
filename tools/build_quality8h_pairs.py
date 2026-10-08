"""Build matched PCM-WAV mix/vocal conditions for the old frozen windows."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


PAIR_POLICY = {
    "schema_version": "quality8h_pair_transform_policy_v1",
    "source_window": "unchanged_original_frozen_18_seconds",
    "source_rate": 44100,
    "crop_samples": "round_relative_source_time_times_44100",
    "shared_gain": "min(1, (1 - 2**-23) / max_abs_across_both_conditions))",
    "output_subtype": "PCM_32",
    "output_channels": 2,
    "original_baseline_is_resampled_control_not_the_historical_raw_input": True,
    "reason": "Legacy PhoneSegment audio provenance requires PCM WAV; preserve float master and avoid clipping either condition.",
    "phone_predictions_or_references_used": False,
    "asr_used_to_choose_crop": False,
}


def paired_pcm(mixed, vocal, first: int, last: int, destination: Path) -> tuple[dict, dict]:
    import numpy as np
    import soundfile as sf

    if mixed.shape != vocal.shape or mixed.ndim != 2 or mixed.shape[1] != 2:
        raise ValueError("paired masters must have identical stereo sample geometry")
    if not 0 <= first < last <= len(mixed) or not 8 * 44100 <= last - first <= 20 * 44100:
        raise ValueError("paired crop is outside the bounded waveform")
    arrays = {"mixed": mixed[first:last], "vocal": vocal[first:last]}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("non-finite paired audio")
    peak = max(float(np.max(np.abs(a))) for a in arrays.values())
    gain = min(1.0, (1.0 - 2 ** -23) / peak) if peak else 1.0
    records = {}
    destination.mkdir(parents=True)
    for role, values in arrays.items():
        scaled = (values * gain).astype(np.float32)
        path = destination / f"{role}-pcm32.wav"
        with path.open("xb") as stream:
            sf.write(stream, scaled, 44100, format="WAV", subtype="PCM_32")
        decoded, rate = sf.read(path, dtype="float32", always_2d=True)
        difference = float(np.max(np.abs(decoded - scaled)))
        if rate != 44100 or decoded.shape != scaled.shape or difference > 2e-7:
            raise ValueError("PCM conversion altered geometry or clipped paired evidence")
        records[role] = {"path": str(path.resolve()), "sha256": sha256_file(path), "sample_rate": rate,
                         "samples": len(decoded), "channels": 2, "subtype": "PCM_32", "roundtrip_max_abs_error": difference}
    return records, {"shared_gain": gain, "pre_gain_pair_peak_abs": peak, "same_gain_for_both_conditions": True,
                     "source_start_sample": first, "source_end_sample_exclusive": last,
                     "both_full_crop_samples_preserved": True, "clipping_performed": False}


def build(run: Path, index_path: Path, destination: Path) -> dict:
    import soundfile as sf
    from run_auxiliary_worker import load_contexts, load_index

    start, contexts = load_contexts(run)
    index = load_index(index_path, sha256_file(index_path), run, "separate", contexts)
    if index["selection"] != "contexts":
        raise ValueError("paired experiment requires all eighteen contexts")
    policy_hash = write_json_new(destination / "pair-policy.json", PAIR_POLICY)
    pairs = []
    for entry in index["entries"]:
        record = read_json(Path(entry["record_path"]))
        context = entry["context"]
        data = {}
        for role in ("mixed", "vocal"):
            path = Path(record[role]["path"])
            if not path.resolve().is_relative_to(run):
                raise ValueError("paired master escapes run")
            verify_file(path, record[role]["sha256"])
            data[role], rate = sf.read(path, dtype="float32", always_2d=True)
            if rate != 44100 or len(data[role]) != record[role]["samples"]:
                raise ValueError("paired master geometry differs from separation evidence")
        origin = record["clock"]["effective_original_start_sec"]
        first = round((context["old_start_sec"] - origin) * 44100)
        last = round((context["old_end_sec"] - origin) * 44100)
        audio, transform = paired_pcm(data["mixed"], data["vocal"], first, last, destination / context["context_id"])
        pairs.append({"context": context, "audio": audio, "transform": transform,
            "effective_original_start_sec": origin + first / 44100, "effective_original_end_sec": origin + last / 44100,
            "separation_record_path": entry["record_path"], "separation_record_sha256": entry["record_sha256"],
            "master_audio_sha256": {role: record[role]["sha256"] for role in ("mixed", "vocal")}})
    result = {"schema_version": "quality8h_paired_inputs_v1", "policy_sha256": start["policy_sha256"],
              "pair_policy_sha256": policy_hash, "separation_index_sha256": sha256_file(index_path), "pairs": pairs,
              "test_songs_loaded": False, "reference_labels_or_score_context_included": False,
              "historical_raw_baseline_replaced": False}
    print(f"paired_inputs_sha256={write_json_new(destination / 'paired-inputs.json', result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--separation-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run) or output.exists():
        parser.error("paired inputs require a new directory under the run")
    build(run, args.separation_index.resolve(), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

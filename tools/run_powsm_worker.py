"""Run pinned audio-only POWSM inference, or replay evidence without loading it."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, sha256_file, verify_file
from any2xsampa.adapters.powsm import emission_payload, PowsmPosteriorRecognizer


def infer(run: Path, requests: Path, output: Path) -> None:
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.powsm_runtime import PowsmRuntime
    from safetensors.torch import save_file

    freeze = read_json(run / "FREEZE.json")
    verify_file(run / "MODEL_LOCK.json", freeze["model_lock_sha256"])
    verify_file(run / "POLICY.json", freeze["policy_sha256"])
    decoding_policy = read_json(run / "DECODING_POLICY.json")
    if decoding_policy.get("schema_version") != "powsm_decoding_policy_v2" or decoding_policy.get("phone_alias_changes") != []:
        raise ValueError("unexpected phone decoding policy")
    rows = read_json(requests)["rows"]
    if not 1 <= len(rows) <= 3 or len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("expected one smoke request or three frozen development requests")
    for row in rows:
        name = row["example_id"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
            raise ValueError("unsafe example ID")
    with serial_candidate_job(), ProcessResourceMonitor(output / "monitor", timeout_seconds=600):
        runtime = PowsmRuntime(run, output / "runtime-cache")
        entries = []
        for row in rows:
            before = time.monotonic()
            logits, audio, crop, timing = runtime.encode(row)
            destination = output / row["example_id"]
            destination.mkdir()
            payload = emission_payload(logits.numpy(), runtime.labels, audio=audio,
                crop=crop, identity=runtime.identity, runtime=timing)
            raw = destination / "logits.safetensors"
            save_file({"frame_logits": logits}, str(raw))
            vocabulary = destination / "vocabulary.json"
            vocabulary.write_bytes((run / "vocabulary.json").read_bytes())
            payload["raw_evidence"] = {"file": raw.name, "sha256": sha256_file(raw),
                "vocabulary_file": vocabulary.name, "vocabulary_sha256": sha256_file(vocabulary),
                "dtype": "float32", "logit_shape": list(logits.shape)}
            posterior = destination / "posterior.json"
            write_json_new(posterior, payload)
            equal, contract_error = False, None
            try:
                real = PowsmPosteriorRecognizer(posterior).recognize(Path(row["audio_path"]))
                fixture = PowsmPosteriorRecognizer(posterior, fixture=True).recognize(Path(row["audio_path"]))
                view = lambda seq: [(asdict(s.time_range), [asdict(p) for p in s.candidate_phones], s.confidence) for s in seq]
                if view(real) != view(fixture):
                    raise RuntimeError("POWSM real/fixture projection differs")
                equal = True
            except ValueError as exc:
                if not payload["decoding"]["untimed_phone_emissions"]:
                    raise
                contract_error = str(exc)
            entries.append({"request": row, "posterior_path": str(posterior.resolve()),
                "posterior_sha256": sha256_file(posterior), "logits_sha256": sha256_file(raw),
                "prediction_phones": payload["decoding"]["prediction_phones"],
                "raw_ctc_ids": payload["decoding"]["collapsed_raw_ids"],
                "nonphone_emissions": payload["decoding"]["nonphone_emissions"],
                "untimed_phone_emissions": payload["decoding"]["untimed_phone_emissions"],
                "real_fixture_equal": equal, "timed_contract_error": contract_error,
                "elapsed_seconds": time.monotonic() - before})
            print(row["example_id"], "emissions", len(entries[-1]["prediction_phones"]),
                "fixture", equal, "peakGiB", round(timing["peak_reserved_bytes"] / 2**30, 3), flush=True)
        write_json_new(output / "RESULT.json", {"complete": True, "entries": entries,
            "model_identity": runtime.identity, "runtime": runtime.runtime,
            "peak_reserved_bytes": runtime.torch.cuda.max_memory_reserved(),
            "requests_sha256": sha256_file(requests), "reference_received": False,
            "decoding_policy_sha256": sha256_file(run / "DECODING_POLICY.json"),
            "source_bindings": {str(p): sha256_file(p) for p in [Path(__file__).resolve(),
                ROOT / "src/any2xsampa/adapters/powsm.py", ROOT / "src/any2xsampa/adapters/powsm_runtime.py"]}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = time.monotonic()
    write_json_new(output / "COMMAND.json", {"argv": sys.argv, "python": sys.executable,
        "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        if args.fixture:
            if args.audio is None or args.requests is not None:
                raise ValueError("fixture requires --audio and forbids --requests")
            segments = PowsmPosteriorRecognizer(args.fixture, fixture=True).recognize(args.audio)
            write_json_new(output / "RESULT.json", {"mode": "fixture", "model_loaded": False,
                "source_sha256": sha256_file(args.fixture), "segments": [s.to_dict() for s in segments]})
        else:
            if args.run is None or args.requests is None or args.audio is not None:
                raise ValueError("real worker requires --run and --requests")
            infer(args.run.resolve(), args.requests.resolve(), output)
        write_json_new(output / "EXIT.json", {"exit_code": 0, "elapsed_seconds": time.monotonic()-before})
    except Exception:
        write_json_new(output / "EXIT.json", {"exit_code": 1, "elapsed_seconds": time.monotonic()-before,
            "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()

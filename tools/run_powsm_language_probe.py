"""Run one fixed language condition on the frozen training cohort, offline."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file


def run_probe(run: Path, condition: str, output: Path, development: bool = False) -> None:
    from any2xsampa.adapters.powsm_language import PowsmLanguageRuntime, PowsmLanguagePosteriorRecognizer
    from any2xsampa.adapters.powsm import emission_payload
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from safetensors.torch import save_file, load_file

    freeze = read_json(run / "FREEZE.json")
    verify_file(run / "POLICY.json", freeze["policy_sha256"])
    parent = Path(freeze["parents"]["powsm"])
    language = {"unknown": "<unk>", "cantonese": "<yue>"}[condition]
    if development:
        selection = read_json(run / "POWSM_LANGUAGE_SELECTION.json")
        if not selection["passed"] or selection["selected_condition"] != condition:
            raise ValueError("development can only evaluate the frozen passing condition")
        requests = parent / "original-development.json"
    else:
        requests = run / "POWSM_AUDIO_REQUESTS.json"
        verify_file(requests, freeze["original_requests_sha256"])
    rows = read_json(requests)["rows"]
    if len(rows) != (3 if development else 20):
        raise ValueError("comparison cohort changed")
    with serial_candidate_job(), ProcessResourceMonitor(output / "monitor", timeout_seconds=900):
        rt = PowsmLanguageRuntime(parent, output / "runtime-cache", language=language)
        smoke = None
        if condition == "unknown" and not development:
            row = read_json(parent / "smoke-8.json")["rows"][0]
            logits, _, _, _ = rt.encode(row)
            old = load_file(str(parent / "smoke8-final/training-smoke-8/logits.safetensors"))["frame_logits"]
            delta = float((old-logits).abs().max())
            if delta > 1e-5 or not rt.torch.equal(old.argmax(-1), logits.argmax(-1)):
                raise ValueError("unknown-language wrapper differs from the historical adapter")
            smoke = {"maximum_logit_difference": delta, "prediction_identical": True}
        entries = []
        for row in rows:
            logits, audio, crop, timing = rt.encode(row)
            folder = output / row["example_id"]
            folder.mkdir()
            raw = folder / "logits.safetensors"
            save_file({"frame_logits": logits}, str(raw))
            vocabulary = folder / "vocabulary.json"
            vocabulary.write_bytes((parent / "vocabulary.json").read_bytes())
            payload = emission_payload(logits.numpy(), rt.labels, audio=audio, crop=crop,
                identity=rt.identity, runtime=timing)
            payload["producer"]["language_hint_used"] = language == "<yue>"
            payload["raw_evidence"] = {"file": raw.name, "sha256": sha256_file(raw),
                "vocabulary_file": vocabulary.name, "vocabulary_sha256": sha256_file(vocabulary),
                "dtype": "float32", "logit_shape": list(logits.shape)}
            posterior = folder / "posterior.json"
            write_json_new(posterior, payload)
            replay = False
            if not payload["decoding"]["untimed_phone_emissions"]:
                segments = PowsmLanguagePosteriorRecognizer(posterior, fixture=True).recognize(Path(row["audio_path"]))
                replay = len(segments) == sum(x.startswith("/") and x.endswith("/") for x in payload["decoding"]["collapsed_raw_labels"])
                if not replay:
                    raise ValueError("language-conditioned fixture lost a phone")
            entries.append({"request": row, "prediction": payload["decoding"]["prediction_phones"],
                "special_emissions": payload["decoding"]["special_emissions"],
                "untimed_phone_emissions": payload["decoding"]["untimed_phone_emissions"],
                "fixture_verified": replay, "posterior_path": str(posterior.resolve()),
                "posterior_sha256": sha256_file(posterior)})
            print(condition, row["example_id"], "phones", len(entries[-1]["prediction"]), flush=True)
        write_json_new(output / "RESULT.json", {"complete": True, "condition": condition,
            "language_condition": language, "development": development, "entries": entries,
            "reference_received": False, "requests_sha256": sha256_file(requests),
            "policy_sha256": sha256_file(run / "POLICY.json"), "historical_wrapper_equivalence": smoke,
            "peak_reserved_bytes": rt.torch.cuda.max_memory_reserved(),
            "code_bindings": {str(p): sha256_file(p) for p in [Path(__file__).resolve(),
                ROOT / "src/any2xsampa/adapters/powsm_language.py"]}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--condition", choices=["unknown", "cantonese"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = time.monotonic()
    write_json_new(args.output / "COMMAND.json", {"argv": sys.argv, "python": sys.executable,
        "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        run_probe(args.run.resolve(), args.condition, args.output.resolve(), args.development)
        write_json_new(args.output / "EXIT.json", {"exit_code": 0, "elapsed_seconds": time.monotonic()-before})
    except Exception:
        write_json_new(args.output / "EXIT.json", {"exit_code": 1, "elapsed_seconds": time.monotonic()-before,
            "traceback": traceback.format_exc()})
        raise

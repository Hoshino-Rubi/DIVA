"""One correctness pass on all eighteen original windows, not a tuning trial."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run) or output.exists():
        parser.error("verification output must be a new directory inside the run")
    sys.path.insert(1, str(ROOT / "tools"))
    from run_auxiliary_worker import load_contexts
    from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime

    start, contexts = load_contexts(run)
    base = ROOT / ".state/runs/limited-data-20260906T050251"
    index_path = base / "original-baselines/index.json"
    verify_file(index_path, "1240f27cbf8aa2b62e6d3f8da21a95fbc1d11921a0ba25099f1e97fdf228a905")
    original = {e["example_id"]: e for e in read_json(index_path)["entries"] if e["split"] != "test"}
    results = []
    with ProcessResourceMonitor(output, timeout_seconds=1200), LockedAudioCache() as cache:
        runtime = PhoneticXeusRuntime(base / "model", cache_dir=output / "runtime-cache", audio_cache=cache)
        vocabulary = runtime.snapshot / "src/model/xeusphoneme/resources/ipa_vocab.json"
        for context in contexts:
            request = {"example_id": context["example_id"], "song_id": context["song_id"], "split": context["split"],
                       "audio_role": "original", "audio_path": context["audio_path"], "audio_sha256": context["audio_sha256"],
                       "start_sec": context["old_start_sec"], "end_sec": context["old_end_sec"]}
            previous = original[request["example_id"]]
            for field in ("song_id", "split", "audio_role", "audio_sha256", "start_sec", "end_sec"):
                if request[field] != previous[field]:
                    raise ValueError("cache validation changed its historical input")
            current = runtime.save_example(request, output / "evidence" / request["example_id"])
            verify_file(Path(previous["evidence_path"]), previous["evidence_sha256"])
            verify_file(Path(previous["posterior_path"]), previous["posterior_sha256"])
            if current["evidence_sha256"] != previous["evidence_sha256"]:
                raise ValueError(f"cache changed original feature/logit bytes: {request['example_id']}")
            old_payload, new_payload = read_json(Path(previous["posterior_path"])), read_json(Path(current["posterior_path"]))
            if old_payload["segments"] != new_payload["segments"] or old_payload["decoding"] != new_payload["decoding"]:
                raise ValueError("cache changed original posterior semantics")
            old = PhoneticXeusPosteriorRecognizer(Path(previous["posterior_path"]), vocabulary_path=vocabulary).recognize(Path(request["audio_path"]))
            new = PhoneticXeusPosteriorRecognizer(Path(current["posterior_path"])).recognize(Path(request["audio_path"]))
            fixture = PhoneticXeusPosteriorRecognizer(Path(current["posterior_path"]), fixture=True).recognize(Path(request["audio_path"]))
            signature = lambda values: [(s.time_range, s.candidate_phones, s.confidence) for s in values]
            if not signature(old) == signature(new) == signature(fixture):
                raise ValueError("cache changed original PhoneSegment contract")
            results.append({"example_id": request["example_id"], "source_sha256": request["audio_sha256"],
                            "evidence_sha256": current["evidence_sha256"], "byte_identical": True, "real_fixture_identical": True})
            print(f"cache_verified={request['example_id']}", flush=True)
        stats = cache.stats()
    if cache.stats()["open_sources"] != 0:
        raise RuntimeError("source read-sharing handles were not released")
    report = {"schema_version": "quality8h_audio_cache_validation_v1", "examples": results, "windows": len(results),
              "source_cache_stats": stats, "all_source_handles_released": True, "model_outputs_byte_identical": True,
              "test_predictions_opened": False, "new_training_performed": False, "not_an_additional_performance_sample": True}
    write_json_new(output / "verification.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

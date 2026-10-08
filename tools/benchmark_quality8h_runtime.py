"""Fixed 8s/20s train-audio benchmark, fresh process plus bounded warm runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
import sys
import time

PROCESS_ENTRY = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new


class StageTimer:
    def __init__(self) -> None:
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def wrap(self, function, name):
        @wraps(function)
        def measured(*args, **kwargs):
            before = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds[name] += time.perf_counter() - before
                self.calls[name] += 1
        return measured

    def reset(self) -> None:
        self.seconds.clear()
        self.calls.clear()


def benchmark(run: Path, output: Path, mode: str, warm_repetitions: int) -> dict:
    # -I deliberately removes the script directory; add only this verified
    # project tools directory to access the existing strict request validator.
    sys.path.insert(1, str(ROOT / "tools"))
    from run_phoneticxeus_worker import validate_requests
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    import any2xsampa.adapters.phoneticxeus_runtime as runtime_module
    import scipy.signal
    import safetensors.torch

    base = ROOT / ".state/runs/limited-data-20260906T050251"
    requests = []
    request_hashes = {}
    for purpose in ("smoke8", "smoke20"):
        path = base / "requests" / f"{purpose}.json"
        document = read_json(path)
        validate_requests(document)
        requests.extend(document["rows"])
        request_hashes[purpose] = sha256_file(path)
    timer = StageTimer()
    runtime_module.read_audio_crop = timer.wrap(runtime_module.read_audio_crop, "audio_preparation")
    runtime_module.describe_audio_file = timer.wrap(runtime_module.describe_audio_file, "audio_hash_metadata")
    runtime_module.emission_payload = timer.wrap(runtime_module.emission_payload, "posterior_projection")
    runtime_module.write_json_new = timer.wrap(runtime_module.write_json_new, "posterior_json_write")
    scipy.signal.resample_poly = timer.wrap(scipy.signal.resample_poly, "resample")
    safetensors.torch.save_file = timer.wrap(safetensors.torch.save_file, "tensor_write")
    cache_context = nullcontext(None)
    if mode == "cached":
        from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
        cache_context = LockedAudioCache(max_bytes=8 * 1024 ** 2, max_sources=8)
    with ProcessResourceMonitor(output, timeout_seconds=600), cache_context as cache:
        if cache is not None:
            cache.read = timer.wrap(cache.read, "audio_preparation")
            cache._describe = timer.wrap(cache._describe, "audio_hash_metadata")
        load_start = time.perf_counter()
        kwargs = {"cache_dir": output / "runtime-cache"}
        if cache is not None:
            kwargs["audio_cache"] = cache
        runtime = runtime_module.PhoneticXeusRuntime(base / "model", **kwargs)
        load_seconds = time.perf_counter() - load_start
        torch = runtime.torch
        batches = []
        for iteration in range(1 + warm_repetitions):
            timer.reset()
            torch.cuda.reset_peak_memory_stats()
            cache_before = cache.stats() if cache else None
            start = time.perf_counter()
            entries = []
            for row in requests:
                entries.append(runtime.save_example(row, output / f"batch-{iteration:02d}" / row["example_id"]))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            first_ready = time.perf_counter() - PROCESS_ENTRY if iteration == 0 else None
            gpu_seconds = sum(read_json(Path(e["posterior_path"]))["producer"]["runtime"]["cuda_event_ms"] / 1000 for e in entries)
            record = {"iteration": iteration, "temperature": "cold_process_first_batch" if iteration == 0 else "warm_model_batch",
                "elapsed_seconds": elapsed, "first_result_ready_since_process_entry_seconds": first_ready,
                "cuda_event_seconds": gpu_seconds, "stage_seconds": dict(timer.seconds), "stage_calls": dict(timer.calls),
                "stage_times_overlap": "audio_preparation includes hash/metadata and resampling; do not sum nested stages",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "cache_before": cache_before, "cache_after": cache.stats() if cache else None, "entries": entries}
            batches.append(record)
            write_json_new(output / f"batch-{iteration:02d}.json", record)
            print(f"mode={mode}; iteration={iteration}; batch_seconds={elapsed:.6f}; gpu_seconds={gpu_seconds:.6f}", flush=True)
        report = {"schema_version": "quality8h_runtime_benchmark_v1", "mode": mode, "request_sha256": request_hashes,
            "load_seconds": load_seconds, "batches": batches, "warm_repetitions": warm_repetitions,
            "model_metadata": runtime.identity, "runtime": runtime.runtime,
            "os_filesystem_cache_flushed": False, "cold_means_new_process_and_new_model_instance": True,
            "neural_feature_or_logit_cache_used": False, "precision": "float32", "test_songs_loaded": False,
            "measurement_only_wrappers": True, "model_inference_runs_for_every_request": True}
        write_json_new(output / "benchmark.json", report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--mode", choices=["baseline", "cached"], required=True)
    parser.add_argument("--warm-repetitions", type=int, choices=[0, 5], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run) or output.exists():
        parser.error("benchmark output must be a new directory inside the run")
    benchmark(run, output, args.mode, args.warm_repetitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare every predeclared benchmark sample, including byte/contract replay."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def samples(records: list[dict]) -> dict:
    cold = [r["batches"][0]["first_result_ready_since_process_entry_seconds"] for r in records]
    first_batches = [r["batches"][0]["elapsed_seconds"] for r in records]
    warm = [b["elapsed_seconds"] for r in records for b in r["batches"] if b["iteration"] > 0]
    if len(records) != 3 or len(warm) != 5 or [r["warm_repetitions"] for r in records] != [5, 0, 0]:
        raise ValueError("benchmark must retain exactly three cold starts and five warm batches")
    return {"cold_process_seconds": cold, "cold_process_median_seconds": statistics.median(cold),
            "first_batch_seconds": first_batches, "first_batch_median_seconds": statistics.median(first_batches),
            "warm_batch_seconds": warm, "warm_batch_median_seconds": statistics.median(warm),
            "warm_batch_range_seconds": [min(warm), max(warm)],
            "maximum_peak_reserved_bytes": max(b["peak_reserved_bytes"] for r in records for b in r["batches"])}


def compare_artifact(base: dict, new: dict, audio_path: Path) -> dict:
    for field in ("example_id", "song_id", "split", "audio_role", "audio_sha256", "start_sec", "end_sec", "frames", "vocabulary_sha256", "model_snapshot_lock_sha256"):
        if base[field] != new[field]:
            raise ValueError(f"benchmark input/evidence identity changed: {field}")
    for entry in (base, new):
        verify_file(Path(entry["evidence_path"]), entry["evidence_sha256"])
        verify_file(Path(entry["posterior_path"]), entry["posterior_sha256"])
    if base["evidence_sha256"] != new["evidence_sha256"]:
        raise ValueError("cached optimization changed raw feature/logit bytes")
    old, current = read_json(Path(base["posterior_path"])), read_json(Path(new["posterior_path"]))
    for field in ("segments", "decoding"):
        if old[field] != current[field]:
            raise ValueError(f"cached optimization changed posterior {field}")
    for field in ("input_audio", "crop", "model_metadata"):
        if old["producer"][field] != current["producer"][field]:
            raise ValueError("cached optimization changed source/model provenance")
    values = []
    for entry, fixture in ((base, False), (new, False), (new, True)):
        segments = PhoneticXeusPosteriorRecognizer(Path(entry["posterior_path"]), fixture=fixture).recognize(audio_path)
        values.append([(s.time_range, s.candidate_phones, s.confidence) for s in segments])
    if not values[0] == values[1] == values[2]:
        raise ValueError("cached optimization differs through the stable PhoneSegment contract")
    return {"example_id": base["example_id"], "raw_features_and_logits_byte_identical": True,
            "posterior_segments_and_decoding_identical": True, "real_and_fixture_contract_identical": True,
            "evidence_sha256": base["evidence_sha256"]}


def collect(run: Path, destination: Path) -> dict:
    records = {mode: [read_json(run / f"performance/{mode}-{i}/benchmark.json") for i in range(3)] for mode in ("baseline", "cached")}
    for mode, group in records.items():
        if any(r["mode"] != mode or r["neural_feature_or_logit_cache_used"] or not r["model_inference_runs_for_every_request"] or r["precision"] != "float32" for r in group):
            raise ValueError("benchmark changed model computation or precision")
    requests = {}
    base = ROOT / ".state/runs/limited-data-20260906T050251"
    for name in ("smoke8", "smoke20"):
        path = base / "requests" / f"{name}.json"
        for row in read_json(path)["rows"]:
            requests[row["example_id"]] = row
        for group in records.values():
            if any(r["request_sha256"][name] != sha256_file(path) for r in group):
                raise ValueError("benchmark request changed")
    comparisons = []
    for cold_index, (old, new) in enumerate(zip(records["baseline"], records["cached"])):
        if old["model_metadata"] != new["model_metadata"] or len(old["batches"]) != len(new["batches"]):
            raise ValueError("benchmark model/batch scope changed")
        for a, b in zip(old["batches"], new["batches"]):
            if a["iteration"] != b["iteration"]:
                raise ValueError("benchmark iteration pairing changed")
            for first, second in zip(a["entries"], b["entries"]):
                compared = compare_artifact(first, second, Path(requests[first["example_id"]]["audio_path"]))
                comparisons.append({"cold_run": cold_index, "iteration": a["iteration"], **compared})
    if len(comparisons) != 16:
        raise ValueError("benchmark must compare every two-input cold/warm batch")
    measured = {mode: samples(group) for mode, group in records.items()}
    old, new = measured["baseline"], measured["cached"]
    warm_reduction = 1 - new["warm_batch_median_seconds"] / old["warm_batch_median_seconds"]
    cold_reduction = 1 - new["cold_process_median_seconds"] / old["cold_process_median_seconds"]
    memory_reduction = 1 - new["maximum_peak_reserved_bytes"] / old["maximum_peak_reserved_bytes"]
    report = {"schema_version": "quality8h_performance_comparison_v1", "measurements": measured,
        "warm_batch_time_reduction": warm_reduction, "cold_process_time_reduction": cold_reduction,
        "peak_reserved_reduction": memory_reduction, "comparisons": comparisons,
        "warm_target_passed": warm_reduction >= .15, "cold_target_passed": cold_reduction >= .15,
        "memory_target_passed": memory_reduction >= .20, "all_outputs_unchanged": True,
        "claim_scope": "fixed_freedom_flower_8s_plus_20s_pair_three_processes_five_warm_batches",
        "os_cache_was_not_flushed": True, "small_sample_not_a_cross_hardware_guarantee": True,
        "cache_is_opt_in": True, "no_neural_features_or_logits_cached": True}
    write_json_new(destination / "comparison.json", report)
    lines = ["# 运行效率验证", "", "固定训练歌曲八秒＋二十秒输入对；每种条件三次独立进程启动、五次热运行。", "",
        "|指标|原实现|只读音频缓存|", "|---|---:|---:|",
        f"|热批处理中位耗时|{old['warm_batch_median_seconds']:.6f}s|{new['warm_batch_median_seconds']:.6f}s|",
        f"|冷进程首批结果中位耗时|{old['cold_process_median_seconds']:.6f}s|{new['cold_process_median_seconds']:.6f}s|",
        f"|最大 GPU reserved|{old['maximum_peak_reserved_bytes'] / 1024 ** 3:.3f} GiB|{new['maximum_peak_reserved_bytes'] / 1024 ** 3:.3f} GiB|", "",
        f"热批处理耗时降低 {warm_reduction:.1%}；冷启动降低 {cold_reduction:.1%}；显存降低 {memory_reduction:.1%}。", "",
        "十六组对应产物的 encoder features/logits 文件逐字节相同，posterior segments/decoding 与 real/fixture PhoneSegment 对照均相同。",
        "每次仍执行完整 FP32 模型推理，只缓存有上限的重采样音频；没有缓存神经特征或 logits。",
        "收益仅适用于这一固定基准，不声称所有歌曲、硬件或冷启动都有相同加速；没有清空操作系统文件缓存。",
        "缓存通过 Windows 只读共享句柄阻止源文件在会话中被写入或删除，退出后释放；新增 --audio-cache 为显式可选项，默认未改变。"]
    with (destination / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run) or output.exists():
        parser.error("performance report must be a new directory inside this run")
    result = collect(run, output)
    print(f"warm_time_reduction={result['warm_batch_time_reduction']:.4%}; identical={result['all_outputs_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

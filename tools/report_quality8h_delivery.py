"""Produce an evidence-bound technical delivery while recording the time failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def build(run: Path, audit_name: str, regression_name: str) -> dict:
    audit_path = run / "delivery-verification-v2.json"
    audit = read_json(audit_path)
    audit_command = read_json(run / f"commands/{audit_name}.command.json")
    tests = read_json(run / f"commands/{regression_name}.command.json")
    if audit_command["return_code"] != 0 or tests["return_code"] != 0:
        raise ValueError("final source audit and full regression must have terminal success")
    if audit_command["source_fingerprints_sha256"] != tests["source_fingerprints_sha256"]:
        raise ValueError("audited source differs from the full-regression source")
    log_path = Path(tests["stdout_path"])
    matches = re.findall(r"(\d+) passed(?:, (\d+) skipped)? in ([\d.]+)s", log_path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError("cannot establish an unambiguous complete regression summary")
    passed, skipped, seconds = matches[0]
    for name, digest in audit["artifact_hashes"].items():
        verify_file(run / name, digest)
    training = read_json(run / "training-summary-v1/summary.json")
    performance = read_json(run / "performance-summary-v1/comparison.json")
    slices = read_json(run / "slices-v1/manifest.json")
    diagnostics = read_json(run / "paired-diagnostics-v1/diagnostics.json")
    matrix = {m["system"] + "/" + m["condition"]: m["metrics"] for m in diagnostics["metrics"]
              if m["reference_factor"] == 4 and m["partition"] == "shalong"}
    statuses = {
        "fixed_sources_and_split": "completed_and_verified",
        "separator_and_qwen_offline_integration": "completed_and_verified",
        "raw_word_timestamp_pipeline": "completed_with_invalid_timing_evidence_retained",
        "automatic_safe_window_admission": "failed_zero_windows",
        "three_seed_ctc_kl_training": "completed",
        "rendered_development_candidate_gate": "passed_not_deployed",
        "paired_original_diagnostics": "completed_conditional_only",
        "warm_runtime_target": "passed" if performance["warm_target_passed"] else "failed",
        "cold_start_target": "passed" if performance["cold_target_passed"] else "failed",
        "gpu_memory_reduction_target": "passed" if performance["memory_target_passed"] else "failed",
        "full_regression": "passed",
        "source_and_historical_artifact_preservation": "verified",
        "human_cut_point_answers": "pending_six_existing_questions",
        "browser_playback_and_rendering": "not_verified_no_browser_available",
        "continuous_eight_hour_execution": "failed_unverified_gap_not_counted_as_work",
    }
    report = {"schema_version": "quality8h_delivery_report_v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "technical_work_delivered": True, "overall_user_goal_fully_satisfied": False,
        "continuous_eight_hour_commitment_met": False, "execution_integrity_incident": "EXECUTION_INTEGRITY.md",
        "statuses": statuses, "training": training, "shalong_conditional_matrix": matrix,
        "automatic_supported_windows": slices["machine_supported_windows"], "official_original_windows": 0,
        "original_candidate_reference_window_coverage": diagnostics["coverage"], "performance": performance["measurements"],
        "warm_batch_time_reduction": performance["warm_batch_time_reduction"],
        "cold_process_time_reduction": performance["cold_process_time_reduction"],
        "peak_reserved_reduction": performance["peak_reserved_reduction"],
        "tests": {"passed": int(passed), "skipped": int(skipped or 0), "elapsed_seconds": float(seconds),
                  "command": f"commands/{regression_name}.command.json", "stdout_sha256": sha256_file(log_path),
                  "source_fingerprints_sha256": tests["source_fingerprints_sha256"]},
        "delivery_audit_sha256": sha256_file(audit_path), "artifact_hashes": audit["artifact_hashes"],
        "review": audit["review"], "default_model_changed": False, "final_test_predictions_opened": False,
        "additional_dataset_training_trials_allowed": False}
    write_json_new(run / "FINAL_REPORT.json", report)
    lines = ["# 本轮技术交付与未兑现的时长承诺", "",
        "**连续工作八小时的承诺没有兑现。** 01:49 至 09:29 的无工作证据间隔没有计入有效工作；后续补做不能追溯改变这点。原因尚未确定，见 [执行记录](EXECUTION_INTEGRITY.md)。", "",
        "## 已完成", "",
        "- 固定分离模型、本地 Qwen ASR／时间戳模型在独立环境运行；完成十八个非测试上下文的混音／分离双路分析。",
        "- 三个 CTC＋KL 适配实验及全部检查点冷重放完成。合成开发门槛通过，冻结候选 Seed 43，但不部署。",
        f"- 合成开发 PER：基线 {training['baseline']['per']:.6f} → Seed43 保护候选 {next(r for r in training['seeds'] if r['seed'] == 43)['best_admissible']['metrics']['per']:.6f}；送气音 PER 0.64 → 0.48。",
        f"- 《沙龙》Seed43 固定候选参考下，匹配混音 PER {matrix['seed43/mixed']['per']:.6f} → 分离人声 {matrix['seed43/vocal']['per']:.6f}。**这是条件诊断，不是正式原唱准确率。**",
        f"- 固定 8s＋20s 基准热批处理耗时降低 {performance['warm_batch_time_reduction']:.1%}；16 组基准和另外 18 个原唱窗口的特征／logits 逐字节相同。",
        f"- 完整回归 **{passed} passed，{skipped or 0} skipped**；跳过项为已有 Tcl/Tk 和已安装 grpc 环境保护。",
        "- 二十四个外部原始文件、三千零六十音符参考重建、旧检查点及历史结果核验通过；默认模型和最终测试预测保持不动。", "",
        "## 未通过／仍待确认", "",
        "- 固定安全切片规则下得到 0 个新窗口。原始时间戳异常保留，没有采用上游插值修补结果冒充边界真值。",
        "- 正式原唱评测仍不能晋级。原唱音素／顺序的人工确认有效，欠缺的是具体切片边缘事实。",
        "- 冷启动仅约 0.5% 改善，GPU reserved 未降低；没有宣称这两个目标达成。",
        "- 浏览器连接不可用，未验证页面实际播放，也没有代理听辨。静态链接、音频与波形文件检查通过。", "",
        "## 交付入口", "",
        "- [全部训练结果](training-summary-v1/RESULTS.md)", "- [原唱／分离条件对照](paired-diagnostics-v1/RESULTS.md)",
        "- [自动切片失败与覆盖报告](slices-v1/REPORT.md)", "- [性能与一致性验证](performance-summary-v1/RESULTS.md)",
        "- [同样六个切点的集中复核](review-v1/index.html)：增加分离辅助预览，不重标全曲。",
        "- [机器可读报告](FINAL_REPORT.json)；全部命令、源码指纹、耗时和退出状态在 `commands/`。"]
    with (run / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    write_json_new(run / "RUN_STATUS.json", {"schema_version": "quality8h_run_status_v1",
        "status": "technical_delivery_completed_continuous_time_commitment_failed", "overall_goal_complete": False,
        "statuses": statuses, "final_report_sha256": sha256_file(run / "FINAL_REPORT.json"),
        "results_sha256": sha256_file(run / "RESULTS.md"), "tests_passed": int(passed), "tests_skipped": int(skipped or 0)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-command", required=True)
    parser.add_argument("--regression-command", required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    if not run.is_relative_to(ROOT / ".state/runs"):
        parser.error("delivery report must remain under .state/runs")
    value = build(run, args.audit_command, args.regression_command)
    print(f"technical_delivery=True; continuous_time=False; tests={value['tests']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

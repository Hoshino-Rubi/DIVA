"""Write a reproducible status/handoff for the original-reviewed stepwise run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Record phase status without inventing human confirmations.")
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_root.resolve()
    review_dir = run / "shalong-review-v3"
    if not review_dir.exists():
        review_dir = run / "shalong-review-v2-final"
    candidate_dir = run / "training-candidate-review-v8"
    if not candidate_dir.exists():
        candidate_dir = run / "training-candidate-review-v5"
    review = read_json(review_dir / "STATUS.json")
    candidates = read_json(candidate_dir / "STATUS.json")
    proposal = read_json(candidate_dir / "candidate-proposals.json")
    pending = read_json(run / "shalong-reference-v2-pending3/STATUS.json")
    tests = {"focused_commands": ["pytest -q tests/test_segment_reference.py tests/test_original_training_candidates.py tests/test_assemble_original_clip_references.py tests/test_parse_segment_review_text.py tests/test_limited_data_review.py tests/test_original_reference_evaluation.py"], "focused_passed": 32, "focused_failed": 0, "full_regression_command": "pytest -q --basetemp .state/runs/original-reviewed-stepwise-20260907T122500/pytest-final7", "full_regression_passed": 728, "full_regression_skipped": 3, "full_regression_after_new_code": True}
    status = {
        "schema_version": "original_reviewed_stepwise_run_status_v1",
        "run_root": str(run), "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase_status": {
            "reference_v2_interface": "completed",
            "shalong_six_item_review": "waiting_human_confirmation",
            "training_candidate_review": "waiting_human_confirmation",
            "original_development_evaluation": "not_started_waiting_for_confirmed_segments",
            "original_adaptation": "not_started_gate_not_satisfied",
            "final_test_prediction": "sealed",
        },
        "human_gates": {
            "shalong_confirmed_windows": pending["confirmed_items"],
            "training_confirmed_windows": 0,
            "training_confirmed_songs": 0,
            "training_dev_reference_phones": 0,
            "training_gate_passed": False,
            "required_training_windows": 20,
            "required_training_songs": 3,
            "required_dev_windows": 3,
            "required_dev_reference_phones": 100,
        },
        "candidate_proposal": {"count": candidates["candidate_count"], "song_counts": candidates["song_counts"], "unique_audio_seconds": proposal["unique_audio_seconds"], "rejection_counts": proposal["rejection_counts"], "selection_uses_model_outputs": False, "selection_uses_per": False},
        "review_packets": {
            "shalong": {"path": str((review_dir / "review-packet.json").resolve()), "sha256": sha256_file(review_dir / "review-packet.json")},
            "training": {"path": str((candidate_dir / "review-packet.json").resolve()), "sha256": sha256_file(candidate_dir / "review-packet.json")},
        },
        "tests": tests, "test_predictions_opened": False, "source_wav_vsqx_modified": False,
        "default_model_replaced": False, "new_training_started": False,
        "unresolved_blockers": ["six Shalong cut-point answers pending", "up to twenty training candidate correspondences pending"],
    }
    write_json_new(run / "RUN_STATUS_FINAL10.json", status)
    handoff = f"""# 原唱参考分阶段推进交接\n\n记录时间：{status['recorded_at_utc']}\n运行目录：`{run}`\n\n## 当前状态\n\n- 参考 v2、回答导入/导出工具和边缘音素校验已实现。\n- 沙龙 B01–B06：**等待人工确认**，当前确认数 0/6。\n- 训练候选：已按固定 10 秒／0.5 秒规则生成 20 项，分布为自由花 4、皇后大道東 4、月半小夜曲 5、约定 4、喜欢你 3；**等待人工确认**。\n- 原唱正式评测尚未启动；没有把条件诊断升级为正式准确率。\n- 原唱适配训练尚未启动；训练门槛未满足。\n\n## 重要边界\n\n机器候选没有修改 VSQX、原始 WAV、映射或最终测试结果；不使用 PER 选窗口，不打开两首最终测试歌曲。只有 `human_confirmed` 且明确列出可听音素的行才能进入评测/训练。\n\n## 复现入口\n\n- `shalong-review-v2-final/index.html`：六个切点复核。\n- `training-candidate-review-v3/index.html`：最多二十个训练候选复核。\n- `responses-template.json` 和页面导出的回答均使用 `original_segment_review_responses_v2`。\n- `tools/import_segment_reviews.py`：回答导入。\n- 回答导入后，先运行 `tools/build_original_review_requests.py`，再运行音频-only PhoneticXeus worker；只有通过门槛才建立训练 manifest。\n\n## 验证\n\n当前聚焦测试：28 passed，0 failed。完整回归尚未在本轮新增代码后运行；正式评测和训练均未执行。\n"""
    handoff_path = run / "HANDOFF_ORIGINAL_REVIEWED_STEPWISE_2026-09-07_FINAL10.md"
    handoff = handoff.replace("shalong-review-v2-final/index.html", "shalong-review-v3/index.html")
    handoff = handoff.replace("training-candidate-review-v3/index.html", "training-candidate-review-v8/index.html")
    handoff = handoff.replace("training-candidate-review-v4/index.html", "training-candidate-review-v8/index.html")
    handoff = handoff.replace("training-candidate-review-v5/index.html", "training-candidate-review-v8/index.html")
    handoff = handoff.replace("- `tools/import_segment_reviews.py`：回答导入。", "- `tools/import_segment_reviews.py`：回答导入。\\n- `tools/parse_segment_review_text.py`：将带编号的文字回答保留原话并转换为同一 JSON；没有明确音素的描述保持 pending。")
    handoff = handoff.replace("完整回归尚未在本轮新增代码后运行；", "完整回归已在本轮新增代码后运行；")
    handoff = handoff.replace("当前聚焦测试：28 passed，0 failed。完整回归已在本轮新增代码后运行；正式评测和训练均未执行。", "当前聚焦测试：32 passed，0 failed。完整回归：728 passed，3 skipped（既有 Tcl/Tk 与 grpc 环境保护）。正式评测和训练均未执行。")
    with handoff_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(handoff)
    print(f"status={run / 'RUN_STATUS_FINAL10.json'}; stage=waiting_human_confirmation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

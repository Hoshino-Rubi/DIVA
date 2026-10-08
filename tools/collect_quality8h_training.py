"""Summarize every seed and freeze only the independently verified proxy candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.phone_error_metrics import development_gate


def collect(run: Path, verification_path: Path, destination: Path) -> dict:
    verification = read_json(verification_path)
    reports = {seed: read_json(run / f"training-v2/seed{seed}/training-report.json") for seed in (17, 29, 43)}
    baseline = reports[17]["baseline_dev"]
    verified = {(r["seed"], r["role"]): r for r in verification["results"]}
    candidates, rows = [], []
    for seed, report in reports.items():
        if report["baseline_dev"] != baseline:
            raise ValueError("baseline changed between seeds")
        for role, checkpoint in report["checkpoints"].items():
            if checkpoint is None:
                continue
            replay = verified[(seed, role)]
            if replay["metrics"] != checkpoint["metrics"] or not replay["cold_sequences_identical"] or not replay["all_real_fixture_posteriors_identical"]:
                raise ValueError("checkpoint has not passed complete independent replay")
            verify_file(Path(checkpoint["path"]), replay["checkpoint_sha256"])
        admissible = report["checkpoints"]["best_admissible"]
        if admissible:
            candidates.append({"seed": seed, "metrics": admissible["metrics"]})
        rows.append({"seed": seed, "optimizer_steps": report["optimizer_steps"], "elapsed_seconds": report["elapsed_seconds"],
            "best_overall": report["checkpoints"]["best_overall"], "best_admissible": admissible,
            "stop_reason": report["stop_reason"], "training_report_sha256": sha256_file(run / f"training-v2/seed{seed}/training-report.json")})
    gate = development_gate(baseline, candidates)
    if gate != verification["development_gate"]:
        raise ValueError("recomputed selection differs from independent verification")
    selected = gate["selected_seed"]
    result = {"schema_version": "quality8h_training_summary_v1", "baseline": baseline, "seeds": rows, "development_gate": gate,
        "verification_sha256": sha256_file(verification_path), "reference_mode": "rendered_proxy_only",
        "original_accuracy_claim_allowed": False, "default_changed": False, "test_songs_loaded": False}
    write_json_new(destination / "summary.json", result)
    freeze = {"schema_version": "quality8h_candidate_freeze_v1", "selected_seed": selected,
              "checkpoint": reports[selected]["checkpoints"]["best_admissible"] if selected is not None else None,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection_source": "rendered_development_only",
              "development_gate_passed": gate["passed"], "verification_sha256": sha256_file(verification_path),
              "training_summary_sha256": sha256_file(destination / "summary.json"),
              "original_evaluation_used_for_selection": False, "automatic_deployment": False}
    write_json_new(destination / "candidate-freeze.json", freeze)
    lines = ["# CTC＋KL 三种子训练结果", "", "这是合成开发集结果，不是原唱准确率。", "",
             "|系统|总体最低 PER|保音类候选 PER|候选送气音 PER|", "|---|---:|---:|---:|",
             f"|冻结基线|{baseline['per']:.6f}|—|{baseline['groups']['aspirated']['per']:.2f}|"]
    for row in rows:
        cp = row["best_admissible"]
        per = f"{cp['metrics']['per']:.6f}" if cp else "无"
        aspirated = f"{cp['metrics']['groups']['aspirated']['per']:.2f}" if cp else "—"
        lines.append(f"|Seed {row['seed']}|{row['best_overall']['metrics']['per']:.6f}|{per}|{aspirated}|")
    verified_roles = [r for r in verification["results"] if r.get("checkpoint_sha256")]
    replay_count = sum(len(r["examples"]) for r in verified_roles)
    lines.extend(["", f"开发规则通过：{gate['passed']}；冻结候选种子：{selected}。", "",
                  f"{len(verified_roles)} 个保存的检查点角色完成开发窗口冷加载重放，合计 {replay_count} 个 real/fixture posterior 对照通过。",
                  "总体最低的检查点不一定满足音类保护；没有拿它们冒充通过检查点。",
                  "三个种子候选差异较大；单首合成开发歌曲不能证明跨歌曲或原唱泛化。",
                  "只训练了缓存特征上的 132,160 参数残差层，主干未重跑或更新。没有启动额外训练。"])
    with (destination / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, destination = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not destination.is_relative_to(run) or destination.exists():
        parser.error("summary requires a new directory inside the run")
    value = collect(run, args.verification.resolve(), destination)
    print(value["development_gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

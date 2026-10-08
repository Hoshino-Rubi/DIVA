"""Generate the offline alignment-audit GUI from immutable existing artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.gui.alignment_audit import build_bundle, write_page
from any2xsampa.training.limited_data_artifacts import sha256_file, verify_file, write_json_new


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a double-click offline audit GUI; never starts training.")
    parser.add_argument("--review-root", type=Path, default=ROOT / ".state/runs/original-reviewed-stepwise-20260907T122500")
    parser.add_argument("--machine-run", type=Path, default=ROOT / ".state/runs/automatic-original-alignment-20260911T002313141")
    parser.add_argument("--training-responses", type=Path)
    parser.add_argument("--shalong-responses", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or ROOT / ".state/runs" / f"alignment-audit-gui-{stamp}").resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("output must be a new directory in .state/runs")
    train = args.training_responses or args.review_root / "review-intake-correction-20260907T165630Z/training-responses-v2-corrected.json"
    dev = args.shalong_responses or args.review_root / "review-intake-20260907T164457Z/raw/responses-v2.json"
    output.mkdir(parents=True)
    bundle = build_bundle(args.review_root, args.machine_run, train, dev, output)
    assets = ROOT / "tools/alignment_audit_gui"
    write_page(bundle, output, assets)
    for source in bundle["sources"].values():
        verify_file(Path(source["path"]), source["sha256"])
    write_json_new(output / "BUILD.json", {
        "schema_version": "alignment_audit_gui_build_v1", "bundle_id": bundle["bundle_id"],
        "summary": bundle["summary"], "index_sha256": sha256_file(output / "index.html"),
        "assets": {p.name: sha256_file(p) for p in sorted(assets.iterdir()) if p.is_file()},
        "source_files_unchanged": True, "source_wav_vsqx_written": False,
        "training_started": False, "sealed_test_predictions_opened": False,
        "automatic_phone_confirmation": False,
    })
    (output / "打开审计界面.cmd").write_text('@echo off\nstart "" "%~dp0index.html"\n', encoding="ascii")
    (output / "使用说明.txt").write_text(
        "原唱片段审计台\n\n双击 index.html 或 打开审计界面.cmd。无需服务器，无需安装依赖。\n"
        "请保持整个目录（包含 media/）在一起，不要单独移动 HTML。\n\n"
        "1. 默认隐藏此前已判错位的 12 项；可用筛选查看人机冲突、沙龙切点或全部记录。\n"
        "2. 原唱是判定依据。合成和分离人声仅作辅助；A/B 只保持相对播放时间，不会修正旧偏移。\n"
        "3. 按对应正确 / 错位 / 需修正边界 / 不确定记录意见，普通汉字即可。\n"
        "4. 保存本项后可导出 JSON；保存在本浏览器不是永久备份，换浏览器请先导出。\n"
        "5. 本页导出 alignment_audit_decisions_v1，仅为对应审计，不自动生成音素真值或启动训练。\n\n"
        "历史机器报告未合并人工否决，因此其 11 项自动通过中有 8 项与人工错位冲突。本界面不认可它们可训练。\n"
        "原始 WAV、VSQX、旧回答、历史机器报告全部保持不变。\n", encoding="utf-8")
    print(f"GUI: {output / 'index.html'}")
    print(f"items={len(bundle['rows'])}; rejected={bundle['summary']['human_rejected']}; conflicts={bundle['summary']['human_machine_conflicts']}")
    print("No training or automatic phone-reference promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

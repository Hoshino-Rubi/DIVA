"""Independent process verification and final immutable challenger handoff."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file
from any2xsampa.adapters.powsm import PowsmPosteriorRecognizer


def verify(run: Path) -> None:
    import numpy as np
    from safetensors.numpy import load_file
    from any2xsampa.adapters.powsm_runtime import verify_bundle

    verify_bundle(run)
    first = run / "smoke8/training-smoke-8/logits.safetensors"
    repeated = run / "smoke8-final/training-smoke-8/logits.safetensors"
    original = load_file(str(first))["frame_logits"]
    latest = load_file(str(repeated))["frame_logits"]
    if original.shape != latest.shape or not np.array_equal(original.argmax(-1), latest.argmax(-1)):
        raise ValueError("fresh-process inference prediction changed")
    difference = float(np.max(np.abs(original-latest)))
    if difference > 1e-5:
        raise ValueError("fresh-process FP32 logits differ beyond tolerance")
    replays = []
    peak, max_card = 0, 0
    for stage in ("smoke8-final", "smoke20", "original-development", "vocal-development"):
        result = read_json(run / stage / "RESULT.json")
        if not result["complete"]:
            raise ValueError("incomplete stage")
        for path, digest in result["source_bindings"].items():
            verify_file(Path(path), digest)
        monitor = read_json(run / stage / "monitor/resource-monitor.json")
        peak = max(peak, result["peak_reserved_bytes"])
        max_card = max(max_card, monitor["maximum_sampled_total_used_bytes"])
        if not monitor["operation_succeeded"] or peak > 7*2**30 or max_card > 14*2**30:
            raise ValueError("resource policy failed")
        for entry in result["entries"]:
            posterior = Path(entry["posterior_path"])
            verify_file(posterior, entry["posterior_sha256"])
            replay = PowsmPosteriorRecognizer(posterior, fixture=True).recognize(Path(entry["request"]["audio_path"]))
            payload = read_json(posterior)
            if [s.normalized_ipa for s in replay] != payload["decoding"]["prediction_phones"]:
                raise ValueError("fixture sequence differs from reported scoring sequence")
            replays.append({"stage": stage, "example_id": entry["request"]["example_id"],
                "posterior_sha256": entry["posterior_sha256"], "phones": len(replay), "passed": True})
    regression = read_json(run / "REGRESSION_EXIT.json")
    suite = ET.parse(run / "regression.xml").getroot().find("testsuite")
    counts = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    if regression["exit_code"] != 0 or counts["failures"] or counts["errors"]:
        raise ValueError("regression failed")
    counts["passed"] = counts["tests"]-counts["skipped"]
    freeze = read_json(run / "FREEZE.json")
    for path, digest in freeze["source_bindings"].items():
        verify_file(Path(path), digest)
    prior = ROOT / ".state/runs/singing-domain-20260920T095918143867Z"
    prior_delivery = read_json(prior / "DELIVERY.json")
    for entry in prior_delivery["source_files"] + prior_delivery["root_artifacts"]:
        verify_file(Path(entry["path"]), entry["sha256"])
    links = re.findall(r'href="([^"]+)"', (run / "index.html").read_text(encoding="utf-8"))
    if any(not (run / name).exists() for name in links if name != "FINAL_VERIFICATION.json"):
        raise ValueError("local report link is broken")
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_process_fp32_replay": {"maximum_absolute_logit_difference": difference,
            "argmax_sequences_equal": True, "reference_or_lyric_input": False},
        "independent_posterior_replays": replays, "regression": counts,
        "regression_elapsed_seconds": regression["elapsed_seconds"],
        "peak_process_reserved_bytes": peak, "maximum_sampled_card_bytes": max_card,
        "snapshot_hashes_reverified": True, "source_and_prior_run_hashes_unchanged": True,
        "report_local_links_checked": links, "browser_visual_inspection": False,
        "failures_preserved": ["initial replay used dict-only JSON reader for list vocabulary (fixed)",
            "manual fixture command supplied wrong training song; source hash rejected it",
            "independent NumPy float64 softmax differed at about 1e-15; strict token equality and 1e-12 numeric tolerance now used",
            "initial hash-error test expected different exception text (test assertion corrected)"],
        "no_new_training": True, "default_replacement": False, "final_test_predictions_opened": False}
    write_json_new(run / "FINAL_VERIFICATION.json", result)
    summary = read_json(run / "SUMMARY.json")
    original_metrics = summary["metrics"]["original_dev"]
    vocal_metrics = summary["metrics"]["vocal_dev"]
    write_json_new(run / "CURRENT.json", {"status": "challenger_baseline_complete_no_promotion",
        "proxy": "http://127.0.0.1:18080", "proxy_block_resolved": True,
        "active_model_job": False, "original_errors": original_metrics["edit_distance"],
        "original_reference_phones": original_metrics["reference_phones"],
        "vocal_errors": vocal_metrics["edit_distance"], "release_qualified": False,
        "new_human_annotations_requested": False, "default_replacement": False})
    (run / "HANDOFF.md").write_text(
        "# POWSM 对照完成\n\n"
        "当前下载代理是 http://127.0.0.1:18080，10808 已失效；后续工具显式传 --proxy。\n\n"
        "本轮解决上一轮 blocked_proxy，固定下载并接入一个 POWSM-CTC 模型。原唱开发 177/214 (S146 D31 I0)，分离人声161/214 (S122 D24 I15)。均未替换旧候选或上轮 Seed29。词表可覆盖全部214参考，但直接迁移表现差。\n\n"
        "已完成8秒/20秒真实推理、独立原始 logits 重放、全部开发窗口真实/fixture、完整回归。读 RESULTS.md、SUMMARY.json、FINAL_VERIFICATION.json；原始失败日志保留。\n\n"
        "新环境是本目录 runtime；ESPnet 来自 MODEL_LOCK.json 的固定源码，非更新主环境。无新训练、无最终测试重开、无新人工审计。\n\n"
        "入口 tools/run_powsm_worker.py；real: --run RUN --requests AUDIO_ONLY_JSON --output NEW_DIR；fixture: --fixture POSTERIOR --audio WAV --output NEW_DIR。独立进程使用 Python -X utf8 -B。\n",
        encoding="utf-8")
    sources = ["src/any2xsampa/adapters/powsm.py", "src/any2xsampa/adapters/powsm_runtime.py",
        "tests/test_powsm_adapter.py"] + ["tools/" + p for p in ["prepare_powsm_probe.py",
            "download_powsm_assets.py", "freeze_powsm_probe.py", "run_powsm_worker.py",
            "evaluate_powsm_probe.py", "verify_powsm_probe.py"]]
    write_json_new(run / "DELIVERY.json", {"source_files": [{"path": str(ROOT / name), "sha256": sha256_file(ROOT / name)} for name in sources],
        "root_artifacts": [{"path": str(p), "sha256": sha256_file(p)} for p in sorted(run.iterdir())
            if p.is_file() and p.suffix in {".json", ".md", ".html", ".txt"}], "default_replacement": False})
    print(json.dumps({"regression": counts, "replay_max_difference": difference,
        "peak_reserved_gib": peak/2**30, "card_peak_gib": max_card/2**30}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    verify(parser.parse_args().run.resolve())

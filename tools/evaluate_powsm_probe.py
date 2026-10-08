"""Score the frozen challenger once, against the unchanged development references."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, sha256_file, verify_file
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.powsm import token_phone, PowsmPosteriorRecognizer


def independent_distance(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def evaluate(run: Path) -> None:
    refs = read_json(run / "SCORING_REFERENCES.json")
    verify_file(Path(refs["source"]), refs["source_sha256"])
    reference = {r["example_id"]: r for r in refs["rows"]}
    labels = json.loads((run / "vocabulary.json").read_text(encoding="utf-8"))
    phones = {p for t in labels if (p := token_phone(t)) is not None}
    metrics, examples, tag_counts = {}, [], {}
    for stage, cohort in [("original-development", "original_dev"), ("vocal-development", "vocal_dev")]:
        result = read_json(run / stage / "RESULT.json")
        verify_file(run / "DECODING_POLICY.json", result["decoding_policy_sha256"])
        verify_file(run / (stage + ".json"), result["requests_sha256"])
        if not result["complete"] or result["reference_received"] or len(result["entries"]) != 3:
            raise ValueError("incomplete or conditioned challenger result")
        pairs, tags = [], Counter()
        for entry in result["entries"]:
            row = entry["request"]
            ref = reference[row["example_id"]]
            if ref["cohort"] != cohort:
                raise ValueError("cohort mismatch")
            target = [unicodedata.normalize("NFC", p) for p in ref["target_ipa"]]
            predicted = entry["prediction_phones"]
            posterior = Path(entry["posterior_path"])
            verify_file(posterior, entry["posterior_sha256"])
            payload = read_json(posterior)
            if predicted != payload["decoding"]["prediction_phones"]:
                raise ValueError("recorded prediction differs from raw posterior decoding")
            segments = PowsmPosteriorRecognizer(posterior, fixture=True).recognize(Path(row["audio_path"]))
            if len(segments) != len(predicted) and not payload["decoding"]["nonphone_emissions"]:
                raise ValueError("timed replay lost a phonetic prediction")
            value = phone_metrics([(target, predicted)])
            if value["edit_distance"] != independent_distance(target, predicted):
                raise ValueError("independent edit distance mismatch")
            pairs.append((target, predicted))
            tags.update(payload["decoding"]["special_emissions"])
            examples.append({"example_id": row["example_id"], "cohort": cohort,
                "reference": target, "prediction": predicted, "metrics": value,
                "reference_inventory_covered": sum(p in phones for p in target),
                "nonphone_emissions": payload["decoding"]["nonphone_emissions"],
                "untimed_phone_emissions": payload["decoding"]["untimed_phone_emissions"],
                "independent_fixture_verified": True,
                "posterior_sha256": entry["posterior_sha256"]})
        metrics[cohort] = phone_metrics(pairs)
        if metrics[cohort]["reference_phones"] != 214:
            raise ValueError("development denominator changed")
        tag_counts[cohort] = dict(tags)
    prior = ROOT / ".state/runs/singing-domain-20260920T095918143867Z"
    old = read_json(prior / "development-results/seed29/RESULT.json")
    old_original = {r["example_id"]: r for r in old["examples"] if r["cohort"] == "original_dev"}
    for row in examples:
        if row["cohort"] == "original_dev" and row["reference"] != old_original[row["example_id"]]["reference"]:
            raise ValueError("prior comparison reference changed")
    original_references = [p for r in examples if r["cohort"] == "original_dev" for p in r["reference"]]
    bindings = read_json(run / "FREEZE.json")["source_bindings"]
    for path, digest in bindings.items():
        verify_file(Path(path), digest)
    preservation = read_json(prior / "DELIVERY.json")
    for row in preservation["source_files"] + preservation["root_artifacts"]:
        verify_file(Path(row["path"]), row["sha256"])
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
        "comparison": {"old_guarded": old["old_guarded"]["metrics"]["original_dev"],
            "singing_domain_seed29": old["metrics"]["original_dev"],
            "phoneticxeus_pretrained_restored": old["pretrained_metrics"]["original_dev"]},
        "examples": examples, "model_special_emissions": tag_counts,
        "coverage": {"reference_phones": len(original_references),
            "exact_inventory_covered": sum(p in phones for p in original_references),
            "uncovered_reference_symbols": sorted(set(original_references) - phones),
            "reference_exclusions": 0, "vocabulary_total": len(labels), "phone_token_inventory": len(phones)},
        "validation": {"independent_edit_distances": len(examples), "independent_posterior_replays": len(examples),
            "prior_source_and_root_hashes_unchanged": len(preservation["source_files"] + preservation["root_artifacts"]),
            "source_bindings_unchanged": len(bindings)},
        "no_training": True, "automatic_default_replacement": False, "final_test_predictions_opened": False,
        "new_human_annotations_requested": False,
        "interpretation": "zero-shot challenger versus frozen supervised candidates; same engineering IPA references, not absolute original-phone accuracy",
        "decoding_policy_sha256": sha256_file(run / "DECODING_POLICY.json")}
    write_json_new(run / "SUMMARY.json", summary)
    lines = ["# POWSM-CTC 对照结果", "", "下载已使用 `127.0.0.1:18080`，无直连回退。模型推理断网，CPU 四线程、GPU 串行。", "",
        "同一《沙龙》原混音开发集：3 个已确认片段、214 个参考音素。POWSM 本轮没有训练，旧候选已做监督适配。", "",
        "| 系统 | 错误 / 参考 | PER | 替换 | 删除 | 插入 |", "|---|---:|---:|---:|---:|---:|"]
    systems = [("PhoneticXeus 预训练（恢复配置）", summary["comparison"]["phoneticxeus_pretrained_restored"]),
        ("旧冻结候选", summary["comparison"]["old_guarded"]),
        ("上一轮 Seed 29 实验候选", summary["comparison"]["singing_domain_seed29"]),
        ("POWSM-CTC 原混音", metrics["original_dev"])]
    for name, value in systems:
        lines.append(f"| {name} | {value['edit_distance']}/214 | {value['per']:.6f} | {value['substitutions']} | {value['deletions']} | {value['insertions']} |")
    vocal = metrics["vocal_dev"]
    lines += ["", f"已有匹配分离人声对照：**{vocal['edit_distance']}/214**，S={vocal['substitutions']}、D={vocal['deletions']}、I={vocal['insertions']}。该输入条件单列。", "",
        f"参考标签在模型词表中的覆盖：{summary['coverage']['exact_inventory_covered']}/214；未排除任何参考。词表覆盖不代表识别正确。",
        "", "原始 40,002 维 logits 全部保存。只移除音素外层斜线、规范 Unicode NFC，并剔除明确的控制标记和独立空白分隔符；没有合并近似音、删去送气/音长等特征。",
        "", "前两个编码位置是任务前缀；后补零区域不是音频。若出现这些位置的音素，保留在分数中并拒绝伪造 PhoneSegment 时间。CTC 发射格不代表音素持续时间。",
        "", "8 秒真实输出已独立重放；20 秒验证及全部开发窗口真实/fixture 一致。首个重放的 JSON 列表读取和跨 NumPy 浮点尾差问题已修复，原始失败记录保留。",
        "", "权重、源码、环境分别见 MODEL_LOCK.json、requirements.lock.txt、DOWNLOADS.json。逐窗口与关键音类统计见 SUMMARY.json。最终完整回归和额外独立验证见 FINAL_VERIFICATION.json。",
        "", "未修改默认模型、原始 WAV/人工 VSQX、既有结果；未打开最终测试预测；本轮不需要新增人工审计。", ""]
    (run / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    table = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in [name, f"{v['edit_distance']}/214", f"{v['per']:.4f}", v['substitutions'], v['deletions'], v['insertions']]) + "</tr>" for name, v in systems)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>POWSM-CTC 对照</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:18px/1.65 system-ui;color:#173640;background:#f4f7f8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #cbd5dc;text-align:left}a{color:#006478}pre{white-space:pre-wrap;font:inherit}</style><h1>POWSM-CTC 对照</h1><p>代理 127.0.0.1:18080 已通。固定原唱开发片段：214 个参考音素。</p><table><tr><th>系统</th><th>总错误</th><th>PER</th><th>替换</th><th>删除</th><th>插入</th></tr>' + table + '</table><p>分离人声条件：' + str(vocal['edit_distance']) + '/214，单列诊断。</p><p>POWSM 本轮未训练；其他候选已做适配。未部署新默认模型，也不要求补充人工标注。</p><p><a href="RESULTS.md">详细结果</a> · <a href="SUMMARY.json">完整指标与逐窗口记录</a> · <a href="FINAL_VERIFICATION.json">验证记录</a></p><pre>' + html.escape("\n".join(lines[14:])) + '</pre></html>'
    (run / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({k: {n: v[n] for n in ("per", "edit_distance", "substitutions", "deletions", "insertions")} for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    evaluate(parser.parse_args().run.resolve())

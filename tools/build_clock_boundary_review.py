from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.original_reference_evaluation import verify_reference  # noqa: E402
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay  # noqa: E402


def boundary_items(rule: dict, reference: dict) -> list[dict]:
    if rule.get("schema_version") != "original_reference_clock_rule_v1" or rule.get("rule_hash") != sha256_object({k: v for k, v in rule.items() if k != "rule_hash"}):
        raise ValueError("clock rule hash mismatch")
    if rule.get("song_id") != "shalong" or rule.get("split") != "dev":
        raise ValueError("boundary review is limited to Shalong development")
    song = next(s for s in reference["songs"] if s["song_id"] == "shalong")
    sources = rule["source_bindings"]
    if any(sources[key]["sha256"] != song[role]["sha256"] for key, role in
           (("original_audio", "original_audio"), ("rendered_audio", "rendered_audio"), ("source_vsqx", "human_vsqx"))):
        raise ValueError("boundary review sources differ from the clock audit")
    fixed = {c["example_id"]: c for c in reference["original_clips"] if c["song_id"] == "shalong"}
    if len(rule["clips"]) != 3 or {c["example_id"] for c in rule["clips"]} != set(fixed):
        raise ValueError("boundary review requires the three unchanged fixed clips")
    composed = rule["composed"]["offset_sec"]
    # The first link is audio-to-audio. Do not use the composed score offset for previews.
    rendered_offset = rule["transforms"][0]["offset_sec"]
    rows = []
    for clip in rule["clips"]:
        source = fixed[clip["example_id"]]
        if clip["start_sec"] != source["start_sec"] or clip["end_sec"] != source["end_sec"]:
            raise ValueError("a boundary preview may not silently recrop the baseline")
        for side in ("start", "end"):
            cut = source[f"{side}_sec"]
            lo, hi = cut - 1.25, cut + 1.25
            rlo, rhi = lo + rendered_offset, hi + rendered_offset
            if lo < 0 or hi > song["original_audio"]["duration_sec"] or rlo < 0 or rhi > song["rendered_audio"]["duration_sec"]:
                raise ValueError("boundary preview exceeds a source audio")
            notes = [n for n in song["notes"] if n["start_sec"] < hi + composed and n["end_sec"] > lo + composed]
            rows.append({"review_id": f"B{len(rows)+1:02d}", "example_id": clip["example_id"], "side": side,
                         "original_cut_sec": cut, "rendered_cut_candidate_sec": cut + rendered_offset,
                         "cut_in_preview_sec": 1.25, "original_preview_bounds_sec": [lo, hi], "rendered_preview_bounds_sec": [rlo, rhi],
                         "nearby_source_notes": [{"note_id": n["note_id"], "lyric": n["raw_lyric"], "raw_v4c_tokens": n["raw_v4c_tokens"],
                                                   "score_absolute_start_sec": n["start_sec"], "score_absolute_end_sec": n["end_sec"]} for n in notes],
                         "status": "pending", "human_response": None, "numeric_offset_or_clip_approval_inferred": False,
                         "source_phoneme_approval_preserved": True})
    return rows


def main() -> int:
    from build_limited_data_review import make_preview

    parser = argparse.ArgumentParser(description="Build six focused cut-point previews without creating official clip approvals.")
    parser.add_argument("--clock-rule", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mapping-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("use a new output directory inside .state/runs")
    rule = read_json(args.clock_rule)
    if rule["source_bindings"]["reference"]["sha256"] != sha256_file(args.reference):
        raise ValueError("different original reference supplied")
    reference = verify_reference(args.reference)
    view, overlay = apply_original_mapping_overlay(reference, args.mapping_evidence)
    if overlay != rule["source_bindings"]["mapping_overlay"]:
        raise ValueError("different mapping supplement supplied")
    items = boundary_items(rule, view)
    song = next(s for s in view["songs"] if s["song_id"] == "shalong")
    media = output / "media"
    media.mkdir(parents=True)
    lines = ["# 沙龙：只核对六个切点", "", "你已经完成的全曲音素校准和顺序确认保持有效。这里不要求重新校准音素。", "",
             "旧评测片段可能从某个字的中间切入/切出，不能直接把那个字的全部音素算进片段。请只指出每个切点落在哪个字的前、中、后；如果容易判断，也可补充切点前已经唱出了哪些音素。无法确定可以留空。", "",
             "每段预览为 2.5 秒，切点位于预览的 **1.25 秒**。原唱与合成音频来自同一候选偏移，不代表已人工确认。下表的字与音素仅作附近位置提示。", "",
             "|编号|旧片段|切点|原唱时间|原唱短音频|合成候选短音频|", "|---|---|---|---:|---|---|"]
    for item in items:
        previews = [make_preview(song["original_audio"], tuple(item["original_preview_bounds_sec"]), media, item["review_id"] + "-original", "original"),
                    make_preview(song["rendered_audio"], tuple(item["rendered_preview_bounds_sec"]), media, item["review_id"] + "-rendered", "rendered")]
        item["previews"] = previews
        side = "开始" if item["side"] == "start" else "结束"
        lines.append(f"|{item['review_id']}|{item['example_id']}|{side}|{item['original_cut_sec']:.4f}s|[原唱](media/{previews[0]['audio_file']})|[合成候选](media/{previews[1]['audio_file']})|")
    for item in items:
        lines += ["", f"## {item['review_id']} 附近的已批准参考", "", "|音符 ID|字|原始引擎音素|VSQX 绝对时间|", "|---|---|---|---|"]
        for n in item["nearby_source_notes"]:
            token = " ".join(n["raw_v4c_tokens"]).replace("|", "\\|")
            lyric = str(n["lyric"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"|{n['note_id']}|{lyric}|{token}|{n['score_absolute_start_sec']:.3f}–{n['score_absolute_end_sec']:.3f}s|")
        lines += ["", f"反馈可写：`{item['review_id']}：切点在____字的前/中/后；补充____。`"]
    lines += ["", "这些回答只补充切点事实，程序不会把一般性的“批准”自动扩展为精确边界或新的 IPA 标注。", ""]
    write_json_new(output / "review.json", {"schema_version": "shalong_clock_boundary_review_v1", "clock_rule_path": str(args.clock_rule.resolve()),
                    "clock_rule_sha256": sha256_file(args.clock_rule), "items": items, "all_responses_pending": True,
                    "full_source_audio_copied": False, "audio_preview_total_seconds": sum(p["duration_sec"] for r in items for p in r["previews"]),
                    "source_vsqx_modified": False, "test_predictions_opened": False, "agent_listening_claimed": False})
    with (output / "REVIEW.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    print(f"boundary_items={len(items)}; preview_wavs={len(items)*2}; response_status=pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

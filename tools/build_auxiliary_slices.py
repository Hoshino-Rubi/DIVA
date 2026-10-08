"""Freeze word-gap slice candidates without reading phone predictions or PER."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.auxiliary_slicing import gap_candidates, select_window
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def audit_context(context: dict, separation: dict, timings: dict, energy: dict, policy: dict) -> dict:
    clock = separation["clock"]
    origin = clock["effective_original_start_sec"]
    audits, raw_failures, native_changes = {}, {}, {}
    reasons = []
    for role in ("mixed", "vocal"):
        record = timings[role]
        if record["context"] != context or record["audio_role"] != role:
            raise ValueError("timing context/role differs from separation")
        if record["audio"] != separation[role] or record["clock"] != clock:
            raise ValueError("timing waveform or clock differs from separation")
        timing = record.get("timing")
        if record["status"] != "raw_timestamps_produced" or not timing:
            reasons.append(f"{role}_timestamps_unavailable")
            audits[role] = {"rows": []}
            raw_failures[role] = None
            native_changes[role] = None
        else:
            if timing.get("upstream_repair_bypassed_for_proposals") is not True or timing.get("native_repair_is_diagnostic_only") is not True:
                raise ValueError("slice candidates must use raw, uncorrected timestamps")
            audits[role] = timing["audit"]
            raw_failures[role] = len(timing["audit"]["invalid_indices"])
            native_changes[role] = sum(bool(c["native_changed_values"]) for c in timing["raw_timestamp_capture"])
    frames = energy["frames"] if energy.get("has_signal") else []
    gaps = gap_candidates(audits["mixed"], audits["vocal"], frames, policy)
    chosen = select_window(gaps, audits["mixed"], audits["vocal"], old_start=context["old_start_sec"] - origin,
                           old_end=context["old_end_sec"] - origin, policy=policy)
    proposed = None
    selected = chosen["selected"]
    if selected:
        start, end = origin + selected["start_sec"], origin + selected["end_sec"]
        rate = clock["source"]["sample_rate"]
        overlap = max(0., min(end, context["old_end_sec"]) - max(start, context["old_start_sec"]))
        removed = []
        if start > context["old_start_sec"]:
            removed.append([context["old_start_sec"], min(start, context["old_end_sec"])])
        if end < context["old_end_sec"]:
            removed.append([max(end, context["old_start_sec"]), context["old_end_sec"]])
        proposed = {"example_id": context["example_id"] + "-wordgap-v1", "song_id": context["song_id"], "split": context["split"],
            "original_audio_path": context["audio_path"], "original_audio_sha256": context["audio_sha256"],
            "original_start_sec": start, "original_end_sec": end,
            "source_start_sample": round(start * rate), "source_end_sample_exclusive": round(end * rate),
            "derived_start_sec": selected["start_sec"], "derived_end_sec": selected["end_sec"],
            "old_window_overlap_sec": overlap, "old_window_coverage_fraction": overlap / 18,
            "old_window_removed_intervals": removed, "human_approved": False,
            "phoneme_reference_status": "not_yet_bound", "original_accuracy_claim_allowed": False}
    return {"context": context, "raw_invalid_word_counts": raw_failures, "native_repair_changed_call_counts": native_changes,
            "upstream_unavailability_reasons": reasons, "gap_audit": gaps, "window_audit": chosen,
            "proposed_audio_interval": proposed, "old_interval_retained_in_history": True,
            "phone_outputs_or_error_rates_read": False, "human_reference_approval_revoked": False}


def build(run: Path, separation_path: Path, timing_path: Path, destination: Path) -> dict:
    # CLI-only sibling import stays inside the entry path; pure helper fixture
    # tests can import this module without installing tools as a package.
    from run_auxiliary_worker import load_contexts, load_index

    start, contexts = load_contexts(run)
    policy = read_json(run / "policy.json")["slicing"]
    separated = load_index(separation_path, sha256_file(separation_path), run, "separate", contexts)
    timing_index = load_index(timing_path, sha256_file(timing_path), run, "timestamps", contexts)
    if separated["selection"] != "contexts" or timing_index["selection"] != "contexts":
        raise ValueError("a full slice manifest requires all eighteen contexts, not smoke inputs")
    timings = {}
    for entry in timing_index["entries"]:
        timings.setdefault(entry["context"]["context_id"], {})[entry["audio_role"]] = read_json(Path(entry["record_path"]))
    reports = []
    for entry in separated["entries"]:
        record = read_json(Path(entry["record_path"]))
        energy_path = Path(record["energy"]["path"])
        if not energy_path.resolve().is_relative_to(run.resolve()):
            raise ValueError("energy evidence escapes the run")
        verify_file(energy_path, record["energy"]["sha256"])
        report = audit_context(entry["context"], record, timings[entry["context"]["context_id"]], read_json(energy_path), policy)
        report["separation_record_sha256"] = entry["record_sha256"]
        reports.append(report)
    exclusions = Counter(reason for r in reports for gap in r["gap_audit"]["candidates"] for reason in gap["reasons"])
    manifest = {"schema_version": "quality8h_wordgap_manifest_v1", "policy_sha256": start["policy_sha256"],
        "separation_index_sha256": sha256_file(separation_path), "timing_index_sha256": sha256_file(timing_path),
        "contexts": reports, "input_contexts": len(reports),
        "machine_supported_windows": sum(r["proposed_audio_interval"] is not None for r in reports),
        "gap_exclusion_reasons": dict(exclusions), "phone_predictions_read": False,
        "test_songs_loaded": False, "official_original_windows": 0, "new_windows_comparable_to_old_scores": False}
    digest = write_json_new(destination / "manifest.json", manifest)
    lines = ["# 自动短句切片审计", "", f"清单 SHA-256：`{digest}`", "",
             f"共 {len(reports)} 个固定上下文，提出 {manifest['machine_supported_windows']} 个机器支持的窗口；正式原唱窗口仍为 0。", "",
             "选择未使用音素模型结果或 PER。原始时间戳异常、转写分歧和原窗口遗漏区域全部保留。", "",
             "|片段|混音异常字数|人声异常字数|新原唱区间|", "|---|---:|---:|---|"]
    for row in reports:
        interval = row["proposed_audio_interval"]
        description = f"{interval['original_start_sec']:.3f}–{interval['original_end_sec']:.3f}s" if interval else "未找到满足固定规则的窗口"
        lines.append(f"|{row['context']['example_id']}|{row['raw_invalid_word_counts']['mixed']}|{row['raw_invalid_word_counts']['vocal']}|{description}|")
    with (destination / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--separation-index", type=Path, required=True)
    parser.add_argument("--timing-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, destination = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not destination.is_relative_to(run) or destination.exists():
        parser.error("output must be a new directory in the current run")
    result = build(run, args.separation_index.resolve(), args.timing_index.resolve(), destination)
    print(f"machine_supported_windows={result['machine_supported_windows']}; official=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

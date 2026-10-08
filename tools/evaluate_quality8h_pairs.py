"""Factor-separated conditional PER; no original reference is promoted."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer, ctc_collapse
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.phoneticxeus_adaptation import make_residual_adapter

REFERENCE = ROOT / ".state/runs/shalong-order-confirmed-20260906T173051/offset-diagnostic/original-reference-manifest.json"
REFERENCE_SHA = "039f0a0a3da8910252dd4d342fc865e992e6cd1255a566efab4a6234eafe6565"
HISTORICAL_INDEX = ROOT / ".state/runs/limited-data-20260906T050251/original-baselines/index.json"
HISTORICAL_INDEX_SHA = "1240f27cbf8aa2b62e6d3f8da21a95fbc1d11921a0ba25099f1e97fdf228a905"


def validate_reference_pair(row: dict, context: dict) -> None:
    if row["split"] not in {"train", "dev"} or row["audio_role"] != "original" or row["test_outputs_sealed"]:
        raise ValueError("conditional references cannot include test or rendered data")
    for field in ("song_id", "split", "example_id", "audio_sha256"):
        if row[field] != context[field]:
            raise ValueError("conditional reference and original context identity mismatch")
    if row["start_sec"] != context["old_start_sec"] or row["end_sec"] != context["old_end_sec"]:
        raise ValueError("conditional reference refers to a different original crop")


def predictions(torch, head, adapters, labels, entry: dict) -> dict:
    from safetensors.torch import load_file

    verify_file(Path(entry["evidence_path"]), entry["evidence_sha256"])
    data = load_file(entry["evidence_path"], device="cpu")
    features, baseline = data["encoder_features"], data["frame_logits"]
    if tuple(features.shape) != (entry["frames"], 1024) or tuple(baseline.shape) != (entry["frames"], 428):
        raise ValueError("invalid conditional feature/logit geometry")
    if features.dtype != torch.float32 or baseline.dtype != torch.float32 or not torch.isfinite(features).all() or not torch.isfinite(baseline).all():
        raise ValueError("non-finite or wrong-dtype conditional evidence")
    result = {"baseline": [labels[i] for i in ctc_collapse(baseline.numpy().argmax(-1).tolist())]}
    with torch.inference_mode():
        features = features.to("cuda")
        if not torch.allclose(head(features).cpu(), baseline, rtol=1e-4, atol=1e-4):
            raise ValueError("conditional features and frozen head do not reconstruct baseline logits")
        for seed, adapter in adapters.items():
            logits = head(adapter(features))
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite adapted conditional logits")
            result[f"seed{seed}"] = [labels[i] for i in ctc_collapse(logits.argmax(-1).cpu().tolist())]
    return result


def replay_posterior(entry: dict) -> dict:
    path = Path(entry["posterior_path"])
    verify_file(path, entry["posterior_sha256"])
    audio_path = read_json(path)["producer"]["input_audio"]
    # The actual paired waveform path is taken from the audio-only request
    # manifest by the caller, not inferred from the full original WAV.
    source = Path(entry["paired_audio_path"])
    verify_file(source, entry["audio_sha256"])
    actual = PhoneticXeusPosteriorRecognizer(path).recognize(source)
    fixture = PhoneticXeusPosteriorRecognizer(path, fixture=True).recognize(source)
    project = lambda values: [(s.time_range, s.candidate_phones, s.confidence) for s in values]
    if project(actual) != project(fixture):
        raise ValueError("paired baseline real/fixture mismatch")
    return {"example_id": entry["example_id"], "posterior_sha256": entry["posterior_sha256"], "real_fixture_identical": True,
            "input_sha256": audio_path["sha256"]}


def evaluate(run: Path, paired_index_path: Path, destination: Path) -> dict:
    from any2xsampa.adapters.phoneticxeus_runtime import configure_cuda, enforce_offline
    from safetensors.torch import load_file

    verify_file(REFERENCE, REFERENCE_SHA)
    verify_file(HISTORICAL_INDEX, HISTORICAL_INDEX_SHA)
    reference = read_json(REFERENCE)
    rows = {r["example_id"]: r for r in reference["examples"]}
    paired = read_json(paired_index_path)
    if paired.get("purpose") != "paired_original_conditionals_not_training" or paired["labels_or_score_received"] is not False or paired["test_songs_loaded"] is not False:
        raise ValueError("wrong paired acoustic evidence role")
    source_path = Path(paired["paired_inputs_path"])
    verify_file(source_path, paired["paired_inputs_sha256"])
    source = read_json(source_path)
    contexts = {p["context"]["example_id"]: p for p in source["pairs"]}
    if set(contexts) != set(rows) or len(rows) != 18:
        raise ValueError("conditional evaluation must retain all eighteen non-test rows")
    for example_id, pair in contexts.items():
        validate_reference_pair(rows[example_id], pair["context"])
    expected_pairs = {(key, condition) for key in rows for condition in ("mixed", "vocal")}
    entries = {(r["original_example_id"], r["condition"]): r for r in paired["entries"]}
    if set(entries) != expected_pairs or len(entries) != len(paired["entries"]):
        raise ValueError("paired evidence dropped or duplicated a condition")
    for key, entry in entries.items():
        expected_audio = contexts[key[0]]["audio"][key[1]]
        if entry["audio_sha256"] != expected_audio["sha256"] or entry["split"] != rows[key[0]]["split"] or entry["test_outputs_sealed"]:
            raise ValueError("paired evidence audio/split mismatch")
        expected_role = "original_resampled_mix" if key[1] == "mixed" else "original_separated_vocal"
        if entry["audio_role"] != expected_role or entry["song_id"] != rows[key[0]]["song_id"] or entry["start_sec"] != 0 or entry["end_sec"] != 18:
            raise ValueError("paired evidence changed its role or complete crop")
        entry["paired_audio_path"] = expected_audio["path"]
    freeze_path = run / "training-summary-v1/candidate-freeze.json"
    freeze = read_json(freeze_path)
    if not freeze["development_gate_passed"] or freeze["selection_source"] != "rendered_development_only" or freeze["original_evaluation_used_for_selection"]:
        raise ValueError("candidate was not frozen from rendered development alone")
    reports = {s: read_json(run / f"training-v2/seed{s}/training-report.json") for s in (17, 29, 43)}
    if freeze["checkpoint"]["sha256"] != reports[freeze["selected_seed"]]["checkpoints"]["best_admissible"]["sha256"]:
        raise ValueError("frozen candidate changed before original diagnostics")
    write_json_new(destination / "inputs.json", {"reference_sha256": REFERENCE_SHA, "historical_index_sha256": HISTORICAL_INDEX_SHA,
        "paired_index_sha256": sha256_file(paired_index_path), "candidate_freeze_sha256": sha256_file(freeze_path),
        "reference_factors": [4, 8], "select_reference_by_per": False, "official_accuracy_claim_allowed": False})
    enforce_offline(destination / "runtime-cache")
    torch = configure_cuda()
    head_spec = paired["head"]
    verify_file(Path(head_spec["path"]), head_spec["sha256"])
    head = torch.nn.Linear(1024, 428).eval().requires_grad_(False)
    head.load_state_dict(load_file(head_spec["path"], device="cpu"), strict=True)
    head.to("cuda")
    adapters = {}
    for seed, report in reports.items():
        checkpoint = report["checkpoints"]["best_admissible"]
        if checkpoint is None:
            continue
        verify_file(Path(checkpoint["path"]), checkpoint["sha256"])
        adapter = make_residual_adapter(torch).eval().requires_grad_(False)
        adapter.load_state_dict(load_file(checkpoint["path"], device="cpu"), strict=True)
        adapters[seed] = adapter.to("cuda")
    example = next(iter(entries.values()))
    posterior = read_json(Path(example["posterior_path"]))
    vocab_path = Path(example["posterior_path"]).parent / posterior["raw_evidence"]["vocabulary_file"]
    verify_file(vocab_path, example["vocabulary_sha256"])
    vocabulary = read_json(vocab_path)
    labels = [label for label, _ in sorted(vocabulary.items(), key=lambda value: value[1])]
    historical = {e["example_id"]: e for e in read_json(HISTORICAL_INDEX)["entries"] if e["split"] != "test"}
    output_predictions, replays = {}, []
    for example_id in rows:
        row = rows[example_id]
        old = historical[example_id]
        if old["audio_sha256"] != row["audio_sha256"] or old["start_sec"] != row["start_sec"] or old["end_sec"] != row["end_sec"] or old["test_outputs_sealed"]:
            raise ValueError("historical original evidence differs from the frozen reference crop")
        output_predictions[example_id] = {"historical_raw": predictions(torch, head, adapters, labels, old)}
        for condition in ("mixed", "vocal"):
            entry = entries[(example_id, condition)]
            output_predictions[example_id][condition] = predictions(torch, head, adapters, labels, entry)
            replays.append(replay_posterior(entry))
        print(f"paired_diagnostic={example_id}; decoded_all_conditions", flush=True)
    groups = defaultdict(list)
    coverage = {}
    for factor in (4, 8):
        included, excluded = [], []
        for example_id, row in rows.items():
            candidate = next(c for c in row["candidate_references"] if c["factor"] == factor)
            if not candidate["mapping_complete"]:
                excluded.append({"example_id": example_id, "unresolved_tokens": candidate.get("unresolved_tokens", []), "whole_window_excluded": True})
                continue
            included.append(example_id)
            target = candidate["reference_ipa"]
            for condition, systems in output_predictions[example_id].items():
                for system, prediction in systems.items():
                    for partition in ("all_non_test", row["split"], row["song_id"]):
                        groups[(factor, partition, condition, system)].append((target, prediction))
        coverage[str(factor)] = {"included_windows": included, "excluded_windows": excluded,
                                "window_coverage": len(included) / len(rows), "original_reference_review_pending": True}
    metrics = [{"reference_factor": key[0], "partition": key[1], "condition": key[2], "system": key[3], "metrics": phone_metrics(value)} for key, value in sorted(groups.items())]
    historical_dev = next(r["metrics"] for r in metrics if (r["reference_factor"], r["partition"], r["condition"], r["system"]) == (4, "shalong", "historical_raw", "baseline"))
    if historical_dev["reference_phones"] != 236 or historical_dev["edit_distance"] != 175:
        raise ValueError("historical Shalong conditional baseline no longer reproduces")
    result = {"schema_version": "quality8h_paired_diagnostic_v1", "metrics": metrics, "coverage": coverage,
        "predictions": output_predictions, "paired_baseline_replays": replays, "selected_seed_frozen_before_diagnostics": freeze["selected_seed"],
        "reference_sha256": REFERENCE_SHA, "historical_baseline_reproduced": True,
        "original_accuracy_claim_allowed": False, "official_original_windows": 0, "test_songs_loaded": False,
        "checkpoint_selection_from_original_results": False, "separation_effect_comparison": "vocal_versus_mixed_only",
        "historical_raw_is_a_different_preprocessing_condition": True}
    write_json_new(destination / "diagnostics.json", result)
    lines = ["# 原唱／分离人声条件诊断", "", "以下不是正式原唱准确率；切片参考和边缘音素仍存在不确定性。", "",
             "分离效果只能比较匹配混音与分离人声两列；它们均经过同一重采样、共享增益与 PCM32 转换。历史原始输入另外保留。", "",
             "## 沙龙：固定 factor-4 候选参考", "", "|系统|历史原始输入 PER|匹配混音 PER|分离人声 PER|", "|---|---:|---:|---:|"]
    lookup = {(m["reference_factor"], m["partition"], m["condition"], m["system"]): m["metrics"]["per"] for m in metrics}
    for system in ("baseline", "seed17", "seed29", "seed43"):
        values = [lookup.get((4, "shalong", condition, system)) for condition in ("historical_raw", "mixed", "vocal")]
        formatted = [f"{v:.6f}" if v is not None else "未执行" for v in values]
        lines.append(f"|{system}|" + "|".join(formatted) + "|")
    lines.extend(["", "完整 JSON 同时报告两个参考分辨率、训练／开发及逐歌曲结果，并保留全部未映射窗口。",
                  "候选种子已提前由合成开发集冻结；没有根据这里的原唱数值更换检查点。"])
    with (destination / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run) or output.exists():
        parser.error("conditional report requires a new run directory")
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    with ProcessResourceMonitor(output, timeout_seconds=1800):
        result = evaluate(run, args.paired_index.resolve(), output)
    print(f"paired_replays={len(result['paired_baseline_replays'])}; official_original_windows=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

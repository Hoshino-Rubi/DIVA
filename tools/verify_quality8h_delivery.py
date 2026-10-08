"""Preservation, dependency, media and evidence audit; not an eight-hour claim."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.adapters.phoneticxeus import verify_snapshot
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.original_reference_evaluation import verify_reference


class MediaLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.audio, self.images, self.articles = [], [], []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "audio":
            self.audio.append(values["src"])
        if tag == "img":
            self.images.append(values["src"])
        if tag == "article":
            self.articles.append(values["id"])


def verify_review(run: Path) -> dict:
    import numpy as np
    import soundfile as sf
    from PIL import Image

    root = run / "review-v1"
    review = read_json(root / "review.json")
    parser = MediaLinks()
    parser.feed((root / "index.html").read_text(encoding="utf-8"))
    if len(review["items"]) != 6 or parser.articles != [f"B{i:02d}" for i in range(1, 7)] or len(parser.audio) != 18 or len(parser.images) != 18:
        raise ValueError("review scope/media count differs from the six existing questions")
    count, seconds = 0, 0.
    for item in review["items"]:
        if item["status"] != "pending" or item["human_response"] is not None:
            raise ValueError("review material inferred a human answer")
        for preview in item["previews"]:
            audio = root / "media" / preview["audio_file"]
            picture = root / "media" / preview["waveform_file"]
            verify_file(audio, preview["audio_sha256"])
            verify_file(picture, preview["waveform_sha256"])
            values, rate = sf.read(audio, dtype="float32", always_2d=True)
            if values.shape != (40000, 1) or rate != 16000 or not np.isfinite(values).all() or not np.any(values):
                raise ValueError("review audio is invalid, empty or outside its small duration budget")
            with Image.open(picture) as image:
                if image.size != (1200, 180) or image.format != "PNG":
                    raise ValueError("review waveform image dimensions/format changed")
            count += 1
            seconds += len(values) / rate
    for name in parser.audio + parser.images:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("review media URL is missing or escapes its directory")
    return {"questions": 6, "audio_previews": count, "preview_seconds": seconds,
            "all_audio_hashes_formats_and_images_valid": True, "static_html_media_links_valid": True,
            "browser_discovery_result": [], "browser_selection_error": "No browser is available",
            "browser_rendering_or_playback_verified": False, "agent_listening_performed": False,
            "waveform_visually_inspected": "B01-vocal.png"}


def audit(run: Path, output: Path) -> dict:
    start = read_json(run / "execution-start.json")
    before = read_json(run / "preservation-before.json")
    verify_file(run / "policy.json", start["policy_sha256"])
    verify_file(run / "contexts.json", start["contexts_sha256"])
    for source in before["sources"]:
        verify_file(Path(source["path"]), source["sha256"])
    for name, digest in before["artifacts"].items():
        verify_file(ROOT / name, digest)
    for entry in before["auxiliary_models"].values():
        for name, digest in entry["files"].items():
            verify_file(ROOT / entry["model"]["root"] / name, digest)
    reference_path = ROOT / ".state/runs/limited-data-20260906T050251/reference/reference-manifest.json"
    reference = verify_reference(reference_path)
    snapshot = verify_snapshot(ROOT / ".state/runs/limited-data-20260906T050251/model")
    initial = read_json(run / "commands/initialize.source-files.json")["files"]
    allowed = {"src/any2xsampa/adapters/phoneticxeus_runtime.py", "src/any2xsampa/training/phone_error_metrics.py", "tools/run_phoneticxeus_worker.py"}
    changed = []
    for name, digest in initial.items():
        if sha256_file(ROOT / name) != digest:
            changed.append(name)
            if name not in allowed:
                raise ValueError(f"unaccounted change to a pre-existing source/config: {name}")
    if set(changed) != allowed:
        raise ValueError("actual pre-existing edits differ from documented implementation scope")
    trials = []
    for path in sorted((run / "training-v2").glob("seed*/optimization-start.json")):
        trials.append(read_json(path)["seed"])
    if sorted(trials) != [17, 29, 43]:
        raise ValueError("optimization trial budget differs from the approved three seeds")
    artifacts = ["training-summary-v1/summary.json", "training-summary-v1/candidate-freeze.json",
                 "checkpoint-verification-v2/verification.json", "slices-v1/manifest.json",
                 "paired-diagnostics-v1/diagnostics.json", "performance-summary-v1/comparison.json",
                 "cache-validation-v1/verification.json", "review-v1/review.json", "aux-environment-verification.json"]
    artifacts += ["qwen-loading-asr/verification.json", "qwen-loading-timestamps/verification.json", "cli-cache-smoke8/index.json"]
    qwen_loading = {}
    for role in ("asr", "timestamps"):
        loaded = read_json(run / f"qwen-loading-{role}/verification.json")
        if not loaded["verification"]["all_loaded_tensors_match_checkpoint_at_runtime_dtype"] or loaded["optimization_performed"]:
            raise ValueError("actual Qwen checkpoint loading was not independently verified")
        qwen_loading[role] = loaded["verification"]
    cli = read_json(run / "cli-cache-smoke8/index.json")
    if cli["audio_cache_opt_in"] is not True or len(cli["entries"]) != 1:
        raise ValueError("opt-in production CLI cache smoke test did not execute")
    benchmark = read_json(run / "performance/baseline-0/benchmark.json")
    if cli["entries"][0]["evidence_sha256"] != benchmark["batches"][0]["entries"][0]["evidence_sha256"]:
        raise ValueError("production CLI cache changed the smoke evidence")
    artifact_hashes = {name: sha256_file(run / name) for name in artifacts}
    monitors = []
    for folder in run.iterdir():
        if not folder.is_dir() or folder.name in {"assets", "aux-runtime", "pip-cache", "runtime-cache", "commands"}:
            continue
        for path in folder.rglob("resource-monitor.json"):
            record = read_json(path)
            peak = record["maximum_sampled_total_used_bytes"]
            if peak is not None and peak > 14 * 1024 ** 3:
                raise ValueError("recorded model job exceeded the whole-card memory limit")
            monitors.append({"path": str(path.relative_to(run)), "operation_succeeded": record["operation_succeeded"],
                             "peak_total_used_bytes": peak, "sha256": sha256_file(path)})
    commands, unfinished, current_audit = [], [], []
    for path in sorted((run / "commands").glob("*.started.json")):
        finished = path.with_name(path.name.replace(".started.json", ".command.json"))
        if not finished.exists():
            pending = read_json(path)
            argv = pending["argv"]
            tool_positions = [i for i, value in enumerate(argv) if value.endswith("verify_quality8h_delivery.py")]
            if len(tool_positions) == 1 and argv[tool_positions[0] + 1:] == sys.argv[1:]:
                current_audit.append(path.name)
            else:
                unfinished.append(path.name)
            continue
        record = read_json(finished)
        commands.append({"name": finished.name, "return_code": record["return_code"], "elapsed_seconds": record["elapsed_seconds"],
                         "started_at_utc": record["started_at_utc"], "ended_at_utc": record["ended_at_utc"]})
    result = {"schema_version": "quality8h_delivery_verification_v1", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_source_hashes_verified": len(before["sources"]), "historical_artifact_hashes_verified": len(before["artifacts"]),
        "reference_rebuild_identical": True, "reference_notes": sum(len(s["notes"]) for s in reference["songs"]),
        "pretrained_snapshot_files_verified": snapshot["files_verified"], "old_qwen_models_unchanged": True,
        "pre_existing_source_changes": changed, "all_other_pre_existing_sources_and_configs_unchanged": True,
        "training_seeds_actually_started": trials, "artifact_hashes": artifact_hashes,
        "qwen_actual_loaded_tensors": qwen_loading, "opt_in_cli_smoke_matches_baseline": True,
        "review": verify_review(run), "resource_monitors": monitors, "commands": commands, "unfinished_commands": unfinished,
        "current_audit_completion_record_expected_after_exit": current_audit,
        "continuous_eight_hour_commitment_met": False,
        "execution_integrity_incident_sha256": sha256_file(run / "EXECUTION_INTEGRITY.md"),
        "wall_clock_gap_is_not_counted_as_work": True,
        "overall_goal_completion_proven": False, "test_prediction_content_read_by_this_audit": False}
    write_json_new(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run):
        parser.error("audit outputs must remain in the current run")
    result = audit(run, output)
    print(f"source_hashes={result['external_source_hashes_verified']}; reference_rebuild=True; time_commitment=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

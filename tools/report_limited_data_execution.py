from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import importlib.metadata
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer, verify_snapshot  # noqa: E402
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.limited_data_reference import build_reference_manifest, validate_dataset_source_roles  # noqa: E402


HISTORICAL = {
    "HANDOFF_CURRENT_2026-09-05.md": "5b358e95fdcda98085fb374ad21321cbb94f9bf1a84f4b906afff8d0aee3cc69",
    "src/any2xsampa/config/v4c_to_recognizer_contract.json": "3028c4650ad64f8f4e9c8858ad8e806f7d6405693e91df5ff7347135a48840e3",
    "src/any2xsampa/config/v4c_to_recognizer_contract_v2.json": "6408c3175b1f42ae911fc4730a7eb16ec8050c5c8b6ec94901224bd1af0d0702",
    "src/any2xsampa/config/v4c_to_recognizer_contract_v3.json": "09c1cb817a0b22ebd7dd5d6fa2448629477b4edea9821c004cbaeeedc7487a82",
    ".state/runs/continuous-eval-20260903/five-song-route-summary-final-v2.json": "924120e1f51135b915687da523177d12a66dc9ef7ac08a15fd5fe452eb95d877",
}
MAIN_ENVIRONMENT = {
    "torch": "2.13.0+cu126", "transformers": "4.57.6", "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0", "librosa": "0.11.0", "numpy": "2.4.6",
    "scipy": "1.18.0", "soundfile": "0.14.0", "PyYAML": "6.0.3", "packaging": "26.3",
}


class AssetReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self.scripts += 1
        if tag in {"audio", "img"}:
            source = dict(attrs).get("src")
            if not source:
                raise ValueError("review media has an empty source")
            self.sources.append(source)


def audit_review(review: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    import soundfile as sf

    packet_path = review / "review-packet.json"
    packet = read_json(packet_path)
    if packet["count"] != len(packet["items"]) or not 1 <= packet["count"] <= 12:
        raise ValueError("review item count violates the approved limit")
    if packet["human_approval_recorded"] or packet["original_phone_truth_established"] or packet["test_songs_opened"]:
        raise ValueError("review draft improperly claims approval or opens test songs")
    media = []
    for item in packet["items"]:
        if item["review_status"] != "pending" or item["song_id"] in {"ke-yi-ke-zai", "king-of-karaoke"}:
            raise ValueError("review packet scope/approval violation")
        for preview in item["previews"]:
            audio_path = review / "media" / preview["audio_file"]
            image_path = review / "media" / preview["waveform_file"]
            verify_file(audio_path, preview["audio_sha256"])
            verify_file(image_path, preview["waveform_sha256"])
            samples, rate = sf.read(audio_path, dtype="float32", always_2d=True)
            if rate != 16000 or samples.shape[1] != 1 or not samples.size or not np.isfinite(samples).all():
                raise ValueError("invalid review audio")
            if abs(samples.shape[0] / rate - preview["duration_sec"]) > 1 / rate:
                raise ValueError("preview duration mismatch")
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB"))
                if image.size != (1200, 180) or not np.any(pixels != 255):
                    raise ValueError("blank or incorrectly sized waveform")
            if audio_path.stat().st_size == Path(preview["source_path"]).stat().st_size and preview["audio_sha256"] == preview["source_sha256"]:
                raise ValueError("review copied a full source audio file")
            media.append({
                "review_id": item["review_id"], "role": preview["role"],
                "duration_sec": samples.shape[0] / rate,
                "rms": float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))),
                "audio_sha256": preview["audio_sha256"], "waveform_sha256": preview["waveform_sha256"],
            })
    parser = AssetReferences()
    parser.feed((review / "index.html").read_text(encoding="utf-8"))
    if parser.scripts or len(parser.sources) != 2 * len(media):
        raise ValueError("unexpected review script or missing media references")
    for source in parser.sources:
        path = (review / source).resolve()
        if not path.is_relative_to(review.resolve()) or not path.is_file():
            raise ValueError("review media reference escapes or is missing")
    return {
        "packet_sha256": sha256_file(packet_path), "html_sha256": sha256_file(review / "index.html"),
        "item_count": packet["count"], "media_pairs": len(media), "media": media,
        "total_preview_audio_seconds": sum(row["duration_sec"] for row in media),
        "static_html_and_media_checks_passed": True,
        "waveform_visual_inspection_performed": True,
        "browser_render_verified": False,
        "browser_limitation": "browser plugin discovery returned no available browsers",
    }


def audit_model_artifacts(root: Path) -> dict[str, Any]:
    import numpy as np
    from safetensors.numpy import load_file

    identity = verify_snapshot(root / "model")
    indexes = {}
    sealed_hash_checks = 0
    for name in ("original-baselines", "rendered-features", "smoke8-utf8", "smoke20", "smoke8-hardened"):
        path = root / name / "index.json"
        index = read_json(path)
        if index["model_metadata"] != identity:
            raise ValueError("artifact base model does not match the approved snapshot")
        verify_file(Path(index["head"]["path"]), index["head"]["sha256"])
        for entry in index["entries"]:
            verify_file(Path(entry["evidence_path"]), entry["evidence_sha256"])
            verify_file(Path(entry["posterior_path"]), entry["posterior_sha256"])
            if entry["test_outputs_sealed"]:
                sealed_hash_checks += 1
        indexes[name] = {"sha256": sha256_file(path), "example_count": len(index["entries"])}
    old_path = root / "smoke8-utf8/evidence/freedom-flower-smoke8/evidence.safetensors"
    new_path = root / "smoke8-hardened/evidence/freedom-flower-smoke8/evidence.safetensors"
    old, new = load_file(str(old_path)), load_file(str(new_path))
    numerical = {}
    for key in ("encoder_features", "frame_logits"):
        if not np.allclose(old[key], new[key], rtol=1e-5, atol=1e-5):
            raise ValueError("hardened cold inference changed acoustic evidence")
        numerical[key] = {"bit_equal": bool(np.array_equal(old[key], new[key])), "maximum_absolute_difference": float(np.max(np.abs(old[key] - new[key])))}
    requests = read_json(root / "requests/smoke8.json")
    audio = Path(requests["rows"][0]["audio_path"])
    posterior = root / "smoke8-hardened/evidence/freedom-flower-smoke8/posterior.json"
    segments = PhoneticXeusPosteriorRecognizer(posterior).recognize(audio)
    return {
        "snapshot": identity, "indexes": indexes,
        "sealed_artifact_hashes_verified": sealed_hash_checks,
        "sealed_prediction_content_parsed": False,
        "hardened_inference_numerical_comparison": numerical,
        "self_contained_posterior_replayed": True, "self_contained_segment_count": len(segments),
        "resource_monitor": read_json(root / "smoke8-hardened/resource-monitor.json"),
    }


def regression_result(root: Path, name: str) -> dict[str, Any]:
    metadata = read_json(root / f"commands/{name}.command.json")
    if metadata["return_code"] != 0:
        raise ValueError("full regression did not pass")
    output_path = Path(metadata["stdout_path"])
    output = output_path.read_text(encoding="utf-8")
    result = re.search(r"(\d+) passed, (\d+) skipped in ([\d.]+)s", output)
    if result is None:
        raise ValueError("full regression summary is missing")
    return {
        "command_name": name,
        "passed": int(result[1]), "skipped": int(result[2]), "seconds": float(result[3]),
        "stdout_sha256": sha256_file(output_path), "command_metadata": metadata,
        "skip_reasons": [line.strip() for line in output.splitlines() if line.startswith("SKIPPED ")],
    }


def build_report(root: Path, *, final: bool, regression_name: str = "full-regression-final") -> dict[str, Any]:
    plan = read_json(root / "execution-plan.json")
    start = datetime.fromisoformat(plan["started_at_utc"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    elapsed = (now - start).total_seconds()
    if final and elapsed < 4 * 3600:
        raise ValueError("minimum four-hour execution window has not elapsed; use a progress report")
    if final:
        regression = read_json(root / f"commands/{regression_name}.command.json")
        fingerprints = read_json(Path(regression["source_fingerprints_path"]))
        verify_file(Path(regression["source_fingerprints_path"]), regression["source_fingerprints_sha256"])
        for name, digest in fingerprints["files"].items():
            verify_file(ROOT / name, digest)
    reference = read_json(root / "reference/reference-manifest.json")
    dataset = read_json(root / "reference/rendered-proxy-dataset.json")
    bindings = reference["bindings"]
    rebuilt_reference, rebuilt_dataset = build_reference_manifest(
        Path(bindings["triplets"]["path"]), Path(bindings["mapping"]["path"]), Path(bindings["vocabulary"]["path"])
    )
    if sha256_object(reference) != sha256_object(rebuilt_reference) or sha256_object(dataset) != sha256_object(rebuilt_dataset):
        raise ValueError("reference/dataset does not rebuild identically from its unchanged sources")
    validate_dataset_source_roles(dataset)
    for name, digest in HISTORICAL.items():
        verify_file(ROOT / name, digest)
    reports = [read_json(root / f"training/seed{seed}/training-report.json") for seed in (17, 29, 43)]
    checkpoint_proof = read_json(root / "verification/checkpoints/verification.json")
    if not all(row["exact_prediction_and_metric_replay"] for row in checkpoint_proof["proofs"]):
        raise ValueError("checkpoint cold replay failed")
    job_names = [
        "create-runtime", "install-torch", "install-model-dependencies", "acquire-model",
        "runtime-pip-check", "runtime-freeze", "prepare-acoustic-requests",
        "smoke8-utf8", "smoke20", "original-baselines", "rendered-features",
        "train-seed17", "train-seed29", "train-seed43", "verify-posterior-replay",
        "verify-checkpoints", "section-correspondence-v2", "build-review-v2",
        "smoke8-hardened", "full-regression",
    ]
    commands = {name: read_json(root / f"commands/{name}.command.json") for name in job_names}
    if any(record["return_code"] != 0 or record["timed_out"] for record in commands.values()):
        raise ValueError("an essential execution command failed or is incomplete")
    main_versions = {name: importlib.metadata.version(name) for name in MAIN_ENVIRONMENT}
    if main_versions != MAIN_ENVIRONMENT:
        raise ValueError("the original main runtime package versions changed during this execution")
    runtime_environment = read_json(root / "runtime-environment.json")
    return {
        "schema_version": "limited_data_execution_report_v1",
        "status": "completed_bounded_experiment" if final else "verification_in_progress",
        "started_at_utc": start.isoformat(), "reported_at_utc": now.isoformat(),
        "elapsed_seconds": elapsed, "minimum_four_hours_elapsed": elapsed >= 4 * 3600,
        "planning_time_included": False,
        "reference": {"manifest_sha256": sha256_file(root / "reference/reference-manifest.json"), "dataset_sha256": sha256_file(root / "reference/rendered-proxy-dataset.json"), "rebuild_identical": True, "external_source_count": 24, "all_external_source_hashes_reverified": True, "summary": dataset["summary"]},
        "model": audit_model_artifacts(root),
        "environments": {
            "main_versions_unchanged": True, "main_versions": main_versions,
            "isolated_runtime": runtime_environment,
            "isolated_environment_sha256": sha256_file(root / "runtime-environment.json"),
            "install_report_bindings": {
                name: sha256_file(root / name)
                for name in ("torch-install-report.json", "dependencies-install-report.json")
            },
        },
        "training": {
            "mode": "rendered_proxy_only", "trial_count": 3,
            "baseline_dev_per": reports[0]["baseline_dev"]["per"],
            "baseline_dev_groups": reports[0]["baseline_dev"]["groups"],
            "runs": [{"seed": row["seed"], "adapted_dev_per": row["adapted_dev"]["per"], "best_epoch": row["best_epoch"], "checkpoint": row["adapter"], "groups": row["adapted_dev"]["groups"]} for row in reports],
            "development_gate": checkpoint_proof["development_gate"],
            "admission_status": "passed_development_only" if checkpoint_proof["development_gate"]["passed"] else "rejected_by_predeclared_development_gate",
            "original_accuracy_available": False, "held_out_test_metrics_computed": False,
            "qualified_recognizer": False, "default_model_replaced": False,
            "interpretation": "Error against derived engine-reference IPA on one rendered development song; improvement can include learning reference conventions, not original-singing phonetic accuracy.",
        },
        "verification": {
            "full_regression": regression_result(root, regression_name),
            "posterior_replay": read_json(root / "verification/posterior-replay.json"),
            "checkpoint_replay": checkpoint_proof,
            "review": audit_review(root / "review-v2"),
            "historical_artifact_sha256_unchanged": HISTORICAL,
        },
        "limitations": [
            "Original-singing phone truth is not independently reviewed; test predictions remain sealed from evaluation/selection.",
            "V4 mapping restores surviving v1 values, not verified acoustic realization.",
            "Aspiration group PER increases from 0.64 to 0.72 for all three seeds (16 to 18 edit errors over 25 references); the predeclared group-regression gate rejects every checkpoint.",
            "Front-rounded references number only six; this is insufficient evidence of robust preservation.",
            "Original/rendered matching is a machine proposal, not human alignment approval.",
            "Browser visual/playback verification was unavailable; static media/HTML checks and waveform image inspection were performed.",
            "Early commands predate automatic source-tree fingerprint capture; later verification records capture their actual launch-time source hashes.",
        ],
        "resolved_failures": [
            "Historical Queen Road East XML requires its existing hash-bound normalized copy.",
            "Isolated Windows Python must use -X utf8; pinned upstream source was not rewritten.",
            "Correspondence v1 rewarded skipped source frames; v2 uses source-frame-weighted costs, verified with a constant-cost regression.",
        ],
        "essential_commands_completed": list(commands),
        "additional_complete_triplets_requested": 0,
        "review_packet": str((root / "review-v2/index.html").resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit actual artifacts and close out the bounded limited-data execution.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--regression-command", default="full-regression-final")
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    if not root.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(root) or output.exists():
        parser.error("use a new report inside the execution run")
    if re.fullmatch(r"[a-z0-9-]+", args.regression_command) is None:
        parser.error("invalid regression command name")
    report = build_report(root, final=args.final, regression_name=args.regression_command)
    print(f"report_sha256={write_json_new(output, report)}; status={report['status']}; elapsed_hours={report['elapsed_seconds'] / 3600:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

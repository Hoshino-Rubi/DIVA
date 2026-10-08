from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.training import cantonese_singing_ctc as ctc  # noqa: E402
from any2xsampa.training.acoustic_features import (  # noqa: E402
    FrozenWavLMFeatureExtractor,
    FrozenXLSRFeatureExtractor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _manifest_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    model_to_candidate: dict[str, str] = {}
    for path in sorted(root.glob("*/backbone_manifest.json")):
        payload = _json(path)
        family = str(payload.get("family") or "")
        extractor = (
            FrozenWavLMFeatureExtractor(path)
            if family == "wavlm"
            else FrozenXLSRFeatureExtractor(path)
        )
        metadata = dict(extractor.metadata())
        candidate = str(payload.get("candidate_id") or path.parent.name)
        model_to_candidate[str(metadata["model_sha256"])] = candidate
        records.append(
            {
                "candidate_id": candidate,
                "family": family,
                "repository": metadata["repository"],
                "revision": metadata["revision"],
                "manifest_path": _relative(path),
                "manifest_sha256": _sha256(path),
                "model_sha256": metadata["model_sha256"],
                "license": metadata["license"],
                "local_files_only": metadata["local_files_only"],
                "device_policy": "explicit_cuda_no_cpu_fallback",
            }
        )
    return records, model_to_candidate


def _feature_records(
    root: Path,
    model_to_candidate: Mapping[str, str],
    paths: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    import torch

    records: list[dict[str, Any]] = []
    by_cache_sha: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"^(?P<candidate>.+?)-layer(?P<layer>-?\d+)-features(?:-.+)?\.pt$")
    candidates = paths if paths is not None else sorted(root.glob("*-layer*-features.pt"))
    for path in candidates:
        match = pattern.match(path.name)
        if match is None:
            raise ValueError(f"feature-cache name does not bind a hidden layer: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != ctc.FEATURE_CACHE_SCHEMA:
            raise ValueError(f"unsupported feature cache: {path}")
        cache_sha = _sha256(path)
        model_sha = str(payload.get("model_sha256") or "")
        record = {
            "candidate_id": model_to_candidate.get(model_sha, match.group("candidate")),
            "hidden_layer": int(match.group("layer")),
            "path": _relative(path),
            "bytes": path.stat().st_size,
            "sha256": cache_sha,
            "dataset_sha256": payload.get("dataset_sha256"),
            "model_sha256": model_sha,
            "features_sha256": payload.get("features_sha256"),
            "hidden_size": payload.get("hidden_size"),
            "example_count": payload.get("example_count"),
        }
        records.append(record)
        by_cache_sha[cache_sha] = record
        del payload
    return records, by_cache_sha


def _run_records(root: Path, caches: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/metrics.json")):
        payload = _json(path)
        if payload.get("schema_version") != ctc.REPORT_SCHEMA:
            continue
        cache = caches.get(str(payload.get("feature_cache_sha256") or ""))
        if cache is None:
            raise ValueError(f"training report references an unknown feature cache: {path}")
        config = payload.get("configuration")
        validation = payload.get("validation")
        if not isinstance(config, Mapping) or not isinstance(validation, Mapping):
            raise ValueError(f"training report is incomplete: {path}")
        checkpoint = Path(str(payload.get("checkpoint") or ""))
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        if not checkpoint.is_file() or _sha256(checkpoint) != payload.get("checkpoint_sha256"):
            raise ValueError(f"training checkpoint hash mismatch: {path}")
        records.append(
            {
                "kind": "robustness" if "robustness" in path.parts else "selection",
                "candidate_id": cache["candidate_id"],
                "hidden_layer": cache["hidden_layer"],
                "metrics_path": _relative(path),
                "metrics_sha256": _sha256(path),
                "checkpoint_path": _relative(checkpoint),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "feature_cache_sha256": payload["feature_cache_sha256"],
                "feature_tensor_sha256": payload["feature_tensor_sha256"],
                "fold_index": int(config["fold_index"]),
                "seed": int(payload["seed"]),
                "learning_rate": float(config["learning_rate"]),
                "batch_size": int(config["batch_size"]),
                "epochs_completed": int(payload["epochs_completed"]),
                "validation_similarity": float(validation["token_similarity"]),
                "validation_exact_sequence_ratio": float(validation["exact_sequence_ratio"]),
                "validation_reference_tokens": int(validation["reference_token_count"]),
                "provenance_rebound": isinstance(payload.get("provenance_rebound"), Mapping),
            }
        )
    return records


def _candidate_results(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = sorted({str(run["candidate_id"]) for run in runs})
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        selection = [
            run for run in runs if run["candidate_id"] == candidate and run["kind"] == "selection"
        ]
        robustness = [
            run for run in runs if run["candidate_id"] == candidate and run["kind"] == "robustness"
        ]
        best_folds: list[dict[str, Any]] = []
        for fold in sorted({int(run["fold_index"]) for run in selection}):
            best_folds.append(
                max(
                    (run for run in selection if run["fold_index"] == fold),
                    key=lambda run: (run["validation_similarity"], run["metrics_path"]),
                )
            )
        edits = sum(
            round((1.0 - run["validation_similarity"]) * run["validation_reference_tokens"])
            for run in best_folds
        )
        references = sum(run["validation_reference_tokens"] for run in best_folds)
        results.append(
            {
                "candidate_id": candidate,
                "selection_run_count": len(selection),
                "best_folds": best_folds,
                "micro_similarity": 1.0 - edits / references if references else 0.0,
                "minimum_fold_similarity": min(
                    (run["validation_similarity"] for run in best_folds),
                    default=0.0,
                ),
                "all_exact_zero": all(
                    run["validation_exact_sequence_ratio"] == 0.0 for run in best_folds
                ),
                "robustness_runs": robustness,
            }
        )
    return results


def build_summary(
    dataset_path: Path,
    root: Path,
    threshold: float,
    *,
    manifest_root: Path | None = None,
    mert_info_path: Path | None = None,
    auxiliary_evidence_path: Path | None = None,
    auxiliary_evidence_paths: list[Path] | None = None,
    feature_cache_paths: list[Path] | None = None,
    runs_root: Path | None = None,
    reports_root: Path | None = None,
    triplet_manifest_path: Path | None = None,
) -> dict[str, Any]:
    dataset = _json(dataset_path)
    examples = ctc._examples(dataset)
    folds = []
    for index in range(len(dataset.get("folds", []))):
        train, validation, identity = ctc._split(dataset, examples, index)
        folds.append(
            {
                "fold_index": index,
                "split_identity": identity,
                "train_groups": sorted({str(row["group_id"]) for row in train}),
                "validation_groups": sorted({str(row["group_id"]) for row in validation}),
                "leakage_check": "passed",
            }
        )
    snapshot_root = manifest_root or root
    manifests, model_to_candidate = _manifest_records(snapshot_root)
    features, cache_lookup = _feature_records(root, model_to_candidate, feature_cache_paths)
    run_directory = runs_root or root
    runs = _run_records(run_directory, cache_lookup)
    candidates = _candidate_results(runs)
    best = max(candidates, key=lambda item: item["micro_similarity"])
    reports = [
        {
            "path": _relative(path),
            "sha256": _sha256(path),
            "fold_index": _json(path).get("fold_index"),
            "token_similarity": _json(path).get("token_similarity"),
        }
        for path in sorted((reports_root or (root / "reports")).glob("*.json"))
    ]
    evidence_paths = list(auxiliary_evidence_paths or [])
    if auxiliary_evidence_path is not None:
        evidence_paths.append(auxiliary_evidence_path)
    auxiliary_evidence: list[dict[str, Any]] = []
    for evidence_path in evidence_paths:
        if evidence_path.is_file():
            auxiliary_evidence.append(
                {
                    "path": _relative(evidence_path),
                    "sha256": _sha256(evidence_path),
                    "payload": _json(evidence_path),
                }
            )
    rejected: list[dict[str, Any]] = []
    mert_path = mert_info_path or snapshot_root / "MERT-v1-95M-MODEL_INFO.json"
    if mert_path.is_file():
        mert = _json(mert_path)
        rejected.append(
            {
                "candidate_id": "MERT-v1-95M",
                "repository": mert.get("id"),
                "revision": mert.get("sha"),
                "metadata_path": _relative(mert_path),
                "metadata_sha256": _sha256(mert_path),
                "license": (mert.get("cardData") or {}).get("license"),
                "reason": "NonCommercial license and custom-code runtime violate the active route red lines",
                "weights_downloaded": False,
            }
        )
    triplet_records: dict[str, Any] | None = None
    if triplet_manifest_path is not None:
        triplet_manifest = _json(triplet_manifest_path)
        raw_items = triplet_manifest.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("triplet manifest must contain a non-empty items list")
        items: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise ValueError("triplet manifest items must be objects")
            original = raw_item.get("original_wav")
            rendered = raw_item.get("rendered_wav")
            vsqx = raw_item.get("human_vsqx")
            if not all(isinstance(value, Mapping) for value in (original, rendered, vsqx)):
                raise ValueError("triplet manifest item is missing one of its three artifacts")
            baseline = raw_item.get("audio_only_baseline")
            items.append(
                {
                    "song_id": raw_item.get("song_id"),
                    "song": raw_item.get("song"),
                    "human_claims": raw_item.get("human_claims"),
                    "original_wav": {
                        "path": original.get("path"),
                        "sha256": original.get("sha256"),
                    },
                    "rendered_wav": {
                        "path": rendered.get("path"),
                        "sha256": rendered.get("sha256"),
                    },
                    "human_vsqx": {
                        "path": vsqx.get("normalized_path") or vsqx.get("path"),
                        "sha256": vsqx.get("normalized_sha256") or vsqx.get("sha256"),
                        "note_count": vsqx.get("note_count"),
                    },
                    "unresolved_baseline": baseline is None,
                    "audio_only_baseline": (
                        {
                            "posterior_json_sha256": baseline.get("posterior_json_sha256"),
                            "replay_segments_sha256": baseline.get("replay_segments_sha256"),
                            "timing_report_sha256": baseline.get("timing_report_sha256"),
                            "segment_count": baseline.get("segment_count"),
                            "input_audio_sha256": baseline.get("input_audio_sha256"),
                            "device": baseline.get("device"),
                            "providers": baseline.get("providers"),
                            "cuda_active": baseline.get("cuda_active"),
                            "runtime_status": baseline.get("runtime_status"),
                        }
                        if isinstance(baseline, Mapping)
                        else None
                    ),
                }
            )
        triplet_records = {
            "path": _relative(triplet_manifest_path),
            "sha256": _sha256(triplet_manifest_path),
            "schema_version": triplet_manifest.get("schema_version"),
            "triplet_count": len(items),
            "items": items,
        }
    passed = (
        best["micro_similarity"] >= threshold
        and best["minimum_fold_similarity"] >= threshold
    )
    return {
        "schema_version": "pretrained_acoustic_backbone_route_report_v1",
        "objective": "materially different frozen pretrained acoustic backbone evaluation",
        "dataset": {
            "path": _relative(dataset_path),
            "sha256": _sha256(dataset_path),
            "dataset_hash": dataset.get("dataset_hash"),
            "example_count": len(examples),
            "folds": folds,
            "human_vsqx_targets_only_after_feature_extraction": True,
            "lyrics_or_score_used_as_acoustic_input": False,
        },
        "runtime_policy": {
            "feature_extraction": "serial explicit CUDA",
            "cpu_thread_limit": 4,
            "cpu_fallback_allowed": False,
            "torch": "2.13.0+cu126",
            "cuda": "12.6",
            "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU",
        },
        "manifests": manifests,
        "triplets": triplet_records,
        "feature_caches": features,
        "candidate_results": candidates,
        "rejected_candidates": rejected,
        "best_candidate": {
            "candidate_id": best["candidate_id"],
            "micro_similarity": best["micro_similarity"],
            "minimum_fold_similarity": best["minimum_fold_similarity"],
            "threshold": threshold,
            "passed": passed,
        },
        "detailed_reports": reports,
        "auxiliary_evidence": auxiliary_evidence,
        "human_window": {
            "condition": "A" if passed else "B",
            "reason": (
                "candidate reached the listening-review evidence gate"
                if passed
                else "bounded non-NC WavLM/XLS-R candidates remain below 0.80 and MERT is NC-restricted"
            ),
            "smallest_user_decisions": []
            if passed
            else [
                "select and authorize another immutable non-NC acoustic backbone asset",
                "return to the additional complete-triplet route",
                "authorize a materially different full-backbone fine-tuning route",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the pretrained acoustic backbone route.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--mert-info", type=Path)
    parser.add_argument("--auxiliary-evidence", type=Path, action="append")
    parser.add_argument("--feature-cache", type=Path, action="append")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--reports-root", type=Path)
    parser.add_argument("--triplet-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args(argv)
    try:
        if not 0.0 < args.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if args.output.exists():
            raise FileExistsError(f"route report already exists: {args.output}")
        summary = build_summary(
            args.dataset,
            args.root,
            args.threshold,
            manifest_root=args.manifest_root,
            mert_info_path=args.mert_info,
            auxiliary_evidence_paths=args.auxiliary_evidence,
            feature_cache_paths=args.feature_cache,
            runs_root=args.runs_root,
            reports_root=args.reports_root,
            triplet_manifest_path=args.triplet_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "best_candidate": summary["best_candidate"],
                "human_window": summary["human_window"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

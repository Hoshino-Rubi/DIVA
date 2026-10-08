from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new  # noqa: E402
from any2xsampa.training.original_adapter_transfer import evaluate_adapter_transfer  # noqa: E402
from any2xsampa.training.original_reference_evaluation import verify_reference, validate_baseline_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed rendered-domain adapters on confirmed original clips.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--baseline-index", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = verify_reference(args.reference)
    evaluation = read_json(args.evaluation_manifest)
    if evaluation.get("schema_version") != "original_reference_evaluation_manifest_v1":
        raise ValueError("unsupported original evaluation manifest")
    baseline_index = read_json(args.baseline_index)
    baseline_entries = validate_baseline_index(baseline_index, reference)
    rows = [row for row in evaluation["examples"] if row["official_evaluation_eligible"]]
    if args.conditional:
        rows = [row for row in evaluation["examples"] if row["song_id"] == "shalong" and row["candidate_references"][0]["mapping_complete"]]
    if not rows or any(row["song_id"] != "shalong" for row in rows):
        raise ValueError("this bounded transfer evaluation requires only Shalong clips")
    reports = {}
    for seed in (17, 29, 43):
        report = read_json(args.training_root / f"seed{seed}/training-report.json")
        reports[seed] = report
    vocabulary = Path(reference["bindings"]["vocabulary"]["path"])
    result = evaluate_adapter_transfer(rows, baseline_entries, reports, vocabulary, conditional=args.conditional)
    result.update({
        "reference_path": str(args.reference.resolve()), "reference_sha256": sha256_file(args.reference),
        "evaluation_manifest_path": str(args.evaluation_manifest.resolve()), "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest),
        "baseline_index_path": str(args.baseline_index.resolve()), "baseline_index_sha256": sha256_file(args.baseline_index),
        "training_reports": {str(seed): {"path": str((args.training_root / f"seed{seed}/training-report.json").resolve()), "sha256": sha256_file(args.training_root / f"seed{seed}/training-report.json")} for seed in reports},
        "confirmed_original_windows": len(rows) if not args.conditional else 0,
        "conditional_diagnostic_windows": len(rows) if args.conditional else 0,
        "original_accuracy_available": not args.conditional,
    })
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    write_json_new(output, result)
    print(f"confirmed_original_windows={len(rows)}; baseline_per={result['metrics']['baseline']['per']:.6f}; seed17_per={result['metrics']['seed17']['per']:.6f}; seed29_per={result['metrics']['seed29']['per']:.6f}; seed43_per={result['metrics']['seed43']['per']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

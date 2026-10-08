from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import canonical_json, write_json_new  # noqa: E402
from any2xsampa.training.limited_data_reference import build_reference_manifest, training_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hash-bound rich-IPA proxy references without rewriting human annotations.")
    parser.add_argument("--triplets", required=True, type=Path)
    parser.add_argument("--mapping", type=Path, default=ROOT / "src/any2xsampa/config/v4c_reference_ipa_v4.json")
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ROOT / ".state/runs"):
        parser.error("generated artifacts must remain under .state/runs")
    names = ("reference-manifest.json", "rendered-proxy-dataset.json", "training-gate.json")
    if any((destination / name).exists() for name in names):
        parser.error("refusing to overwrite an existing reference run")
    manifest, dataset = build_reference_manifest(args.triplets, args.mapping, args.vocabulary)
    gate = training_gate(dataset)
    hashes = {
        names[0]: write_json_new(destination / names[0], manifest),
        names[1]: write_json_new(destination / names[1], dataset),
        names[2]: write_json_new(destination / names[2], gate),
    }
    print(canonical_json({"artifact_sha256": hashes, "summary": dataset["summary"], "training_gate": gate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

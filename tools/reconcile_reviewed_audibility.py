"""Freeze audited edge-sequence corrections before reading any model predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new
from any2xsampa.training.reviewed_audibility_reference import build_reviewed_reference
from import_original_edge_review import load_verified_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / ".state/runs"):
        raise ValueError("output must be a new workspace run directory")
    packet, checks = load_verified_packet(args.packet.resolve())
    raw = args.answers.read_bytes()
    payload = read_json(args.answers)
    result = build_reviewed_reference(packet, payload)
    output.mkdir(parents=True)
    (output / "raw-answers.json").write_bytes(raw)
    reference_hash = write_json_new(output / "reviewed-reference.json", result)
    write_json_new(output / "INPUTS.json", {
        "packet": {"path": str(args.packet.resolve()), "sha256": sha256_file(args.packet)},
        "answers": {"path": str(args.answers.resolve()), "sha256": sha256_file(args.answers)},
        "source_checks": checks, "reference_sha256": reference_hash,
        "prediction_files_read": False, "training_started": False})
    print(result["coverage"])


if __name__ == "__main__":
    main()

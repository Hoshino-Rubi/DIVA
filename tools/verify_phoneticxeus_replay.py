from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer  # noqa: E402
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently replay unsealed raw CTC evidence through the stable phone contract.")
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / ".state/runs") or args.output.exists():
        parser.error("use a new report path under .state/runs")
    verified, sealed = [], []
    for index_path in args.index:
        index = read_json(index_path)
        for entry in index["entries"]:
            if entry["test_outputs_sealed"]:
                sealed.append(entry["example_id"])
                continue
            path = Path(entry["posterior_path"])
            verify_file(path, entry["posterior_sha256"])
            payload = read_json(path)
            if payload["producer"]["input_audio"]["sha256"] != entry["audio_sha256"]:
                raise ValueError("index and posterior audio identity disagree")
            audio = Path(payload["producer"]["crop"].get("audio_path", ""))
            # Audio paths are deliberately not inferred from rendered/original filenames.
            request_source = next(
                candidate for candidate in (index_path.parent.parent / "requests").glob("*.json")
                if sha256_file(candidate) == index["requests_sha256"]
            )
            requests = read_json(request_source)
            request = next(row for row in requests["rows"] if row["example_id"] == entry["example_id"])
            audio = Path(request["audio_path"])
            real = PhoneticXeusPosteriorRecognizer(path, vocabulary_path=args.vocabulary).recognize(audio)
            fixture = PhoneticXeusPosteriorRecognizer(path, fixture=True, vocabulary_path=args.vocabulary).recognize(audio)
            contract = lambda rows: [
                {"time_range": asdict(row.time_range), "candidate_phones": [asdict(c) for c in row.candidate_phones], "confidence": row.confidence}
                for row in rows
            ]
            if contract(real) != contract(fixture):
                raise ValueError("real/fixture replay changed the stable phone contract")
            verified.append({
                "example_id": entry["example_id"], "posterior_sha256": entry["posterior_sha256"],
                "phone_segment_count": len(real), "phone_contract_sha256": sha256_object(contract(real)),
                "source_audio_sha256": entry["audio_sha256"], "raw_logits_recomputed": True,
                "real_fixture_equal": True,
            })
    report = {
        "schema_version": "phoneticxeus_replay_verification_v1",
        "index_bindings": [{"path": str(path), "sha256": sha256_file(path)} for path in args.index],
        "vocabulary_sha256": sha256_file(args.vocabulary),
        "verified_count": len(verified), "verified": verified,
        "sealed_skipped": sealed, "sealed_outputs_opened": False,
        "qualified_recognizer": False,
    }
    print(f"verified={len(verified)}; sealed_skipped={len(sealed)}; report_sha256={write_json_new(args.output, report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

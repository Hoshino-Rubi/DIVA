"""Verify a frozen edge packet and import explicit human selections, offline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.original_edge_review import assemble_references, validate_packet
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_sources(packet: dict, directory: Path) -> dict:
    bindings = packet["source_bindings"]
    checked = {}
    for info in bindings.values():
        path = Path(info["path"]).resolve()
        if path not in checked:
            checked[path] = sha256_file(path)
        _require(checked[path] == info["sha256"], f"source hash mismatch: {path}")
    reference_path = Path(bindings["reference"]["path"])
    reference, overlay = apply_original_mapping_overlay(read_json(reference_path), Path(bindings["mapping_evidence"]["path"]))
    _require(overlay["sha256"] == bindings["mapping_overlay"]["sha256"], "overlay source differs from verified runtime mapping")
    song = next(s for s in reference["songs"] if s["song_id"] == "shalong")
    source_clips = {c["example_id"]: c for c in reference["original_clips"] if c["song_id"] == "shalong"}
    triplets = read_json(Path(bindings["triplets"]["path"]))
    approval = next(t for t in triplets["items"] if t["song_id"] == "shalong")
    _require(approval["human_vsqx"]["sha256"] == song["human_vsqx"]["sha256"], "source VSQX approval hash mismatch")
    _require(approval.get("human_claims", {}).get("phoneme_correctness") == "user_asserted_correct", "source phoneme approval missing")
    catalog = {n["note_id"]: n for n in song["notes"]}
    positions = {n["note_id"]: i for i, n in enumerate(song["notes"])}
    for clip in packet["clips"]:
        _require(clip["example_id"] in source_clips, "unknown original clip")
        original = source_clips[clip["example_id"]]
        _require(clip["start_sec"] == original["start_sec"] and clip["end_sec"] == original["end_sec"], "original crop changed")
        _require(clip["audio_sha256"] == song["original_audio"]["sha256"] and clip["source_vsqx_sha256"] == song["human_vsqx"]["sha256"], "clip role/source hashes differ")
        _require(clip["reference_manifest_sha256"] == bindings["reference"]["sha256"], "clip reference hash differs")
        _require(Path(clip["audio_path"]).resolve() == Path(song["original_audio"]["path"]).resolve(), "original/rendered audio role mismatch")
        verify_file(Path(clip["audio_path"]), clip["audio_sha256"])
        _require(clip.get("source_sequence_approval_binding") == "triplets", "sequence approval not bound to triplets")
        seq = clip.get("source_sequence_approval", {})
        _require(seq.get("scope") == "source_vsqx_phone_sequence_only" and seq.get("source_vsqx_sha256") == clip["source_vsqx_sha256"], "sequence approval scope/hash differs")
        selected_positions = []
        for note in clip["notes"]:
            _require(note["note_id"] in catalog, "unknown source note")
            source = catalog[note["note_id"]]
            _require(note["phones"] == source["model_target_ipa"] and note["mapping_status"] == source["mapping_status"] and note["lyric"] == source["raw_lyric"], "packet note differs from verified mapping")
            selected_positions.append(positions[note["note_id"]])
        _require(selected_positions == list(range(selected_positions[0], selected_positions[-1] + 1)), "clip note catalog omitted internal source notes")
    media_count = 0
    for item in packet["items"]:
        for media in item["media"]:
            for field, digest in (("audio_file", "audio_sha256"), ("waveform_file", "waveform_sha256")):
                if field not in media:
                    continue
                path = (directory / media[field]).resolve()
                _require(path.is_relative_to((directory / "media").resolve()), "media escapes packet directory")
                verify_file(path, media[digest])
                media_count += 1
    return {"source_files_verified": len(checked), "media_files_verified": media_count,
            "source_catalog_verified": True, "source_sequence_approval_verified": True}


def load_verified_packet(path: Path) -> tuple[dict, dict]:
    packet = read_json(path)
    validate_packet(packet)
    _require(packet.get("policy", {}).get("test_songs_opened") is False, "packet did not keep test songs sealed")
    return packet, _verify_sources(packet, path.parent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    packet, checks = load_verified_packet(args.packet.resolve())
    if args.validate_only:
        print(f"packet verified: {len(packet['items'])} edges; {checks}; no references created")
        return 0
    if not args.answers or not args.output_dir:
        parser.error("--answers and --output-dir are required unless --validate-only")
    output = args.output_dir.resolve()
    _require(output.is_relative_to(ROOT / ".state/runs") and not output.exists(), "output must be a new .state/runs directory")
    raw = args.answers.read_bytes()
    answers = read_json(args.answers)
    result = assemble_references(packet, answers)
    original_count = sum(r["status"] == "human_confirmed" for r in result["examples"])
    reference_phones = sum(len(r["phones"]) for r in result["examples"] if r["status"] == "human_confirmed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw-answers.json").write_bytes(raw)
    reference_hash = write_json_new(output / "segment-reference-v2.json", result)
    status = {"schema_version": "original_edge_review_intake_v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "packet_path": str(args.packet.resolve()), "packet_sha256": packet["packet_sha256"],
              "answers_sha256": sha256_file(args.answers), "reference_sha256": reference_hash, "checks": checks,
              "formal_dev_windows": original_count, "formal_dev_reference_phones": reference_phones,
              "dev_reference_count_gate": original_count >= 3 and reference_phones >= 100,
              "training_windows": 0, "training_gate_passed": False, "training_started": False,
              "evaluation_executed": False, "test_songs_opened": False,
              "pending_or_rejected": [{"example_id": r["example_id"], "status": r["status"], "reasons": r["blocking_reasons"]} for r in result["examples"] if r["status"] != "human_confirmed"],
              "new_default_model_deployed": False}
    write_json_new(output / "STATUS.json", status)
    print(f"formal_dev_windows={original_count}; reference_phones={reference_phones}; training_started=false; output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

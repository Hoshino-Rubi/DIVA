"""Read-only global correspondence audit; no score-based label repair."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file


def audit(run: Path) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
    enforce_offline(run / "whole-audit-cache")
    import numpy as np
    from any2xsampa.training.whole_sequence_reference import audit_song, check_clock_anchors
    from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
    from any2xsampa.training.original_reference_correspondence import global_path
    from any2xsampa.training.limited_data_reference import _notes, load_mapping, load_vocabulary
    from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline
    spec = read_json(run / "POLICY.json")
    base = ROOT / ".state/runs/limited-data-20260906T050251"
    manifest = base / "reference/reference-manifest.json"
    raw = read_json(manifest)
    evidence = ROOT / ".state/runs/original-reference-eval-20260906T154652/manual-evidence/symbol-evidence.json"
    # Locate the already verified evidence object by schema, not a guessed filename.
    evidence_candidates = []
    for path in evidence.parent.glob("*.json"):
        obj = read_json(path)
        if obj.get("schema_version") == "vocaloid_manual_symbol_evidence_v1":
            evidence_candidates.append(path)
    if len(evidence_candidates) != 1:
        raise ValueError("manual symbol evidence is ambiguous")
    view, overlay = apply_original_mapping_overlay(raw, evidence_candidates[0])
    refp = ROOT / ".state/runs/original-adaptation-20260919T012000/reviewed-training-reference.json"
    reviewed = [r for r in read_json(refp)["examples"] if r["split"] == "train"]
    correspondence = read_json(ROOT / "src/any2xsampa/config/original_reference_evaluation_v1.json")["correspondence"]
    acceptance = {"edge_tolerance_seconds": 2.0, "resolution_tolerance_seconds": 0.5,
        "minimum_anchors": spec["whole_sequence"]["minimum_independent_confirmed_anchors"],
        "constant_offset_required": False, "existing_reviewed_anchor_occurrences_not_reassigned": True,
        "mapping_exclusion": "complete_song", "machine_evidence_is_not_human_confirmation": True}
    write_json_new(run / "WHOLE_AUDIT_POLICY.json", acceptance)
    output = run / "whole-audit"
    output.mkdir()
    reports, eligible = [], []
    for song in view["songs"]:
        if song["song_id"] not in spec["songs"]:
            continue
        for role in ("original_audio", "rendered_audio"):
            verify_file(Path(song[role]["path"]), song[role]["sha256"])
        source = song["human_vsqx"]
        verify_file(Path(source["path"]), source["sha256"])
        verify_file(Path(source["normalized_path"]), source["normalized_sha256"])
        timeline = parse_vsqx_timeline(Path(source["normalized_path"]))
        original_song = next(s for s in raw["songs"] if s["song_id"] == song["song_id"])
        if (len(timeline["notes"]) != len(song["notes"])
                or [n["phonemes"] for n in timeline["notes"]] != [n["raw_v4c_tokens"] for n in song["notes"]]):
            raise ValueError("source VSQX note order/tokens differ from the frozen manifest")
        report = audit_song(song, reviewed, minimum_anchors=acceptance["minimum_anchors"])
        if report["source_structure_passed"]:
            old = read_json(base / "correspondence-v2" / (song["song_id"] + ".json"))
            for role in ("original", "rendered"):
                if old[role + "_audio_sha256"] != song[role + "_audio"]["sha256"]:
                    raise ValueError("descriptor source role/hash differs")
            descriptor = old["descriptor_artifact"]
            verify_file(Path(descriptor["path"]), descriptor["sha256"])
            with np.load(descriptor["path"], allow_pickle=False) as saved:
                paths = [global_path(saved["original"], saved["rendered"], factor=f, policy=correspondence) for f in (4,8)]
            write_json_new(output / (song["song_id"] + "-paths.json"), {"paths": paths, "descriptor": descriptor})
            report = check_clock_anchors(report, paths)
        else:
            report["machine_sequence_training_eligible"] = False
        reports.append(report)
        if report["machine_sequence_training_eligible"]:
            eligible.append({"song_id": song["song_id"], "audio": song["original_audio"],
                "source_vsqx": source, "target_ids": [i for n in song["notes"] for i in n["model_target_ids"]],
                "target_ipa": [p for n in song["notes"] for p in n["model_target_ipa"]],
                "reference_nature": spec["whole_sequence"]["reference_nature"], "human_full_alignment_confirmed": False})
        print(song["song_id"], report["machine_sequence_training_eligible"], report["exclusion_reasons"], flush=True)
    write_json_new(run / "WHOLE_SEQUENCE_AUDIT.json", {"songs": reports, "eligible": eligible,
        "source_manifest_sha256": sha256_file(manifest), "reviewed_reference_sha256": sha256_file(refp),
        "overlay": overlay, "acceptance_policy_sha256": sha256_file(run / "WHOLE_AUDIT_POLICY.json"),
        "final_test_audio_opened": False, "old_references_unchanged": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    audit(parser.parse_args().run.resolve())

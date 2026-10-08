from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.original_reference_correspondence import load_policy  # noqa: E402
from any2xsampa.training.original_reference_evaluation import (  # noqa: E402
    build_original_manifest, evaluate_original, replay_prediction, reviewed_songs, validate_baseline_index, verify_reference,
)
from any2xsampa.training.original_reference_offset import offset_probe  # noqa: E402
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay  # noqa: E402
from build_limited_data_review import make_preview  # noqa: E402
from evaluate_original_reference import mapping_audit, review_template, write_summary  # noqa: E402


def premeasure_metadata(song: dict) -> dict:
    path = Path(song["human_vsqx"]["normalized_path"])
    verify_file(path, song["human_vsqx"]["normalized_sha256"])
    root = ET.parse(path).getroot()
    ns = {"v": root.tag[1:].split("}", 1)[0]}
    master = root.find("v:masterTrack", ns)
    if master is None:
        return {"available": False}
    measures = master.findtext("v:preMeasure", namespaces=ns)
    tempos, signatures = master.findall("v:tempo", ns), master.findall("v:timeSig", ns)
    if measures is None or len(tempos) != 1 or len(signatures) != 1:
        return {"available": False, "reason": "not_a_single_tempo_signature_premeasure"}
    bpm = float(tempos[0].findtext("v:v", namespaces=ns)) / 100
    numerator = int(signatures[0].findtext("v:nu", namespaces=ns))
    denominator = int(signatures[0].findtext("v:de", namespaces=ns))
    duration = int(measures) * numerator * 4 / denominator * 60 / bpm
    return {"available": True, "premeasure_count": int(measures), "bpm": bpm, "duration_sec": duration,
            "source_vsqx_sha256": song["human_vsqx"]["sha256"],
            "used_to_fit_audio_offset": False, "metadata_alone_establishes_audio_correspondence": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose independent audio clock offset outside fixed original evaluation clips.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--review-evidence", type=Path, required=True)
    parser.add_argument("--baseline-index", type=Path, required=True)
    parser.add_argument("--descriptor-index", type=Path, required=True)
    parser.add_argument("--mapping-evidence", type=Path)
    parser.add_argument("--clip-bindings", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("use a new run output directory")
    import numpy as np

    before = time.monotonic()
    reference = verify_reference(args.reference)
    evidence = read_json(args.review_evidence)
    reviewed = reviewed_songs(evidence, reference)
    entries = validate_baseline_index(read_json(args.baseline_index), reference)
    descriptor_index = read_json(args.descriptor_index)
    if descriptor_index["reference_manifest_sha256"] != sha256_file(args.reference) or descriptor_index["baseline_index_sha256"] != sha256_file(args.baseline_index):
        raise ValueError("descriptor index source binding mismatch")
    previous = {r["song_id"]: r for r in descriptor_index["songs"]}
    expected = {s["song_id"] for s in reference["songs"] if s["split"] != "test"}
    if len(previous) != len(descriptor_index["songs"]) or set(previous) != expected:
        raise ValueError("descriptor index must contain the fixed train/development groups")
    probe_path = ROOT / "src/any2xsampa/config/original_reference_offset_probe_v1.json"
    probe_policy = read_json(probe_path)
    policy = load_policy()
    overlay_binding = None
    if args.mapping_evidence:
        reference, overlay_binding = apply_original_mapping_overlay(reference, args.mapping_evidence)
        policy["reference_overlay"] = overlay_binding
    policy["correspondence"] = {"method": "held_out_audio_offset_probe_v1", "probe_policy": probe_policy,
                                "downsample_factors": probe_policy["downsample_factors"], "note_boundary_margin_sec": 0.15}
    output.mkdir(parents=True)
    write_json_new(output / "policy.json", policy)
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in
              (("reference", args.reference), ("human_review", args.review_evidence), ("baseline_index", args.baseline_index), ("descriptor_index", args.descriptor_index), ("probe_policy", probe_path))}
    if args.mapping_evidence:
        inputs["mapping_evidence"] = {"path": str(args.mapping_evidence.resolve()), "sha256": sha256_file(args.mapping_evidence)}
    clip_bindings = []
    if args.clip_bindings:
        review = read_json(args.clip_bindings)
        if review.get("schema_version") != "original_clip_binding_reviews_v1":
            raise ValueError("unsupported clip-binding review schema")
        clip_bindings = review["items"]
        inputs["clip_bindings"] = {"path": str(args.clip_bindings.resolve()), "sha256": sha256_file(args.clip_bindings)}
    write_json_new(output / "inputs.json", inputs)
    reports, records = [], []
    for song in reference["songs"]:
        if song["split"] == "test":
            continue
        old_path = Path(previous[song["song_id"]]["path"])
        verify_file(old_path, previous[song["song_id"]]["sha256"])
        old = read_json(old_path)
        for role in ("original", "rendered"):
            if old[f"{role}_audio_sha256"] != song[f"{role}_audio"]["sha256"]:
                raise ValueError("offset descriptor source mismatch")
        cache = Path(old["descriptor_artifact"]["path"])
        verify_file(cache, old["descriptor_artifact"]["sha256"])
        if old["policy"]["sample_rate"] != 22050 or old["policy"]["hop_length"] != 512 or old["policy"]["descriptor"] != "librosa_0.11.0_chroma_cens":
            raise ValueError("offset descriptor timebase mismatch")
        with np.load(cache, allow_pickle=False) as data:
            original, rendered = data["original"], data["rendered"]
        clips = [c for c in reference["original_clips"] if c["song_id"] == song["song_id"]]
        probes = [offset_probe(original, rendered, clips, factor=f, frame_seconds=512/22050, policy=probe_policy) for f in probe_policy["downsample_factors"]]
        difference = abs(probes[0]["best"]["offset_sec"] - probes[1]["best"]["offset_sec"])
        clip_rows = [{"example_id": c["example_id"], "original_start_sec": c["start_sec"], "original_end_sec": c["end_sec"],
                      "candidates": [{"factor": p["factor"], "rendered_start_sec": c["start_sec"] + p["best"]["offset_sec"], "rendered_end_sec": c["end_sec"] + p["best"]["offset_sec"]} for p in probes],
                      "resolution_disagreement_sec": difference, "resolution_consistent": difference <= 0.5,
                      "alignment_status": "machine_proposed", "human_reviewed": False} for c in clips]
        report = {"song_id": song["song_id"], "split": song["split"], "probes": probes, "clips": clip_rows,
                  "descriptor_artifact": old["descriptor_artifact"], "score_metadata_posthoc_only": premeasure_metadata(song),
                  "audio_only_fit": True, "original_evaluation_clips_excluded_from_fit": True, "phone_predictions_used": False}
        path = output / "correspondence" / f"{song['song_id']}.json"
        records.append({"song_id": song["song_id"], "path": str(path), "sha256": write_json_new(path, report)})
        reports.append(report)
        print(f"offset_probe={song['song_id']}; offset4={probes[0]['best']['offset_sec']:.6f}; offset8={probes[1]['best']['offset_sec']:.6f}", flush=True)
    write_json_new(output / "correspondence/index.json", {"schema_version": "original_offset_correspondence_index_v1", "songs": records,
                    "reference_manifest_sha256": sha256_file(args.reference), "policy_sha256": sha256_object(policy), "test_audio_processed": False})
    manifest = build_original_manifest(reference, evidence, reference_digest=sha256_file(args.reference), evidence_digest=sha256_file(args.review_evidence), correspondence=reports, policy=policy, clip_bindings=clip_bindings)
    manifest_hash = write_json_new(output / "original-reference-manifest.json", manifest)
    write_json_new(output / "clip-binding-review-template.json", review_template(manifest))
    audit = mapping_audit(reference, evidence)
    if overlay_binding is not None:
        audit["mapping_overlay"] = overlay_binding
    write_json_new(output / "mapping-audit.json", audit)
    print(f"offset_references_frozen={manifest_hash}", flush=True)
    vocabulary = Path(reference["bindings"]["vocabulary"]["path"])
    predictions = {r["example_id"]: replay_prediction(r, entries[r["example_id"]], vocabulary) for r in manifest["examples"]}
    report = evaluate_original(manifest, predictions)
    report.update(inputs=inputs, reference_rebuild_identical=True, all_external_source_hashes_verified=True,
                  original_manifest_file_sha256=manifest_hash, elapsed_seconds=time.monotonic() - before)
    write_json_new(output / "evaluation-report.json", report)
    write_summary(output, report, manifest, audit)
    media = output / "review-media"
    media.mkdir()
    previews, lines = [], ["# Focused Original Clip Correspondence", "", "Existing phoneme approval is preserved. These audio-only offsets are proposals, not human confirmations.", ""]
    for song in reference["songs"]:
        if song["song_id"] not in reviewed or song["split"] == "test":
            continue
        for row in [r for r in manifest["examples"] if r["song_id"] == song["song_id"]]:
            candidate = row["candidate_references"][0]
            if "rendered_start_sec" not in candidate:
                continue
            pair = [make_preview(song["original_audio"], (row["start_sec"], row["end_sec"]), media, row["example_id"] + "-original", "original"),
                    make_preview(song["rendered_audio"], (candidate["rendered_start_sec"], candidate["rendered_end_sec"]), media, row["example_id"] + "-rendered", "rendered")]
            previews.append({"example_id": row["example_id"], "previews": pair, "human_reviewed": False})
            lines += [f"## {row['example_id']}", "", f"Original: {row['start_sec']:.3f}-{row['end_sec']:.3f}s; rendered proposal: {candidate['rendered_start_sec']:.3f}-{candidate['rendered_end_sec']:.3f}s.",
                      f"[Original audio](review-media/{pair[0]['audio_file']}) | [Rendered audio](review-media/{pair[1]['audio_file']})", ""]
    write_json_new(output / "review-media.json", {"items": previews, "all_responses_pending": True, "full_source_audio_copied": False})
    with (output / "FOCUSED_REVIEW.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    print(f"diagnostic_windows={report['coverage']['diagnostic_mapping_complete_windows']}; official_windows={report['coverage']['official_windows']}; preview_pairs={len(previews)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

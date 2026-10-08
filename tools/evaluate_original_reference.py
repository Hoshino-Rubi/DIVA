from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.original_reference_correspondence import load_policy, match_cached_song  # noqa: E402
from any2xsampa.training.original_reference_evaluation import (  # noqa: E402
    build_original_manifest, evaluate_original, replay_prediction, reviewed_songs, validate_baseline_index, verify_reference,
)
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay  # noqa: E402


def correspondence_reports(reference: dict, reference_path: Path, baseline_path: Path, previous_path: Path, policy: dict, output: Path) -> list[dict]:
    index = read_json(previous_path)
    if index["reference_manifest_sha256"] != sha256_file(reference_path) or index["baseline_index_sha256"] != sha256_file(baseline_path):
        raise ValueError("previous correspondence index has different source bindings")
    records = {r["song_id"]: r for r in index["songs"]}
    expected = {s["song_id"] for s in reference["songs"] if s["split"] != "test"}
    if len(records) != len(index["songs"]) or set(records) != expected:
        raise ValueError("descriptor index must contain exactly the train/development songs")
    reports, entries = [], []
    for song in reference["songs"]:
        if song["split"] == "test":
            continue
        record = records[song["song_id"]]
        path = Path(record["path"])
        verify_file(path, record["sha256"])
        old = read_json(path)
        old_policy = old["policy"]
        if old_policy.get("descriptor") != "librosa_0.11.0_chroma_cens" or any(old_policy[k] != policy["correspondence"][k] for k in ("sample_rate", "hop_length")):
            raise ValueError("descriptor extraction policy differs")
        before = time.monotonic()
        clips = [c for c in reference["original_clips"] if c["song_id"] == song["song_id"]]
        report = match_cached_song(song, old, clips, policy["correspondence"])
        report["source_correspondence_sha256"] = record["sha256"]
        report["elapsed_seconds"] = time.monotonic() - before
        report_path = output / "correspondence" / f"{song['song_id']}.json"
        entries.append({"song_id": song["song_id"], "path": str(report_path), "sha256": write_json_new(report_path, report)})
        reports.append(report)
        print(f"global_correspondence={song['song_id']}; seconds={report['elapsed_seconds']:.2f}", flush=True)
    write_json_new(output / "correspondence/index.json", {
        "schema_version": "original_global_correspondence_index_v1", "songs": entries,
        "reference_manifest_sha256": sha256_file(reference_path), "policy_sha256": sha256_object(policy),
        "baseline_index_sha256": sha256_file(baseline_path), "test_audio_processed": False,
        "human_reviewed": False, "phone_model_outputs_used": False,
    })
    return reports


def mapping_audit(reference: dict, evidence: dict) -> dict:
    reviewed = reviewed_songs(evidence, reference)
    rows = []
    for song in reference["songs"]:
        if song["split"] == "test":
            continue
        unresolved = Counter(token for n in song["notes"] for token in n["unresolved_tokens"])
        examples = [{"note_id": n["note_id"], "raw_lyric": n["raw_lyric"], "raw_v4c_tokens": n["raw_v4c_tokens"],
                     "unresolved_tokens": n["unresolved_tokens"], "rendered_start_sec": n["start_sec"], "rendered_end_sec": n["end_sec"]}
                    for n in song["notes"] if n["unresolved_tokens"]]
        total = len(song["notes"])
        resolved = sum(n["mapping_status"] == "resolved_proxy" for n in song["notes"])
        rows.append({"song_id": song["song_id"], "split": song["split"], "human_original_review_method_bound": song["song_id"] in reviewed,
                     "source_claims": song["source_claims"], "total_notes": total, "resolved_notes": resolved,
                     "note_mapping_coverage": resolved / total if total else 0, "unresolved_token_counts": dict(unresolved),
                     "unresolved_notes": examples})
    return {"schema_version": "original_reference_mapping_audit_v1", "mapping": reference["bindings"]["mapping"],
            "songs": rows, "new_token_meanings_guessed": False, "raw_human_annotations_modified": False,
            "meaning_of_unresolved": "The software mapping is incomplete; the human annotation is not rejected.",
            "test_song_details_omitted": True}


def review_template(manifest: dict) -> dict:
    items = []
    for row in manifest["examples"]:
        if not row["human_original_review_established"] or row["clip_binding"] is not None:
            continue
        items.append({
            "example_id": row["example_id"], "song_id": row["song_id"], "split": row["split"],
            "reference_manifest_sha256": manifest["reference_manifest_sha256"], "review_evidence_sha256": manifest["review_evidence_sha256"],
            "original_audio_sha256": row["audio_sha256"], "source_vsqx_sha256": row["source_vsqx_sha256"],
            "start_sec": row["start_sec"], "end_sec": row["end_sec"], "status": "pending", "reviewer": None,
            "review_statement": None, "complete_sequence_confirmed": False,
            "note_ids": row["candidate_references"][0]["note_ids"],
            "suggestion_only": True, "existing_phoneme_approval_preserved": True,
        })
    return {"schema_version": "original_clip_binding_reviews_v1", "items": items,
            "scope": "Exact generated clip/sequence correspondence only; not a request to reannotate the complete VSQX."}


def write_summary(output: Path, report: dict, manifest: dict, audit: dict) -> None:
    lines = ["# Original Reference Evaluation", "", "## Outcome", "",
             "Human one-by-one calibration against the original is now recorded as reference evidence.",
             "The immutable audio-only PhoneticXeus outputs were replayed; no model was trained or deployed.",
             "", f"Official original score available: {report['original_score_available']}.",
             "Unconfirmed clip correspondence remains diagnostic, even when two timing resolutions agree.",
             "No original recognition-accuracy or improvement claim follows from conditional scores.", "",
             "## Coverage", "", "| Song | Original Review Method Bound | Mapped Notes | Total Notes |", "| --- | --- | ---: | ---: |"]
    for row in audit["songs"]:
        lines.append(f"| {row['song_id']} | {row['human_original_review_method_bound']} | {row['resolved_notes']} | {row['total_notes']} |")
    lines += ["", "Only the current three-song review statement is hash-bound; older song claims are preserved, not upgraded or rejected.",
              "Two of those three songs remain final test and are excluded from this development evaluation.", "",
              "## Timing And Conditional Diagnostics", "",
              "Both resolutions were fixed before scoring. The first is the primary diagnostic, never chosen by lowest PER.",
              "Boundary notes are retained and flagged. A complete window containing an unresolved token receives no score.", "",
              "| Clip | Original Seconds | Rendered Proposal Seconds | Resolution Difference | Conditional PER (4 / 8) |", "| --- | --- | --- | ---: | --- |"]
    details = {r["example_id"]: r for r in report["examples"]}
    for row in manifest["examples"]:
        candidate = row["candidate_references"][0]
        scores = details[row["example_id"]]["conditional_diagnostics"]
        formatted = " / ".join(f"{r['metrics']['per']:.4f}" if r["metrics"] is not None else "unresolved" for r in scores)
        interval = f"{candidate['rendered_start_sec']:.3f}-{candidate['rendered_end_sec']:.3f}" if "rendered_start_sec" in candidate else "collapsed"
        lines.append(f"| {row['example_id']} | {row['start_sec']:.3f}-{row['end_sec']:.3f} | {interval} | {row['resolution_disagreement_sec']:.3f}s | {formatted} |")
    confirmed = report["coverage"]["official_windows"]
    pending = report["coverage"]["fixed_windows"] - confirmed
    lines += ["", "## Remaining Specific Work", "",
              f"Confirmed exact clip bindings: {confirmed}; remaining fixed clips without official bindings: {pending}.",
              "The source VSQX phoneme approval is preserved separately from crop correspondence.",
              "Unknown token occurrences and exact source note IDs are in mapping-audit.json.",
              "The original-source mapping and crop binding must be complete before additional official sequence PER is emitted.",
              "No per-phone duration annotation is required for sequence PER.", "",
              "## Preservation", "", "All external sources were hash-verified during the identical reference rebuild.",
              "Historical reports, mappings, rendered scores and checkpoints are unchanged. Final-test predictions were not opened.", ""]
    with (output / "RESULTS.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay original-audio baselines against separately reviewed reference and timing evidence.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--review-evidence", type=Path, required=True)
    parser.add_argument("--baseline-index", type=Path, required=True)
    parser.add_argument("--previous-correspondence-index", type=Path, required=True)
    parser.add_argument("--clip-bindings", type=Path)
    parser.add_argument("--mapping-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("use a new output directory under .state/runs")
    before = time.monotonic()
    reference = verify_reference(args.reference)
    evidence = read_json(args.review_evidence)
    reviewed_songs(evidence, reference)
    index = read_json(args.baseline_index)
    entries = validate_baseline_index(index, reference)
    policy = load_policy()
    overlay_binding = None
    if args.mapping_evidence:
        reference, overlay_binding = apply_original_mapping_overlay(reference, args.mapping_evidence)
        policy["reference_overlay"] = overlay_binding
    bindings = []
    if args.clip_bindings:
        review = read_json(args.clip_bindings)
        if review.get("schema_version") != "original_clip_binding_reviews_v1":
            raise ValueError("unsupported clip-binding review schema")
        bindings = review["items"]
    output.mkdir(parents=True)
    write_json_new(output / "policy.json", policy)
    inputs = {key: {"path": str(path.resolve()), "sha256": sha256_file(path)} for key, path in
              (("reference", args.reference), ("human_review", args.review_evidence), ("baseline_index", args.baseline_index), ("previous_correspondence", args.previous_correspondence_index))}
    if args.clip_bindings:
        inputs["clip_bindings"] = {"path": str(args.clip_bindings.resolve()), "sha256": sha256_file(args.clip_bindings)}
    if args.mapping_evidence:
        inputs["mapping_evidence"] = {"path": str(args.mapping_evidence.resolve()), "sha256": sha256_file(args.mapping_evidence)}
    write_json_new(output / "inputs.json", inputs)
    reports = correspondence_reports(reference, args.reference, args.baseline_index, args.previous_correspondence_index, policy, output)
    manifest = build_original_manifest(reference, evidence, reference_digest=sha256_file(args.reference), evidence_digest=sha256_file(args.review_evidence),
                                       correspondence=reports, policy=policy, clip_bindings=bindings)
    manifest_file_hash = write_json_new(output / "original-reference-manifest.json", manifest)
    audit = mapping_audit(reference, evidence)
    if overlay_binding is not None:
        audit["mapping_overlay"] = overlay_binding
    write_json_new(output / "mapping-audit.json", audit)
    write_json_new(output / "clip-binding-review-template.json", review_template(manifest))
    print(f"reference_frozen={manifest_file_hash}; fixed_windows={len(manifest['examples'])}", flush=True)
    vocabulary = Path(reference["bindings"]["vocabulary"]["path"])
    predictions = {}
    for row in manifest["examples"]:
        predictions[row["example_id"]] = replay_prediction(row, entries[row["example_id"]], vocabulary)
        print(f"replayed_original={row['example_id']}", flush=True)
    report = evaluate_original(manifest, predictions)
    report.update(inputs=inputs, reference_rebuild_identical=True, all_external_source_hashes_verified=True,
                  original_manifest_file_sha256=manifest_file_hash, elapsed_seconds=time.monotonic() - before)
    report_hash = write_json_new(output / "evaluation-report.json", report)
    write_summary(output, report, manifest, audit)
    print(f"report_sha256={report_hash}; official_windows={report['coverage']['official_windows']}; diagnostic_windows={report['coverage']['diagnostic_mapping_complete_windows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

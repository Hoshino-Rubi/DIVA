from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.original_reference_evaluation import reviewed_songs, verify_reference  # noqa: E402
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay  # noqa: E402
from any2xsampa.training.reference_clock import ClockOffset, bind_score_notes, boundary_audit, clock_binding_record, exact_sample, rational, score_clocks, safe_expanding_crop  # noqa: E402
from any2xsampa.training.reference_clock_audio import common_support_offset, rendered_activity, score_activity_offset, validate_offset_probes  # noqa: E402


def normalize_order_evidence(payload: dict, song: dict, reference_digest: str, review_digest: str) -> dict:
    if payload.get("schema_version") != "original_clip_binding_reviews_v1":
        raise ValueError("unsupported order evidence schema")
    source = payload["evidence"]
    statement = source["statement"]
    if not isinstance(statement, str) or not statement or hashlib.sha256(statement.encode("utf-8")).hexdigest() != source["statement_sha256"]:
        raise ValueError("order statement hash mismatch")
    if source.get("source") != "direct_user_statement_in_current_conversation":
        raise ValueError("order evidence must cite the user")
    items = payload["items"]
    if len(items) != 3 or len({r["example_id"] for r in items}) != 3:
        raise ValueError("order evidence requires the original three distinct source records")
    for row in items:
        if row.get("song_id") != "shalong" or row.get("split") != "dev" or row.get("status") != "human_confirmed_order_only":
            raise ValueError("only Shalong whole-song order is established")
        if row.get("complete_sequence_confirmed") is not False or row.get("suggestion_only") is not True:
            raise ValueError("order evidence must not promote a machine crop")
        expected = {"reference_manifest_sha256": reference_digest, "review_evidence_sha256": review_digest,
                    "source_vsqx_sha256": song["human_vsqx"]["sha256"], "original_audio_sha256": song["original_audio"]["sha256"]}
        if any(row.get(k) != v for k, v in expected.items()) or row.get("review_statement") != statement:
            raise ValueError("order evidence is bound to different sources")
    return {"schema_version": "human_whole_song_order_evidence_v1", "song_id": "shalong", "split": "dev",
            "statement_verbatim": statement, "statement_sha256": source["statement_sha256"],
            "source_vsqx_sha256": song["human_vsqx"]["sha256"], "original_audio_sha256": song["original_audio"]["sha256"],
            "full_song_order_confirmed": True, "opening_offset_exists": True, "numeric_offset_confirmed": False,
            "machine_crops_confirmed": False, "legacy_suggested_note_ids_used_as_reviewed_labels": False,
            "legacy_scope_text_not_used_to_override_order_only_status": True}


def read_descriptors(path: Path, song: dict, policy: dict) -> tuple[object, object, dict]:
    import numpy as np

    report = read_json(path)
    if report.get("song_id") != "shalong" or report.get("split") != "dev":
        raise ValueError("clock audit may not read a test or different song descriptor")
    for role in ("original", "rendered"):
        if report[f"{role}_audio_sha256"] != song[f"{role}_audio"]["sha256"]:
            raise ValueError("descriptor source identity changed")
    config = report["policy"]
    if config.get("descriptor") != "librosa_0.11.0_chroma_cens" or config.get("sample_rate") != policy["descriptor_sample_rate"] or config.get("hop_length") != policy["descriptor_hop_samples"]:
        raise ValueError("unsupported descriptor extraction clock")
    artifact = report["descriptor_artifact"]
    cache_path = Path(artifact["path"])
    verify_file(cache_path, artifact["sha256"])
    if cache_path.stat().st_size > 64 * 1024**2:
        raise ValueError("descriptor cache exceeds resource bound")
    with np.load(cache_path, allow_pickle=False) as cache:
        if set(cache.files) != {"original", "rendered"}:
            raise ValueError("unexpected cached descriptor roles")
        original, rendered = cache["original"], cache["rendered"]
    for role, values in (("original", original), ("rendered", rendered)):
        expected = int(song[f"{role}_audio"]["duration_sec"] * config["sample_rate"] / config["hop_length"]) + 1
        if values.ndim != 2 or values.shape[0] != 12 or abs(values.shape[1] - expected) > 2:
            raise ValueError("descriptor frames disagree with the source audio duration")
    return original, rendered, artifact


def clip_details(clips: list[dict], notes: list[dict], low: float, high: float, pad: float, sample_rate: int) -> list[dict]:
    by_id = {row["note_id"]: row for row in notes}
    rows = []
    for clip in clips:
        if clip["song_id"] != "shalong" or clip["split"] != "dev" or clip["test_outputs_sealed"]:
            raise ValueError("clock crop is not a Shalong development clip")
        audit = boundary_audit(notes, start=clip["start_sec"], end=clip["end_sec"], offset_low=low, offset_high=high, uncertainty=pad)
        details = [{"note_id": n, "raw_lyric": by_id[n]["raw_lyric"], "raw_v4c_tokens": by_id[n]["raw_v4c_tokens"],
                    "score_absolute_start_sec": by_id[n]["start_sec"], "score_absolute_end_sec": by_id[n]["end_sec"]} for n in audit["partial_or_uncertain_edge_note_ids"]]
        rows.append({"example_id": clip["example_id"], "song_id": "shalong", "split": "dev", **audit,
                     "boundary_note_details": details,
                     "original_source_start_sample": exact_sample(clip["start_sec"], sample_rate, side="floor"),
                     "original_source_end_sample_exclusive": exact_sample(clip["end_sec"], sample_rate, side="ceil"),
                     "sample_rounding_policy": "outward_floor_start_ceil_end",
                     "official_evaluation_eligible": False})
    return rows


def write_summary(output: Path, result: dict) -> None:
    clock = result["clock_rule"]
    gates = clock["clock_validation"]
    a = result["original_rendered_validation"]
    b = result["rendered_score_validation"]
    lines = ["# Shalong Clock Rule Audit", "", "## Outcome", "",
             "The clock equations and their source bindings are versioned. Versioning does not create human crop approval.",
             f"Audio clock validation passed: {gates['passed']}.",
             "Official baseline upgrade performed: false. The existing crop/phone reference gate is unchanged.", "",
             "## Clock Equations", "", "```text",
             f"rendered = original + ({a['point_offset_sec']:.9f})",
             f"rendered = score_absolute + ({b['point_offset_sec']:.9f})",
             f"score_absolute = original + ({clock['composed']['offset_sec']:.9f})",
             f"score_display = score_absolute - ({result['score_clock']['premeasure_seconds']:.9f})", "```", "",
             "The first two offsets are audio estimates, not human-confirmed times. The last is exact VSQX metadata.",
             "Premeasure duration is not added a second time to the waveform clock.", "",
             "## Validation", "", "- Original/rendered lag candidates now use identical source-frame support.",
             "- Fixed original crops and a four-second guard are excluded from fitting.",
             "- Full-song cached descriptors retain temporal context; no strict waveform-context independence is claimed.",
             "- Both resolutions and both rendered activity thresholds are retained, with no PER-based selection.",
             "- The composed offset envelope is observed sensitivity, not a statistical confidence interval.", "",
             "Failure reasons: " + (", ".join(gates["failure_reasons"]) or "none"), "",
             "## Fixed Crop Boundary Audit", "", "| Crop | Original Seconds | Full Notes | Partial/Uncertain Notes |", "| --- | --- | ---: | --- |"]
    for row in clock["clips"]:
        lines.append(f"| {row['example_id']} | {row['start_sec']:.3f}-{row['end_sec']:.3f} | {len(row['fully_contained_note_ids'])} | {', '.join(row['partial_or_uncertain_edge_note_ids']) or 'none'} |")
    lines += ["", "The partial edge notes remain in the evidence. They are not discarded to obtain a better score.",
              "Whole-song order and engine-phone approval are preserved. They do not tell which subset of a cut note's phones is audible.", "",
              "## Preservation", "", "No new training, checkpoint selection, model download, default change, or final-test prediction access occurred.",
              "Source WAV/VSQX files, mappings, previous diagnostics and superseded results were not modified.",
              "The earlier baseline/adapter PER figures remain conditional diagnostics, not formal original accuracy.", ""]
    with (output / "RESULTS.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Version and independently audit the Shalong audio-to-score clock without promoting order-only approval.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--order-evidence", type=Path, required=True)
    parser.add_argument("--mapping-evidence", type=Path, required=True)
    parser.add_argument("--descriptor-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("use a new output directory inside .state/runs")
    before = time.monotonic()
    policy_path = ROOT / "src/any2xsampa/config/shalong_clock_audit_v1.json"
    policy = read_json(policy_path)
    reference = verify_reference(args.reference)
    review = read_json(args.human_review)
    if "shalong" not in reviewed_songs(review, reference):
        raise ValueError("human-reviewed original phonemes are not bound to Shalong")
    view, overlay = apply_original_mapping_overlay(reference, args.mapping_evidence)
    song = next(s for s in view["songs"] if s["song_id"] == "shalong")
    order = normalize_order_evidence(read_json(args.order_evidence), song, sha256_file(args.reference), sha256_file(args.human_review))
    clips = [c for c in reference["original_clips"] if c["song_id"] == "shalong"]
    score = score_clocks(Path(song["human_vsqx"]["normalized_path"]))
    bind_score_notes(score, song["notes"])
    original, rendered, cache = read_descriptors(args.descriptor_report, song, policy["audio_offset"])
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in
              (("reference", args.reference), ("human_review", args.human_review), ("order_evidence", args.order_evidence),
               ("mapping_evidence", args.mapping_evidence), ("descriptor_report", args.descriptor_report), ("policy", policy_path))}
    output.mkdir(parents=True)
    write_json_new(output / "inputs.json", inputs)
    write_json_new(output / "policy.json", policy)
    write_json_new(output / "human-order-evidence.json", order)
    write_json_new(output / "score-clock.json", score)
    probes = [common_support_offset(original, rendered, clips, factor=f, policy=policy["audio_offset"]) for f in policy["audio_offset"]["factors"]]
    a = validate_offset_probes(probes, policy["audio_offset"])
    write_json_new(output / "original-rendered-offset.json", {"probes": probes, "validation": a, "descriptor_cache": cache})
    print(f"original_rendered_offset={a['point_offset_sec']:.9f}; gate={a['passed']}", flush=True)
    rms, step = rendered_activity(song["rendered_audio"], step_sec=policy["rendered_to_score"]["sample_step_sec"])
    b = score_activity_offset(rms, song["notes"], step, policy["rendered_to_score"])
    write_json_new(output / "rendered-score-offset.json", b)
    print(f"rendered_score_offset={b['point_offset_sec']:.9f}; gate={b['passed']}", flush=True)
    low = a["observed_offset_envelope_sec"][0] - b["observed_offset_envelope_sec"][1]
    high = a["observed_offset_envelope_sec"][1] - b["observed_offset_envelope_sec"][0]
    clip_audits = clip_details(clips, song["notes"], low, high, policy["rendered_to_score"]["nominal_edge_uncertainty_sec"], song["original_audio"]["sample_rate"])
    for row in clip_audits:
        row["safe_expansion_proposal"] = safe_expanding_crop(
            song["notes"], start=row["start_sec"], end=row["end_sec"], offset_low=low, offset_high=high,
            uncertainty=policy["rendered_to_score"]["nominal_edge_uncertainty_sec"], policy=policy["recrop_proposals"],
        )
    diagnostics = {"passed": a["passed"] and b["passed"], "failure_reasons": a["failure_reasons"] + b["failure_reasons"],
                   "composed_offset_envelope_sec": [low, high], "envelope_is_confidence_interval": False}
    transforms = [ClockOffset("original", "rendered", rational(a["point_offset_sec"], "original/rendered offset")),
                  ClockOffset("rendered", "score_absolute", -rational(b["point_offset_sec"], "rendered/score offset"))]
    rule = clock_binding_record(transforms=transforms, source_bindings={**inputs, "original_audio": song["original_audio"],
                                "rendered_audio": song["rendered_audio"], "source_vsqx": song["human_vsqx"], "mapping_overlay": overlay,
                                "projected_shalong_notes_sha256": sha256_object(song["notes"])},
                                order_evidence=order, diagnostics=diagnostics, clips=clip_audits)
    write_json_new(output / "clock-rule.json", rule)
    result = {"schema_version": "shalong_clock_audit_report_v1", "inputs": inputs, "clock_rule": rule,
              "score_clock": {k: v for k, v in score.items() if k != "notes"},
              "original_rendered_validation": a, "rendered_score_validation": {k: v for k, v in b.items() if k != "fits"},
              "boundary_complete_fixed_clips": sum(c["boundary_complete"] for c in clip_audits),
              "official_original_windows": 0, "official_baseline_upgrade_performed": False,
              "full_song_order_approval_preserved": True, "legacy_note_suggestions_promoted": False,
              "source_rebuild_identical": True, "new_training_started": False, "test_predictions_opened": False,
              "elapsed_seconds": time.monotonic() - before, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json_new(output / "audit-report.json", result)
    write_summary(output, result)
    print(f"clock_rule_sha256={sha256_file(output / 'clock-rule.json')}; boundary_complete={result['boundary_complete_fixed_clips']}; official=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only, two-stage audio/score provenance and frozen note-count audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.local.common import read, sha  # noqa: E402
from any2xsampa.music.note_correspondence import audit_local_lag, note_count_ceiling  # noqa: E402
from any2xsampa.training.limited_data_artifacts import verify_file  # noqa: E402
from any2xsampa.training.reference_clock import bind_score_notes, score_clocks  # noqa: E402
from any2xsampa.training.reference_clock_audio import rendered_activity, score_activity_offset  # noqa: E402


CODE = [
    "tools/audit_game_note_correspondence.py",
    "src/any2xsampa/music/note_correspondence.py",
    "src/any2xsampa/training/reference_clock.py",
    "src/any2xsampa/training/reference_clock_audio.py",
    "src/any2xsampa/training/limited_data_artifacts.py",
    "src/any2xsampa/vocaloid/vsqx_alignment.py",
]
POLICY_PATH = ROOT / "src/any2xsampa/config/shalong_clock_audit_v1.json"
REPORT_NAME = "SOURCE_CLOCK_REVIEW.json"


def write_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def inputs(run: Path, reference: Path) -> tuple[dict, dict, dict, dict[str, dict]]:
    selected = read(run / "SELECTED_INPUTS.json")
    if sha(Path(selected["reference_file"])) != selected["reference_sha256"]:
        raise ValueError("reviewed selection source hash changed")
    reviewed = read(Path(selected["reference_file"]))
    songs = {s["song_id"]: s for s in read(reference)["songs"]}
    jobs = [j for j in selected["jobs"] if j["id"].startswith("original-")]
    if len(jobs) != 5 or len({j["song_id"] for j in jobs}) != 5:
        raise ValueError("the five original training songs must remain fixed")
    examples = {e["example_id"]: e for e in reviewed["examples"]}
    for job in jobs:
        ex, song = examples[job["source_example_id"]], songs[job["song_id"]]
        if (ex["song_id"] != song["song_id"] or ex["split"] != "train"
                or ex["status"] != "human_confirmed" or song["split"] != "train"
                or ex["audio_sha256"] != song["original_audio"]["sha256"]
                or ex["source_vsqx_sha256"] != song["human_vsqx"]["sha256"]
                or ex["reference_manifest_sha256"] != sha(reference)):
            raise ValueError("song, split, review or source binding changed")
        for side, note_id in (("start", ex["note_ids"][0]), ("end", ex["note_ids"][-1])):
            edge = ex["edge_evidence"][side]
            if (edge["status"] != "confirmed" or edge["note_id"] != note_id
                    or edge["audible_identity_confirmed"] is not True):
                raise ValueError("reviewed boundary note identity is not confirmed")
    return selected, reviewed, songs, examples


def source_paths(run: Path, reference: Path, selected: dict) -> list[Path]:
    paths = [run / "SELECTED_INPUTS.json", Path(selected["reference_file"]), reference,
             run / "BASELINE_REVIEW.json", POLICY_PATH,
             ROOT / ".state/runs/limited-data-20260906T050251/correspondence-v2/index.json",
             ROOT / ".state/runs/original-reference-eval-20260906T154652/offset-final/inputs.json"]
    previous = ROOT / ".state/runs/original-reference-eval-20260906T154652/offset-final/correspondence"
    paths.extend(previous / f"{j['song_id']}.json" for j in selected["jobs"] if j["id"].startswith("original-"))
    paths.extend(ROOT / ".state/runs/limited-data-20260906T050251/correspondence-v2" / f"{j['song_id']}.json"
                 for j in selected["jobs"] if j["id"].startswith("original-"))
    return paths


def test_count(xml_path: Path) -> int:
    root = ET.parse(xml_path).getroot()
    tests = int(root.attrib.get("tests", 0)) or sum(int(s.attrib.get("tests", 0)) for s in root.findall("testsuite"))
    failures = int(root.attrib.get("failures", 0)) + int(root.attrib.get("errors", 0))
    failures += sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in root.findall("testsuite"))
    if tests <= 0 or failures:
        raise ValueError("preflight tests have failures or no passing cases")
    return tests


def preflight(run: Path, reference: Path, output: Path, tests_xml: Path) -> dict:
    selected, _, _, _ = inputs(run, reference)
    if not output.is_relative_to(ROOT / ".state/runs"):
        raise ValueError("output must be in a new .state/runs directory")
    tests = test_count(tests_xml)
    bindings = {str(path.resolve()): sha(path) for path in source_paths(run, reference, selected)}
    code = {str((ROOT / name).resolve()): sha(ROOT / name) for name in CODE}
    report = {"schema_version": "game_note_clock_review_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "mode": "read_only_audio_clock_then_count_ceiling", "tests": {"path": str(tests_xml), "sha256": sha(tests_xml), "passed": tests},
              "code_sha256": code, "input_sha256": bindings,
              "sampling": "fixed earliest confirmed 10s window per training song; no new selection",
              "search": "rendered RMS vs score, then cached original/rendered CENS fixed lag within +/-4s of human-bound note-span midpoint",
              "prediction_access_during_clock_stage": False, "model_forwards": 0, "optimizer_updates": 0,
              "new_downloads": 0, "training_or_final_test_access": False,
              "failure_policy": "no automatic retries; no clock promotion from chroma", "reference_boundary_precision_assumed": False}
    write_new(output / REPORT_NAME, report)
    return report


def verify_review(run: Path, reference: Path, output: Path) -> tuple[dict, dict, dict, dict[str, dict]]:
    review = read(output / REPORT_NAME)
    selected, reviewed, songs, examples = inputs(run, reference)
    for role in ("code_sha256", "input_sha256"):
        for file, digest in review[role].items():
            verify_file(Path(file), digest)
    verify_file(Path(review["tests"]["path"]), review["tests"]["sha256"])
    if review["tests"]["passed"] <= 0 or review["model_forwards"] != 0:
        raise ValueError("invalid reviewed execution budget")
    return selected, reviewed, songs, examples


def clock_stage(run: Path, reference: Path, output: Path) -> dict:
    import numpy as np

    selected, _, songs, examples = verify_review(run, reference, output)
    policy = read(POLICY_PATH)["rendered_to_score"]
    previous = ROOT / ".state/runs/original-reference-eval-20260906T154652/offset-final/correspondence"
    descriptor_root = ROOT / ".state/runs/limited-data-20260906T050251/correspondence-v2"
    old_inputs = read(previous.parent / "inputs.json")
    descriptor_index_file = descriptor_root / "index.json"
    verify_file(descriptor_index_file, old_inputs["descriptor_index"]["sha256"])
    descriptor_index = read(descriptor_index_file)
    if descriptor_index["reference_manifest_sha256"] != sha(reference):
        raise ValueError("descriptor index reference differs")
    indexed = {r["song_id"]: r for r in descriptor_index["songs"]}
    rows = []
    for job in selected["jobs"]:
        if not job["id"].startswith("original-"):
            continue
        song, ex = songs[job["song_id"]], examples[job["source_example_id"]]
        vsqx = song["human_vsqx"]
        score_path = Path(vsqx.get("normalized_path", vsqx["path"]))
        verify_file(score_path, vsqx.get("normalized_sha256", vsqx["sha256"]))
        score = score_clocks(score_path)
        bind_score_notes(score, song["notes"])
        verify_file(Path(song["original_audio"]["path"]), song["original_audio"]["sha256"])
        rms, step = rendered_activity(song["rendered_audio"], step_sec=policy["sample_step_sec"])
        activity = score_activity_offset(rms, song["notes"], step, policy)
        old = read(previous / f"{job['song_id']}.json")
        prior_file = descriptor_root / f"{job['song_id']}.json"
        verify_file(prior_file, indexed[job["song_id"]]["sha256"])
        prior = read(prior_file)
        if (old["descriptor_artifact"] != prior["descriptor_artifact"]
                or prior["original_audio_sha256"] != song["original_audio"]["sha256"]
                or prior["rendered_audio_sha256"] != song["rendered_audio"]["sha256"]
                or prior["split"] != "train" or old["split"] != "train"):
            raise ValueError("previous audio-only descriptor source does not match")
        artifact = old["descriptor_artifact"]
        verify_file(Path(artifact["path"]), artifact["sha256"])
        with np.load(artifact["path"], allow_pickle=False) as cache:
            if set(cache.files) != {"original", "rendered"}:
                raise ValueError("unexpected descriptor roles")
            original, rendered = cache["original"], cache["rendered"]
        notes = {n["note_id"]: n for n in song["notes"]}
        first, last = notes[ex["note_ids"][0]], notes[ex["note_ids"][-1]]
        span_mid = (first["start_sec"] + last["end_sec"]) / 2
        crop_mid = (ex["start_sec"] + ex["end_sec"]) / 2
        expected = span_mid - crop_mid + activity["point_offset_sec"]
        probe = audit_local_lag(original, rendered, start_sec=ex["start_sec"], end_sec=ex["end_sec"],
                                expected_lag_sec=expected, frame_sec=512/22050)
        old_lags = [p["best"]["offset_sec"] for p in old["probes"]]
        rows.append({"song_id": song["song_id"], "example_id": ex["example_id"],
                     "audio_sha256": song["original_audio"]["sha256"],
                     "rendered_sha256": song["rendered_audio"]["sha256"],
                     "source_vsqx_sha256": vsqx["sha256"],
                     "score": {"bpm_x100": score["bpm_x100"], "premeasure_seconds": score["premeasure_seconds"],
                               "note_count": len(score["notes"]), "clock_bound_to_reference": True},
                     "score_to_rendered_activity": {k: v for k, v in activity.items() if k != "fits"},
                     "audio_only_local": probe, "previous_global_lags_sec": old_lags,
                     "previous_global_lags_promoted": False,
                     "register_transform_source_verified": False,
                     "model_or_pitch_predictions_used_for_clock": False,
                     "exact_original_note_boundaries_proven": False})
    result = {"schema_version": "game_note_source_clock_audit_v1", "rows": rows,
              "formal_windows": 0, "model_forwards": 0, "predictions_opened": False,
              "notes": "Render and mix matching yields candidates, not a 50ms original note annotation. No offset is applied to a model prediction."}
    write_new(output / "SOURCE_CLOCKS.json", result)
    return result


def ceiling_stage(run: Path, reference: Path, output: Path) -> dict:
    selected, _, songs, examples = verify_review(run, reference, output)
    clocks = read(output / "SOURCE_CLOCKS.json")
    if [r["song_id"] for r in clocks["rows"]] != [j["song_id"] for j in selected["jobs"] if j["id"].startswith("original-")]:
        raise ValueError("clock audit does not cover the frozen five songs")
    baseline = read(run / "BASELINE_REVIEW.json")
    if baseline["input_sha256"] != sha(run / "SELECTED_INPUTS.json"):
        raise ValueError("frozen prediction review mismatches input")
    records = []
    for job in selected["jobs"]:
        if not job["id"].startswith("original-"):
            continue
        ex = examples[job["source_example_id"]]
        prediction_path = Path(job["output"]) / "prediction.json"
        verify_file(prediction_path, baseline["predictions"][job["id"]])
        prediction = read(prediction_path)
        if (prediction["input"]["audio_sha256"] != songs[job["song_id"]]["original_audio"]["sha256"]
                or len(ex["note_ids"]) != len(set(ex["note_ids"]))):
            raise ValueError("prediction audio or reference note IDs changed")
        records.append({"song_id": job["song_id"], "example_id": ex["example_id"],
                        "prediction_sha256": baseline["predictions"][job["id"]],
                        **note_count_ceiling(ex["note_ids"], len(prediction["notes"]))})
    n_ref = sum(r["reference_notes"] for r in records)
    n_pred = sum(r["predicted_notes"] for r in records)
    possible_matches = sum(r["maximum_one_to_one_matches"] for r in records)
    required = math.ceil(.95 * n_ref)
    report = {"schema_version": "game_note_frozen_count_ceiling_v1", "rows": records,
              "frozen_reference_notes": n_ref, "frozen_predicted_notes": n_pred,
              "maximum_possible_matches": possible_matches, "minimum_missing_notes": n_ref - possible_matches,
              "maximum_possible_recall": possible_matches / n_ref,
              "matches_needed_for_95pct": required, "additional_matches_needed_even_if_every_existing_note_were_exact": max(0, required-possible_matches),
              "ninety_five_pct_joint_gate_mathematically_possible": possible_matches >= required,
              "actual_original_note_boundary_metrics_reported": False,
              "source_clock_review_sha256": sha(output / "SOURCE_CLOCKS.json"),
              "model_forwards": 0, "new_training": False, "default_model_changed": False}
    write_new(output / "COUNT_CEILING.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "clock", "ceiling"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tests-xml", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage == "preflight":
        if args.tests_xml is None:
            parser.error("--tests-xml is required for preflight")
        result = preflight(args.run, args.reference, out, args.tests_xml)
    elif args.stage == "clock":
        result = clock_stage(args.run, args.reference, out)
    else:
        result = ceiling_stage(args.run, args.reference, out)
    print(json.dumps({"stage": args.stage, "rows": len(result.get("rows", [])),
                      "maximum_possible_recall": result.get("maximum_possible_recall")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

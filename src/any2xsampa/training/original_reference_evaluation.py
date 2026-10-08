"""Original-audio evaluation with separate annotation and clip-binding evidence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256, PhoneticXeusPosteriorRecognizer
from any2xsampa.training.limited_data_artifacts import finite_number, read_json, sha256_file, sha256_object, verify_file
from any2xsampa.training.limited_data_reference import SONG_SPLIT, SPLITS, build_reference_manifest
from any2xsampa.training.phone_error_metrics import phone_metrics


SCHEMA = "original_reference_evaluation_manifest_v1"


def verify_reference(reference_path: Path) -> dict[str, Any]:
    reference = read_json(reference_path)
    unsigned = {k: v for k, v in reference.items() if k != "manifest_hash"}
    if reference.get("manifest_hash") != sha256_object(unsigned):
        raise ValueError("reference manifest hash mismatch")
    if reference.get("split_sha256") != sha256_object(SPLITS):
        raise ValueError("reference song split changed")
    for binding in reference["bindings"].values():
        verify_file(Path(binding["path"]), binding["sha256"])
    bindings = reference["bindings"]
    rebuilt, _ = build_reference_manifest(*(Path(bindings[name]["path"]) for name in ("triplets", "mapping", "vocabulary")))
    if rebuilt != reference:
        raise ValueError("reference rebuild differs from immutable source notes")
    return reference


def reviewed_songs(evidence: Mapping[str, Any], reference: Mapping[str, Any]) -> set[str]:
    if evidence.get("schema_version") != "human_reference_review_clarification_v1":
        raise ValueError("unsupported human review evidence")
    if evidence.get("evidence_source") != "direct_user_statement_in_current_conversation":
        raise ValueError("human review must cite the user's statement")
    if evidence.get("evidence_status") != "human_reviewed_against_original_as_reported_by_user":
        raise ValueError("human review method was not established")
    if not evidence.get("reviewer") or not evidence.get("user_statement_verbatim"):
        raise ValueError("review evidence requires its reviewer and exact statement")
    if evidence["scope"]["source_manifest_sha256"] != reference["bindings"]["triplets"]["sha256"]:
        raise ValueError("review evidence refers to a different triplet revision")
    songs = {s["song_id"]: s for s in reference["songs"]}
    reviewed: set[str] = set()
    for item in evidence["scope"]["items"]:
        song_id = item["song_id"]
        if song_id in reviewed or song_id not in songs:
            raise ValueError("duplicate or unknown human-review song")
        for field, role in (("original_wav_sha256", "original_audio"), ("rendered_wav_sha256", "rendered_audio"), ("human_vsqx_sha256", "human_vsqx")):
            if item[field] != songs[song_id][role]["sha256"]:
                raise ValueError("human review is bound to a different source revision")
        reviewed.add(song_id)
    return reviewed


def note_sequence(notes: Sequence[Mapping[str, Any]], note_ids: Sequence[str]) -> dict[str, Any]:
    by_id = {note["note_id"]: (i, note) for i, note in enumerate(notes)}
    if len(note_ids) != len(set(note_ids)) or any(n not in by_id for n in note_ids):
        raise ValueError("unknown or duplicate note IDs")
    indices = [by_id[n][0] for n in note_ids]
    if indices and indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("a reference sequence must retain contiguous source notes")
    rows = [by_id[n][1] for n in note_ids]
    unresolved = sorted({token for row in rows for token in row["unresolved_tokens"]})
    unsupported = sorted({symbol for row in rows for symbol in row["unsupported_model_ipa"]})
    complete = bool(rows) and not unresolved and not unsupported and all(row["mapping_status"] == "resolved_proxy" for row in rows)
    return {
        "note_ids": list(note_ids), "raw_v4c_tokens_by_note": [r["raw_v4c_tokens"] for r in rows],
        "mapping_complete": complete, "unresolved_tokens": unresolved,
        "unsupported_model_ipa": unsupported,
        "reference_ipa": [p for r in rows for p in r["model_target_ipa"]] if complete else [],
        "reference_ids": [p for r in rows for p in r["model_target_ids"]] if complete else [],
        "raw_annotations_modified": False,
    }


def propose_sequence(song: Mapping[str, Any], interval: Mapping[str, Any], margin: float) -> dict[str, Any]:
    start = finite_number(interval["rendered_start_sec"], "rendered proposal start")
    end = finite_number(interval["rendered_end_sec"], "rendered proposal end")
    margin = finite_number(margin, "boundary margin", minimum=0)
    if start < 0 or end <= start or end > song["rendered_audio"]["duration_sec"]:
        return {"status": "invalid_or_collapsed_timing", "note_ids": [], "mapping_complete": False,
                "reference_ipa": [], "boundary_ambiguous_note_ids": [], "unresolved_tokens": []}
    # Retain every overlapping note; never trim away a hard or unknown edge phone.
    selected = [n for n in song["notes"] if n["start_sec"] < end and n["end_sec"] > start]
    boundary_notes = [n["note_id"] for n in song["notes"] if any(n["start_sec"] - margin <= boundary <= n["end_sec"] + margin for boundary in (start, end))]
    return {"status": "machine_proposed", **note_sequence(song["notes"], [n["note_id"] for n in selected]),
            "boundary_ambiguous_note_ids": boundary_notes, "rendered_start_sec": start, "rendered_end_sec": end}


def validate_clip_binding(binding: Mapping[str, Any], clip: Mapping[str, Any], song: Mapping[str, Any], reference_digest: str, evidence_digest: str) -> None:
    if binding.get("status") != "human_confirmed" or not binding.get("reviewer") or not binding.get("review_statement"):
        raise ValueError("explicit clip review is required; a machine proposal is not approval")
    expected = {
        "reference_manifest_sha256": reference_digest, "review_evidence_sha256": evidence_digest,
        "example_id": clip["example_id"], "song_id": song["song_id"], "split": song["split"],
        "original_audio_sha256": song["original_audio"]["sha256"],
        "source_vsqx_sha256": song["human_vsqx"]["sha256"],
    }
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError("clip review source/split binding mismatch")
    if song["split"] == "test" or clip["audio_role"] != "original":
        raise ValueError("sealed test or rendered data cannot enter original development evaluation")
    for field in ("start_sec", "end_sec"):
        if finite_number(binding.get(field), field, minimum=0) != clip[field]:
            raise ValueError("clip review time offset differs from the frozen original window")
    if binding.get("complete_sequence_confirmed") is not True:
        raise ValueError("the complete crop sequence, including its edges, must be confirmed")
    if not binding.get("note_ids"):
        raise ValueError("clip review requires source note IDs")
    note_sequence(song["notes"], binding["note_ids"])


def build_original_manifest(reference: Mapping[str, Any], evidence: Mapping[str, Any], *, reference_digest: str, evidence_digest: str, correspondence: Sequence[Mapping[str, Any]], policy: Mapping[str, Any], clip_bindings: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    reviewed = reviewed_songs(evidence, reference)
    songs = {s["song_id"]: s for s in reference["songs"]}
    proposals: dict[str, Any] = {}
    for report in correspondence:
        song = songs[report["song_id"]]
        if song["split"] == "test" or report["split"] != song["split"]:
            raise ValueError("correspondence has a sealed or mismatched split")
        for row in report["clips"]:
            if row["example_id"] in proposals:
                raise ValueError("duplicate correspondence example")
            proposals[row["example_id"]] = row
    expected_ids = {c["example_id"] for c in reference["original_clips"] if c["split"] != "test"}
    if set(proposals) != expected_ids:
        raise ValueError("correspondence must cover every fixed train/development clip")
    bindings = {b["example_id"]: b for b in clip_bindings}
    if len(bindings) != len(clip_bindings) or not set(bindings) <= expected_ids:
        raise ValueError("duplicate, unknown, or sealed clip review")
    rows = []
    for clip in reference["original_clips"]:
        if clip["split"] == "test":
            continue
        song = songs[clip["song_id"]]
        proposed = proposals[clip["example_id"]]
        if proposed["original_start_sec"] != clip["start_sec"] or proposed["original_end_sec"] != clip["end_sec"]:
            raise ValueError("correspondence uses a different original clip")
        candidates = [{"factor": c["factor"], **propose_sequence(song, c, policy["correspondence"]["note_boundary_margin_sec"])} for c in proposed["candidates"]]
        if [c["factor"] for c in candidates] != policy["correspondence"]["downsample_factors"]:
            raise ValueError("correspondence candidates differ from the fixed policy")
        binding = bindings.get(clip["example_id"])
        reviewed_reference = None
        if binding is not None:
            validate_clip_binding(binding, clip, song, reference_digest, evidence_digest)
            reviewed_reference = note_sequence(song["notes"], binding["note_ids"])
        reasons = []
        if song["song_id"] not in reviewed:
            reasons.append("original_review_method_not_bound_to_this_song")
        if binding is None:
            reasons.append("exact_original_clip_correspondence_pending")
        if reviewed_reference is not None and not reviewed_reference["mapping_complete"]:
            reasons.append("whole_window_contains_unresolved_mapping")
        rows.append({
            "example_id": clip["example_id"], "song_id": song["song_id"], "split": song["split"],
            "audio_role": "original", "audio_path": song["original_audio"]["path"],
            "audio_sha256": song["original_audio"]["sha256"], "start_sec": clip["start_sec"], "end_sec": clip["end_sec"],
            "source_vsqx_sha256": song["human_vsqx"]["sha256"], "source_claims": song["source_claims"],
            "human_original_review_established": song["song_id"] in reviewed,
            "reference_kind": "human_calibrated_vsqx_derived_ipa",
            "candidate_references": candidates, "resolution_disagreement_sec": proposed["resolution_disagreement_sec"],
            "resolution_consistent": proposed["resolution_consistent"],
            "candidate_note_sequences_agree": candidates[0]["note_ids"] == candidates[1]["note_ids"],
            "clip_binding": binding, "reviewed_reference": reviewed_reference,
            "official_evaluation_eligible": not reasons, "official_exclusion_reasons": reasons,
            "training_eligible": False, "test_outputs_sealed": False,
        })
    result = {
        "schema_version": SCHEMA, "reference_manifest_sha256": reference_digest,
        "review_evidence_sha256": evidence_digest, "split_sha256": reference["split_sha256"],
        "policy": dict(policy), "examples": rows,
        "review_method_bound_songs": sorted(reviewed),
        "test_groups_omitted": list(SPLITS["test"]),
        "test_predictions_opened": False, "historical_dataset_modified": False,
    }
    result["manifest_hash"] = sha256_object(result)
    return result


def validate_baseline_index(index: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if index.get("purpose") != "original-baselines" or index.get("source_dataset_hash") != reference["manifest_hash"]:
        raise ValueError("baseline must bind the original source reference")
    if index.get("split_sha256") != reference["split_sha256"] or index.get("no_labels_or_score_received") is not True:
        raise ValueError("baseline split or audio-only provenance mismatch")
    model = index["model_metadata"]
    if (model.get("model_id"), model.get("revision"), model.get("weights_sha256")) != (MODEL_ID, REVISION, WEIGHT_SHA256):
        raise ValueError("unexpected baseline model")
    entries = {e["example_id"]: e for e in index["entries"]}
    if len(entries) != len(index["entries"]) or set(entries) != {c["example_id"] for c in reference["original_clips"]}:
        raise ValueError("baseline must cover exactly the frozen original clips")
    for clip in reference["original_clips"]:
        entry = entries[clip["example_id"]]
        for field in ("song_id", "split", "audio_role", "audio_sha256", "start_sec", "end_sec", "test_outputs_sealed"):
            if entry[field] != clip[field]:
                raise ValueError(f"baseline binding mismatch: {field}")
    return entries


def replay_prediction(row: Mapping[str, Any], entry: Mapping[str, Any], vocabulary_path: Path) -> dict[str, Any]:
    if row["split"] == "test" or entry["test_outputs_sealed"]:
        raise ValueError("sealed prediction cannot be opened")
    path = Path(entry["posterior_path"])
    verify_file(path, entry["posterior_sha256"])
    payload = read_json(path)
    for field in ("example_id", "song_id", "split", "audio_role", "test_outputs_sealed"):
        if payload[field] != row[field]:
            raise ValueError("posterior identity differs from its original reference")
    producer = payload["producer"]
    metadata = producer["model_metadata"]
    if (metadata.get("model_id"), metadata.get("revision"), metadata.get("weights_sha256")) != (MODEL_ID, REVISION, WEIGHT_SHA256):
        raise ValueError("posterior model differs from the pinned original baseline")
    raw_path = (path.parent / payload["raw_evidence"]["file"]).resolve()
    if raw_path != Path(entry["evidence_path"]).resolve() or payload["raw_evidence"]["sha256"] != entry["evidence_sha256"]:
        raise ValueError("posterior raw cache differs from its baseline index")
    if any(producer.get(field) is not False for field in ("lyrics_used", "score_used", "rendered_reference_used")):
        raise ValueError("original baseline must be audio-only")
    for field in ("start_sec", "end_sec"):
        if finite_number(producer["crop"][f"requested_{field}"], field) != row[field]:
            raise ValueError("posterior crop time mismatch")
        if abs(finite_number(producer["crop"][f"effective_{field}"], field) - row[field]) > 1 / producer["input_audio"]["sample_rate"]:
            raise ValueError("posterior effective crop time mismatch")
    if producer["input_audio"]["sha256"] != row["audio_sha256"]:
        raise ValueError("posterior source audio hash mismatch")
    recognizer = PhoneticXeusPosteriorRecognizer(path, vocabulary_path=vocabulary_path)
    segments = recognizer.recognize(Path(row["audio_path"]))
    fixture = PhoneticXeusPosteriorRecognizer(path, fixture=True, vocabulary_path=vocabulary_path).recognize(Path(row["audio_path"]))
    if [(s.time_range, s.candidate_phones, s.confidence) for s in segments] != [(s.time_range, s.candidate_phones, s.confidence) for s in fixture]:
        raise ValueError("real and fixture replay differ")
    return {"phones": payload["decoding"]["collapsed_raw_labels"], "segments": len(segments),
            "special_emissions": payload["decoding"]["special_emissions"], "raw_logits_reverified": True,
            "real_fixture_match": True, "posterior_sha256": entry["posterior_sha256"]}


def evaluate_original(manifest: Mapping[str, Any], predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA or manifest.get("manifest_hash") != sha256_object({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("original evaluation manifest hash mismatch")
    if set(predictions) != {r["example_id"] for r in manifest["examples"]}:
        raise ValueError("predictions must cover the fixed train/development set")
    if manifest.get("split_sha256") != sha256_object(SPLITS):
        raise ValueError("original evaluation split differs from the frozen experiment")
    details, official = [], []
    per_song: dict[str, list] = {}
    exclusion_counts: Counter[str] = Counter()
    diagnostic_windows = 0
    for row in manifest["examples"]:
        if SONG_SPLIT.get(row["song_id"]) != row["split"] or row["split"] == "test" or row["audio_role"] != "original":
            raise ValueError("original evaluation song/split/role violation")
        predicted = predictions[row["example_id"]]["phones"]
        if not isinstance(predicted, list) or any(not isinstance(p, str) or not p for p in predicted):
            raise ValueError("invalid predicted phone sequence")
        diagnostic = []
        for candidate in row["candidate_references"]:
            metric = phone_metrics([(candidate["reference_ipa"], predicted)]) if candidate["mapping_complete"] else None
            diagnostic.append({"factor": candidate["factor"], "metrics": metric,
                               "mapping_complete": candidate["mapping_complete"], "original_accuracy_claim_allowed": False})
        diagnostic_windows += diagnostic[0]["metrics"] is not None
        metric = None
        if row["official_evaluation_eligible"]:
            if not row["human_original_review_established"] or not row["clip_binding"] or row["clip_binding"]["status"] != "human_confirmed" or not row["reviewed_reference"]["mapping_complete"] or row["official_exclusion_reasons"]:
                raise ValueError("official eligibility bypasses reference review gates")
            binding = row["clip_binding"]
            for field in ("example_id", "song_id", "split", "start_sec", "end_sec"):
                if binding[field] != row[field]:
                    raise ValueError("official clip binding differs from its evaluation row")
            if binding["original_audio_sha256"] != row["audio_sha256"] or binding["source_vsqx_sha256"] != row["source_vsqx_sha256"]:
                raise ValueError("official clip source hashes differ")
            if binding["reference_manifest_sha256"] != manifest["reference_manifest_sha256"] or binding["review_evidence_sha256"] != manifest["review_evidence_sha256"]:
                raise ValueError("official clip evidence bindings differ")
            if binding.get("complete_sequence_confirmed") is not True or binding["note_ids"] != row["reviewed_reference"]["note_ids"]:
                raise ValueError("official reference differs from the reviewed complete sequence")
            pair = (row["reviewed_reference"]["reference_ipa"], predicted)
            official.append(pair)
            per_song.setdefault(row["song_id"], []).append(pair)
            metric = phone_metrics([pair])
        exclusion_counts.update(row["official_exclusion_reasons"])
        details.append({"example_id": row["example_id"], "song_id": row["song_id"], "split": row["split"],
                        "prediction": predictions[row["example_id"]], "official_metrics": metric,
                        "conditional_diagnostics": diagnostic, "exclusion_reasons": row["official_exclusion_reasons"]})
    return {
        "schema_version": "original_reference_evaluation_report_v1", "manifest_hash": manifest["manifest_hash"],
        "metric_scope": manifest["policy"]["official_metric_scope"],
        "official_metrics": phone_metrics(official) if official else None,
        "official_per_song": {song: phone_metrics(pairs) for song, pairs in per_song.items()},
        "original_score_available": bool(official), "model_improvement_measured": False,
        "coverage": {"fixed_windows": len(details), "human_review_method_bound_windows": sum(r["human_original_review_established"] for r in manifest["examples"]),
                     "official_windows": len(official), "official_window_fraction": len(official) / len(details) if details else 0.0,
                     "diagnostic_mapping_complete_windows": diagnostic_windows, "exclusion_reason_counts": dict(exclusion_counts)},
        "diagnostic_scope": manifest["policy"]["diagnostic_metric_scope"], "examples": details,
        "phone_duration_evaluated": False, "test_predictions_opened": False,
        "new_training_started": False, "default_model_replaced": False,
    }

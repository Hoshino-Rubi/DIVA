from __future__ import annotations

import pytest

from any2xsampa.training.limited_data_artifacts import sha256_object
from any2xsampa.training.limited_data_reference import SPLITS
from any2xsampa.training.original_reference_correspondence import load_policy
from any2xsampa.training.original_reference_evaluation import (
    build_original_manifest, evaluate_original, note_sequence, propose_sequence, reviewed_songs, validate_clip_binding,
)


@pytest.fixture
def inputs() -> tuple[dict, dict, list[dict]]:
    notes = [{"note_id": f"shalong-n{i:03d}", "start_sec": i, "end_sec": i + 0.8,
              "raw_v4c_tokens": ["k_h"], "model_target_ipa": ["k\u02b0"], "model_target_ids": [4],
              "unresolved_tokens": [], "unsupported_model_ipa": [], "mapping_status": "resolved_proxy"} for i in range(1, 4)]
    song = {"song_id": "shalong", "split": "dev", "notes": notes, "source_claims": {"phoneme_correctness": "user_asserted_correct"},
            "original_audio": {"path": "original.wav", "sha256": "a" * 64, "duration_sec": 40},
            "rendered_audio": {"path": "rendered.wav", "sha256": "b" * 64, "duration_sec": 40},
            "human_vsqx": {"sha256": "c" * 64}}
    clip = {"example_id": "shalong-original-01", "song_id": "shalong", "split": "dev", "audio_role": "original", "start_sec": 10, "end_sec": 28}
    reference = {"songs": [song], "original_clips": [clip], "split_sha256": sha256_object(SPLITS), "bindings": {"triplets": {"sha256": "d" * 64}}}
    evidence = {"schema_version": "human_reference_review_clarification_v1", "evidence_source": "direct_user_statement_in_current_conversation",
                "evidence_status": "human_reviewed_against_original_as_reported_by_user", "reviewer": "user", "user_statement_verbatim": "Reviewed one by one against original.",
                "scope": {"source_manifest_sha256": "d" * 64, "items": [{"song_id": "shalong", "original_wav_sha256": "a" * 64, "rendered_wav_sha256": "b" * 64, "human_vsqx_sha256": "c" * 64}]}}
    correspondence = [{"song_id": "shalong", "split": "dev", "clips": [{"example_id": clip["example_id"], "original_start_sec": 10, "original_end_sec": 28,
                        "candidates": [{"factor": f, "rendered_start_sec": 0.9, "rendered_end_sec": 3.9} for f in (4, 8)],
                        "resolution_disagreement_sec": 0.0, "resolution_consistent": True}]}]
    return reference, evidence, correspondence


def build(inputs, bindings=()) -> dict:
    reference, evidence, correspondence = inputs
    return build_original_manifest(reference, evidence, reference_digest="e" * 64, evidence_digest="f" * 64,
                                   correspondence=correspondence, policy=load_policy(), clip_bindings=bindings)


def approval(inputs) -> dict:
    reference = inputs[0]
    return {"status": "human_confirmed", "reviewer": "user", "review_statement": "This exact crop contains these complete notes.",
            "reference_manifest_sha256": "e" * 64, "review_evidence_sha256": "f" * 64,
            "example_id": "shalong-original-01", "song_id": "shalong", "split": "dev",
            "original_audio_sha256": "a" * 64, "source_vsqx_sha256": "c" * 64,
            "start_sec": 10, "end_sec": 28, "complete_sequence_confirmed": True,
            "note_ids": [n["note_id"] for n in reference["songs"][0]["notes"]]}


def test_manual_original_review_is_preserved_but_machine_timing_not_promoted(inputs) -> None:
    result = build(inputs)
    row = result["examples"][0]
    assert row["human_original_review_established"]
    assert not row["official_evaluation_eligible"]
    assert row["official_exclusion_reasons"] == ["exact_original_clip_correspondence_pending"]
    assert not row["training_eligible"]


def test_verified_sequence_can_be_scored_without_per_phone_timestamps(inputs) -> None:
    manifest = build(inputs, [approval(inputs)])
    result = evaluate_original(manifest, {"shalong-original-01": {"phones": ["k\u02b0"] * 3}})
    assert result["official_metrics"]["per"] == 0
    assert result["original_score_available"]
    assert not result["phone_duration_evaluated"]
    assert not result["model_improvement_measured"]


def test_unconfirmed_scores_are_only_conditional(inputs) -> None:
    result = evaluate_original(build(inputs), {"shalong-original-01": {"phones": ["k\u02b0"] * 3}})
    assert result["official_metrics"] is None
    assert result["examples"][0]["conditional_diagnostics"][0]["metrics"]["per"] == 0
    assert not result["original_score_available"]


def test_complete_unresolved_window_excluded_not_spliced(inputs) -> None:
    n = inputs[0]["songs"][0]["notes"][1]
    n.update(mapping_status="unresolved", raw_v4c_tokens=["7"], unresolved_tokens=["7"], model_target_ipa=[], model_target_ids=[])
    result = build(inputs, [approval(inputs)])
    row = result["examples"][0]
    assert len(row["reviewed_reference"]["note_ids"]) == 3
    assert row["reviewed_reference"]["reference_ipa"] == []
    assert row["official_exclusion_reasons"] == ["whole_window_contains_unresolved_mapping"]


@pytest.mark.parametrize("field,value", [("start_sec", 10.1), ("end_sec", float("nan")), ("original_audio_sha256", "0" * 64), ("source_vsqx_sha256", "0" * 64), ("split", "train"), ("status", "pending"), ("complete_sequence_confirmed", False)])
def test_clip_binding_rejects_unreviewed_or_mismatched_inputs(inputs, field, value) -> None:
    binding = approval(inputs)
    binding[field] = value
    with pytest.raises(ValueError):
        build(inputs, [binding])


def test_hard_note_cannot_be_deleted_from_reference(inputs) -> None:
    notes = inputs[0]["songs"][0]["notes"]
    with pytest.raises(ValueError, match="contiguous"):
        note_sequence(notes, [notes[0]["note_id"], notes[2]["note_id"]])


def test_cropped_edge_note_retained_and_flagged(inputs) -> None:
    result = propose_sequence(inputs[0]["songs"][0], {"rendered_start_sec": 1.5, "rendered_end_sec": 3.5}, 0.15)
    assert len(result["note_ids"]) == 3
    assert result["boundary_ambiguous_note_ids"] == ["shalong-n001", "shalong-n003"]


def test_outside_proposal_is_excluded_without_clamping_the_audio_window(inputs) -> None:
    result = propose_sequence(inputs[0]["songs"][0], {"rendered_start_sec": -1, "rendered_end_sec": 3}, 0.15)
    assert result["status"] == "invalid_or_collapsed_timing"
    assert not result["mapping_complete"]
    assert result["reference_ipa"] == []


def test_review_evidence_does_not_cover_other_revisions(inputs) -> None:
    reference, evidence, _ = inputs
    evidence["scope"]["items"][0]["human_vsqx_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different source"):
        reviewed_songs(evidence, reference)


def test_empty_prediction_reports_deletions(inputs) -> None:
    result = evaluate_original(build(inputs, [approval(inputs)]), {"shalong-original-01": {"phones": []}})
    assert result["official_metrics"]["deletions"] == 3
    assert result["official_metrics"]["per"] == 1


def test_manifest_tamper_rejected(inputs) -> None:
    manifest = build(inputs)
    manifest["examples"][0]["audio_role"] = "rendered"
    with pytest.raises(ValueError, match="hash"):
        evaluate_original(manifest, {"shalong-original-01": {"phones": []}})


def test_test_group_rejected_even_with_rehashed_manifest(inputs) -> None:
    manifest = build(inputs)
    manifest["examples"][0].update(song_id="king-of-karaoke", split="test")
    manifest["manifest_hash"] = sha256_object({k: v for k, v in manifest.items() if k != "manifest_hash"})
    with pytest.raises(ValueError, match="split"):
        evaluate_original(manifest, {"shalong-original-01": {"phones": []}})


def test_unknown_or_duplicate_clip_review_rejected(inputs) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build(inputs, [approval(inputs), approval(inputs)])


def test_unrecognized_old_claim_not_silently_upgraded(inputs) -> None:
    inputs[1]["scope"]["items"] = []
    row = build(inputs)["examples"][0]
    assert not row["human_original_review_established"]
    assert row["source_claims"]["phoneme_correctness"] == "user_asserted_correct"


def test_official_score_cannot_rebind_reviewed_timing_by_rehashing(inputs) -> None:
    manifest = build(inputs, [approval(inputs)])
    manifest["examples"][0]["start_sec"] = 11
    manifest["manifest_hash"] = sha256_object({k: v for k, v in manifest.items() if k != "manifest_hash"})
    with pytest.raises(ValueError, match="binding differs"):
        evaluate_original(manifest, {"shalong-original-01": {"phones": []}})

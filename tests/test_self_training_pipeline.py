from __future__ import annotations

import json
from pathlib import Path

import pytest

from any2xsampa.training.self_training import SelfTrainingError, build_self_training_batch


def _candidate(candidate_id: str, *, confidence: float = 0.95, singer: str = "s1") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "singer_id": singer,
        "song_id": "song-1",
        "phrase_id": f"phrase-{candidate_id}",
        "model_output": True,
        "phone_hypothesis": ["x", "ei"],
        "normalized_ipa": ["h", "ei"],
        "articulatory_features": [
            {"phone": "x", "manner": "fricative"},
            {"phone": "ei", "class": "diphthong"},
        ],
        "confidence": confidence,
        "model_provenance": {
            "model_id": "ctc-toy",
            "revision": "r1",
            "source": "local-checkpoint",
            "checkpoint_sha256": "a" * 64,
            "inference_config_sha256": "b" * 64,
            "input_audio_sha256": "c" * 64,
            "vocabulary_sha256": "d" * 64,
            "calibration_sha256": "e" * 64,
            "policy_version": "fixture-policy-v1",
            "generated_at_utc": "2026-08-11T00:00:00Z",
        },
        "acoustic_evidence": {"posterior_margin": 0.44},
        "asr_transcript": {"text": "喜", "model_output": True},
    }


def test_pseudo_labels_preserve_truth_boundaries_and_reasons(tmp_path: Path) -> None:
    human = _candidate("human")
    human["human_label"] = {"phone_tokens": ["n", "ei"]}
    low = _candidate("low", confidence=0.4)
    no_provenance = _candidate("no-provenance")
    del no_provenance["model_provenance"]
    manifest = build_self_training_batch(
        candidates=[_candidate("accepted"), human, low, no_provenance],
        output_root=tmp_path,
        min_confidence=0.9,
        seed=11,
    )
    assert manifest["accepted_count"] == 1
    assert manifest["rejected_count"] == 3
    reasons = {row["reason"] for row in manifest["failure_reasons"]}
    assert reasons == {"human_truth_protected", "confidence_below_threshold", "missing_model_provenance"}
    accepted = json.loads((tmp_path / "accepted_pseudo_labels.jsonl").read_text(encoding="utf-8"))
    assert accepted["label_source"] == "pseudo"
    assert accepted["human_truth_overwritten"] is False
    assert accepted["asr_transcript_is_phone_truth"] is False
    assert accepted["phone_tokens"] == ["x", "ei"]
    assert accepted["example_id"] == "accepted"
    assert accepted["group_id"] == "s1/song-1/phrase-accepted"
    assert accepted["target_phone_tokens"] == accepted["phone_tokens"]
    assert accepted["internal_ipa_tokens"] == accepted["normalized_ipa"]
    assert accepted["pseudo_label"] is True


def test_resume_is_reproducible_and_rejects_changed_candidates(tmp_path: Path) -> None:
    first = build_self_training_batch(
        candidates=[_candidate("a")], output_root=tmp_path / "first", min_confidence=0.8, seed=19
    )
    resumed = build_self_training_batch(
        candidates=[_candidate("a"), _candidate("b")],
        output_root=tmp_path / "second",
        min_confidence=0.8,
        seed=19,
        resume_manifest=tmp_path / "first" / "self_training_manifest.json",
    )
    assert resumed["accepted_count"] == 2
    assert resumed["resume_parent_sha256"] == first["manifest_sha256"]
    repeat = build_self_training_batch(
        candidates=[_candidate("b"), _candidate("a")],
        output_root=tmp_path / "repeat",
        min_confidence=0.8,
        seed=19,
    )
    assert resumed["accepted_sha256"] == repeat["accepted_sha256"]
    changed = _candidate("a")
    changed["phone_hypothesis"] = ["changed"]
    with pytest.raises(SelfTrainingError, match="resume candidate changed"):
        build_self_training_batch(
            candidates=[changed],
            output_root=tmp_path / "changed",
            min_confidence=0.8,
            seed=19,
            resume_manifest=tmp_path / "first" / "self_training_manifest.json",
        )


def test_candidate_identity_and_duplicate_ids_fail_explicitly(tmp_path: Path) -> None:
    missing = _candidate("missing")
    del missing["singer_id"]
    manifest = build_self_training_batch(candidates=[missing], output_root=tmp_path / "missing")
    assert manifest["failure_reasons"] == [{"candidate_id": "missing", "reason": "missing_identity"}]
    with pytest.raises(SelfTrainingError, match="duplicate candidate IDs"):
        build_self_training_batch(
            candidates=[_candidate("same"), _candidate("same")], output_root=tmp_path / "duplicate"
        )


def test_human_flags_empty_evidence_and_existing_outputs_fail_closed(tmp_path: Path) -> None:
    human = _candidate("human-case")
    human["label_source"] = "HuMaN"
    locked = _candidate("locked")
    locked["human_authored_annotation_evidence"] = True
    nested_human = _candidate("nested-human")
    nested_human["label_provenance"] = {"kind": "human"}
    empty = _candidate("empty")
    empty["acoustic_evidence"] = {}
    output = tmp_path / "batch"
    manifest = build_self_training_batch(candidates=[human, locked, nested_human, empty], output_root=output)
    assert {row["reason"] for row in manifest["failure_reasons"]} == {
        "human_truth_protected",
        "missing_acoustic_evidence",
    }
    assert not list(output.glob("*.tmp"))
    with pytest.raises(SelfTrainingError, match="output already exists"):
        build_self_training_batch(candidates=[_candidate("new")], output_root=output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manual_label", False),
        ("manual_training_label", {"phones": ["a"]}),
        ("manual_positive", True),
        ("manual_negative", True),
        ("manual_label_locked", "true"),
    ],
)
def test_manual_truth_aliases_are_never_promoted(
    tmp_path: Path, field: str, value: object
) -> None:
    candidate = _candidate(field)
    candidate[field] = value
    manifest = build_self_training_batch(candidates=[candidate], output_root=tmp_path / field)
    assert manifest["accepted_count"] == 0
    assert manifest["failure_reasons"] == [
        {"candidate_id": field, "reason": "human_truth_protected"}
    ]


def test_resume_rejects_candidates_removed_from_current_input(tmp_path: Path) -> None:
    build_self_training_batch(
        candidates=[_candidate("a"), _candidate("b")], output_root=tmp_path / "first"
    )
    with pytest.raises(SelfTrainingError, match="absent from the current input"):
        build_self_training_batch(
            candidates=[_candidate("a")],
            output_root=tmp_path / "second",
            resume_manifest=tmp_path / "first" / "self_training_manifest.json",
        )


def test_lineage_mutation_changes_pseudo_label_identity(tmp_path: Path) -> None:
    first = _candidate("a")
    second = _candidate("a")
    second["model_provenance"] = dict(second["model_provenance"])
    second["model_provenance"]["inference_config_sha256"] = "f" * 64
    one = build_self_training_batch(candidates=[first], output_root=tmp_path / "one")
    two = build_self_training_batch(candidates=[second], output_root=tmp_path / "two")
    assert one["accepted_sha256"] != two["accepted_sha256"]


def test_resume_rejects_tampered_pseudo_label_even_with_resigned_manifest(tmp_path: Path) -> None:
    first = build_self_training_batch(candidates=[_candidate("a")], output_root=tmp_path / "first")
    manifest_path = tmp_path / "first" / "self_training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decisions"][0]["pseudo_label"]["phone_tokens"] = ["tampered", "ei"]
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = __import__("hashlib").sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    tampered = tmp_path / "tampered-manifest.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SelfTrainingError, match="pseudo label SHA-256 mismatch"):
        build_self_training_batch(
            candidates=[_candidate("a")],
            output_root=tmp_path / "second",
            resume_manifest=tampered,
        )
    assert first["accepted_count"] == 1

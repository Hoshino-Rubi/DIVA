from __future__ import annotations

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("audit_reference_dataset", Path(__file__).resolve().parents[1] / "tools" / "audit_reference_dataset.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inputs() -> dict:
    return {
        "triplets": {"items": [{"song_id": "song", "original_wav": {"sha256": "a" * 64}, "rendered_wav": {"sha256": "b" * 64}}]},
        "mapping": {"tokens": {"a": {"phones": ["a"], "ipa": ["?"]}}, "target_inventory": ["a"]},
        "reference": {"rows": [{"song_id": "song", "v4c_tokens": ["a"], "mapping_status": "mapped"}]},
        "dataset": {"examples": [{"song_id": "song", "audio_sha256": "b" * 64, "example_id": "one"}]},
    }


def test_rendered_audio_is_not_original_audio_evidence() -> None:
    report = MODULE.build_audit(**_inputs())
    assert report["feature_audio_roles"] == {"rendered": 1}
    assert report["gates"]["original_song_evaluation"] is False
    assert report["ipa_corruption_affected_notes"] == 1
    assert report["gates"]["ipa_integrity_passed"] is False


def test_unknown_and_identical_source_hashes_are_not_guessed() -> None:
    inputs = _inputs()
    inputs["dataset"]["examples"][0]["audio_sha256"] = "c" * 64
    assert MODULE.build_audit(**inputs)["unknown_or_ambiguous_examples"] == ["one"]
    inputs["triplets"]["items"][0]["original_wav"]["sha256"] = "b" * 64
    inputs["dataset"]["examples"][0]["audio_sha256"] = "b" * 64
    assert MODULE.build_audit(**inputs)["feature_audio_roles"] == {"ambiguous": 1}


def test_known_corruption_markers_are_not_confused_with_valid_ipa() -> None:
    assert MODULE.ipa_corruption_markers("k\u5906") == ["cjk_character_in_ipa_field"]
    assert MODULE.ipa_corruption_markers("\ufffd") == ["placeholder_or_replacement_character"]
    assert MODULE.ipa_corruption_markers("t\u0361s\u02b0\u0259\u014b\u0294") == []


def test_audit_preserves_input_and_records_projection_collisions() -> None:
    inputs = _inputs()
    inputs["mapping"]["tokens"]["a2"] = {"phones": ["a"], "ipa": ["a"], "loss": "collapsed"}
    report = MODULE.build_audit(**inputs)
    assert report["projection_collisions"] == [{"phones": ["a"], "raw_tokens": ["a", "a2"]}]
    assert inputs["mapping"]["tokens"]["a"]["ipa"] == ["?"]
    assert inputs["reference"]["rows"][0]["mapping_status"] == "mapped"

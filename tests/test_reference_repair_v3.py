from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest

SPEC = spec_from_file_location("reference_repair_v3", Path(__file__).parents[1] / "tools/build_reference_repair_v3.py")
MOD = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


def _note(note_id, lyric, ipa=("a", "n"), mapping="resolved_proxy"):
    return {"note_id": note_id, "raw_lyric": lyric, "model_target_ipa": list(ipa),
            "mapping_status": mapping, "start_sec": 0.0, "end_sec": 1.0}


def _row(review_id, decision, song_id="half-moon-night-song", split="train"):
    return {"review_id": review_id, "decision": decision, "song_id": song_id, "split": split,
            "source_example_id": review_id, "start_sec": 0.0, "end_sec": 10.0,
            "original_cut_sec": None, "cut_in_preview_sec": None, "side": None,
            "source_audio_sha256": "a" * 64, "source_vsqx_sha256": "b" * 64,
            "rendered_preview_binding": {"source_crop": {"effective_start_sec": 0.0, "effective_end_sec": 10.0}}}


def _song(song_id, notes):
    return {"song_id": song_id, "notes": notes, "original_audio": {"sha256": "a" * 64},
            "human_vsqx": {"sha256": "b" * 64}}


def test_historical_misalignment_never_revived():
    notes = {"half-moon-night-song-n001": _note("half-moon-night-song-n001", "甲")}
    candidate = MOD._make_candidate(_row("C06", "misaligned"), {}, notes, {"half-moon-night-song": _song("half-moon-night-song", list(notes.values()))})
    assert candidate["candidate_kind"] == "historical_misalignment_excluded"
    assert candidate["retained_as_candidate"] is False
    assert candidate["training_eligible"] is False


def test_c20_missing_lyric_risk_is_explicit_and_not_counted():
    notes = {"half-moon-night-song-n323": _note("half-moon-night-song-n323", "說")}
    row = _row("C20", "corresponds")
    item = {"bound_existing_notes": [{"note_id": "half-moon-night-song-n323", "source_lyric": "說"}]}
    candidate = MOD._make_candidate(row, item, notes, {"half-moon-night-song": _song("half-moon-night-song", list(notes.values()))})
    assert candidate["retained_as_candidate"] is False
    assert any("C20_missing" in reason for reason in candidate["excluded_reasons"])
    assert candidate["notes"][0]["excluded_reason"] == "C20_missing_or_severely_truncated_lyric_risk"


def test_boundary_word_evidence_keeps_all_named_phones_unresolved():
    notes = {
        "like-you-n081": _note("like-you-n081", "笑", ("s", "i", "u")),
        "like-you-n082": _note("like-you-n082", "聲", ("s", "i", "n")),
        "like-you-n083": _note("like-you-n083", "更", ("k", "a", "n")),
    }
    row = _row("C05", "boundary_issue", song_id="like-you")
    row["boundary_state"] = "truncated_start"
    item = {"bound_existing_notes": [{"note_id": n, "source_lyric": n} for n in notes]}
    candidate = MOD._make_candidate(row, item, notes, {"like-you": _song("like-you", list(notes.values()))})
    assert candidate["candidate_kind"] == "boundary_repair"
    for note in candidate["notes"][:2]:
        assert all(phone["status"] == "unresolved_boundary" for phone in note["phones"])
    assert candidate["formal_phone_reference_created"] is False


def test_dev_boundary_evidence_is_never_full_dev():
    notes = {"shalong-n043": _note("shalong-n043", "量")}
    row = _row("B01", "corresponds", song_id="shalong", split="dev")
    item = {"bound_existing_notes": [{"note_id": "shalong-n043", "source_lyric": "量"}]}
    candidate = MOD._make_candidate(row, item, notes, {"shalong": _song("shalong", list(notes.values()))})
    assert candidate["candidate_kind"] == "dev_boundary_evidence"
    assert candidate["dev_full_window"] is False
    assert candidate["formal_phone_reference_created"] is False


def test_source_hash_mismatch_is_rejected():
    notes = [_note("half-moon-night-song-n001", "?")]
    songs = {"half-moon-night-song": _song("half-moon-night-song", notes)}
    row = _row("C03", "corresponds")
    row["source_audio_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source audio hash mismatch"):
        MOD._verify_source_bindings([row], songs)

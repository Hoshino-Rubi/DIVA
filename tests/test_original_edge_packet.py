from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import wave

import pytest

SPEC = spec_from_file_location("original_edge_packet_builder", Path(__file__).parents[1] / "tools/build_original_edge_review.py")
MOD = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


@pytest.fixture
def inputs():
    if not MOD.BUNDLE.exists():
        pytest.skip("frozen local audit artifacts unavailable")
    audit, bundle = MOD.read_json(MOD.AUDIT), MOD.read_json(MOD.BUNDLE)
    reference = MOD.read_json(Path(bundle["sources"]["reference"]["path"]))
    view, _ = MOD.apply_original_mapping_overlay(reference, Path(bundle["sources"]["mapping_evidence"]["path"]))
    song = next(s for s in view["songs"] if s["song_id"] == "shalong")
    return audit, bundle, song, bundle["sources"], {"bound": True, "scope": "source_vsqx_phone_sequence_only"}


def test_six_original_cuts_three_full_candidate_sequences(inputs):
    packet = MOD.build_packet(*inputs)
    assert [i["review_id"] for i in packet["items"]] == list(MOD.REVIEW_IDS)
    assert [(c["start_sec"], c["end_sec"]) for c in packet["clips"]] == [
        (51.595, 69.595), (128.9875, 146.9875), (206.38, 224.38)]
    assert [(c["candidate_note_ids"][0], c["candidate_note_ids"][-1]) for c in packet["clips"]] == [
        ("shalong-n043", "shalong-n067"), ("shalong-n150", "shalong-n180"), ("shalong-n251", "shalong-n288")]
    assert [len(c["notes"]) for c in packet["clips"]] == [25, 31, 38]
    assert all(c["split"] == "dev" for c in packet["clips"])
    assert not packet["policy"]["automatic_training"]
    assert not packet["policy"]["test_songs_opened"]
    assert packet["packet_sha256"] == MOD.sha256_object({k: v for k, v in packet.items() if k != "packet_sha256"})


def test_previous_audit_verbatim_and_internal_tan_preserved(inputs):
    packet = MOD.build_packet(*inputs)
    audits = MOD.indexed(inputs[0]["rows"])
    assert all(i["previous_audit"] == audits[i["review_id"]] for i in packet["items"])
    tan = next(n for n in packet["clips"][1]["notes"] if n["note_id"] == "shalong-n174")
    assert tan["lyric"] == "坦"
    assert tan["phones"]


def test_unknown_mapping_not_filtered_and_machine_candidates_ignored(inputs):
    audit, bundle, song, sources, approval = deepcopy(inputs)
    note = next(n for n in song["notes"] if n["note_id"] == "shalong-n160")
    note.update(mapping_status="unresolved", model_target_ipa=[])
    for row in bundle["rows"]:
        row["candidate_note_ids"] = ["made-up-model-note"]
        row["machine"] = {"suggested_note_ids": []}
    packet = MOD.build_packet(audit, bundle, song, sources, approval)
    retained = next(n for n in packet["clips"][1]["notes"] if n["note_id"] == note["note_id"])
    assert retained["mapping_status"] == "unresolved" and retained["phones"] == []
    assert "made-up-model-note" not in str(packet["clips"])


def test_rendered_clock_cannot_replace_original_clock(inputs):
    args = deepcopy(inputs)
    row = MOD.indexed(args[1]["rows"])["B01"]
    original = next(m for m in row["media"] if m["role"] == "original")
    original["source_crop"] = next(m for m in row["media"] if m["role"] == "rendered")["source_crop"]
    with pytest.raises(ValueError, match="original preview clock"):
        MOD.build_packet(*args)


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "mismatched_pair", "not_18_seconds"])
def test_reject_invalid_packet_shape(inputs, change):
    args = deepcopy(inputs)
    rows = args[1]["rows"]
    if change == "duplicate":
        rows.append(deepcopy(rows[-1]))
    elif change == "missing":
        rows[:] = [r for r in rows if r["review_id"] != "B06"]
    elif change == "extra":
        extra = deepcopy(rows[-1]); extra["review_id"] = "B07"; rows.append(extra)
    else:
        row = MOD.indexed(rows)["B02"]; audit = MOD.indexed(args[0]["rows"])["B02"]
        if change == "mismatched_pair":
            row["source_example_id"] = audit["source_example_id"] = "wrong-pair"
        else:
            row["original_cut_sec"] += 1; audit["original_cut_sec"] += 1
            crop = next(m for m in row["media"] if m["role"] == "original")["source_crop"]
            for key in ("requested_start_sec", "requested_end_sec", "effective_start_sec", "effective_end_sec"):
                crop[key] += 1
    with pytest.raises(ValueError):
        MOD.build_packet(*args)


def test_bound_sources_and_actual_existing_media(inputs):
    MOD.verify_file(MOD.AUDIT, MOD.AUDIT_SHA256)
    MOD.verify_file(MOD.BUNDLE, MOD.BUNDLE_SHA256)
    MOD.verify_file(MOD.GUI / "bundle.json", MOD.BUNDLE_SHA256)
    packet = MOD.build_packet(*inputs)
    media = MOD.verify_media(packet, MOD.GUI)
    assert len(media) == 36
    assert len({m["file"] for m in media}) == 36


def test_media_hash_mismatch_and_path_escape_fail_closed(inputs):
    packet = MOD.build_packet(*inputs)
    packet["items"][0]["media"][0]["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MOD.verify_media(packet, MOD.GUI)
    with pytest.raises(ValueError, match="within the media"):
        MOD.media_path(MOD.GUI, "media/../../not-allowed.wav")


def test_no_overwriting_old_directory():
    with pytest.raises(ValueError, match="new directory"):
        MOD.write_run(MOD.GUI)


def test_overlay_projects_authorized_compound(inputs):
    song = inputs[2]
    compound = [n for n in song["notes"] if "ua_n" in n["raw_v4c_tokens"]]
    assert compound
    assert all(n["mapping_status"] == "resolved_proxy" for n in compound)
    assert all("ua_n" not in n["unresolved_tokens"] for n in compound)

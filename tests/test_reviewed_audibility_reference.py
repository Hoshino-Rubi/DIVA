from copy import deepcopy

import pytest
from test_original_edge_review import packet, observation_payload
from any2xsampa.training.limited_data_artifacts import sha256_object
from any2xsampa.training.reviewed_audibility_reference import build_reviewed_reference, validate_reviewed_reference


def test_explicit_end_absence_is_accounted_without_cut_cause(packet):
    data = observation_payload(packet)
    data["answers"][1]["phones"][0]["edge_status"] = "absent"
    original = deepcopy(packet)
    result = build_reviewed_reference(packet, data)
    row = result["examples"][0]
    assert row["status"] == "human_confirmed"
    assert [p["phone"] for p in row["phones"]] == ["m", "a", "a", "k", "i", "o"]
    assert row["omitted_phones"][0]["phone"] == "h"
    assert row["omitted_phones"][0]["review_ids"] == ["B02"]
    assert row["missing_phone_cause"] == "not_assessed"
    assert row["source_phone_count"] == 7
    assert packet == original


def test_partial_is_kept_once_and_repeated_symbols_are_not_deduplicated(packet):
    data = observation_payload(packet)
    data["answers"][0]["phones"][0]["edge_status"] = "absent"
    data["answers"][0]["phones"][1]["edge_status"] = "partial"
    row = build_reviewed_reference(packet, data)["examples"][0]
    assert [p["phone"] for p in row["phones"][:2]] == ["a", "a"]
    assert row["partial_phone_count"] == 1
    assert row["phones"][0]["audibility"] == "partial"


@pytest.mark.parametrize("mode", ["uncertain", "unreviewed", "all_absent", "unmapped", "unapproved"])
def test_incomplete_or_untrusted_references_do_not_get_scored(packet, mode):
    data = observation_payload(packet)
    if mode == "uncertain":
        data["answers"][0]["phones"][0]["edge_status"] = "uncertain"
    elif mode == "unreviewed":
        data["answers"][0]["phones"].pop()
    elif mode == "all_absent":
        for phone in data["answers"][0]["phones"]:
            phone["edge_status"] = "absent"
    elif mode == "unmapped":
        packet["clips"][0]["notes"][1]["mapping_status"] = "unresolved"
    else:
        packet["clips"][0]["source_sequence_approval_bound"] = False
    packet["packet_sha256"] = sha256_object({k: v for k, v in packet.items() if k != "packet_sha256"})
    data["packet_sha256"] = packet["packet_sha256"]
    row = build_reviewed_reference(packet, data)["examples"][0]
    assert row["status"] == "pending" and not row["phones"]


def test_tampered_source_or_internal_omission_rejected(packet):
    result = build_reviewed_reference(packet, observation_payload(packet))
    broken = deepcopy(result)
    broken["examples"][0]["phones"].pop(3)
    with pytest.raises(ValueError, match="hash"):
        validate_reviewed_reference(broken)
    broken["manifest_hash"] = sha256_object({k: v for k, v in broken.items() if k != "manifest_hash"})
    with pytest.raises(ValueError, match="unaccounted"):
        validate_reviewed_reference(broken)

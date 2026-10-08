from copy import deepcopy
import pytest

from any2xsampa.training.limited_data_artifacts import sha256_object
from any2xsampa.training.original_edge_review import (
    assemble_references, validate_answers, validate_packet,
)
from any2xsampa.training.segment_reference import validate_segment_reference


def seal(packet):
    packet["packet_sha256"] = sha256_object({k: v for k, v in packet.items() if k != "packet_sha256"})
    return packet


@pytest.fixture
def packet():
    clips, items = [], []
    for i in range(3):
        example = f"shalong-original-{i + 1:02d}"
        notes = [{"note_id": f"shalong-n{3*i+j+1:03d}", "lyric": "字", "phones": phones,
                  "mapping_status": "resolved_proxy"} for j, phones in enumerate((["m", "a", "a"], ["k", "i"], ["h", "o"]))]
        start, end = 10.0 + i * 30, 28.0 + i * 30
        clips.append({"example_id": example, "song_id": "shalong", "split": "dev", "audio_path": "fixture.wav",
                      "audio_sha256": "a"*64, "source_vsqx_sha256": "b"*64, "reference_manifest_sha256": "c"*64,
                      "start_sec": start, "end_sec": end, "notes": notes, "candidate_note_ids": [n["note_id"] for n in notes],
                      "source_sequence_approval_bound": True})
        for side, j in (("start", 0), ("end", 2)):
            items.append({"review_id": f"B{2*i+(1 if side=='start' else 2):02d}", "example_id": example,
                          "side": side, "original_cut_sec": start if side == "start" else end,
                          "cut_in_preview_sec": 1.25, "duration_sec": 2.5, "notes": [notes[j]], "media": []})
    return seal({"schema_version": "original_edge_review_packet_v1", "clips": clips, "items": items,
                 "source_bindings": {}, "policy": {"starts_training": False}})


def payload(packet, ids=("B01", "B02")):
    answers = []
    for item in packet["items"]:
        if item["review_id"] not in ids:
            continue
        indices = [1, 2] if item["side"] == "start" else [0]
        answers.append({"review_id": item["review_id"], "status": "confirmed", "note_id": item["notes"][0]["note_id"],
                        "phones": [{"phone_index": j, "edge_status": "full"} for j in indices],
                        "statement": "中线内此字及所选音素确实可听", "audible_identity_confirmed": True})
    return {"schema_version": "original_edge_review_answers_v1", "packet_sha256": packet["packet_sha256"],
            "reviewer": "fixture only", "exported_at": "2026-09-15T00:00:00Z", "answers": answers}


def test_empty_answers_do_not_create_truth(packet):
    result = assemble_references(packet, payload(packet, ()))
    assert result["human_confirmed_count"] == 0
    assert all(r["status"] == "pending" and r["phones"] == [] for r in result["examples"])


def test_one_pair_keeps_duplicate_symbols_internal_phones_and_original_times(packet):
    original = deepcopy(packet)
    result = assemble_references(packet, payload(packet))
    row = result["examples"][0]
    assert result["human_confirmed_count"] == 1
    assert row["start_sec"] == 10 and row["end_sec"] == 28
    assert [p["phone"] for p in row["phones"]] == ["a", "a", "k", "i", "h"]
    assert row["boundary_state"] == "truncated_both"
    assert not row["training_eligible"]
    assert packet == original
    validate_segment_reference(result)


def test_single_answer_cannot_assemble_clip(packet):
    result = assemble_references(packet, payload(packet, ("B01",)))
    assert result["human_confirmed_count"] == 0
    assert "end_edge_not_confirmed" in result["examples"][0]["blocking_reasons"]


@pytest.mark.parametrize("status", ["uncertain", "wrong_correspondence"])
def test_uncertain_and_rejected_never_create_phone_rows(packet, status):
    data = payload(packet)
    data["answers"][0].update(status=status, note_id=None, phones=[], audible_identity_confirmed=False)
    result = assemble_references(packet, data)
    assert result["human_confirmed_count"] == 0
    assert result["examples"][0]["status"] == ("pending" if status == "uncertain" else "rejected")


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(packet_sha256="0"*64),
    lambda p: p["answers"].append(deepcopy(p["answers"][0])),
    lambda p: p["answers"][0].update(audible_identity_confirmed=False),
    lambda p: p["answers"][0].update(note_id="shalong-n999"),
    lambda p: p["answers"][0].update(phones=[{"phone_index": 0, "edge_status": "full"}, {"phone_index": 2, "edge_status": "full"}]),
    lambda p: p["answers"][0].update(phones=[{"phone_index": 1, "edge_status": "full"}, {"phone_index": 1, "edge_status": "full"}]),
    lambda p: p["answers"][0]["phones"][0].update(phone_index=True),
    lambda p: p["answers"][0]["phones"][-1].update(edge_status="truncated"),
])
def test_invalid_explicit_answers_are_rejected_atomically(packet, mutate):
    data = payload(packet)
    mutate(data)
    with pytest.raises(ValueError):
        validate_answers(packet, data)


def test_unresolved_internal_note_excludes_whole_window(packet):
    packet["clips"][0]["notes"][1].update(mapping_status="unresolved", phones=[])
    seal(packet)
    result = assemble_references(packet, payload(packet))
    assert result["human_confirmed_count"] == 0
    assert "unresolved mapping" in result["examples"][0]["blocking_reasons"][0]
    assert result["examples"][0]["phones"] == []


def test_sequence_approval_is_required_not_guessed(packet):
    packet["clips"][0]["source_sequence_approval_bound"] = False
    seal(packet)
    result = assemble_references(packet, payload(packet))
    assert result["human_confirmed_count"] == 0


@pytest.mark.parametrize("change", ["offset", "split", "hash"])
def test_packet_rejects_wrong_audio_time_split_or_self_hash(packet, change):
    if change == "offset":
        packet["items"][0]["original_cut_sec"] += 0.2
        seal(packet)
    elif change == "split":
        packet["clips"][0]["split"] = "test"
        seal(packet)
    else:
        packet["packet_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_packet(packet)


def test_full_index_list_with_cut_phone_still_marks_truncation(packet):
    data = payload(packet)
    data["answers"][0]["phones"] = [{"phone_index": i, "edge_status": "truncated" if i == 0 else "full"} for i in range(3)]
    row = assemble_references(packet, data)["examples"][0]
    assert row["boundary_state"] == "truncated_both"
    assert row["phones"][0]["edge_status"] == "truncated"


def test_gui_export_roundtrips_into_python_importer(packet):
    import json
    from pathlib import Path
    import subprocess
    state = Path(__file__).resolve().parents[1] / "tools/original_edge_review_gui/state.js"
    program = r"""
const fs = require('node:fs');
const state = require(process.argv[1]);
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const output = state.exportAnswers(input.packet, input.answers.answers, input.answers.reviewer, input.answers.exported_at);
process.stdout.write(JSON.stringify(output));
"""
    process = subprocess.run(["node", "-e", program, str(state)], input=json.dumps({"packet": packet, "answers": payload(packet)}),
                             text=True, encoding="utf-8", capture_output=True, timeout=20, check=True)
    exported = json.loads(process.stdout)
    result = assemble_references(packet, exported)
    assert result["human_confirmed_count"] == 1
    assert [p["phone"] for p in result["examples"][0]["phones"]] == ["a", "a", "k", "i", "h"]


def observation_payload(packet):
    data = payload(packet)
    data["schema_version"] = "original_edge_review_answers_v2"
    for answer in data["answers"]:
        item = next(i for i in packet["items"] if i["review_id"] == answer["review_id"])
        answer["observation_schema"] = "phone_audibility_v2"
        answer["phones"] = [{"phone_index": i, "edge_status": "full"} for i in range(len(item["notes"][0]["phones"]))]
    return data


@pytest.mark.parametrize("state", ["absent", "partial", "uncertain"])
def test_v2_observations_are_accepted_preserved_and_not_silently_used_as_phone_truth(packet, state):
    data = observation_payload(packet)
    data["answers"][1]["phones"][0]["edge_status"] = state
    original = deepcopy(data)
    assert validate_answers(packet, data)["B02"]["phones"][0]["edge_status"] == state
    result = assemble_references(packet, data)
    row = result["examples"][0]
    assert row["status"] == "pending" and row["phones"] == []
    assert row["blocking_reasons"] == ["audibility_observations_require_reference_reconciliation"]
    assert row["edge_evidence"]["end"]["phones"][0]["edge_status"] == state
    assert data == original


def test_v2_unselected_is_not_inferred_absent(packet):
    data = observation_payload(packet)
    data["answers"][1]["phones"] = data["answers"][1]["phones"][1:]
    row = assemble_references(packet, data)["examples"][0]
    assert row["status"] == "pending"
    assert row["edge_evidence"]["end"]["phones"] == [{"phone_index": 1, "edge_status": "full"}]


def test_v2_all_full_can_use_existing_reference_contract(packet):
    row = assemble_references(packet, observation_payload(packet))["examples"][0]
    assert row["status"] == "human_confirmed"
    assert [p["phone"] for p in row["phones"]] == ["m", "a", "a", "k", "i", "h", "o"]
    assert row["boundary_state"] == "full"


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(schema_version="original_edge_review_answers_v1"),
    lambda p: p["answers"][0]["phones"][0].update(edge_status="truncated"),
    lambda p: p["answers"][0]["phones"][0].update(phone_index=True),
    lambda p: p["answers"][0]["phones"].append(deepcopy(p["answers"][0]["phones"][0])),
    lambda p: p["answers"][0].update(observation_schema="unsupported"),
])
def test_v2_still_enforces_identity_and_version_contract(packet, mutate):
    data = observation_payload(packet)
    mutate(data)
    with pytest.raises(ValueError):
        validate_answers(packet, data)


def test_js_absent_full_end_observation_replays_in_python(packet):
    import json
    from pathlib import Path
    import subprocess
    state = Path(__file__).resolve().parents[1] / "tools/original_edge_review_gui/state.js"
    data = observation_payload(packet)
    data["answers"][1]["phones"][0]["edge_status"] = "absent"
    program = r"""
const fs = require('fs'), S = require(process.argv[1]), p = JSON.parse(fs.readFileSync(0, 'utf8'));
const answers = p.data.answers.map(a => S.commitDraft(p.packet, p.packet.items.find(i => i.review_id === a.review_id), a, p.data.reviewer));
process.stdout.write(JSON.stringify(S.exportAnswers(p.packet, answers, p.data.reviewer)));
"""
    output = subprocess.run(["node", "-e", program, str(state)],
                            input=json.dumps({"packet": packet, "data": data}), encoding="utf-8",
                            capture_output=True, check=True, timeout=20)
    result = assemble_references(packet, json.loads(output.stdout))
    row = result["examples"][0]
    assert row["edge_evidence"]["end"]["phones"][0]["edge_status"] == "absent"
    assert row["status"] == "pending"

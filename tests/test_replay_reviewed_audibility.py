from copy import deepcopy
import pytest
from test_original_edge_review import packet, observation_payload
from any2xsampa.training.reviewed_audibility_reference import build_reviewed_reference
from tools.replay_reviewed_audibility import validate_entry, score


def entry(row):
    return {**{k: row[k] for k in ("example_id", "song_id", "split", "audio_sha256", "start_sec", "end_sec")},
            "audio_role": "original", "test_outputs_sealed": False}


@pytest.mark.parametrize("field,value", [
    ("song_id", "king-of-karaoke"), ("split", "test"), ("audio_role", "rendered"),
    ("audio_sha256", "0"*64), ("start_sec", 10.5), ("test_outputs_sealed", True)])
def test_replay_refuses_wrong_role_source_clock_or_split(packet, field, value):
    row = build_reviewed_reference(packet, observation_payload(packet))["examples"][0]
    e = entry(row); e[field] = value
    with pytest.raises(ValueError):
        validate_entry(row, e, "historical_raw")


def test_paired_clock_is_bound_to_same_original_window(packet):
    row = build_reviewed_reference(packet, observation_payload(packet))["examples"][0]
    pair = {"context": {"example_id": row["example_id"], "audio_sha256": row["audio_sha256"],
                       "old_start_sec": row["start_sec"], "old_end_sec": row["end_sec"]},
            "audio": {"mixed": {"sha256": "d"*64}}}
    e = {**entry(row), "example_id": row["example_id"]+"-paired-mixed",
         "audio_role": "original_resampled_mix", "audio_sha256": "d"*64, "start_sec": 0, "end_sec": 18}
    validate_entry(row, e, "mixed", pair)
    pair["context"]["old_start_sec"] += 1
    with pytest.raises(ValueError):
        validate_entry(row, e, "mixed", pair)


def test_scoring_uses_identical_denominator_and_preserves_empty_prediction(packet):
    reference = build_reviewed_reference(packet, observation_payload(packet))
    rows = [r for r in reference["examples"] if r["status"] == "human_confirmed"]
    predictions = {r["example_id"]: {c: {"baseline": [], "seed43": [p["phone"] for p in r["phones"]]}
                                   for c in ("historical_raw", "mixed", "vocal")} for r in rows}
    result = score(reference, predictions)
    for systems in result["metrics"].values():
        assert systems["baseline"]["per"] == 1
        assert systems["seed43"]["per"] == 0
        assert systems["baseline"]["reference_phones"] == systems["seed43"]["reference_phones"] == 7
    bad = deepcopy(predictions); bad["extra-window"] = {}
    with pytest.raises(ValueError):
        score(reference, bad)

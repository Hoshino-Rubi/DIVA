"""Review whole-song content/order independently of phone boundary timing."""
from __future__ import annotations
from typing import Any


def validate_answers(packet: dict, answers: dict, *, packet_sha256: str) -> dict[str, Any]:
    if (packet.get("schema_version") != "whole_sequence_review_packet_v1"
            or answers.get("schema_version") != "whole_sequence_review_answers_v1"
            or answers.get("packet_sha256") != packet_sha256):
        raise ValueError("whole-sequence answer schema/packet hash mismatch")
    items = {r["review_id"]: r for r in packet["items"]}
    rows = answers.get("answers")
    if not isinstance(rows, list) or len(rows) != len(items) or {r.get("review_id") for r in rows} != set(items):
        raise ValueError("whole-sequence answers must preserve every unique review item")
    accepted, pending, conflicts = [], [], []
    for row in rows:
        status = row.get("status")
        if status not in {"sequence_confirmed", "sequence_differs", "pending"}:
            raise ValueError("unsupported whole-sequence decision")
        if not isinstance(row.get("statement", ""), str):
            raise ValueError("review statement must be text")
        if row.get("phone_timing_confirmed") is not False:
            raise ValueError("whole-sequence review does not establish phone timing")
        item = items[row["review_id"]]
        if row.get("source_bindings") != item["source_bindings"]:
            raise ValueError("whole-sequence source versions changed")
        decision = {"review_id":row["review_id"], "song_id":item["song_id"], "status":status,
            "statement":row.get("statement", ""), "source_bindings":item["source_bindings"],
            "approval_scope":"complete source phone/lyric sequence and repeated-section order/count only",
            "phone_timing_confirmed":False, "mapping_extended":False, "original_answer":row}
        if status == "sequence_confirmed":
            if row.get("explicit_sequence_scope_confirmed") is not True:
                raise ValueError("sequence approval requires explicit scope affirmation")
            accepted.append(decision)
        elif status == "sequence_differs":
            conflicts.append(decision)
        else:
            pending.append(decision)
    return {"accepted":accepted, "pending":pending, "conflicts":conflicts,
        "starts_training":False, "reference_times_modified":False, "default_replacement":False}

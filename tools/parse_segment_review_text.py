"""Convert one-line reviewer notes into the v2 response JSON.

This parser intentionally treats free-form text as evidence only.  A line such
as ``B01：切点在“住”的中间`` remains pending; it is never promoted to an
audio/phone label.  Structured fields may be supplied when the reviewer wants
to confirm a row without using the HTML form.

Accepted form::

    B01：status=pending；reviewer=Leo；说明=切点在住字中间
    C03:status=human_confirmed;boundary=full;notes=n001,n002;phones=n001|0|m|full,n002|0|a|full;说明=...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new  # noqa: E402


_FIELD_ALIASES = {"status": "status", "reviewer": "reviewer", "说明": "review_statement", "原话": "review_statement", "boundary": "boundary_state", "边界": "boundary_state", "notes": "note_ids", "音符": "note_ids", "phones": "phones", "音素": "phones"}


def _split_lines(text: str) -> list[tuple[str, str]]:
    result = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*)\s*[:：]\s*(.*)$", line)
        if not match:
            raise ValueError(f"invalid review line (expected ID: text): {raw}")
        result.append((match.group(1), match.group(2).strip()))
    return result


def _parse_value(key: str, value: str):
    if key == "note_ids":
        return [part.strip() for part in re.split(r"[,，]", value) if part.strip()]
    if key == "phones":
        phones = []
        for part in re.split(r"[,，]", value):
            part = part.strip()
            if not part:
                continue
            fields = [field.strip() for field in part.split("|")]
            if len(fields) not in {3, 4} or not fields[1].isdigit() or not fields[0] or not fields[2]:
                raise ValueError(f"invalid phone field: {part}")
            phones.append({"note_id": fields[0], "phone_index": int(fields[1]), "phone": fields[2], "audible": True, "edge_status": fields[3] if len(fields) == 4 else "full"})
        return phones
    return value


def parse_text(packet: dict, text: str) -> dict:
    items = {item["review_id"]: item for item in packet.get("items", [])}
    lines = _split_lines(text)
    values = {review_id: {"review_id": review_id, "status": "pending", "reviewer": None, "review_statement": None, "boundary_state": None, "note_ids": [], "phones": []} for review_id in items}
    for review_id, body in lines:
        if review_id not in values:
            raise ValueError(f"unknown review ID: {review_id}")
        # Preserve the complete unparsed answer regardless of structured fields.
        values[review_id]["review_statement"] = body
        for part in re.split(r"[;；]", body):
            part = part.strip()
            if "=" not in part:
                continue
            raw_key, raw_value = [value.strip() for value in part.split("=", 1)]
            key = _FIELD_ALIASES.get(raw_key)
            if key is None:
                continue
            values[review_id][key] = _parse_value(key, raw_value)
    return {"schema_version": "original_segment_review_responses_v2", "packet_sha256": packet["packet_sha256"], "responses": list(values.values()), "automatic_annotation_updates_allowed": False, "source_text_preserved": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ID-prefixed text answers into v2 response JSON without inferring labels.")
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    packet = read_json(args.packet)
    if packet.get("test_songs_opened") is True:
        raise ValueError("test review items cannot be imported")
    responses = parse_text(packet, args.text.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    digest = write_json_new(output, responses)
    write_json_new(output.with_name(output.stem + "-provenance.json"), {"schema_version": "original_segment_text_response_provenance_v1", "packet_sha256": packet["packet_sha256"], "packet_file_sha256": sha256_file(args.packet), "text_file_sha256": sha256_file(args.text), "response_sha256": digest, "automatic_inference": False})
    print(f"responses={len(responses['responses'])}; response_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

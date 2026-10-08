"""Build a read-only audit bundle without promoting machine or text-only labels."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

from any2xsampa.training.limited_data_artifacts import (
    read_json, sha256_file, sha256_object, verify_file, write_json_new,
)
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay

SCHEMA = "alignment_audit_bundle_v1"
SONGS = {
    "freedom-flower": "自由花", "queen-road-east": "皇后大道東",
    "half-moon-night-song": "月半小夜曲", "promise": "约定",
    "like-you": "喜欢你", "shalong": "沙龙",
}
SPLITS = {key: ("dev" if key == "shalong" else "train") for key in SONGS}


def checked_object(path: Path, field: str) -> dict[str, Any]:
    value = read_json(path)
    if value.get(field) != sha256_object({k: v for k, v in value.items() if k != field}):
        raise ValueError(f"{field} mismatch: {path}")
    return value


def bind_responses(packet: dict, path: Path) -> dict[str, dict]:
    value = read_json(path)
    if value.get("schema_version") != "original_segment_review_responses_v2":
        raise ValueError("unsupported response schema")
    if value.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("response packet hash mismatch")
    rows = value.get("responses", [])
    answers = {row["review_id"]: row for row in rows}
    if len(answers) != len(rows) or set(answers) != {r["review_id"] for r in packet["items"]}:
        raise ValueError("response IDs do not match packet")
    if any(a.get("status") not in {"pending", "rejected", "human_confirmed"} for a in rows):
        raise ValueError("unknown human response status")
    return answers


def audit_state(human: dict, machine: dict) -> dict[str, Any]:
    """Human rejection takes precedence; old text-only approval is not phone truth."""
    rejected = human.get("status") == "rejected"
    conflict = rejected and machine.get("status") == "machine_aligned"
    return {
        "status": "human_rejected" if rejected else "needs_audit",
        "human_machine_conflict": conflict,
        "human_rejected": rejected,
        "machine_flagged": machine.get("status") == "needs_review",
        "text_approval_only": human.get("status") == "human_confirmed" and not human.get("phones"),
        "formal_reference_validated": False,
        "training_eligible": False,
    }


def note_view(note_id: str, song: dict) -> dict:
    note = next((n for n in song["notes"] if n["note_id"] == note_id), None)
    if note is None:
        raise ValueError(f"unknown note: {note_id}")
    return {"note_id": note_id, "lyric": note.get("raw_lyric", ""),
            "engine_tokens": note.get("raw_v4c_tokens", []),
            "phones": note.get("model_target_ipa", []),
            "mapping_status": note.get("mapping_status"),
            "score_start_sec": note.get("start_sec"), "score_end_sec": note.get("end_sec")}


def copy_previews(packet_dir: Path, item: dict, output: Path) -> list[dict]:
    result = []
    for preview in item.get("previews", []):
        row = {"role": preview["role"], "duration_sec": preview.get("duration_sec"),
               "source_path": preview.get("source_path"), "source_sha256": preview.get("source_sha256"),
               "source_crop": preview.get("source_crop"), "preview_gain": preview.get("preview_gain")}
        for field, hash_key in (("audio_file", "audio_sha256"), ("waveform_file", "waveform_sha256")):
            name = preview[field]
            if not isinstance(name, str) or Path(name).name != name or ":" in name:
                raise ValueError("invalid media filename")
            source = (packet_dir / "media" / name).resolve()
            if not source.is_relative_to((packet_dir / "media").resolve()):
                raise ValueError("media escapes packet directory")
            verify_file(source, preview[hash_key])
            target = output / "media" / f"{item['review_id']}-{name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(source, target)
            verify_file(target, preview[hash_key])
            row[field] = target.relative_to(output).as_posix()
            row[hash_key] = preview[hash_key]
        result.append(row)
    return result


def audit_row(item: dict, packet: dict, human: dict, machine: dict, song: dict, media: list[dict]) -> dict:
    sid = song["song_id"]
    if sid not in SPLITS or song["split"] != SPLITS[sid]:
        raise ValueError("sealed or mismatched song/split")
    kind = "boundary" if "original_cut_sec" in item else "training"
    if kind == "training" and (item.get("song_id") != sid or item.get("split") != "train"):
        raise ValueError("training item song/split mismatch")
    refs = item.get("candidate_references", [])
    ids = list(dict.fromkeys(n for ref in refs for n in ref["note_ids"])) if refs else item["candidate_note_ids"]
    if kind == "boundary":
        ids = list(dict.fromkeys([n["note_id"] for n in item["nearby_source_notes"]] + ids))
    positions = {n["note_id"]: i for i, n in enumerate(song["notes"])}
    ids.sort(key=lambda n: positions.get(n, -1))
    state = audit_state(human, machine)
    warnings = []
    if state["human_machine_conflict"]:
        warnings.append("你已判定错位，但旧机器报告标为通过；以人工意见为先，不能进入训练。")
    if state["text_approval_only"]:
        warnings.append("已保留你的文字通过意见；尚无完整边缘音素参考，不等于正式音素评测已通过。")
    if kind == "boundary":
        warnings.append("这是切点附近预览，不是整段开发音频；只审计中线处的开头或结尾。")
    if len({tuple(ref["note_ids"]) for ref in refs}) > 1:
        warnings.append("两套历史对应候选的首尾音符范围不同，不能自动合并为正确答案。")
    statement = human.get("review_statement") or ""
    if "????" in statement:
        warnings.append("历史补充文件中的说明文字已损坏；原始状态保留。聊天补充原话：3待定均为错位。")
    notes = [note_view(n, song) for n in ids]
    if any(n['note_id'] in machine.get('reason', '') and n['mapping_status'] == 'resolved_proxy' for n in notes):
        warnings.append('此映射异常来自旧报告；本页按已有补充规则可解析该项，但没有重新运行对齐或批准训练。')
    return {
        "review_id": item["review_id"], "kind": kind, "song_id": sid, "song_name": SONGS[sid],
        "split": song["split"], "packet_sha256": packet["packet_sha256"],
        "candidate_id": item.get("candidate_id"), "source_example_id": item.get("example_id"),
        "start_sec": item.get("start_sec"), "end_sec": item.get("end_sec"),
        "original_cut_sec": item.get("original_cut_sec"), "cut_in_preview_sec": item.get("cut_in_preview_sec"),
        "side": item.get("side"), "human": human, "machine": machine, "state": state,
        "notes": notes, "nearby_note_ids": [n["note_id"] for n in item.get("nearby_source_notes", [])],
        "candidate_references": [{k: r.get(k) for k in ("factor", "note_ids", "rendered_start_sec", "rendered_end_sec")} for r in refs],
        "candidate_note_ids": item.get("candidate_note_ids", []),
        "media": media, "warnings": warnings,
        "source_audio_sha256": song["original_audio"]["sha256"],
        "source_vsqx_sha256": song["human_vsqx"]["sha256"],
        "mapping_resolved_notes": sum(n["mapping_status"] == "resolved_proxy" for n in notes),
    }


def build_bundle(review_root: Path, machine_run: Path, training_responses: Path,
                 shalong_responses: Path, output: Path) -> dict:
    paths = {
        "training_packet": review_root / "training-candidate-review-v8/review-packet.json",
        "shalong_packet": review_root / "shalong-review-v3/review-packet.json",
        "proposals": review_root / "training-candidate-review-v8/candidate-proposals.json",
        "machine": machine_run / "machine-aligned-reference.json",
        "machine_edges": machine_run / "shalong-machine-edge-suggestions.json",
        "training_responses": training_responses, "shalong_responses": shalong_responses,
    }
    packets = {key: checked_object(paths[key], "packet_sha256") for key in ("training_packet", "shalong_packet")}
    for packet in packets.values():
        if packet.get("test_songs_opened") is not False:
            raise ValueError("sealed test review packet")
    if {i["review_id"] for i in packets["training_packet"]["items"]} != {f"C{i:02d}" for i in range(1, 21)}:
        raise ValueError("expected the existing C01-C20 only")
    if {i["review_id"] for i in packets["shalong_packet"]["items"]} != {f"B{i:02d}" for i in range(1, 7)}:
        raise ValueError("expected the existing B01-B06 only")
    proposals = read_json(paths["proposals"])
    binding = proposals["source_bindings"]
    for key in ("reference", "mapping_evidence"):
        paths[key] = Path(binding[key]["path"])
        verify_file(paths[key], binding[key]["sha256"])
    reference, overlay = apply_original_mapping_overlay(read_json(paths["reference"]), paths["mapping_evidence"])
    songs = {s["song_id"]: s for s in reference["songs"] if s["split"] != "test"}
    machine = checked_object(paths["machine"], "manifest_hash")
    edges = checked_object(paths["machine_edges"], "manifest_hash")
    for key, source in (("reference", "reference"), ("proposals", "candidate_proposals")):
        verify_file(paths[key], machine["source_bindings"][source]["sha256"])
    verify_file(paths["shalong_packet"], edges["source_bindings"]["shalong_packet"]["sha256"])
    verify_file(paths["reference"], edges["source_bindings"]["reference"]["sha256"])
    train_human = bind_responses(packets["training_packet"], training_responses)
    dev_human = bind_responses(packets["shalong_packet"], shalong_responses)
    machine_map = {row.get("candidate_id", row.get("example_id")): row for row in machine["examples"] + machine["errors"]}
    edge_map = {row["review_id"]: row for row in edges["items"] + edges["errors"]}
    rows = []
    for kind, humans, machines in (("training_packet", train_human, machine_map), ("shalong_packet", dev_human, edge_map)):
        packet = packets[kind]
        for item in packet["items"]:
            sid = item.get("song_id", "shalong")
            if sid not in songs or sid not in SPLITS:
                raise ValueError("unknown or sealed song in packet")
            for preview in item["previews"]:
                if preview.get("role") == "original" and preview.get("source_sha256") != songs[sid]["original_audio"]["sha256"]:
                    raise ValueError("preview original source hash mismatch")
            key = item.get("candidate_id", item["review_id"])
            media = copy_previews(paths[kind].parent, item, output)
            rows.append(audit_row(item, packet, humans[item["review_id"]], machines.get(key, {}), songs[sid], media))
    sources = {k: {"path": str(p.resolve()), "sha256": sha256_file(p)} for k, p in paths.items()}
    result = {
        "schema_version": SCHEMA, "built_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows, "sources": sources, "mapping_overlay": overlay,
        "summary": {"items": len(rows), "training": 20, "boundaries": 6,
                    "human_rejected": sum(r["state"]["human_rejected"] for r in rows),
                    "human_machine_conflicts": sum(r["state"]["human_machine_conflict"] for r in rows),
                    "training_text_approvals": sum(a["status"] == "human_confirmed" for a in train_human.values()),
                    "machine_reported": machine["counts"], "new_formal_references": 0},
        "policy": {"source_read_only": True, "test_songs_omitted": True,
                   "automatic_training": False, "audit_export_is_not_phone_truth": True,
                   "machine_score_is_not_accuracy": True},
    }
    result["bundle_id"] = sha256_object({k: v for k, v in result.items() if k != "built_at"})
    return result


def write_page(bundle: dict, output: Path, assets: Path) -> None:
    """Inline code/data so file:// needs no server, CDN, or fetch."""
    import json
    template = (assets / "index.html").read_text(encoding="utf-8")
    payload = json.dumps(bundle, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    document = template.replace("__AUDIT_CSS__", (assets / "style.css").read_text(encoding="utf-8"))
    document = document.replace("__AUDIT_CORE__", (assets / "audit_core.js").read_text(encoding="utf-8"))
    document = document.replace("__AUDIT_APP__", (assets / "app.js").read_text(encoding="utf-8"))
    document = document.replace("__AUDIT_DATA__", payload)
    with (output / "index.html").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
    write_json_new(output / "bundle.json", bundle)

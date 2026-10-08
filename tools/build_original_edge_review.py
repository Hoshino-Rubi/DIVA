"""Freeze six existing original-audio cut previews without creating phone truth."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
import wave
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay, OVERLAY_PATH

AUDIT = ROOT / ".state/runs/alignment-audit-intake-20260915T132143Z/accepted/effective-audit.json"
BUNDLE = AUDIT.parent.parent / "raw/bundle.json"
GUI = ROOT / ".state/runs/alignment-audit-gui-20260915T084940Z"
AUDIT_SHA256 = "c6966ecdffd1d64d13202b5136d3f50ff8557c997aea4c0aecf33822f375dfa1"
BUNDLE_SHA256 = "7eede459410921fbb262dafc3319f5f4f35b1026a3daa7deb6ea3c220d167c9c"
REVIEW_IDS = tuple(f"B{i:02d}" for i in range(1, 7))


def binding(path: Path, expected: str | None = None) -> dict[str, str]:
    digest = verify_file(path, expected) if expected else sha256_file(path)
    return {"path": str(path.resolve()), "sha256": digest}


def indexed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {row["review_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate review_id")
    return result


def note_view(note: Mapping[str, Any]) -> dict[str, Any]:
    return {"note_id": note["note_id"], "lyric": note["raw_lyric"],
            "phones": list(note.get("model_target_ipa", [])),
            "mapping_status": note.get("mapping_status", "unresolved")}


def nearby_notes(row: Mapping[str, Any], song: Mapping[str, Any]) -> list[dict[str, Any]]:
    wanted = set(row["nearby_note_ids"])
    result = [note_view(note) for note in song["notes"] if note["note_id"] in wanted]
    if not wanted or len(result) != len(wanted):
        raise ValueError("nearby notes missing from full reference")
    return result


def continuous_notes(start: Mapping[str, Any], end: Mapping[str, Any], song: Mapping[str, Any]) -> list[dict[str, Any]]:
    # The frozen score-nearby ranges are navigation candidates, never a model/PER selection.
    wanted = {n["note_id"] for row in (start, end) for n in nearby_notes(row, song)}
    indices = [i for i, note in enumerate(song["notes"]) if note["note_id"] in wanted]
    return [note_view(note) for note in song["notes"][min(indices):max(indices) + 1]]


def validate_edge(row: Mapping[str, Any], audit: Mapping[str, Any], song: Mapping[str, Any]) -> None:
    for key in ("review_id", "source_example_id", "side", "original_cut_sec", "cut_in_preview_sec",
                "source_audio_sha256", "source_vsqx_sha256", "packet_sha256"):
        if row[key] != audit[key]:
            raise ValueError(f"audit/GUI binding mismatch: {key}")
    if row["song_id"] != "shalong" or row["split"] != "dev" or audit["decision"] != "corresponds":
        raise ValueError("only accepted shalong dev correspondence may enter this packet")
    if row["source_audio_sha256"] != song["original_audio"]["sha256"] or row["source_vsqx_sha256"] != song["human_vsqx"]["sha256"]:
        raise ValueError("source audio/VSQX binding mismatch")
    originals = [m for m in row["media"] if m["role"] == "original"]
    if len(originals) != 1 or row["cut_in_preview_sec"] != 1.25:
        raise ValueError("exactly one original preview and a 1.25s cut are required")
    media = originals[0]
    if media["source_sha256"] != song["original_audio"]["sha256"]:
        raise ValueError("original preview source mismatch")
    crop = media["source_crop"]
    for key, value in (("requested_start_sec", row["original_cut_sec"] - 1.25),
                       ("requested_end_sec", row["original_cut_sec"] + 1.25),
                       ("effective_start_sec", row["original_cut_sec"] - 1.25),
                       ("effective_end_sec", row["original_cut_sec"] + 1.25)):
        if not math.isclose(crop[key], value, abs_tol=1 / song["original_audio"]["sample_rate"], rel_tol=0):
            raise ValueError("original preview clock mismatch")
    if not math.isclose(media["duration_sec"], 2.5, abs_tol=1e-8):
        raise ValueError("original preview must be 2.5 seconds")


def build_packet(audit: dict[str, Any], bundle: dict[str, Any], song: dict[str, Any],
                 sources: dict[str, Any], approval: dict[str, Any]) -> dict[str, Any]:
    audits, rows = indexed(audit["rows"]), indexed(bundle["rows"])
    if {rid for rid in rows if rid.startswith("B")} != set(REVIEW_IDS):
        raise ValueError("exactly B01-B06 are required")
    if {rid for rid in audits if rid.startswith("B")} != set(REVIEW_IDS):
        raise ValueError("audit must contain exactly B01-B06")
    if audit["bundle_id"] != bundle["bundle_id"]:
        raise ValueError("audit does not bind the frozen GUI bundle")
    items, clips = [], []
    for rid in REVIEW_IDS:
        row = rows[rid]
        validate_edge(row, audits[rid], song)
        items.append({"review_id": rid, "example_id": row["source_example_id"],
                      "side": row["side"], "original_cut_sec": row["original_cut_sec"],
                      "cut_in_preview_sec": 1.25, "duration_sec": 2.5,
                      "notes": nearby_notes(row, song), "media": deepcopy(row["media"]),
                      "previous_audit": deepcopy(audits[rid]),
                      "question": "此前对应审计已保留，无需重答。只听原唱预览1.25秒中线：选择切点处的字及片段内仍可听见的音素；不清楚可留待复核。开头看中线右侧，结尾看中线左侧。"})
    for first, last in zip(REVIEW_IDS[::2], REVIEW_IDS[1::2]):
        start, end = rows[first], rows[last]
        if start["side"] != "start" or end["side"] != "end" or start["source_example_id"] != end["source_example_id"]:
            raise ValueError("clip requires matched start/end edges")
        if not math.isclose(end["original_cut_sec"] - start["original_cut_sec"], 18, abs_tol=1e-8):
            raise ValueError("paired original cuts must span 18 seconds")
        notes = continuous_notes(start, end, song)
        clips.append({"example_id": start["source_example_id"], "song_id": "shalong", "split": "dev",
                      "audio_path": song["original_audio"]["path"], "audio_sha256": song["original_audio"]["sha256"],
                      "source_vsqx_sha256": song["human_vsqx"]["sha256"],
                      "reference_manifest_sha256": sources["reference"]["sha256"],
                      "start_sec": start["original_cut_sec"], "end_sec": end["original_cut_sec"],
                      "candidate_note_ids": [n["note_id"] for n in notes], "notes": notes,
                      "source_sequence_approval_bound": approval["bound"],
                      "source_sequence_approval_binding": "triplets",
                      "source_sequence_approval": deepcopy(approval)})
    if len({c["example_id"] for c in clips}) != 3:
        raise ValueError("three distinct clips are required")
    result = {"schema_version": "original_edge_review_packet_v1", "source_bindings": sources,
              "policy": {"automatic_training": False, "test_songs_opened": False, "training_c_items_paused": True,
                         "misaligned_c_items_restored": False, "formal_phone_reference_created": False,
                         "machine_or_per_note_selection": False, "prior_correspondence_reasked": False,
                         "source_sequence_scope": "source_vsqx_phone_sequence_only",
                         "original_audio_clock_inferred_from_rendered": False,
                         "internal_notes": "沿用已校准连续音符；不是重新认可全曲时轴；必须由显式首尾答案界定保留范围。"},
              "clips": clips, "items": items}
    result["packet_sha256"] = sha256_object(result)
    return result


def load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    sources = {"audit": binding(AUDIT, AUDIT_SHA256),
               "bundle": binding(BUNDLE, BUNDLE_SHA256),
               "gui_bundle": binding(GUI / "bundle.json", BUNDLE_SHA256)}
    audit, bundle = read_json(AUDIT), read_json(BUNDLE)
    for name, item in bundle["sources"].items():
        sources[name] = binding(Path(item["path"]), item["sha256"])
    sources["mapping_overlay"] = binding(OVERLAY_PATH, bundle["mapping_overlay"]["sha256"])
    reference = read_json(Path(sources["reference"]["path"]))
    for name, item in reference["bindings"].items():
        sources[name] = binding(Path(item["path"]), item["sha256"])
    overlay = read_json(OVERLAY_PATH)
    legacy = Path(sources["mapping"]["path"]).parent / overlay["legacy_convention_source"]
    sources["mapping_legacy_convention"] = binding(legacy, overlay["legacy_convention_sha256"])
    evidence = read_json(Path(sources["mapping_evidence"]["path"]))
    sources["mapping_manual"] = binding(Path(evidence["source_path"]), evidence["source_sha256"])
    for index, item in enumerate(evidence["page_images"]):
        sources[f"mapping_page_{index}"] = binding(Path(item["path"]), item["sha256"])
    view, _ = apply_original_mapping_overlay(reference, Path(sources["mapping_evidence"]["path"]))
    song = next(s for s in view["songs"] if s["song_id"] == "shalong")
    for name in ("original_audio", "rendered_audio", "human_vsqx"):
        item = song[name]
        sources[f"shalong_{name}"] = binding(Path(item["path"]), item["sha256"])
    triplets = read_json(Path(sources["triplets"]["path"]))
    claim = next(s for s in triplets["items"] if s.get("song_id") == "shalong")
    bound = (claim["human_vsqx"]["sha256"] == song["human_vsqx"]["sha256"]
             and claim["human_claims"].get("phoneme_correctness") == "user_asserted_correct")
    approval = {"bound": bound, "scope": "source_vsqx_phone_sequence_only",
                "song_id": "shalong", "source_binding": "triplets",
                "field": "items[song_id=shalong].human_claims.phoneme_correctness",
                "human_claims": claim["human_claims"], "pairing_status": claim["pairing_status"],
                "source_vsqx_sha256": claim["human_vsqx"]["sha256"]}
    return audit, bundle, song, sources, approval


def media_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to((root / "media").resolve()) or path.suffix.lower() not in {".wav", ".png"}:
        raise ValueError("media must remain within the media directory")
    return path


def verify_media(packet: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    result = []
    for item in packet["items"]:
        for media in item["media"]:
            for kind in ("audio", "waveform"):
                relative = media[f"{kind}_file"]
                path = media_path(root, relative)
                verify_file(path, media[f"{kind}_sha256"])
                if kind == "audio":
                    with wave.open(str(path), "rb") as wav:
                        if not math.isclose(wav.getnframes() / wav.getframerate(), 2.5, abs_tol=1e-8):
                            raise ValueError("preview WAV duration mismatch")
                result.append({"file": relative, "sha256": media[f"{kind}_sha256"]})
    return result


def write_run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_relative_to((ROOT / ".state/runs").resolve()) or output.exists():
        raise ValueError("output must be a new directory under .state/runs")
    packet = build_packet(*load_sources())
    media = verify_media(packet, GUI)
    output.mkdir(parents=True)
    for item in media:
        destination = media_path(output, item["file"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(media_path(GUI, item["file"]), destination)
    verify_media(packet, output)
    packet_file_hash = write_json_new(output / "packet.json", packet)
    for item in packet["source_bindings"].values():
        verify_file(Path(item["path"]), item["sha256"])
    report = ["# 沙龙六个原唱切点数据包", "",
              "仅 B01–B06：保留此前对应审计，不重问对应；C 训练项全部暂停，13 个错位项不恢复。",
              "18 个既有 WAV 与 18 个波形 PNG 逐字节复制、前后 SHA-256 校验；未重采样、未生成新音频。",
              "每个原唱预览 2.5 秒，中线 1.25 秒；合成预览仅辅助，不能当作原唱时钟。",
              "候选取冻结预览附近音符之间的完整连续 VSQX 序列，并应用已验证 ua+n overlay。未知映射不删除；不读模型预测、不按 PER 筛选。",
              "预览中缺某字的旧意见只作用于该预览，不据此删除完整片段内的坦等字。",
              "源 VSQX 音素批准绑定 triplets 中 user_asserted_correct；不是重新认可全曲时轴。内部沿用连续音符，必须由显式首尾可听音素回答界定保留范围。",
              "未创建正式音素参考、未训练、未打开测试音频/预测、未修改旧源文件。", "",
              "|片段|原唱开始秒|原唱结束秒|连续候选音符|", "|---|---:|---:|---|"]
    report.extend(f"|{c['example_id']}|{c['start_sec']}|{c['end_sec']}|{c['candidate_note_ids'][0]}–{c['candidate_note_ids'][-1]}|" for c in packet["clips"])
    (output / "DATA_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json_new(output / "BUILD.json", {"schema_version": "original_edge_review_build_v1",
                   "packet_sha256": packet["packet_sha256"], "packet_file_sha256": packet_file_hash,
                   "builder": binding(Path(__file__)), "media": media, "items": 6, "clips": 3,
                   "source_files_unchanged": True, "source_wav_vsqx_written": False,
                   "training_started": False, "test_songs_opened": False,
                   "formal_phone_reference_created": False})
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ROOT / ".state/runs" / f"original-edge-review-{stamp}"
    packet = write_run(output)
    print(f"packet={output.resolve() / 'packet.json'}; items=6; sha256={packet['packet_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

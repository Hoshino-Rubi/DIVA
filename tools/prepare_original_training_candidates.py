"""Prepare at most twenty deterministic original-song review candidates."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from any2xsampa.adapters.phoneticxeus_runtime import read_audio_crop  # noqa: E402
from any2xsampa.training.limited_data_artifacts import (  # noqa: E402
    read_json,
    sha256_file,
    sha256_object,
    verify_file,
    write_json_new,
)
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay  # noqa: E402
from any2xsampa.training.original_training_candidates import propose_training_candidates  # noqa: E402


def _preview(audio: dict[str, Any], row: dict[str, Any], output: Path, stem: str, role: str) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    from tools.build_limited_data_review import waveform_png

    values, _, crop = read_audio_crop(Path(audio["path"]), audio["sha256"], row["start_sec"], row["end_sec"])
    peak = float(np.max(np.abs(values)))
    gain = min(1.0, 0.98 / peak) if peak else 1.0
    values = values * gain
    wav, png = output / f"{stem}.wav", output / f"{stem}.png"
    sf.write(wav, values, 16000, subtype="PCM_16")
    waveform_png(values, len(values) / 16000, png, "#167d6d" if role == "original" else "#b24758")
    return {
        "role": role, "audio_file": wav.name, "waveform_file": png.name,
        "audio_sha256": sha256_file(wav), "waveform_sha256": sha256_file(png),
        "source_path": audio["path"], "source_sha256": audio["sha256"], "source_crop": crop,
        "duration_sec": len(values) / 16000, "preview_gain": gain,
        "preview_transform": "derived_crop_mono_16khz_pcm16", "full_source_audio_copied": False,
    }


def _html(candidates: list[dict[str, Any]], notes: dict[str, dict[str, Any]]) -> str:
    cards = []
    for row in candidates:
        review_id = row.get("review_id", row["candidate_id"])
        refs = row["candidate_references"]
        ids = refs[0]["note_ids"] if refs else []
        note_rows_list = []
        for n in ids:
            if n not in notes:
                continue
            note = notes[n]
            phones = note.get("model_target_ipa", [])
            phone_controls = " ".join(
                f'<label class="phone"><input type="checkbox" data-phone-check="{html.escape(review_id, quote=True)}" '
                f'data-note-id="{html.escape(n, quote=True)}" data-phone-index="{index}" '
                f'data-phone="{html.escape(phone, quote=True)}" checked>'
                f'<code>{index}={html.escape(phone)}</code></label>'
                for index, phone in enumerate(phones)
            )
            note_rows_list.append(
                f'<tr data-note-row="{html.escape(review_id, quote=True)}"><td><label><input type="checkbox" '
                f'data-note-check="{html.escape(review_id, quote=True)}" data-note-id="{html.escape(n, quote=True)}" checked>'
                f'<code>{html.escape(n)}</code></label></td><td>{html.escape(str(note.get("raw_lyric", "")))}</td>'
                f'<td><code>{html.escape(" ".join(note.get("raw_v4c_tokens", [])))}</code></td>'
                f'<td>{phone_controls}</td></tr>'
            )
        note_rows = "".join(note_rows_list)
        media = []
        for preview in row["previews"]:
            label = "原唱" if preview["role"] == "original" else "合成对应辅助"
            media.append(
                f'<section><h3>{label}</h3><img src="media/{html.escape(preview["waveform_file"], quote=True)}" alt="波形">'
                f'<audio controls loop preload="none" src="media/{html.escape(preview["audio_file"], quote=True)}"></audio></section>'
            )
        cards.append(
            f'<article><h2>{html.escape(review_id)} · {html.escape(row["candidate_id"])} · {html.escape(row["song_id"])}</h2>'
            f'<p>原唱 {row["start_sec"]:.3f}–{row["end_sec"]:.3f}s；来源上下文 {html.escape(row["context_id"])}。</p>'
            '<p class="notes">默认已勾选整段候选。请先听原唱：完整对应就保持勾选；若开头/结尾被截断，只取消实际听不到的边缘音素。无需手写 nxxx 或 phone_index。</p>'
            '<table><thead><tr><th>听到的音符</th><th>字</th><th>原始引擎音素</th><th>可听音素（索引自动生成）</th></tr></thead>'
            f'<tbody>{note_rows}</tbody></table>'
            f'<div class="previews">{"".join(media)}</div>'
            f'<label>状态<select data-id="{html.escape(review_id, quote=True)}"><option>pending</option><option>human_confirmed</option><option>rejected</option></select></label>'
            f'<label>复核人<input data-reviewer="{html.escape(review_id, quote=True)}"></label>'
            f'<label>复核原话<textarea data-statement="{html.escape(review_id, quote=True)}" rows="2"></textarea></label>'
            f'<label>边界状态<select data-boundary="{html.escape(review_id, quote=True)}"><option value="">未确认</option><option>full</option><option>truncated_start</option><option>truncated_end</option><option>truncated_both</option></select></label>'
            '</article>'
        )
    ids = [row.get("review_id", row["candidate_id"]) for row in candidates]
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'self' file:; img-src 'self' file:; media-src 'self' file:; script-src 'unsafe-inline'; style-src 'unsafe-inline'"><title>训练原唱候选复核</title><style>body{margin:0;background:#f3f5f7;color:#202830;font:15px/1.55 system-ui,"Segoe UI",sans-serif}header,main{max-width:1240px;margin:auto;padding:20px}article{background:#fff;border:1px solid #d4dce4;border-radius:9px;padding:16px;margin:16px 0}.previews{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}img{display:block;width:100%%;height:auto}audio{width:100%%}textarea,input,select{display:block;width:100%%;box-sizing:border-box;padding:7px;margin:4px 0 10px}.notes{color:#556575}.hint{background:#fff4d5;border-left:4px solid #c78c22;padding:10px}table{border-collapse:collapse;width:100%%;margin:10px 0;font-size:13px;background:#fafcfd}th,td{border:1px solid #cbd5df;padding:6px;text-align:left;vertical-align:top}th{background:#e9eef3}.phone{display:inline-flex;gap:3px;margin:2px 8px 2px 0;white-space:nowrap}button{padding:9px 13px}@media(max-width:850px){.previews{grid-template-columns:1fr}table{font-size:12px}}</style></head><body><header><h1>最多二十个训练原唱候选</h1><p class="hint">默认整段已勾选。听原唱后，完整对应就保持不变；截断时取消听不到的边缘音素。不要手写 nxxx 或 phone_index，页面会自动导出。</p><button onclick="download()">导出回答 JSON</button></header><main>""" + "".join(cards) + """</main><script>const IDS=%s;const PACKET='__PACKET__';document.querySelectorAll('[data-note-check]').forEach(box=>box.addEventListener('change',()=>{const id=box.dataset.noteCheck;document.querySelectorAll(`[data-phone-check="${id}"][data-note-id="${box.dataset.noteId}"]`).forEach(x=>x.checked=box.checked);}));function download(){const responses=IDS.map(id=>{const status=document.querySelector(`[data-id="${id}"]`).value;const reviewer=document.querySelector(`[data-reviewer="${id}"]`).value;const statement=document.querySelector(`[data-statement="${id}"]`).value;const boundary=document.querySelector(`[data-boundary="${id}"]`).value;const selectedNotes=[...document.querySelectorAll(`[data-note-check="${id}"]:checked`)].map(x=>x.dataset.noteId);const phones=[...document.querySelectorAll(`[data-phone-check="${id}"]:checked`)].map(x=>{const first=x.dataset.noteId===selectedNotes[0];const last=x.dataset.noteId===selectedNotes[selectedNotes.length-1];const edge=(boundary==='truncated_start'&&first)?'truncated_start':(boundary==='truncated_end'&&last)?'truncated_end':(boundary==='truncated_both'&&(first||last))?(first?'truncated_start':'truncated_end'):'full';return {note_id:x.dataset.noteId,phone_index:Number(x.dataset.phoneIndex),phone:x.dataset.phone,audible:true,edge_status:edge};});return {review_id:id,status,reviewer:reviewer||null,review_statement:statement||null,boundary_state:boundary||null,note_ids:status==='human_confirmed'?selectedNotes:[],phones:status==='human_confirmed'?phones:[]};});const out={schema_version:'original_segment_review_responses_v2',packet_sha256:PACKET,responses,automatic_annotation_updates_allowed:false};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)+'\\n'],{type:'application/json'}));a.download='training-responses-v2.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500);}</script></body></html>""" % (str(ids).replace("'", '"'))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare twenty bounded training review candidates.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mapping-evidence", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--correspondence", required=True, type=Path, help="offset diagnostic original-reference-manifest.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("output must be a new directory inside .state/runs")
    reference = read_json(args.reference)
    correspondence = read_json(args.correspondence)
    if correspondence.get("reference_manifest_sha256") != sha256_file(args.reference):
        raise ValueError("correspondence is bound to a different reference manifest")
    examples = correspondence.get("examples", [])
    songs: dict[str, dict[str, Any]] = {}
    for row in examples:
        songs.setdefault(row["song_id"], {"song_id": row["song_id"], "clips": []})["clips"].append({
            "example_id": row["example_id"],
            "requested_original_start_sec": row["start_sec"], "requested_original_end_sec": row["end_sec"],
            "candidates": row["candidate_references"],
        })
    correspondence_view = {"songs": list(songs.values())}
    base, overlay = apply_original_mapping_overlay(reference, args.mapping_evidence)
    contexts = read_json(args.contexts)["contexts"]
    proposal = propose_training_candidates(contexts=contexts, reference=base, correspondence=correspondence_view)
    output.mkdir(parents=True)
    proposal["source_bindings"] = {
        "reference": {"path": str(args.reference.resolve()), "sha256": sha256_file(args.reference)},
        "mapping_evidence": {"path": str(args.mapping_evidence.resolve()), "sha256": sha256_file(args.mapping_evidence)},
        "contexts": {"path": str(args.contexts.resolve()), "sha256": sha256_file(args.contexts)},
        "correspondence": {"path": str(args.correspondence.resolve()), "sha256": sha256_file(args.correspondence)},
        "mapping_overlay": overlay,
    }
    proposal["proposal_hash"] = sha256_object(proposal)
    write_json_new(output / "candidate-proposals.json", proposal)
    songs_view = {song["song_id"]: song for song in base["songs"]}
    notes = {note["note_id"]: note for song in base["songs"] for note in song["notes"]}
    media = output / "media"
    media.mkdir()
    for row in proposal["candidates"]:
        song = songs_view[row["song_id"]]
        row["previews"] = [
            _preview(song["original_audio"], row, media, row["candidate_id"] + "-original", "original"),
            _preview(song["rendered_audio"], {**row, "start_sec": row["candidate_references"][0]["rendered_start_sec"], "end_sec": row["candidate_references"][0]["rendered_end_sec"]}, media, row["candidate_id"] + "-rendered", "rendered"),
        ]
    packet = {
        "schema_version": "original_training_candidate_review_packet_v2",
        "proposal_hash": proposal["proposal_hash"], "items": [], "count": len(proposal["candidates"]),
        "test_songs_opened": False, "human_confirmation_required": True,
        "response_schema": "original_segment_review_responses_v2",
    }
    for index, candidate in enumerate(proposal["candidates"], 1):
        item = dict(candidate)
        item["review_id"] = f"C{index:02d}"
        packet["items"].append(item)
    packet["packet_sha256"] = sha256_object(packet)
    write_json_new(output / "review-packet.json", packet)
    write_json_new(output / "responses-template.json", {
        "schema_version": "original_segment_review_responses_v2",
        "packet_sha256": packet["packet_sha256"],
        "responses": [{"review_id": item["review_id"], "status": "pending", "reviewer": None, "review_statement": None, "boundary_state": None, "note_ids": [], "phones": []} for item in packet["items"]],
        "automatic_annotation_updates_allowed": False,
    })
    write_json_new(output / "STATUS.json", {"stage": "waiting_human_review", "candidate_count": len(packet["items"]), "song_counts": proposal["song_counts"], "packet_sha256": packet["packet_sha256"], "training_eligible": False})
    with (output / "index.html").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_html(packet["items"], notes).replace("__PACKET__", packet["packet_sha256"]))
    print(f"candidates={len(packet['items'])}; song_counts={proposal['song_counts']}; status=waiting_human_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

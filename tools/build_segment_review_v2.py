"""Build a bounded, local-only review packet for exact clip correspondence.

The page is intentionally an annotation aid, not an automatic labeler.  It
exports the same response schema consumed by ``import_segment_reviews.py``.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import (  # noqa: E402
    read_json,
    sha256_file,
    write_json_new,
)


def _copy_media(source_dir: Path, output_dir: Path, items: list[dict[str, Any]]) -> None:
    destination = output_dir / "media"
    destination.mkdir(parents=True, exist_ok=True)
    for item in items:
        for preview in item.get("previews", []):
            for field in ("audio_file", "waveform_file"):
                name = preview.get(field)
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("review media name must be a plain filename")
                source = source_dir / "media" / name
                target = destination / name
                if target.exists():
                    if sha256_file(target) != sha256_file(source):
                        raise ValueError("review media filename collision")
                else:
                    shutil.copyfile(source, target)


def _response_template(items: list[dict[str, Any]], packet_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "original_segment_review_responses_v2",
        "packet_sha256": packet_hash,
        "responses": [
            {
                "review_id": item["review_id"],
                "status": "pending",
                "reviewer": None,
                "review_statement": None,
                "boundary_state": None,
                "note_ids": [],
                "phones": [],
            }
            for item in items
        ],
        "automatic_annotation_updates_allowed": False,
    }


def _html(items: list[dict[str, Any]], note_catalog: dict[str, dict[str, Any]] | None = None) -> str:
    note_catalog = note_catalog or {}
    cards: list[str] = []
    for item in items:
        previews: list[str] = []
        for preview in item.get("previews", []):
            role = {
                "original": "原始原唱",
                "rendered": "合成参考（仅对应辅助）",
                "separated_vocal_auxiliary": "分离人声（仅辅助）",
            }.get(preview.get("role"), str(preview.get("role")))
            previews.append(
                f'<section><h3>{html.escape(role)}</h3>'
                f'<img src="media/{html.escape(preview["waveform_file"], quote=True)}" alt="波形">'
                f'<audio controls loop preload="none" src="media/{html.escape(preview["audio_file"], quote=True)}"></audio></section>'
            )
        nearby = item.get("nearby_source_notes", [])
        note_text = "；".join(
            f"{note.get('lyric', '')} ({' '.join(note.get('raw_v4c_tokens', []))})"
            for note in nearby
        )
        note_rows: list[str] = []
        for nearby_note in nearby:
            note_id = str(nearby_note.get("note_id", ""))
            source_note = note_catalog.get(note_id, {})
            lyric = source_note.get("raw_lyric", nearby_note.get("lyric", ""))
            tokens = source_note.get("raw_v4c_tokens", nearby_note.get("raw_v4c_tokens", []))
            phones = source_note.get("model_target_ipa", nearby_note.get("model_target_ipa", []))
            phone_controls = " ".join(
                f'<label class="phone"><input type="checkbox" data-phone-check="{html.escape(item["review_id"], quote=True)}" '
                f'data-note-id="{html.escape(note_id, quote=True)}" data-phone-index="{index}" data-phone="{html.escape(phone, quote=True)}">'
                f'<code>{index}={html.escape(phone)}</code></label>'
                for index, phone in enumerate(phones)
            )
            note_rows.append(
                f'<tr><td><label><input type="checkbox" data-note-check="{html.escape(item["review_id"], quote=True)}" '
                f'data-note-id="{html.escape(note_id, quote=True)}"><code>{html.escape(note_id)}</code></label></td>'
                f'<td>{html.escape(str(lyric))}</td><td><code>{html.escape(" ".join(tokens))}</code></td>'
                f'<td>{phone_controls}</td></tr>'
            )
        cards.append(
            f'<article id="{html.escape(item["review_id"], quote=True)}" data-side="{html.escape(str(item.get("side", "")), quote=True)}">'
            f'<h2>{html.escape(item["review_id"])} · {html.escape(item.get("song_id", ""))}</h2>'
            f'<p>旧片段 {item.get("example_id", "")}；切点 {item.get("original_cut_sec", 0):.4f}s。请只确认切点附近的字、是否位于字前/中/后，以及能明确听到的首尾音素。</p>'
            f'<p class="notes">附近参考：{html.escape(note_text)}</p>'
            '<p class="notes">请在原唱中勾选实际听到的边缘音符和音素。页面会自动生成 note_id 与 phone_index，不用手写编号。</p>'
            '<table><thead><tr><th>听到的边缘音符</th><th>字</th><th>原始引擎音素</th><th>可听音素（索引自动生成）</th></tr></thead>'
            f'<tbody>{"".join(note_rows)}</tbody></table>'
            f'<div class="previews">{"".join(previews)}</div>'
            f'<label>状态<select data-review="{html.escape(item["review_id"], quote=True)}" data-field="status"><option value="pending">pending</option><option value="human_confirmed">human_confirmed</option><option value="rejected">rejected</option></select></label>'
            f'<label>复核人<input data-review="{html.escape(item["review_id"], quote=True)}" data-field="reviewer"></label>'
            f'<label>原话<textarea data-review="{html.escape(item["review_id"], quote=True)}" data-field="statement" rows="2"></textarea></label>'
            f'<label>边界状态<select data-review="{html.escape(item["review_id"], quote=True)}" data-field="boundary"><option value="">未确认</option><option value="full">full</option><option value="truncated_start">truncated_start</option><option value="truncated_end">truncated_end</option><option value="truncated_both">truncated_both</option></select></label>'
            '<p>无法确认时请填写 pending；不要凭推测补全音素。</p></article>'
        )
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self' file:; img-src 'self' file:; media-src 'self' file:; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>原唱片段参考 v2 复核</title><style>
body{margin:0;background:#f3f5f7;color:#202830;font:15px/1.55 system-ui,"Segoe UI",sans-serif}header,main{max-width:1240px;margin:auto;padding:20px}article{background:#fff;border:1px solid #d4dce4;border-radius:9px;padding:16px;margin:16px 0}.previews{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}img{display:block;width:100%%;height:auto}audio{width:100%%}.notes{color:#556575}textarea,input,select{width:100%%;box-sizing:border-box;padding:7px;border:1px solid #aebac5;border-radius:4px}button{padding:9px 13px;margin:4px 4px 4px 0;border:1px solid #697887;border-radius:5px;background:#fff;cursor:pointer}.hint{background:#fff4d5;border-left:4px solid #c78c22;padding:10px}@media(max-width:850px){.previews{grid-template-columns:1fr}}
</style></head><body><header><h1>原唱片段参考 v2</h1>
<p class="hint">全曲音素校准保持有效。本页只确认片段对应和边缘音素，不修改 VSQX、不打开最终测试歌曲。填写后点击“导出回答 JSON”。</p>
<button type="button" onclick="downloadResponses()">导出回答 JSON</button> <button type="button" onclick="loadTemplate()">恢复空白</button></header><main>""" + "".join(cards) + """</main><script>
const ids=%s;
function field(id,name){return document.querySelector(`[data-review="${id}"][data-field="${name}"]`)}
document.querySelectorAll('[data-note-check]').forEach(box=>box.addEventListener('change',()=>{const id=box.closest('article').id;document.querySelectorAll(`[data-phone-check="${id}"][data-note-id="${box.dataset.noteId}"]`).forEach(x=>x.checked=box.checked);}));
function values(id){const status=field(id,'status').value;const article=document.getElementById(id);const selectedNotes=[...document.querySelectorAll(`[data-note-check="${id}"]:checked`)].map(x=>x.dataset.noteId);const boundary=field(id,'boundary').value;const phones=[...document.querySelectorAll(`[data-phone-check="${id}"]:checked`)].filter(x=>selectedNotes.includes(x.dataset.noteId)).map(x=>({note_id:x.dataset.noteId,phone_index:Number(x.dataset.phoneIndex),phone:x.dataset.phone,audible:true,edge_status:article.dataset.side==='start'?'truncated_start':article.dataset.side==='end'?'truncated_end':'full'}));return {review_id:id,status,reviewer:field(id,'reviewer').value||null,review_statement:field(id,'statement').value||null,boundary_state:boundary||null,note_ids:status==='human_confirmed'?selectedNotes:[],phones:status==='human_confirmed'?phones:[]};}
function downloadResponses(){const packet=window.PACKET_SHA256;const out={schema_version:'original_segment_review_responses_v2',packet_sha256:packet,responses:ids.map(values),automatic_annotation_updates_allowed:false};const blob=new Blob([JSON.stringify(out,null,2)+'\\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='responses-v2.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500);}
function loadTemplate(){for(const id of ids){for(const n of ['status','reviewer','statement','boundary']){const x=field(id,n);if(n==='status')x.value='pending';else x.value='';}document.querySelectorAll(`[data-note-check="${id}"],[data-phone-check="${id}"]`).forEach(x=>x.checked=false);}}
window.PACKET_SHA256='__PACKET_SHA256__';
</script></body></html>""" % (str([item["review_id"] for item in items]).replace("'", "\"") )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an explicit human clip-reference v2 packet.")
    parser.add_argument("--source-review", required=True, type=Path, help="quality8h review-v1/review.json")
    parser.add_argument("--evaluation-manifest", type=Path, help="order-confirmed original-reference manifest; adds full candidate note sequences")
    parser.add_argument("--reference", type=Path, help="reference manifest used to show IPA phone choices")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("output must be a new directory inside .state/runs")
    source = args.source_review.resolve()
    source_packet = read_json(source)
    items = source_packet.get("items", [])
    if len(items) != 6 or {item.get("review_id") for item in items} != {f"B{i:02d}" for i in range(1, 7)}:
        raise ValueError("the v2 Shalong packet must contain exactly B01-B06")
    if any(item.get("status") not in {"pending", None} for item in items):
        raise ValueError("source packet already contains a response")
    output.mkdir(parents=True)
    _copy_media(source.parent, output, items)
    normalized: list[dict[str, Any]] = []
    candidate_rows = {}
    note_catalog: dict[str, dict[str, Any]] = {}
    if args.reference:
        reference = read_json(args.reference)
        note_catalog = {note["note_id"]: note for song in reference.get("songs", []) for note in song.get("notes", [])}
    if args.evaluation_manifest:
        evaluation = read_json(args.evaluation_manifest)
        candidate_rows = {row["example_id"]: row for row in evaluation.get("examples", [])}
    for item in items:
        row = dict(item)
        row["previews"] = [dict(preview) for preview in item.get("previews", [])]
        proposed = candidate_rows.get(item.get("example_id"))
        if proposed:
            candidate = proposed.get("candidate_references", [{}])[0]
            row["candidate_note_ids"] = list(candidate.get("note_ids", []))
            row["candidate_reference_mapping_complete"] = bool(candidate.get("mapping_complete"))
            row["candidate_reference_factor"] = candidate.get("factor")
        normalized.append(row)
    packet = {
        "schema_version": "original_segment_review_packet_v2",
        "source_review_sha256": sha256_file(source),
        "items": normalized,
        "count": len(normalized),
        "human_approval_recorded": False,
        "test_songs_opened": False,
        "source_phoneme_approval_preserved": True,
        "response_schema": "original_segment_review_responses_v2",
    }
    from any2xsampa.training.limited_data_artifacts import sha256_object
    packet["packet_sha256"] = sha256_object(packet)
    packet_hash = packet["packet_sha256"]
    write_json_new(output / "review-packet.json", packet)
    template = _response_template(normalized, packet_hash)
    write_json_new(output / "responses-template.json", template)
    write_json_new(output / "STATUS.json", {"stage": "waiting_human_review", "confirmed_items": 0, "total_items": 6, "packet_sha256": packet_hash})
    document = _html(normalized, note_catalog).replace("__PACKET_SHA256__", packet_hash)
    with (output / "index.html").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
    print(f"packet={output / 'review-packet.json'}; items=6; status=waiting_human_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

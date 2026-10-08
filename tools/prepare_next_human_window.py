"""Prepare a minimal, read-only human review window from repair proposals.

This intentionally emits review prompts, not phone truth.  In particular,
correspondence-only rows and boundary evidence never receive phone indices.
"""
from __future__ import annotations

import argparse, html, json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GROUP_PROMPTS = {
    "correspondence": "确认原曲与合成片段是否仅在语义/片段层对应；不能据此建立正式 phone reference。若歌词缺失/严重截断，标记 lyric-risk。",
    "boundary": "确认边界窗口的首尾音频是否可听且与审计描述一致；这不是完整 dev 窗口，不得转为训练样本。",
    "boundary_dev": "确认 B01-B06 是否确为边界对应证据，并记录可听首/末边界；不要从 correspondence 推断 phone_index。",
}

def _latest(root: Path) -> Path:
    paths = sorted(root.glob(".state/runs/reference-repair-v3-*/repair-proposals.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError("no reference-repair-v3 repair-proposals.json found")
    return paths[0]

def _kind(review_id: str, row: dict[str, Any]) -> str:
    if review_id.startswith("B"):
        return "boundary_dev"
    if review_id in {"C05", "C10", "C16"}:
        return "boundary"
    if review_id in {"C03", "C11", "C13", "C20"}:
        return "correspondence"
    return "excluded"

def build(proposals: dict[str, Any], source: Path) -> dict[str, Any]:
    rows = []
    for row in proposals.get("proposals", []):
        rid = row.get("review_id", "")
        kind = _kind(rid, row)
        if kind == "excluded":
            continue
        notes = row.get("notes") or []
        rows.append({
            "review_id": rid,
            "group": kind,
            "song_id": row.get("song_id"),
            "split": row.get("split"),
            "decision": row.get("decision"),
            "source_example_id": row.get("source_example_id"),
            "window": row.get("rendered_audio_window") or {},
            "original_audio_window": row.get("original_audio_window") or {},
            "notes": [{"note_id": n.get("note_id"), "lyric": n.get("lyric"), "mapping_status": n.get("mapping_status"), "ipa": n.get("ipa", [])} for n in notes],
            "confirm": GROUP_PROMPTS[kind],
            "phone_index_policy": "manual_only; do not infer or auto-fill phone_index",
            "formal_phone_reference_created": False,
        })
    return {"schema_version": "next_human_window_v1", "source": str(source.resolve()), "formal_phone_reference_created": False, "starts_training": False, "items": rows, "groups": {k: [r["review_id"] for r in rows if r["group"] == k] for k in GROUP_PROMPTS}}

def render(data: dict[str, Any]) -> str:
    sections = []
    for item in data["items"]:
        notes = ", ".join(f'{n["note_id"]} ({n["lyric"]})' for n in item["notes"][:12])
        sections.append(f'<article><h2>{html.escape(item["review_id"])} <small>{html.escape(item["group"])}</small></h2><p><b>确认：</b>{html.escape(item["confirm"])}</p><p>song={html.escape(str(item["song_id"]))}; split={html.escape(str(item["split"]))}; notes={html.escape(notes or "（无）")}</p><p><b>phone policy:</b> manual_only; 不自动猜 phone_index。</p></article>')
    return '<!doctype html><meta charset="utf-8"><title>Next human window</title><style>body{font:15px sans-serif;max-width:1000px;margin:2em auto}article{border:1px solid #ccc;padding:1em;margin:1em 0}small{color:#666}</style><h1>下一人工窗口</h1><p>仅供人工确认；correspondence-only 不构成正式 phone truth。</p>' + ''.join(sections)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--out-dir", type=Path, default=ROOT / ".state/next-human-window")
    args = ap.parse_args()
    source = args.input or _latest(ROOT)
    data = build(json.loads(source.read_text(encoding="utf-8")), source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "next-human-window.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "next-human-window.html").write_text(render(data), encoding="utf-8")
    print(args.out_dir / "next-human-window.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

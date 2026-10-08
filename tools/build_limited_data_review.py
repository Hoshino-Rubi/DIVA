from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import importlib.metadata
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.adapters.phoneticxeus_runtime import read_audio_crop  # noqa: E402
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new  # noqa: E402


def centered_range(center: float, duration: float, source_duration: float) -> tuple[float, float]:
    duration = min(duration, source_duration)
    start = min(max(0.0, center - duration / 2), source_duration - duration)
    return start, start + duration


def choose_review_items(reference: Mapping[str, Any], matches: Mapping[str, Any]) -> list[dict[str, Any]]:
    songs = {song["song_id"]: song for song in reference["songs"] if song["split"] != "test"}
    items = []
    for song_id, song in songs.items():
        clips = [row for row in matches[song_id]["clips"] if row["candidates"]]
        if not clips:
            continue
        chosen = min(clips, key=lambda row: (
            not row["ambiguous"], row["relative_cost_gap"] if row["relative_cost_gap"] is not None else -1,
            row["example_id"],
        ))
        candidate = chosen["candidates"][0]
        original_center = (chosen["requested_original_start_sec"] + chosen["requested_original_end_sec"]) / 2
        rendered_center = (candidate["rendered_start_sec"] + candidate["rendered_end_sec"]) / 2
        items.append({
            "kind": "section_correspondence", "song_id": song_id, "song_name": song["display_name"],
            "question": "这两段是否对应同一句、同一处演唱？",
            "answer_scope": "section_identity_only_not_phone_transcription",
            "original_range": centered_range(original_center, 12.0, song["original_audio"]["duration_sec"]),
            "rendered_range": centered_range(rendered_center, min(12.0, candidate["rendered_end_sec"] - candidate["rendered_start_sec"]), song["rendered_audio"]["duration_sec"]),
            "machine_candidate": candidate,
            "ambiguous": chosen["ambiguous"], "reference_example_id": chosen["example_id"],
            "review_status": "pending", "raw_tokens": [],
        })
    occurrences: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for song in songs.values():
        for note in song["notes"]:
            for token in note["unresolved_tokens"]:
                occurrences.setdefault(token, []).append((song, note))
    for token in sorted(occurrences, key=lambda value: (-len(occurrences[value]), value))[:6]:
        song, note = min(occurrences[token], key=lambda pair: (pair[0]["split"] != "dev", pair[0]["song_id"], pair[1]["note_index"]))
        center = (note["start_sec"] + note["end_sec"]) / 2
        items.append({
            "kind": "engine_token_semantics", "song_id": song["song_id"], "song_name": song["display_name"],
            "question": "这里的引擎音素实际对应什么发音？可用熟悉的读音描述，不必重做 VSQX。",
            "answer_scope": "engine_reference_semantics_only_not_original_truth",
            "original_range": None,
            "rendered_range": centered_range(center, 6.0, song["rendered_audio"]["duration_sec"]),
            "raw_tokens": note["raw_v4c_tokens"], "unresolved_token": token,
            "note_id": note["note_id"], "note_start_sec": note["start_sec"], "note_end_sec": note["end_sec"],
            "raw_lyric": note.get("raw_lyric"),
            "occurrence_count": len(occurrences[token]), "review_status": "pending",
        })
    for index, item in enumerate(items[:12], 1):
        item["review_id"] = f"R{index:02d}"
    return items[:12]


def waveform_png(values: Any, duration: float, path: Path, color: str) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1200, 180), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    left, right, center, scale = 42, 1180, 75, 58
    draw.line((left, center, right, center), fill="#cbd5d1", width=1)
    for column, block in enumerate(np.array_split(values, right - left)):
        if block.size:
            draw.line((left + column, center - float(block.max()) * scale, left + column, center - float(block.min()) * scale), fill=color, width=1)
    for i in range(5):
        x = left + (right - left) * i / 4
        draw.line((x, 140, x, 146), fill="#89918d")
        label = f"{duration * i / 4:.1f}s"
        draw.text((min(x, 1140), 151), label, fill="#4b5550", font=font, anchor="mm")
    canvas.save(path, format="PNG")


def make_preview(audio: Mapping[str, Any], bounds: tuple[float, float], output: Path, stem: str, role: str) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    values, _, crop = read_audio_crop(Path(audio["path"]), audio["sha256"], *bounds)
    peak = float(np.max(np.abs(values)))
    gain = min(1.0, 0.98 / peak) if peak else 1.0
    values = values * gain
    wav_path, image_path = output / f"{stem}.wav", output / f"{stem}.png"
    if wav_path.exists() or image_path.exists():
        raise FileExistsError("review preview already exists")
    sf.write(wav_path, values, 16000, subtype="PCM_16")
    waveform_png(values, len(values) / 16000, image_path, "#167d6d" if role == "original" else "#b24758")
    info = sf.info(wav_path)
    if info.frames != len(values) or info.samplerate != 16000 or info.channels != 1:
        raise ValueError("review preview failed its audio-format check")
    return {
        "role": role, "audio_file": wav_path.name, "waveform_file": image_path.name,
        "audio_sha256": sha256_file(wav_path), "waveform_sha256": sha256_file(image_path),
        "source_path": audio["path"], "source_sha256": audio["sha256"],
        "source_crop": crop, "duration_sec": len(values) / 16000,
        "preview_gain": gain, "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
        "preview_transform": "derived_crop_mono_16khz_pcm16", "full_source_audio_copied": False,
    }


def render_html(items: list[dict[str, Any]]) -> str:
    cards = []
    for item in items:
        previews = []
        for preview in item["previews"]:
            label = "原始录音" if preview["role"] == "original" else "合成参考"
            start = preview["source_crop"]["effective_start_sec"]
            end = preview["source_crop"]["effective_end_sec"]
            previews.append(
                f'<div class="preview"><div class="source"><strong>{label}</strong><span>{start:.2f}–{end:.2f} s</span></div>'
                f'<img src="media/{html.escape(preview["waveform_file"], quote=True)}" alt="{label}波形" width="1200" height="180">'
                f'<audio controls preload="none" src="media/{html.escape(preview["audio_file"], quote=True)}"></audio></div>'
            )
        tokens = " ".join(item.get("raw_tokens", []))
        detail = f'<div class="tokens">{html.escape(tokens)}</div>' if tokens else ""
        if item.get("note_id"):
            detail += f'<div class="detail">{html.escape(item["note_id"])} · {html.escape(str(item.get("raw_lyric") or ""))}</div>'
        kind = "段落对应" if item["kind"] == "section_correspondence" else "音素含义"
        cards.append(
            f'<article id="{item["review_id"]}"><div class="item-head"><h2>{item["review_id"]} · {html.escape(item["song_name"] or item["song_id"])}</h2>'
            f'<span class="status">{kind} · 待复核</span></div><p>{html.escape(item["question"])}</p>{detail}'
            f'<div class="previews">{"".join(previews)}</div></article>'
        )
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; media-src 'self' file:; style-src 'unsafe-inline'">
<title>原唱与参考复核</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f5;color:#1b2924;font:15px/1.55 system-ui,"Segoe UI",sans-serif;letter-spacing:0}
header{background:#fff;border-bottom:1px solid #d7dfda;padding:24px max(20px,calc((100% - 1100px)/2))}
h1{font-size:25px;margin:0 0 6px}header p{margin:0;color:#59645f}main{max-width:1140px;margin:auto;padding:20px}
article{background:#fff;border:1px solid #d7dfda;border-radius:6px;padding:18px;margin-bottom:16px}
.item-head,.source{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
h2{font-size:18px;margin:0}.status{color:#8a4a23;font-size:13px}.previews{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
.preview{min-width:0}.source span,.detail{font-size:13px;color:#69746e}.source strong{font-size:14px}
img{width:100%;height:auto;display:block}audio{width:100%;height:40px;display:block}
.tokens{font:16px/1.8 ui-monospace,Consolas,monospace;overflow-wrap:anywhere;color:#a23147}.detail{margin:4px 0 12px}
article p{margin:8px 0 14px;overflow-wrap:anywhere}@media(max-width:650px){main{padding:12px}article{padding:14px}.previews{grid-template-columns:1fr;gap:18px}header{padding:18px 16px}h1{font-size:22px}}
</style></head><body><header><h1>原唱与参考复核</h1>
<p>''' + str(len(items)) + ''' 项待复核记录。自动对应仅为候选；现有人工 VSQX 保持不变。最终测试歌曲未参与。</p>
</header><main>''' + "".join(cards) + "</main></body></html>"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build at most twelve clearly unapproved review items from existing songs.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--correspondence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("use a new review directory under .state/runs")
    reference, match_index = read_json(args.reference), read_json(args.correspondence)
    if match_index["reference_manifest_sha256"] != sha256_file(args.reference):
        raise ValueError("correspondence references a different manifest")
    matches = {}
    for row in match_index["songs"]:
        path = Path(row["path"])
        verify_file(path, row["sha256"])
        matches[row["song_id"]] = read_json(path)
    items = choose_review_items(reference, matches)
    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    songs = {song["song_id"]: song for song in reference["songs"]}
    for item in items:
        song = songs[item["song_id"]]
        if song["split"] == "test":
            raise ValueError("review packet cannot expose sealed test songs")
        previews = []
        for role in ("original", "rendered"):
            if item[f"{role}_range"] is not None:
                previews.append(make_preview(song[f"{role}_audio"], tuple(item[f"{role}_range"]), media, f"{item['review_id']}-{role}", role))
        item["previews"] = previews
    packet = {
        "schema_version": "limited_data_human_review_packet_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_manifest_sha256": sha256_file(args.reference), "correspondence_index_sha256": sha256_file(args.correspondence),
        "items": items, "count": len(items), "human_approval_recorded": False,
        "original_phone_truth_established": False, "test_songs_opened": False,
        "preview_runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "soundfile", "pillow")},
    }
    write_json_new(output / "review-packet.json", packet)
    write_json_new(output / "review-responses-template.json", {
        "schema_version": "limited_data_review_responses_v1", "packet_sha256": sha256_file(output / "review-packet.json"),
        "responses": [{"review_id": item["review_id"], "review_status": "pending", "answer": None, "ipa": None, "comment": None} for item in items],
        "automatic_annotation_updates_allowed": False,
    })
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(render_html(items))
    print(f"review_items={len(items)}; full_source_audio_copies=0; html={output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

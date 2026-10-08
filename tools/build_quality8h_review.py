"""Keep the same six human questions; add time-matched separated-vocal previews."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new

OLD = ROOT / ".state/runs/shalong-clock-rule-20260906T232630/boundary-review"


def vocal_preview(record: dict, cut_sec: float, output: Path, stem: str) -> dict:
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf
    from PIL import Image, ImageDraw
    from build_limited_data_review import waveform_png

    source = record["vocal"]
    path = Path(source["path"])
    verify_file(path, source["sha256"])
    local_cut = cut_sec - record["clock"]["effective_original_start_sec"]
    first, last = round((local_cut - 1.25) * 44100), round((local_cut + 1.25) * 44100)
    if first < 0 or last > source["samples"]:
        raise ValueError("review crop does not fit the derived vocal context")
    with sf.SoundFile(path) as stream:
        stream.seek(first)
        values = stream.read(last - first, dtype="float32", always_2d=True)
    values = resample_poly(values.mean(axis=1, dtype=np.float32), 160, 441).astype(np.float32)
    if len(values) != 40000 or not np.isfinite(values).all():
        raise ValueError("invalid 2.5-second separated review preview")
    peak = float(np.max(np.abs(values)))
    gain = min(1., .98 / peak) if peak else 1.
    values *= gain
    wav, picture = output / f"{stem}.wav", output / f"{stem}.png"
    with wav.open("xb") as stream:
        sf.write(stream, values, 16000, format="WAV", subtype="PCM_16")
    waveform_png(values, 2.5, picture, "#315fbe")
    with Image.open(picture) as original:
        canvas = original.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.line((611, 5, 611, 140), fill="#c62828", width=2)
    draw.text((620, 8), "cut: 1.25s", fill="#c62828")
    canvas.save(picture)
    return {"role": "separated_vocal_auxiliary", "audio_file": wav.name, "waveform_file": picture.name,
        "audio_sha256": sha256_file(wav), "waveform_sha256": sha256_file(picture), "source_path": str(path),
        "source_sha256": source["sha256"], "source_start_sample": first, "source_end_sample_exclusive": last,
        "duration_sec": 2.5, "sample_rate": 16000, "channels": 1, "preview_gain": gain,
        "original_cut_sec": cut_sec, "cut_in_preview_sec": local_cut - first / 44100,
        "reference_audio_not_replaced": True, "not_a_model_input": True}


def build(run: Path, destination: Path) -> dict:
    old = read_json(OLD / "review.json")
    if len(old["items"]) != 6 or any(item["status"] != "pending" for item in old["items"]):
        raise ValueError("review scope/status changed; do not overwrite human responses")
    index_path = run / "separation-contexts/index.json"
    index = read_json(index_path)
    records = {}
    for entry in index["entries"]:
        if entry["context"]["song_id"] == "shalong":
            verify_file(Path(entry["record_path"]), entry["record_sha256"])
            records[entry["context"]["example_id"]] = read_json(Path(entry["record_path"]))
    media = destination / "media"
    media.mkdir(parents=True)
    items, cards = [], []
    labels = {"original": "原始原唱", "rendered": "合成候选（估计对齐）", "separated_vocal_auxiliary": "分离人声（仅辅助）"}
    for item in old["items"]:
        previews = []
        for previous in item["previews"]:
            for file_key, hash_key in (("audio_file", "audio_sha256"), ("waveform_file", "waveform_sha256")):
                source = OLD / "media" / previous[file_key]
                verify_file(source, previous[hash_key])
                with (media / previous[file_key]).open("xb") as stream:
                    stream.write(source.read_bytes())
            previews.append(previous)
        preview = vocal_preview(records[item["example_id"]], item["original_cut_sec"], media, item["review_id"] + "-vocal")
        previews.append(preview)
        rows = []
        for p in previews:
            rows.append(f'<section><h3>{labels[p["role"]]}</h3><img src="media/{html.escape(p["waveform_file"], quote=True)}" alt="{html.escape(labels[p["role"]])}波形" width="1200" height="180"><audio controls loop preload="none" src="media/{html.escape(p["audio_file"], quote=True)}"></audio></section>')
        notes = " · ".join(f"{n['lyric']} ({' '.join(n['raw_v4c_tokens'])})" for n in item["nearby_source_notes"])
        cards.append(f'<article id="{item["review_id"]}"><h2>{item["review_id"]} · 原唱 {item["original_cut_sec"]:.4f}s · 待具体回复</h2><p>预览中心 1.25 秒是切点。请说明在什么字的前／中／后；能分辨时再补充边缘音素。</p><p class="notes">附近已批准参考：{html.escape(notes)}</p><div class="previews">{"".join(rows)}</div></article>')
        items.append({**item, "previews": previews, "status": "pending", "human_response": None,
                      "source_phoneme_approval_preserved": True, "automated_timestamp_approval_inferred": False})
    content = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; media-src 'self' file:; style-src 'unsafe-inline'">
<title>沙龙六个切点：原唱与分离人声辅助复核</title><style>
body{margin:0;background:#f4f6f8;color:#202b36;font:16px/1.6 system-ui,'Segoe UI',sans-serif}header,main{max-width:1280px;margin:auto;padding:20px}header{padding-bottom:0}h1{font-size:27px}h2{font-size:20px;margin-top:0}h3{font-size:15px}article{background:white;border:1px solid #d7dfe7;border-radius:10px;padding:18px;margin:20px 0}.previews{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}img{display:block;width:100%;height:auto}audio{width:100%}.notes{color:#556373;font-size:14px}.warning{padding:14px;background:#fff4d8;border-left:4px solid #cf8a17}a{color:#174ea6}@media(max-width:850px){.previews{grid-template-columns:1fr}}
</style></head><body><header><h1>仍是六个切点，不重标全曲</h1><p>整曲音素和顺序的人工确认保持有效。每段 2.5 秒，切点在中间；点击播放可循环听。</p><p class="warning">以原始原唱为准。分离模型可能改变或削弱辅音，分离版仅作辅助；合成候选仍使用估计偏移。自动分析、页面播放和结构验证都不等于人工批准。</p></header><main>'''
    content += "".join(cards) + "</main></body></html>"
    with (destination / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(content)
    report = {"schema_version": "quality8h_boundary_review_v1", "previous_review_sha256": sha256_file(OLD / "review.json"),
              "separation_index_sha256": sha256_file(index_path), "items": items, "questions": 6,
              "audio_previews": 18, "total_preview_seconds": 45, "new_questions_added": 0,
              "all_human_responses_pending": True, "agent_listening_performed": False}
    write_json_new(destination / "review.json", report)
    with (destination / "REVIEW.md").open("x", encoding="utf-8") as stream:
        stream.write("# 六个切点的集中复核\n\n打开 [本地复核页面](index.html)。仍使用 B01–B06，不新增问题或要求重标全曲。\n\n保留原来的原唱／合成候选预览，另附六段分离人声辅助预览；共十八个音频、四十五秒素材。每段切点在 1.25 秒附近。\n\n以原始原唱为准；分离模型不能替代人工参考。所有具体切点回答仍待确认。\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_root.resolve(), args.output_dir.resolve()
    if not run.is_relative_to(ROOT / ".state/runs") or not output.is_relative_to(run) or output.exists():
        parser.error("review requires a new directory inside the current run")
    value = build(run, output)
    print(f"review_questions={value['questions']}; new_questions=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a consolidated, offline two-song structure review with existing media."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import html
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,verify_file


def build(run:Path)->None:
    audit=read_json(run/'WHOLE_SEQUENCE_AUDIT.json')
    reference_path=ROOT/'.state/runs/limited-data-20260906T050251/reference/reference-manifest.json'
    reference=read_json(reference_path)
    songs={s['song_id']:s for s in reference['songs'] if s['split']=='train'}
    destination=run/'whole-sequence-review';destination.mkdir()
    media=destination/'media';media.mkdir()
    items=[]
    names={'freedom-flower':'自由花','queen-road-east':'皇后大道東'}
    for report in audit['songs']:
        if not report['mapping_complete'] or report['machine_sequence_training_eligible']:continue
        song=songs[report['song_id']]
        bindings={role:{'path':song[role]['path'],'sha256':song[role]['sha256']}
                  for role in ('original_audio','rendered_audio','human_vsqx')}
        bindings['reference_manifest']={'path':str(reference_path),'sha256':sha256_file(reference_path)}
        assets={}
        for role in ('original_audio','rendered_audio'):
            source=Path(song[role]['path']);verify_file(source,song[role]['sha256'])
            target=media/(song['song_id']+'-'+role+'.wav')
            shutil.copyfile(source,target);verify_file(target,song[role]['sha256'])
            assets[role]={'file':target.relative_to(destination).as_posix(),'sha256':sha256_file(target),
                          'duration_sec':song[role]['duration_sec'],'bit_exact_source_copy':True}
        phrases=[];current=[]
        for note in song['notes']:
            if current and (note['start_sec']-current[-1]['end_sec']>.65 or len(current)>=24):
                phrases.append(current);current=[]
            current.append(note)
        if current:phrases.append(current)
        items.append({'review_id':'S'+str(len(items)+1).zfill(2),'song_id':song['song_id'],
            'display_name':names.get(song['song_id'],song['song_id']),'source_bindings':bindings,'assets':assets,
            'known_reviewed_windows':len(report['anchors']),'conflicts':report.get('clock_checks',[]),
            'phrases':[{'score_start_sec':p[0]['start_sec'],'score_end_sec':p[-1]['end_sec'],
                'text':''.join(str(n.get('raw_lyric') or '□') for n in p),
                'first_note_id':p[0]['note_id'],'last_note_id':p[-1]['note_id']} for p in phrases]})
    packet={'schema_version':'whole_sequence_review_packet_v1','items':items,
        'source_audit_sha256':sha256_file(run/'WHOLE_SEQUENCE_AUDIT.json'),
        'scope':'whole original versus calibrated VSQX sequence/order/repetition completeness; not fine timing',
        'model_predictions_or_PER_shown':False,'defaults_confirmed':False,'pending_export_allowed':True,
        'new_mapping_approval_requested':False,'starts_training':False}
    write_json_new(destination/'packet.json',packet)
    digest=sha256_file(destination/'packet.json')
    data=json.dumps({'packet_sha256':digest,'packet':packet},ensure_ascii=False).replace('<','\\u003c')
    cards=[]
    for item in items:
        rid=item['review_id'];name=html.escape(item['display_name'])
        audio=''.join(f'<label>{label}<audio id="{rid}-{role}" controls preload="metadata" src="{item["assets"][role]["file"]}"></audio></label>'
            for role,label in [('original_audio','原唱（原文件逐字节复制）'),('rendered_audio','Vocaloid 合成参考（辅助核对顺序）')])
        phrases=''.join(f'<tr><td>{p["score_start_sec"]:.2f}–{p["score_end_sec"]:.2f}</td><td>{html.escape(p["text"])}</td><td><button type="button" data-play="{rid}-rendered_audio" data-time="{p["score_start_sec"]}">听合成位置</button></td></tr>' for p in item['phrases'])
        anchors=''.join(f'<tr><td>{c["original_start_sec"]:.2f}–{c["original_end_sec"]:.2f}</td><td>{c["first_note_range"][0]:.2f}–{c["last_note_range"][1]:.2f}</td><td><button type="button" data-play="{rid}-original_audio" data-time="{c["original_start_sec"]}">定位原唱</button> <button type="button" data-play="{rid}-rendered_audio" data-time="{c["first_note_range"][0]}">定位合成</button></td></tr>' for c in item['conflicts'])
        binding=item['source_bindings']['human_vsqx']
        cards.append(f'''<article id="{rid}"><h2>{rid} · {name}</h2>
<p>本次只确认：这份原唱与已校准 VSQX 的整曲发音／歌词顺序、唱段重复次数是否一致。开头偏移、BPM、音符时长可以不同。</p>
<p class="note">自动匹配未能通过已有片段的核验。这是自动方法的限制或对应疑点，不是在宣布你的人工校准错误。已有 {item['known_reviewed_windows']} 个确认短窗仍保留。</p>
<div class="audio">{audio}</div>
<details><summary>查看绑定的文件版本</summary><p>{html.escape(Path(binding['path']).name)}</p><code>VSQX SHA-256: {binding['sha256']}</code><p>原唱 SHA-256: <code>{item['source_bindings']['original_audio']['sha256']}</code></p></details>
<details><summary>已有确认片段，供定位和回忆</summary><table><tr><th>原唱时间</th><th>已选 VSQX／合成范围</th><th>定位</th></tr>{anchors}</table></details>
<details><summary>展开 VSQX 全曲文字顺序</summary><p>文字逐音符展开；重复字、空字保持原记录。左列是 VSQX 时间，不是原唱时间。</p><table><tr><th>VSQX 时间（秒）</th><th>文字</th><th>辅助播放</th></tr>{phrases}</table></details>
<fieldset><legend>整曲对应结论</legend>
<label><input type="radio" name="{rid}-decision" value="sequence_confirmed">顺序与唱段重复次数一致，没有多段或少段</label>
<label><input type="radio" name="{rid}-decision" value="sequence_differs">存在多段、少段或唱段顺序差异</label>
<label><input type="radio" name="{rid}-decision" value="pending">暂不能确定</label></fieldset>
<label class="affirm"><input type="checkbox" id="{rid}-affirm">我确认的是这里绑定的原唱和 VSQX 版本，以及完整发音顺序和重复次数；这不包含逐音素时间边界确认。</label>
<label>补充原话／差异位置（可直接写歌词或大致时间，无须填写音符编号）<textarea id="{rid}-statement" rows="4"></textarea></label>
<p id="{rid}-status" class="status">待定，可随时导出。</p></article>''')
    script=(ROOT/'tools/whole_sequence_review_state.js').read_text(encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>整曲顺序集中核对</title><style>
body{font:18px/1.7 system-ui,sans-serif;background:#edf3f4;color:#173e45;max-width:1080px;margin:28px auto;padding:0 20px}article,header,footer{background:white;border:1px solid #bccfd2;border-radius:12px;padding:24px;margin:22px 0}h1,h2{line-height:1.3}label{display:block;margin:12px 0}audio{display:block;width:100%;margin:10px 0}.audio{display:grid;grid-template-columns:1fr 1fr;gap:20px}button{font:inherit;padding:8px 15px;border:1px solid #738f94;border-radius:7px;background:#f5fafb;cursor:pointer}button.primary{background:#155c65;color:white}textarea{width:100%;box-sizing:border-box;font:inherit;padding:10px;border:1px solid #8da6ab;border-radius:7px}input[type=radio],input[type=checkbox]{width:20px;height:20px;margin-right:10px}fieldset{border:1px solid #b5cbd0;border-radius:8px;margin-top:20px}details{margin:14px 0}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:16px}td,th{text-align:left;padding:8px;border-bottom:1px solid #d9e2e4}code{word-break:break-all;font-size:13px}.note{background:#eef6f7;padding:12px}.affirm{background:#eaf5ef;padding:12px}.status{font-weight:bold}footer{position:sticky;bottom:8px;box-shadow:0 2px 15px #2343}@media(max-width:750px){.audio{grid-template-columns:1fr}body{font-size:16px}article{padding:16px}table button{font-size:13px}}
</style></head><body><header><h1>整曲顺序集中核对 · 两首歌</h1><p>这次核对整段内容和重复顺序。你可以依据已经做过的校准直接作答，不需要重标音素。所有待定回答都能导出。</p><p>回答只保存在本机浏览器；需要点击导出，再把 JSON 发回任务。</p><label>复核人 <input id="reviewer" value="user"></label><button type="button" id="import">导入回答／草稿</button><input type="file" id="import-file" accept="application/json" hidden></header>'''+''.join(cards)+'''<footer><span id="progress"></span> <button type="button" class="primary" id="export">导出全部回答（允许待定）</button><p id="message" role="status"></p></footer><script id="packet" type="application/json">'''+data+'</script><script>'+script+'</script></body></html>'
    (destination/'index.html').write_text(page,encoding='utf-8')
    write_json_new(destination/'BUILD.json',{'packet_sha256':digest,'index_sha256':sha256_file(destination/'index.html'),
        'items':len(items),'media_copies_bit_exact':True,'model_predictions_shown':False,'browser_opened':False})
    print(str(destination/'index.html'),digest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    build(parser.parse_args().run.resolve())

"""Build one bounded listening batch for unresolved train-only error attribution."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse,hashlib,html,json,sys,wave
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,sha256_object,verify_file,write_json_new

NAMES={'freedom-flower':'自由花','queen-road-east':'皇后大道東','half-moon-night-song':'月半小夜曲','promise':'约定','like-you':'喜欢你'}


def candidates(run:Path,notes:dict) -> list[dict]:
    diagnosis=read_json(run/'ERROR_DIAGNOSIS.json')['errors'];bad={}
    for error in diagnosis:
        if error['split']!='train':continue
        key=(error['cache_id'],error['note_id']);bad.setdefault(key,[]).append(error)
    output=[];rows={}
    for (cache_id,note_id),errors in bad.items():
        if cache_id not in rows:rows[cache_id]=read_json(run/'examples'/f'{cache_id}.json')
        row=rows[cache_id];positions=[i for i,p in enumerate(row['source_phones']) if p['note_id']==note_id]
        if not positions or min(positions)==0 or max(positions)==len(row['source_phones'])-1:continue
        reference=[row['reference'][i] for i in positions]
        pred=[e['prediction'] for e in row['edit_operations']['candidate'] if e['reference_index'] in positions and e['prediction'] is not None]
        note=notes[note_id];lyric=note['raw_lyric']
        if reference==pred or not pred or any(x.startswith('<') for x in pred) or any(c in lyric for c in ('?','\ufffd')):continue
        if any(e['phone']=='h' and e['regression_from_teacher_match'] for e in errors):category='h_regression'
        elif any('\u02b0' in e['phone'] for e in errors):category='aspiration_conflict'
        elif any(e['phone'] in {'i','u','e'} and e['candidate_operation']=='deletion' and e['all_systems_disagree_with_reference'] for e in errors):category='vowel_proxy_conflict'
        else:continue
        note_ids=list(dict.fromkeys(p['note_id'] for p in row['source_phones']));ni=note_ids.index(note_id)
        context=[{'note_id':n,'lyric':notes[n]['raw_lyric'],'selected':n==note_id} for n in note_ids[max(0,ni-3):ni+4]]
        output.append({'cache_id':cache_id,'note_id':note_id,'song_id':row['song_id'],'category':category,'lyric':lyric,'context':context,
                       'reference_ipa':reference,'candidate_ipa':pred,'row':row,'positions':positions,'raw_v4c_tokens':note['raw_v4c_tokens']})
    return output


def select(pool:list[dict]) -> list[dict]:
    songs=Counter();groups=Counter();selected=[];used_clips=set();used_notes=set()
    while len(selected)<12:
        available=[x for x in pool if x['cache_id'] not in used_clips and x['note_id'] not in used_notes]
        if not available:break
        winner=min(available,key=lambda x:(songs[x['song_id']],groups[x['category']],list(NAMES).index(x['song_id']),x['cache_id'],x['positions'][0]))
        selected.append(winner);used_clips.add(winner['cache_id']);used_notes.add(winner['note_id']);songs[winner['song_id']]+=1;groups[winner['category']]+=1
    return selected


def copy_interval(source:Path,output:Path,start:float,end:float) -> dict:
    with wave.open(str(source),'rb') as src,wave.open(str(output),'wb') as dst:
        rate=src.getframerate();first,last=round(start*rate),round(end*rate)
        src.setpos(first);frames=src.readframes(last-first)
        if len(frames)!=(last-first)*src.getnchannels()*src.getsampwidth():raise ValueError('truncated review media')
        dst.setparams(src.getparams());dst.writeframes(frames)
    return {'file':output.name,'sha256':sha256_file(output),'source_start_sample':first,'source_end_sample_exclusive':last,'sample_rate':rate,'duration_sec':(last-first)/rate}


def render(packet:dict,output:Path):
    cards=[]
    for item in packet['items']:
        context=' '.join('<mark>'+html.escape(x['lyric'])+'</mark>' if x['selected'] else html.escape(x['lyric']) for x in item['context'])
        options=''.join(f'<label class="option"><input type="radio" name="{item["review_id"]}" value="{key}"> {label}</label>' for key,label in [
            ('a','A 更接近我听到的： /'+html.escape(' '.join(item['choices']['a']))+'/'),
            ('b','B 更接近我听到的： /'+html.escape(' '.join(item['choices']['b']))+'/'),
            ('neither','两种都不准确（可用下方原话说明）'),('uncertain','无法可靠区分，保留待定'),('wrong_location','声音／附近文字不对应，定位不对'),('pending','暂不回答')])
        cards.append(f'''<section class="question" id="{item['review_id']}"><h2>{item['review_id']} · {html.escape(item['song_name'])} ·「{html.escape(item['lyric'])}」</h2>
<p class="context">{context}</p><p>听已确认的完整训练短片段，判断标出的字。两种记录都可能有误；无需填写编号或音素索引。</p>
<audio controls preload="none" src="media/{item['media']['file']}"></audio><p class="minor">原文件 {item['start_sec']:.3f}–{item['end_sec']:.3f} 秒 · {item['media']['duration_sec']:.1f} 秒</p>
<fieldset><legend>哪种记录更接近原唱实际声音？</legend>{options}</fieldset>
<label>可选：你的原话或更合适的读法<textarea rows="2" placeholder="不需要写 IPA，直接描述也可以；原话原样保留。"></textarea></label></section>''')
    source=ROOT/'tools/phone_diagnostic_review';js=(source/'state.js').read_text(encoding='utf-8-sig')+'\n'+(source/'app.js').read_text(encoding='utf-8-sig')
    embedded=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>集中试听 · 音素差异</title>
<style>body{font:18px/1.7 system-ui,sans-serif;color:#153e43;background:#edf4f4;max-width:1020px;margin:0 auto;padding:24px}h1{font-size:32px}h2{font-size:24px}section,.intro{background:white;border:1px solid #d1e2e4;padding:24px;border-radius:16px;margin:24px 0}header{position:sticky;top:0;background:#edf4ffef;padding:12px;z-index:1;border-bottom:1px solid #afc5d0}button,.import{display:inline-block;padding:10px 18px;border:1px solid #15646b;border-radius:8px;background:#13616a;color:white;font-size:18px;cursor:pointer}#count{margin:8px 0}fieldset{border:0;padding:0;margin:18px 0}.option{display:block;padding:10px;border:1px solid #c5d5d7;border-radius:8px;margin:8px 0;cursor:pointer}.option:has(input:checked){background:#daf0ed;border-color:#17665e}input{margin-right:10px}textarea{display:block;box-sizing:border-box;width:100%;padding:12px;font:inherit;border:1px solid #b4c8ca;border-radius:8px}audio{width:100%}.context{font-size:27px;letter-spacing:.12em}mark{background:#ffedb5;padding:4px;border-radius:6px}.minor,#status{color:#496772;font-size:15px}.import input{display:none}a{color:#075e78}</style>
<h1>一次完成的集中试听</h1><div class="intro"><p>这次只核对少量存在分歧的字，区分“模型识别错误”与“Vocaloid 音素展开和原唱实际声音的差异”。不重做全曲校准，也不确认新的切点。</p><p>每题播放一段已经确认过的原唱，黄色标出目标字。A、B 顺序已打散；原唱只有一份，A/B 是两种音素记录，不是两段合成音频。若很难区分，请直接选“无法可靠区分”；若对应不对，直接选“定位不对”。</p><p><b>任意完成数量都可以导出。</b>未答和待定项不会阻止导出，也不会自动当作已确认参考。</p></div>
<header><button id="export" type="button">导出全部回答 JSON</button> <label class="import">导入之前的回答<input id="import" type="file" accept=".json,application/json"></label><p id="count"></p><span id="status">回答会自动保存到当前浏览器。</span></header>
'''+''.join(cards)+'<script id="packet" type="application/json">'+embedded+'</script><script>'+js.replace('</script','<\\/script')+'</script></html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def build(run:Path):
    manifest_path=ROOT/'.state/runs/limited-data-20260906T050251/reference/reference-manifest.json'
    verify_file(manifest_path,'120b45d2970e98d0bccba61d827004b72d50e531b739cfc52ac9f19bc10751c5')
    manifest=read_json(manifest_path);songs={s['song_id']:s for s in manifest['songs'] if s['split']=='train'}
    notes={n['note_id']:n for s in songs.values() for n in s['notes']};pool=candidates(run,notes);chosen=select(pool)
    if len(chosen)!=12 or len({x['song_id'] for x in chosen})!=5:raise ValueError('cannot assemble the predeclared balanced listening batch')
    output=run/'listening-review';output.mkdir(exist_ok=False);(output/'media').mkdir()
    for sid in {x['song_id'] for x in chosen}:
        verify_file(Path(songs[sid]['human_vsqx']['path']),songs[sid]['human_vsqx']['sha256'])
    items=[]
    for i,value in enumerate(chosen,1):
        row=value['row'];source=Path(row['audio_path']);verify_file(source,row['audio_sha256']);rid=f'D{i:02d}'
        media=copy_interval(source,output/'media'/f'{rid}.wav',row['start_sec'],row['end_sec'])
        flipped=int(hashlib.sha256(value['note_id'].encode()).hexdigest(),16)%2==1
        a,b=(value['candidate_ipa'],value['reference_ipa']) if flipped else (value['reference_ipa'],value['candidate_ipa'])
        items.append({'review_id':rid,'song_id':value['song_id'],'song_name':NAMES[value['song_id']],'note_id':value['note_id'],'lyric':value['lyric'],'context':value['context'],
                      'start_sec':row['start_sec'],'end_sec':row['end_sec'],'source_audio_path':str(source),'source_audio_sha256':row['audio_sha256'],
                      'source_example_id':row['example_id'],'source_example_sha256':sha256_file(run/'examples'/f"{value['cache_id']}.json"),'source_vsqx_sha256':songs[value['song_id']]['human_vsqx']['sha256'],
                      'category':value['category'],'choices':{'a':a,'b':b},'choice_sources':{'a':'candidate' if flipped else 'proxy_reference','b':'proxy_reference' if flipped else 'candidate'},
                      'raw_v4c_tokens':value['raw_v4c_tokens'],'media':media,'reference_update_allowed':False,'acoustic_phone_boundaries_confirmed':False})
    packet={'schema_version':'phone_diagnostic_review_packet_v1','items':items,'scope':'diagnostic_only_no_reference_or_training_update','only_training_songs':True,
            'policy_sha256':sha256_file(run/'POLICY.json'),'diagnosis_sha256':sha256_file(run/'ERROR_DIAGNOSIS.json'),'total_audio_seconds':sum(x['media']['duration_sec'] for x in items),
            'selection':'up to 12 distinct existing original training windows; balance song then conflict category; never recut references','counts_by_song':dict(Counter(x['song_id'] for x in items)),
            'counts_by_category':dict(Counter(x['category'] for x in items))}
    packet['packet_sha256']=sha256_object(packet);write_json_new(output/'packet.json',packet);render(packet,output)
    write_json_new(output/'BUILD.json',{'packet_sha256':packet['packet_sha256'],'page_sha256':sha256_file(output/'index.html'),'candidate_notes':len(pool),'items':len(items),'all_original_hashes_verified':True,'default_changed':False})
    print(json.dumps({'packet_sha256':packet['packet_sha256'],'items':len(items),'song_counts':packet['counts_by_song'],'category_counts':packet['counts_by_category']},ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run.resolve())

"""Score and report the three fixed systems only after independent final replay."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import html
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.final_proxy_evaluation import verify_release,score_predictions,SYSTEMS
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,sha256_object,verify_file
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer

NAMES={'pretrained':'预训练基线','old_seed43':'旧 Seed 43','frozen_guarded_candidate':'冻结 guarded 候选'}
SONGS={'ke-yi-ke-zai':'可一可再','king-of-karaoke':'K歌之王'}
GROUPS={'aspirated':'送气音','front_rounded':'前圆唇音','labiodental_f':'/f/','glottal_h':'/h/'}


def validated_predictions(root:Path,gate:dict,gate_sha:str,replay_path:Path)->dict:
    complete=read_json(root/'TEST_COMPLETE.json')
    if complete['gate_sha256']!=gate_sha:raise ValueError('completion release mismatch')
    evidence_path=Path(complete['evidence']['path']);verify_file(evidence_path,complete['evidence']['sha256'])
    model=read_json(evidence_path);replay=read_json(replay_path)
    if (replay.get('schema_version')!='final_proxy_independent_replay_v1' or replay.get('status')!='complete' or
        replay.get('gate_sha256')!=gate_sha or replay.get('independent_replay_source')!=complete['evidence']):
        raise ValueError('independent replay must cover this exact completed evaluation')
    ids=gate['included_example_ids']
    if model.get('gate_sha256')!=gate_sha or model.get('systems')!=SYSTEMS or model.get('status')!='complete':
        raise ValueError('wrong or incomplete final systems')
    if len(model['entries'])!=len(ids) or len(replay['entries'])!=len(ids) or {e['example_id'] for e in replay['entries']}!=set(ids):
        raise ValueError('independent replay coverage changed')
    if model.get('frozen_parameters_unchanged') is not True or replay.get('frozen_parameters_unchanged') is not True:
        raise ValueError('frozen model verification missing')
    vocab=read_json(Path(gate['bindings']['vocabulary']['path']))
    predictions={s:{} for s in SYSTEMS}
    for entry in model['entries']:
        eid=entry['example_id'];checked=next(r for r in replay['entries'] if r['example_id']==eid)
        if len(entry['systems'])!=3 or {r['system'] for r in entry['systems']}!=set(SYSTEMS):raise ValueError('system coverage missing')
        for record in entry['systems']:
            other=next(r for r in checked['systems'] if r['system']==record['system'])
            if any(other.get(k) is not True for k in ('logits_exact','prediction_equal','real_fixture_equal')):
                raise ValueError('independent checkpoint replay failed')
            if other['contract_sha256']!=record['contract_sha256']:raise ValueError('replay posterior contract differs')
            for key in ('evidence','posterior'):verify_file(Path(record[key+'_path']),record[key+'_sha256'])
            posterior=read_json(Path(record['posterior_path']))
            if posterior['evaluation_release']['sha256']!=gate_sha or posterior['split']!='test' or not posterior['final_evaluation_only']:
                raise ValueError('posterior not from authorized final release')
            PhoneticXeusPosteriorRecognizer._verify_raw_projection(posterior,Path(record['evidence_path']),vocab)
            actual=posterior['decoding']['collapsed_raw_labels']
            if record['prediction']!=actual:raise ValueError('prediction does not match raw evidence')
            if eid in predictions[record['system']]:raise ValueError('duplicate final prediction')
            predictions[record['system']][eid]=actual
    return predictions


def delta(old:float|None,new:float|None)->float|None:
    return (old-new)/old if old is not None and old>0 and new is not None else None


def decorate(result:dict,gate:dict)->None:
    candidate=result['systems']['frozen_guarded_candidate']['overall']
    comparisons={}
    for baseline in ['pretrained','old_seed43']:
        base=result['systems'][baseline]['overall']
        comparisons[baseline]=dict(relative_per_reduction=delta(base['per'],candidate['per']),
            absolute_per_change=candidate['per']-base['per'],edit_count_change=candidate['edit_distance']-base['edit_distance'],
            substitutions_change=candidate['substitutions']-base['substitutions'],
            deletions_change=candidate['deletions']-base['deletions'],insertions_change=candidate['insertions']-base['insertions'],
            group_comparison={name:dict(reference_count=g['reference_count'],baseline_per=g['per'],candidate_per=candidate['groups'][name]['per'],
                enough_samples=g['reference_count']>=20,
                regression_over_003=(candidate['groups'][name]['per']>g['per']+.03 if g['reference_count']>=20 else None)) for name,g in base['groups'].items()})
    result['candidate_comparisons']=comparisons
    result['frozen_coverage']=gate['coverage']
    result['final_test_is_now_opened']=True
    result['future_training_on_test_or_reselection_prohibited']=True
    result['generalization_evidence_limit']='two previously held-out songs, six fixed clips, explicitly approved engine-derived proxy reference'


def write_reports(root:Path,result:dict)->None:
    lines=['# 冻结候选的最终代理参考评测','',
        '本次使用两首此前保留歌曲的原始混音，六个固定 18 秒窗口；对照用户批准的工程 IPA 代理参考。该 PER 衡量对这些代理单位的差异，不能换算为精确原唱音素准确率。','',
        '| 系统 | PER ↓ | 替换 | 删除 | 插入 | 错误总数 |',
        '|---|---:|---:|---:|---:|---:|']
    for system in SYSTEMS:
        m=result['systems'][system]['overall'];lines.append(f"| {NAMES[system]} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {m['edit_distance']} |")
    coverage=result['frozen_coverage']
    lines += ['',f"共同分母：{coverage['reference_phone_count']} 个代理音素，{coverage['included_windows']}/6 个窗口，{coverage['included_audio_seconds']:g}/108 秒。明确 absent {coverage['absent_phones']} 个、partial {coverage['partial_phones']} 个；partial 计一次。",'',
              '## 逐歌曲与逐窗口','', '| 片段 | 歌曲 | 参考数 | 预训练 PER | 旧 Seed43 PER | 候选 PER |','|---|---|---:|---:|---:|---:|']
    first=result['systems'][SYSTEMS[0]]['examples']
    for row in first:
        values=[next(r for r in result['systems'][s]['examples'] if r['example_id']==row['example_id'])['metrics']['per'] for s in SYSTEMS]
        lines.append(f"| {row['window_id']} | {SONGS[row['song_id']]} | {row['metrics']['reference_phones']} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
    for song in SONGS:
        if song in result['systems'][SYSTEMS[0]]['by_song']:
            ms=[result['systems'][s]['by_song'][song] for s in SYSTEMS]
            lines.append(f"| 合计 | {SONGS[song]} | {ms[0]['reference_phones']} | "+' | '.join(f"{m['per']:.6f}" for m in ms)+' |')
    lines += ['','## 关键音类','', '| 音类 | 参考数 | 预训练 PER | 旧 Seed43 PER | 候选 PER | 样本是否足够 |','|---|---:|---:|---:|---:|---|']
    for group,label in GROUPS.items():
        ms=[result['systems'][s]['overall']['groups'][group] for s in SYSTEMS]
        values=['—' if m['per'] is None else f"{m['per']:.6f}" for m in ms]
        lines.append(f"| {label} | {ms[0]['reference_count']} | "+' | '.join(values)+(' | ≥20 |' if ms[0]['reference_count']>=20 else ' | 不足20 |'))
    lines += ['','## 限制与复现','',
        '- 只比较预先冻结的三个系统，没有根据最终测试换权重、改映射、移动窗口或删除难点。没有再次训练，默认模型未替换。',
        '- 主干特征和预训练 logits 直接复用原冻结缓存；残差层在 CUDA / FP32 离线重放，CPU 四线程。',
        '- 18 份 posterior 均通过 real/fixture 比较；独立进程对三个系统逐窗口重放并核对原始 logits、预测和输出契约。',
        '- CTC 发射时间只作诊断，不当作音素持续时间。只有两首测试歌，不能据此声称广泛泛化。',
        '- 完整替换/删除/插入混淆及逐窗口序列见 COMPARISON.json。冻结条件见 RELEASE.json、POLICY.json、COVERAGE.json。',
        '- 这些最终测试结果现在已被查看；后续不得在其上调参再宣称独立最终验证。','']
    text='\n'.join(lines)
    with (root/'RESULTS.md').open('x',encoding='utf-8') as f:f.write(text)
    esc=html.escape
    table='<table><thead><tr><th>系统</th><th>代理 PER ↓</th><th>替换</th><th>删除</th><th>插入</th></tr></thead><tbody>'
    for system in SYSTEMS:
        m=result['systems'][system]['overall'];table+='<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in [NAMES[system],f"{m['per']:.6f}",m['substitutions'],m['deletions'],m['insertions']])+'</tr>'
    table+='</tbody></table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最终代理评测结果</title><style>body{font:17px/1.65 system-ui,sans-serif;background:#edf3f3;color:#20333c;margin:0}main{max-width:1080px;margin:30px auto;background:white;border-radius:14px;padding:28px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #dde6e6}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.7 system-ui,sans-serif}.notice{padding:18px;background:#e6f3f0}</style><main><h1>冻结候选：最终代理参考评测</h1><p class="notice">同一原始混音、同一窗口、同一代理参考；没有自动部署。此页无需重新审计。</p>'+table+'<p>共同分母 '+str(coverage['reference_phone_count'])+' 个代理音素。不能把 1−PER 称为原唱准确率。</p><details open><summary>逐窗口、关键音类与验证</summary><pre>'+esc(text)+'</pre></details></main></html>'
    with (root/'index.html').open('x',encoding='utf-8') as f:f.write(page)


def run(gate_path:Path,gate_sha:str,replay_path:Path,output:Path,verify:Path|None=None)->dict:
    gate,_,_=verify_release(gate_path,gate_sha)
    predictions=validated_predictions(gate_path.parent,gate,gate_sha,replay_path)
    reference=read_json(Path(gate['bindings']['reference']['path']))
    result=score_predictions(reference,predictions);decorate(result,gate)
    result['gate_sha256']=gate_sha
    result['source_model_evidence']=read_json(gate_path.parent/'TEST_COMPLETE.json')['evidence']
    result['independent_replay']={'path':str(replay_path.resolve()),'sha256':sha256_file(replay_path)}
    if verify:
        if result!=read_json(verify):raise ValueError('independent metric recomputation differs')
        write_json_new(output/'METRIC_REPLAY.json',dict(identical=True,comparison_sha256=sha256_file(verify),all_three_systems=True,
            all_window_and_song_metrics=True,recomputed_utc=datetime.now(timezone.utc).isoformat()))
    else:
        write_json_new(output/'COMPARISON.json',result);write_reports(output,result)
    for system in SYSTEMS:print(system,result['systems'][system]['overall']['per'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate',required=True,type=Path);p.add_argument('--gate-sha256',required=True)
    p.add_argument('--replay',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--verify',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'.state/runs'):raise ValueError('workspace run output required')
    out.mkdir(parents=True,exist_ok=True)
    run(a.gate.resolve(),a.gate_sha256,a.replay.resolve(),out,a.verify.resolve() if a.verify else None)

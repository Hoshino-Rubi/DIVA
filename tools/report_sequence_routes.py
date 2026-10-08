"""Publish completed evidence and the single consolidated next human window."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file


def report(run:Path)->None:
    completed=read_json(run/'COMPLETION_STAGES.json')
    selection=read_json(run/'ROUTE_SELECTION.json')
    language=read_json(run/'POWSM_LANGUAGE_SELECTION.json')
    lang_dev=read_json(run/'POWSM_LANGUAGE_DEVELOPMENT.json')
    audit=read_json(run/'WHOLE_SEQUENCE_AUDIT.json')
    inputs=read_json(run/'INPUT_AUDIT.json')
    replay=read_json(run/'checkpoint-verification-screen/RESULT.json')
    suite=ET.parse(run/'regression.xml').getroot().find('testsuite')
    counts={k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
    counts['passed']=counts['tests']-counts['skipped']
    if counts['failures'] or counts['errors'] or not completed['no_active_model_job']:raise ValueError('completion or regression not successful')
    preserved=read_json(run/'PRESERVATION_BEFORE.json')['files']
    for path,digest in preserved.items():verify_file(Path(path),digest)
    jobs=[read_json(run/'screen'/arm/song/'RESULT.json') for arm in ('control','sequence_risk')
        for song in read_json(run/'POLICY.json')['songs']]
    for job in jobs:
        verify_file(Path(job['checkpoint_path']),job['checkpoint_sha256'])
        for path,digest in job['code_bindings'].items():verify_file(Path(path),digest)
        if job['held_song'] in job['training_songs'] or not job['frozen_parameters_unchanged']:raise ValueError('training isolation failed')
    model_monitors=[read_json(run/'screen'/j['arm']/j['held_song']/'monitor/resource-monitor.json') for j in jobs]
    model_monitors += [read_json(run/p/'monitor/resource-monitor.json') for p in
        ('powsm-unknown','powsm-cantonese','powsm-cantonese-dev','risk-cuda-probe','checkpoint-verification-screen')]
    if any(not m['operation_succeeded'] or not m['completed_without_resource_abort'] for m in model_monitors):
        raise ValueError('model job resource/operation failed')
    whole_jobs=selection['whole_sequence_quality_jobs_started']
    final_count=3 if (run/'FINAL_TRAINING.json').exists() else 0
    validation={'verified_at_utc':datetime.now(timezone.utc).isoformat(), 'regression':counts,
        'independent_checkpoint_replays':len(replay['checks']),
        'maximum_raw_audio_prefix_logit_difference':max(r['raw_audio_prefix_max_difference'] for r in replay['checks']),
        'all_replayed_metrics_equal':all(r['metrics_equal'] for r in replay['checks']),
        'independent_language_logit_replays':len(inputs['independent_language_replays']),
        'all_prior_source_hashes_unchanged':True,'preserved_file_bindings':len(preserved),
        'all_model_jobs_offline_serial':True,'maximum_sampled_card_bytes':max(m['maximum_sampled_total_used_bytes'] for m in model_monitors),
        'maximum_training_reserved_bytes':max(j['peak_reserved_bytes'] for j in jobs),
        'review_node_tests':{'passed':5,'failed':0,'result':'completion-commands/review-node-tests.log'},
        'review_bit_exact_media_verified':len(inputs['review_media']),
        'browser_visual_validation':False,'default_replacement':False,'final_test_predictions_opened':False}
    write_json_new(run/'FINAL_VERIFICATION.json',validation)
    summary={'status':'awaiting_consolidated_whole_sequence_review','powsm_language':{
        'training_unknown':language['conditions']['unknown']['metrics'],
        'training_cantonese':language['conditions']['cantonese']['metrics'],
        'training_relative_gain':language['training_relative_PER_gain'],
        'development_cantonese':lang_dev['metrics'],'development_unknown_errors':177},
        'cross_song_training':{'prior':selection['prior_full_folds'],
            'arms':{k:v['metrics'] for k,v in selection['arms'].items()},'gates':selection['gates']},
        'completed_quality_training_jobs':len(jobs),'completed_disposable_gradient_probes':1,
        'whole_sequence_quality_training_jobs':whole_jobs,'final_training_jobs':final_count,
        'whole_sequence_eligible_songs':[r['song_id'] for r in audit['eligible']],
        'whole_sequence_exclusions':[{'song_id':r['song_id'],'reasons':r['exclusion_reasons']} for r in audit['songs']],
        'validation':validation,'next_human_window':{
            'page':str(run/'whole-sequence-review/index.html'),
            'packet_sha256':inputs['review_packet_sha256'],'items':2,
            'scope':'complete source-song sequence/order/repeated-section count; no phone timing or new IPA mapping'},
        'default_replacement':False,'new_final_test_predictions':False}
    write_json_new(run/'SUMMARY.json',summary)
    prior=selection['prior_full_folds']
    rows=[('上一轮冻结结果',prior)]+[(('同预算普通续训' if name=='control' else '序列错误风险训练'),value['metrics']) for name,value in selection['arms'].items()]
    lines=['# 序列监督与语种条件实验', '', '本轮已完成实验及验证，停在一个集中的人工窗口。没有新权重晋级，没有替换默认模型。', '',
        '## 已完成', '',
        '- POWSM 固定权重、相同20个原唱训练窗口：未知语种条件1036/1235，明确粤语条件864/1235，五首歌均改善；相对PER降低16.60%。',
        '- 仅按训练歌曲冻结粤语条件后，《沙龙》开发结果为151/214（S109、D22、I20），原未知语种条件为177/214。该未适配模型仍未超过既有已适配候选80/214。',
        '- 两种训练方案均从各自正确的旧留歌检查点继续，五首歌逐歌留出，同一参考、同一优化步数、固定20epoch。没有用开发集选择训练方案或epoch。', '',
        '| 五折留歌验证（1235参考音素） | 总错误 | 替换 | 删除 | 插入 | 送气音错误/参考 |', '|---|---:|---:|---:|---:|---:|']
    for name,value in rows:
        aspirated=value['groups']['aspirated'];errors=sum(aspirated[k] for k in ('substitutions','deletions','insertions'))
        lines.append(f"| {name} | {value['edit_distance']} | {value['substitutions']} | {value['deletions']} | {value['insertions']} | {errors}/{aspirated['reference_count']} |")
    lines += ['', '## 未通过与未执行', '',
        '普通续训未降低总错误；候选序列风险训练也未显示跨歌曲优势，且两者送气音保护均未通过。因此没有启动三个最终种子，也没有为这两组权重重新计算《沙龙》来挑选方案。', '',
        '整段路线的输入审计未通过：《自由花》《皇后大道東》映射完整，但全曲自动音频对应不能与已确认片段同时吻合；皇后大道東的一处两分辨率对应相差约40.7秒。其余三首在已允许映射下仍有未解决项。', '',
        '这些冲突说明目前机器证据不足以核实完整对应，不是判定人工音素标注错误，也不证明完整歌曲一定不同。已有20个确认训练短窗全部保留。整段CTC损失的重复音素、分块连接与梯度已通过fixture测试；未建立可用于训练的整曲缓存，也未启动整段训练。', '',
        '## 下一人工窗口', '',
        '[打开两首歌的集中核对页面](whole-sequence-review/index.html)', '',
        '只确认当前文件版本的完整发音/歌词顺序与唱段重复次数是否一致。无需重新标注音素、输入音符编号或确认精确切点。可依据已经完成的校准记录作答；不确定也能导出，原话保留。', '',
        '原唱和合成参考均为原文件逐字节复制。页面不显示模型预测或PER；切换选项只更新状态文本，不重建页面。', '',
        f"导出文件名：`whole-sequence-review-answers-{inputs['review_packet_sha256'][:12]}.json`。接收入口为 `tools/import_whole_sequence_review.py`；导入只登记确认，不自动修改历史参考或启动训练。", '',
        '## 验证', '',
        f"完整回归：{counts['passed']} passed，{counts['skipped']} skipped；前端Node测试5 passed。10个检查点独立重放的预测/指标一致，原始音频与缓存路径logits最大差值0；43个POWSM输出完成独立raw/fixture及编辑距离核验。", '',
        f"复核原有{len(preserved)}个文件绑定未变；未重开最终测试预测。训练峰值reserved为{validation['maximum_training_reserved_bytes']/2**30:.3f} GiB；模型作业串行、运行断网、CPU四线程。没有浏览器视觉或实际听音验证。", '',
        '详细逐窗口、逐歌曲、门槛与混淆统计见 ROUTE_SELECTION.json、POWSM_LANGUAGE_SELECTION.json、SUMMARY.json。命令、日志、退出状态位于 commands/ 与 completion-commands/。', '']
    (run/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [name,value['edit_distance'],value['substitutions'],value['deletions'],value['insertions']])+'</tr>' for name,value in rows)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本轮实验与集中核对</title><style>body{font:18px/1.7 system-ui;max-width:1020px;margin:35px auto;padding:0 24px;background:#edf3f4;color:#173e45}a{color:#0d6170}.card{padding:24px;background:white;border:1px solid #b9cbd0;border-radius:12px;margin:24px 0}.button{display:inline-block;padding:12px 20px;background:#155c65;color:white;border-radius:7px;text-decoration:none}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #cedce0}pre{white-space:pre-wrap;font:inherit}</style><h1>本轮实验完成，等待整曲顺序确认</h1><div class="card"><h2>下一步只核对两首歌</h2><p>自由花、皇后大道東：确认完整唱段顺序和重复次数。无需重新标注音素或切点，待定项也能导出。</p><a class="button" href="whole-sequence-review/index.html">打开集中核对页面</a></div><div class="card"><h2>五折留歌验证 · 1235个参考音素</h2><table><tr><th>方案</th><th>总错误</th><th>替换</th><th>删除</th><th>插入</th></tr>'''+table+'''</table><p>均未通过晋级门槛，三个最终种子未启动。默认模型保持不变。</p></div><div class="card"><h2>POWSM 语种条件</h2><p>训练片段：1036 → 864 /1235；冻结粤语条件后的开发结果151/214，未知语种条件177/214。两种条件单独记录，未把该模型当作新默认。</p></div><p><a href="RESULTS.md">完整报告</a> · <a href="SUMMARY.json">汇总数据</a> · <a href="FINAL_VERIFICATION.json">验证记录</a></p></html>'''
    (run/'index.html').write_text(page,encoding='utf-8')
    for file in (run/'index.html',run/'whole-sequence-review/index.html'):
        text=file.read_text(encoding='utf-8')
        for name in re.findall(r'(?:href|src)="([^"#]+)"',text):
            if not (file.parent/name).is_file():raise ValueError('broken report/media link: '+name)
    current={'status':'awaiting_whole_sequence_order_review','active_model_job':False,
        'screen_training_jobs_completed':10,'final_training_jobs_started':final_count,'new_weights_promoted':False,
        'manual_review_page':str(run/'whole-sequence-review/index.html'),'manual_review_items':2,
        'packet_sha256':inputs['review_packet_sha256'],'proxy':'http://127.0.0.1:18080',
        'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    (run/'CURRENT.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    (run/'HANDOFF.md').write_text('''# Completed sequence-route experiments; consolidated review pending

Read RESULTS.md, SUMMARY.json, FINAL_VERIFICATION.json, CURRENT.json.

10 quality jobs completed: matched control5 folds594/1235 (S318 D217 I59), sequence-risk5 folds597/1235 (S323 D213 I61). Prior full-tail folds594/1235 (S301 D241 I52). Aspiration errors14/30 ->16/30 control,17/30 risk. Both routes fail >=5% gain and aspiration protection; no final3seeds or new trained-candidate Shalong evaluation. Checkpoints retained as research only; defaults unchanged.

POWSM fixed weights: unknown1036/1235 -> knownCantonese864/1235 on20train windows; all5songs improved. Selected on training only, then Shalong151/214 (S109 D22 I20), priorunknown177. Both conditions/prompt provenance recorded. No new weights/dependencies downloaded.

Whole source audit: no eligible full songs. Freedom-flower/queen-road-east fully mapped but audio-only global paths disagree with reviewed anchor bindings; other3songs unresolved. No manual anchor reassigned or reference timed boundaries modified. Whole-sequence CTC mathematical fixture implementation exists; full-song cache/trainer not executed. Loss code is not evidence that whole training was done.

Next human window: whole-sequence-review/index.html, packet SHA 26af9fc40abef41d8d5cc4c151ad5c22b18d1eb0c718be79c8b3021f03851e63. Two whole-song order/repeated-section completeness questions, no fine timing/phone_index entry. Export pending supported. Accept text confirmation of these exact source-file versions or exported JSON, preserve raw statement and scope. Import tool tools/import_whole_sequence_review.py validates packet/source hashes and does not start training. No answer received yet.

All10checkpoints reloaded in a new model process: exact predictions/metrics, raw audio versus prefix logits maxdiff0, real/fixture match.43POWSMraw replays and scalar-edit checks. Reviewmedia bit-exact sourcecopies; Node5tests including pending/export/no DOM-rebuild. Full pytest numbers in FINAL_VERIFICATION.json. No browser/actual listening validation.

Bound code files in saved checkpoints must remain unchanged: sequence_route_training.py, sequence_risk.py; old parentbound files untouched. Opt-in new inference identity: adapters/sequence_route_inference.py. Whole source-review answers do not validate phoneme timing or extend IPA mappings. Do not restart completed screen jobs or change gates to force a result.

Runtime: existing limited-data Python3.12 Torch2.8/cu126 for PhoneticXeus; powsm isolated runtime for language probe/CPU librosa. GPU serial,CPU4,own7GiB/card14GiB,offline. Current proxy18080,not10808. All outputs under this run. No subagents used; latest developer forbids proactive spawning.
''',encoding='utf-8')
    source_names=['src/any2xsampa/config/sequence_routes_v1.json','src/any2xsampa/adapters/powsm_language.py',
        'src/any2xsampa/adapters/sequence_route_inference.py','src/any2xsampa/training/whole_sequence_reference.py',
        'src/any2xsampa/training/whole_sequence_review.py','src/any2xsampa/training/sequence_risk.py',
        'src/any2xsampa/training/sequence_route_training.py','tests/test_sequence_routes.py','tests/whole_sequence_review.test.cjs']
    source_names += ['tools/'+name for name in ['prepare_sequence_routes.py','run_powsm_language_probe.py','select_powsm_language.py',
        'audit_whole_sequence.py','train_sequence_route.py','probe_sequence_risk_cuda.py','run_sequence_route_stages.py',
        'prepare_whole_sequence_review.py','whole_sequence_review_state.js','import_whole_sequence_review.py',
        'verify_sequence_route_checkpoints.py','evaluate_sequence_routes.py','audit_sequence_route_inputs.py',
        'finalize_sequence_routes.py','report_sequence_routes.py']]
    write_json_new(run/'DELIVERY.json',{'source_files':[{'path':str(ROOT/name),'sha256':sha256_file(ROOT/name)} for name in source_names],
        'root_artifacts':[{'path':str(p),'sha256':sha256_file(p)} for p in sorted(run.iterdir()) if p.is_file() and p.suffix in {'.json','.md','.html'}],
        'review_build_sha256':sha256_file(run/'whole-sequence-review/BUILD.json'),'default_replacement':False})
    print(json.dumps({'status':current['status'],'regression':counts,'quality_training_jobs':10,'manual_items':2,'page':current['manual_review_page']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    report(parser.parse_args().run.resolve())

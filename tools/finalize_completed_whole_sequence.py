"""Finalize the resumed whole-source experiment, leaving the blocked snapshot intact."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,html,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file

NAMES={'prior':'上轮适配模型','control':'普通续训对照','sequence_risk':'序列风险续训','whole_sequence':'整曲序列监督续训'}
SONGS={'freedom-flower':'自由花','half-moon-night-song':'月半小夜曲','like-you':'喜欢你','promise':'约定','queen-road-east':'皇后大道東'}
GROUPS={'aspirated':'送气音','front_rounded':'前圆唇音','labiodental_f':'f','glottal_h':'h'}


def text_new(path:Path,value:str)->None:
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(value)


def finalize(run:Path)->None:
    output=run/'completion';analysis=read_json(output/'ERROR_ANALYSIS.json')
    selection=read_json(run/'ROUTE_SELECTION.json');spec=read_json(run/'POLICY.json')
    replay=read_json(run/'checkpoint-verification-screen/RESULT.json')
    cache=read_json(output/'CACHE_AUDIT.json');code=read_json(run/'CODE_FREEZE.json')
    recovered=(run/'SCREEN_JOB_INDEX.json').exists()
    index=read_json(run/'SCREEN_JOB_INDEX.json') if recovered else None
    screen_dirs=[Path(r['job_path']) for r in index['rows']] if index else [run/'screen/whole_sequence'/song for song in spec['songs']]
    if len(replay['checks'])!=5 or not all(r['metrics_equal'] and r['real_fixture_equal'] and r['raw_audio_prefix_max_difference']<=2e-5 for r in replay['checks']):
        raise ValueError('independent whole checkpoint replay is incomplete')
    for path,digest in code['files'].items():verify_file(Path(path),digest)
    if recovered:
        for path,digest in read_json(run/'NUMERICAL_FIX_FREEZE.json')['files'].items():verify_file(Path(path),digest)
    preserved=read_json(run/'PRESERVATION_BEFORE.json')['files']
    for path,digest in preserved.items():verify_file(Path(path),digest)
    previous_delivery=read_json(run/'DELIVERY.json')
    for row in previous_delivery['source_files']+previous_delivery['root_artifacts']:
        verify_file(Path(row['path']),row['sha256'])
    regression_acceptance=read_json(output/'REGRESSION_ACCEPTED.json')
    verify_file(Path(regression_acceptance['junit_path']),regression_acceptance['junit_sha256'])
    verify_file(Path(regression_acceptance['run_path']),regression_acceptance['run_sha256'])
    suite=next(ET.parse(regression_acceptance['junit_path']).iter('testsuite'))
    stats={k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')}
    stats['passed']=stats['tests']-stats['skipped']-stats['errors']-stats['failures']
    stats['seconds']=float(suite.attrib['time'])
    stats['accepted_attempt']=regression_acceptance['attempt']
    stats['junit_path']=regression_acceptance['junit_path']
    stats['junit_sha256']=regression_acceptance['junit_sha256']
    stats['skips']=[{'test':t.attrib.get('classname','')+'.'+t.attrib.get('name',''),
        'reason':t.find('skipped').attrib.get('message','')} for t in suite.iter('testcase') if t.find('skipped') is not None]
    required_gpu={'test_cuda_training_checkpoint_and_reload','test_ranker_cuda_epoch_checkpoint_and_reload_are_deterministic'}
    gpu_cases=[t for t in suite.iter('testcase') if t.attrib.get('name') in required_gpu]
    if len(gpu_cases)!=2 or any(t.find(tag) is not None for t in gpu_cases for tag in ('skipped','failure','error')):
        raise ValueError('restored CUDA acceptance tests did not both pass')
    stats['gpu_acceptance_tests_passed']=sorted(required_gpu)
    if stats['failures'] or stats['errors']:raise ValueError('full regression failed')
    final_count=0;development=None;transfer=None
    if selection['passed']:
        final=read_json(run/'FINAL_TRAINING.json');final_count=len(final['seeds'])
        development=read_json(run/'DEVELOPMENT_SELECTION.json');transfer=read_json(run/'VOCAL_TRANSFER.json')
        final_replay=read_json(run/'checkpoint-verification-final/RESULT.json')
        if final_count!=3 or len(final_replay['checks'])!=3:raise ValueError('passing route requires all final seeds and replay')
    else:
        reason=read_json(run/'FINAL_TRAINING_NOT_STARTED.json')
        if reason['new_final_seeds']!=0 or (run/'FINAL_TRAINING.json').exists():
            raise ValueError('final training started despite a failed screening gate')
    smoke=read_json(run/'SMOKE_ACCEPTED.json');verify_file(Path(smoke['result_path']),smoke['result_sha256'])
    smoke_result=read_json(Path(smoke['result_path']));probe=read_json(run/'whole-cuda-probe/RESULT.json')
    if not probe['gradients_finite'] or not probe['frozen_parameters_unchanged'] or probe['checkpoint_saved']:
        raise ValueError('disposable whole-gradient probe contract changed')
    monitors=[];maximum_total=0;quality_jobs=[]
    directories=[Path(smoke['result_path']).parent,run/'whole-cache',run/'whole-cuda-probe',run/'checkpoint-verification-screen']
    directories += screen_dirs
    if recovered:directories += [run/'numerical-diagnosis',run/'whole-cuda-probe-v2']
    if final_count:
        directories += [run/'final/whole_sequence'/('seed'+str(seed)) for seed in spec['training']['final_seeds']]
        directories += [run/'checkpoint-verification-final',run/'development']
    for directory in directories:
        monitor=read_json(directory/'monitor/resource-monitor.json')
        if not monitor['completed_without_resource_abort'] or not monitor['operation_succeeded']:
            raise ValueError('resumed model stage has an unacknowledged runtime failure')
        maximum_total=max(maximum_total,monitor['maximum_sampled_total_used_bytes'])
        monitors.append({'stage':str(directory.relative_to(run)),
            'peak_sampled_card_bytes':monitor['maximum_sampled_total_used_bytes'],
            'seconds':monitor['elapsed_seconds']})
    failed_attempts=[]
    if recovered:
        for failed in index['failed_attempts']:
            monitor=read_json(Path(failed['path'])/'monitor/resource-monitor.json')
            failed_attempts.append({**failed,'monitor':monitor})
            maximum_total=max(maximum_total,monitor['maximum_sampled_total_used_bytes'])
    for directory in screen_dirs+([run/'final/whole_sequence'/('seed'+str(seed)) for seed in spec['training']['final_seeds']] if final_count else []):
        result=read_json(directory/'RESULT.json');verify_file(Path(result['checkpoint_path']),result['checkpoint_sha256'])
        quality_jobs.append({'directory':str(directory),'checkpoint_sha256':result['checkpoint_sha256'],
            'seconds':result['elapsed_seconds'],'peak_reserved_bytes':result['peak_reserved_bytes']})
    own_peak=max([smoke_result['peak_reserved_bytes'],read_json(run/'WHOLE_CACHE.json')['peak_reserved_bytes'],
        probe['peak_reserved_bytes'],replay['peak_reserved_bytes']]+[r['peak_reserved_bytes'] for r in quality_jobs])
    if final_count:
        own_peak=max(own_peak,final_replay['peak_reserved_bytes'],transfer['peak_reserved_bytes'])
    if recovered:
        own_peak=max(own_peak,read_json(run/'numerical-diagnosis/RESULT.json')['peak_reserved_bytes'],
            read_json(run/'whole-cuda-probe-v2/RESULT.json')['peak_reserved_bytes'])
    if own_peak>7*2**30 or maximum_total>14*2**30:raise ValueError('recorded resource limit was exceeded')
    commands=[]
    for path in sorted((run/'commands').glob('*.exit.json')):
        result=read_json(path);commands.append({'name':path.name.removesuffix('.exit.json'),**result})
    resumed=next(r for r in commands if r['name']==('recover-screen-v2' if recovered else 'resume-after-resource-release'))
    if resumed['exit_code']!=0:raise ValueError('resumption did not finish cleanly')
    statuses={'reference_confirmation':'completed','real_inference_and_cache':'completed','whole_gradient_probe':'completed',
        'five_fold_training':'completed','independent_checkpoint_replay':'completed',
        'screening_gate':'passed' if selection['passed'] else 'not_passed',
        'final_three_seeds':'completed' if final_count else 'not_executed_gate_failed',
        'development_evaluation':'completed' if final_count else 'not_executed_gate_failed',
        'full_regression':'completed','additional_human_annotation':'not_required'}
    summary={'schema_version':'whole_sequence_completed_v1','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'completed','stages':statuses,'screen_quality_jobs':5,'final_quality_jobs':final_count,
        'screen_process_attempts':6 if recovered else 5,'failed_attempts':failed_attempts,
        'technical_retry_exceeded_original_process_attempt_budget_by':1 if recovered else 0,
        'additional_recipe_or_seed_candidates':0,
        'quality_jobs':quality_jobs,'screening':analysis['systems'],'gate':analysis['independent_gate'],
        'coverage':cache['coverage'],'regression':stats,'resource_monitors':monitors,
        'maximum_recorded_process_reserved_bytes':own_peak,'maximum_sampled_card_used_bytes':maximum_total,
        'independent_screen_replays':len(replay['checks']),'source_history_hashes_verified':len(preserved),
        'original_blocked_delivery_preserved':True,'runtime_commands':commands,
        'default_model_replaced':False,'final_test_predictions_reopened':False,
        'development':development,'vocal_transfer':transfer,'new_human_review_requested':False,
        'old_root_CURRENT_is_a_preserved_blocked_snapshot':True}
    write_json_new(output/'SUMMARY.json',summary)
    verification={'independent_prediction_and_metric_replays':len(replay['checks']),
        'independent_scalar_distances':analysis['independent_scalar_distances_verified'],
        'same_fold_parents_short_cohort_and_optimizer_budget':True,'whole_held_song_exclusion_verified':True,
        'all32_empirical_caches_verified':True,'streamed_recompute_max_difference':max(r['maximum_recompute_logit_difference'] for r in analysis['fold_integrity_checks']),
        'regression':stats,'default_replacement':False,'old_source_history_preserved':len(preserved),
        'old_blocked_delivery_sha256':sha256_file(run/'DELIVERY.json')}
    write_json_new(output/'FINAL_VERIFICATION.json',verification)
    metrics=analysis['systems'];new=metrics['whole_sequence'];prior=metrics['prior'];control=metrics['control']
    rows='\n'.join(f"| {NAMES[name]} | {m['edit_distance']}/{m['reference_phones']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} |" for name,m in metrics.items())
    per_song='\n'.join(f"| {SONGS[song]} | {values['prior']['reference_phones']} | {values['prior']['edit_distance']} | {values['control']['edit_distance']} | {values['whole_sequence']['edit_distance']} |" for song,values in analysis['by_song'].items())
    group_rows=[]
    for name,group in new['groups'].items():
        def errors(value):return sum(value[k] for k in ('substitutions','deletions','insertions'))
        group_rows.append(f"| {GROUPS[name]} | {group['reference_count']} | {errors(prior['groups'][name])} | {errors(control['groups'][name])} | {errors(group)} | {'纳入保护门槛' if group['reference_count']>=20 else '样本不足20，不作保护通过声明'} |")
    outcome='通过五折筛选' if selection['passed'] else '未通过五折筛选'
    seed_text=f"已完成 {final_count} 个最终种子及原唱/合成开发保护评测；详细结果见 DEVELOPMENT_SELECTION.json。" if final_count else '未启动三个最终种子，未新增沙龙开发评测或最终测试推理。'
    numerical_text=('实现修复记录：第四折在第18个epoch因前向/回传重算差异超过2e-5而中止，没有保存检查点。常规重放未复现该异常，具体CUDA内核原因仍未证实；v2将两遍都改为记录梯度的前向路径，逐块丢弃首遍计算图，并保留2e-5阈值。完整计算图梯度对照和真实CUDA探针通过。前三折保留已完成的v1结果；第四折从相同初始化重跑，第五折使用v2。五个完整候选实际用了六次训练进程尝试，比原计划多一次故障重试；没有增加配方、种子或候选选择次数。失败日志与17轮历史均保留。' if recovered else '')
    dev_section=''
    if development:
        dev_rows='\n'.join(f"| {r['seed']} | {r['metrics']['original_dev']['edit_distance']}/214 | {r['metrics']['original_dev']['insertions']} | {r['metrics']['rendered_dev']['edit_distance']}/679 | {r['gate']['passed']} |" for r in development['seeds'])
        dev_section=f'''\n条件式开发结果（同一原始混音与参考）：

| Seed | 原唱错误 | 原唱插入 | 合成错误 | 全部门槛 |
|---|---:|---:|---:|---|
{dev_rows}

晋级资格：{development['release_qualified']}；通过种子：{development['passing_seeds']}。权重未自动部署。选择之后的一次分离人声迁移诊断：{transfer['metrics']['edit_distance']}/214；该分数没有用于选择种子。
'''
    report=f'''已完成显存恢复后的整曲监督实验：{outcome}。

五折都按预先固定的20个epoch完成，分别留出一首训练歌，仍用同样20个已确认原唱短窗、1235个参考音素评估。普通续训与新方法的初始化、短片段数据和优化步数逐折核对一致。默认模型未替换。

| 系统 | 音素编辑错误 | PER | 替换 | 删除 | 插入 |
|---|---:|---:|---:|---:|---:|
{rows}

相对上轮适配基线，PER变化对应改善 {100*(1-new['per']/prior['per']):.3f}%；固定晋级要求至少5%，且参考数不少于20的关键音类不能退化超过0.03。独立整数运算验证门槛结论：{analysis['independent_gate']}。

| 留出歌曲 | 参考音素 | 上轮基线错误 | 普通续训错误 | 整曲监督错误 |
|---|---:|---:|---:|---:|
{per_song}

| 音类 | 参考数 | 上轮基线错误 | 普通续训错误 | 整曲监督错误 | 证据范围 |
|---|---:|---:|---:|---:|---|
{chr(10).join(group_rows)}

与普通续训相比，20个窗口中 {analysis['improved_windows']} 个改善、{analysis['tied_windows']} 个持平、{analysis['worse_windows']} 个退化。替换/删除/插入和完整混淆清单保留在 ERROR_ANALYSIS.json；没有删除难例或更改参考分母。

数据与训练实际完成情况：

- 已消费两项整曲顺序确认，共508.596秒、3211个派生IPA音素。精确音素时间仍未确认；训练使用整序列CTC，不沿用失败的自动恒定偏移，不把模型发射位置当作人工边界。
- 完成32块真实冻结特征缓存，覆盖全部25428个全局帧且各归属一次。训练分块带上下文，损失始终按整条序列计算。
- 加上其他三首歌的已确认短片段，原唱唯一输入时长由200秒增到{cache['coverage']['unique_original_seconds']:.3f}秒；这些秒数包含间奏与停顿，并不全部是新增发声音素。
- 留出自由花或皇后大道東时只使用另一首的整曲；另外三折使用这两首整曲。留出歌曲所有原唱、合成及衍生数据均不参与训练。
- 前15层及428标签输出层冻结，续训最后4层。整曲CTC系数0.25，每首可用整曲每epoch一遍；原有短片段训练与优化步数保持相同。额外计算时间逐作业记录。
- 整曲训练损失下降不等于跨歌识别提高；这五首歌已被用于路线探索，本轮不是独立新增测试集。工程派生IPA参考也不等于独立声学音标真值。

{seed_text}
{dev_section}
{numerical_text}

验证与资源：

- 8秒、20秒真实推理与fixture一致；整曲CUDA梯度探针通过，冻结参数未变。
- 五个保存检查点在独立进程重放：预测、指标一致；原始音频与缓存路径logits一致性通过，real/fixture契约一致。
- 32份缓存完成独立CPU形状/哈希/样本区间验证；80个系统-窗口组合的编辑距离完成独立标量验证。
- 完整回归：{stats['passed']} passed，{stats['skipped']} skipped，{stats['failures']} failed；具体跳过原因见 FINAL_VERIFICATION.json。本次没有人为禁用CUDA验收。
- 回归启动器首次遗漏项目根目录，导致测试收集找不到 tools 包；修正入口路径后重跑。首次收集失败记录保持原样，最终通过的 JUnit 文件由 REGRESSION_ACCEPTED.json 明确绑定。
- 恢复后模型作业记录的最大进程reserved：{own_peak/2**30:.3f} GiB；整卡最大采样占用：{maximum_total/2**30:.3f} GiB。均满足原定7/14 GiB限制；GPU作业串行、CPU四线程、运行断网。
- 120项来源/历史绑定及前次阻塞交付全部保持原哈希。旧的资源不足记录保留，主目录 CURRENT.json/RESULTS.md 是当时的快照；本目录是本次完成状态。

当前没有新增听音审计要求。固定五折实验已经完成；未通过的方案不自动晋级，也不改阈值补跑。所有新权重保留为研究产物。
'''
    text_new(output/'RESULTS.md',report)
    title='整曲监督：'+outcome
    table='<table><thead><tr><th>系统</th><th>编辑错误</th><th>PER</th><th>S / D / I</th></tr></thead><tbody>'
    for name,m in metrics.items():
        table+=f"<tr><td>{html.escape(NAMES[name])}</td><td>{m['edit_distance']}/{m['reference_phones']}</td><td>{m['per']:.4f}</td><td>{m['substitutions']} / {m['deletions']} / {m['insertions']}</td></tr>"
    table+='</tbody></table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{font:17px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:20px;background:#f3f7f7;color:#173b40}section{background:white;border:1px solid #d6e2e3;border-radius:12px;padding:24px;margin:22px 0}table{width:100%;border-collapse:collapse}th,td{padding:12px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#096b74}</style><h1>'+html.escape(title)+'</h1><p>固定预算已完成，默认模型未替换。无需再次提交已有整曲确认。</p><section>'+table+'</section><p><a href="RESULTS.md">完整报告</a> · <a href="ERROR_ANALYSIS.json">错误分析</a> · <a href="FINAL_VERIFICATION.json">验证记录</a></p><section><pre>'+html.escape(report)+'</pre></section></html>'
    text_new(output/'index.html',page)
    handoff=f'''Completed resumed whole-source stage. Read completion/SUMMARY.json and completion/RESULTS.md; root CURRENT/RESULTS/DELIVERY are an intentionally preserved earlier resource-blocked snapshot.

Whole fivefold errors {new['edit_distance']}/1235 (S{new['substitutions']} D{new['deletions']} I{new['insertions']}); prior {prior['edit_distance']}, matched control {control['edit_distance']}. Screen passed={selection['passed']}; final seeds executed={final_count}. No default replacement/final-test reopening/new human annotation. All fivefold quality jobs and independent checkpoint replays complete. Do not restart existing control/risk/whole trials or change gates to create a passing outcome.

Numerical recovery applied={recovered}. Screen job paths are defined in SCREEN_JOB_INDEX.json; original promise/ remains a failed17-complete-epoch attempt, and its complete replacement is promise-recovered-v2/. Never confuse them or count both in metrics. Original code freeze28 remains intact; NUMERICAL_FIX_FREEZE.json binds additive v2 files. First3folds use v1, last2v2; objective, data, initialization, optimizer, epochs and tolerance unchanged. Exact CUDA-kernel cause is unproven; v2 makes both passes grad-enabled and discards first-pass graphs. Six actual process attempts for five completed candidates; one explicit technical budget exception. The original resume command is no longer the correct recovery entry; do not re-run it against incomplete promise/.

Human whole order/repeat approval remains bound to exact freedom-flower and queen-road-east source versions; phone timing false. Two whole sequences now have real32-chunk frozen-prefix caches,508.596sec/3211targets; unique original training-source union618.596sec. Held-song exclusion, matching short cohort/optimizer budgets, same-parent step-zero replay, finite gradients and frozen params checked. Whole recomputation max difference={verification['streamed_recompute_max_difference']}. Whole loss uses exact logit-gradient chain rule across chunk cores, no per-chunk target truncation; the child layers are deterministic in this branch.

CPU-independent80distance checks; real/fixture/raw-prefix checks5; full regression {stats['passed']}passed,{stats['skipped']}skipped,0failed with CUDA available. Own recorded peak{own_peak/2**30:.3f}GiB; card sampled peak{maximum_total/2**30:.3f}GiB. Previous blocked delivery preserved; code freeze28unchanged and source/history120hashes unchanged. No model job remains running at completion.

Use saved results for further analysis. The fixed candidate comparison is finished; any new recipe/model/data-expansion experiment must be separately versioned, not silently added to this budget. Current note-level phones and source waves remain untouched. Only two whole songs eligible; remaining3whole mappings/order not silently approved. No browser/computer-use/subagents were used. Runtime uses existing Python3.12 torch2.8/cu126 FP32 offline,CPU4,GPU serial; proxy18080 only for any separately authorized future download.
'''
    text_new(output/'HANDOFF.md',handoff)
    write_json_new(run/'COMPLETION.json',{'status':'completed','report_path':str(output/'RESULTS.md'),
        'summary_path':str(output/'SUMMARY.json'),'summary_sha256':sha256_file(output/'SUMMARY.json'),
        'completed_at_utc':summary['completed_at_utc'],'screen_passed':selection['passed'],'default_replaced':False,
        'supersedes_runtime_status_but_preserves_root_snapshot':True})
    print('FINALIZED',new['edit_distance'],'/',1235,'regression',stats['passed'],stats['skipped'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);finalize(p.parse_args().run.resolve())

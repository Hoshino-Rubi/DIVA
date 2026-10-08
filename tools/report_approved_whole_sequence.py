"""Close a resource-blocked implementation stage without claiming GPU work passed."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,html,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file


def text_new(path:Path,value:str)->None:
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(value)


def report(run:Path)->None:
    if (run/'WHOLE_CACHE.json').exists() or (run/'ROUTE_SELECTION.json').exists():
        raise ValueError('this report closes only a pre-cache resource block; report new model results separately')
    verified=read_json(run/'INTAKE_VERIFICATION.json');freeze=read_json(run/'FREEZE.json')
    preserved=read_json(run/'PRESERVATION_BEFORE.json')['files']
    for path,digest in preserved.items():verify_file(Path(path),digest)
    code=read_json(run/'CODE_FREEZE.json')['files']
    for path,digest in code.items():verify_file(Path(path),digest)
    suite=next(ET.parse(run/'cpu-regression.xml').iter('testsuite'))
    stats={k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')}
    stats['passed']=stats['tests']-stats['skipped']-stats['errors']-stats['failures']
    stats['elapsed_seconds']=float(suite.attrib['time']);stats['mode']='explicit_cpu_only_full_test_collection'
    stats['junit_sha256']=sha256_file(run/'cpu-regression.xml')
    stats['skips']=[{'test':t.attrib.get('classname','')+'.'+t.attrib.get('name',''),
        'reason':t.find('skipped').attrib.get('message','')} for t in suite.iter('testcase') if t.find('skipped') is not None]
    if stats['errors'] or stats['failures']:raise ValueError('regression is not passing')
    focused=next(ET.parse(run/'focused-tests.xml').iter('testsuite'))
    failure=read_json(run/'whole-smoke/monitor/resource-failure.json')
    latest_path=sorted((run/'resource-prechecks').glob('*.json'))[-1];latest=read_json(latest_path)
    if latest['passed']:raise ValueError('resource check passed; continue rather than publishing a blocked stage')
    command=f'& "{ROOT / ".venv/Scripts/python.exe"}" -X utf8 -B "{ROOT / "tools/resume_approved_whole_sequence.py"}" --run "{run}"'
    now=datetime.now(timezone.utc).isoformat()
    summary={'schema_version':'whole_sequence_approved_stage_v1','snapshot_at_utc':now,
        'status':'awaiting_gpu_headroom','review_accepted':2,'review_pending':0,'review_conflicts':0,
        'source_seconds':sum(r['source_seconds'] for r in verified['checks']),
        'reference_phones':sum(r['target_phones'] for r in verified['checks']),
        'planned_acoustic_chunks':sum(r['chunks'] for r in verified['checks']),
        'real_smoke_completed':False,'whole_feature_cache_created':False,
        'whole_quality_jobs_started':0,'new_final_seeds_started':0,'new_model_quality_metrics':None,
        'training_data_gate':'passed_sequence_only; precise boundaries neither required nor claimed',
        'model_execution_gate':'failed_available_gpu_resources',
        'human_annotation_requested':False,'next_user_action':'release GPU memory; existing order answers are sufficient',
        'latest_resource_precheck':latest,'first_smoke_failure':failure,
        'focused_passed':int(focused.attrib['tests'])-int(focused.attrib['skipped']),
        'regression':stats,'source_and_history_hashes_unchanged':len(preserved),
        'frozen_code_files_checked':len(code),'resume_command':command,
        'new_models_or_dependencies':False,'default_replacement':False,'final_test_predictions_reopened':False}
    write_json_new(run/'CURRENT.json',summary)
    write_json_new(run/'TRAINING_NOT_STARTED.json',{'reason':'gpu_card_occupancy_exceeds_resource_policy',
        'review_intake_sha256':freeze['review_intake_sha256'],'whole_references_sha256':freeze['whole_references_sha256'],
        'available_resource_check_path':str(latest_path),'available_resource_check_sha256':sha256_file(latest_path),
        'whole_quality_jobs_started':0,'final_seed_jobs_started':0,'human_data_confirmation_is_not_the_blocker':True})
    write_json_new(run/'FINAL_VERIFICATION.json',{'at_utc':now,'regression':stats,'focused_passed':summary['focused_passed'],
        'preserved_file_count':len(preserved),'all_preserved_hashes_equal':True,'code_freeze_verified':len(code),
        'real_gpu_acceptance':'not_completed_resource_abort','new_feature_cache':'not_created',
        'real_fixture_whole_worker_replay':'not_executed','quality_improvement_claimed':False})
    rows='\n'.join(f"| {r['song_id']} | {r['source_seconds']:.3f} | {r['note_count']} | {r['target_phones']} | {r['chunks']} |" for r in verified['checks'])
    md=f'''整曲顺序确认已接收；当前暂停在 GPU 资源窗口。

快照 UTC：{now}

已完成：

- 用户 JSON 的两项 sequence_confirmed 均通过原始音频、人工 VSQX、参考清单及复核包哈希校验。原话为空，保留显式勾选和原始 JSON；不编造用户原话。
- 共 {summary['source_seconds']:.3f} 秒原唱、{summary['reference_phones']} 个派生 IPA 参考音素。整曲音素顺序获确认，精确时间及独立声学 IPA 未因此获确认。
- 冻结 32 个处理块的音频请求与参考；16 秒核心、左右最多 2 秒上下文；所有原唱样本覆盖，全部 25,428 个全局发射帧各归属一次。这里只是清单和几何验证，尚未提取模型特征。
- 实现全序列 CTC 的分块回传；与完整计算图的梯度一致。整曲分支关闭随机 dropout，保持激活检查点；额外梯度加入既定优化步骤，不增加步骤数。
- 实现缓存来源检查、整首留出隔离、训练器、条件式最终种子、独立重放、开发保护评测和续跑入口。未修改旧 hash-bound 训练代码。
- {summary['focused_passed']} 项聚焦测试通过。完整 CPU 测试收集：{stats['passed']} passed、{stats['skipped']} skipped、0 failed；其中 2 项 GPU 验收因显式禁用 CUDA 暂未执行，另 3 项沿用环境跳过。
- 120 项原始来源/历史绑定和 28 项冻结代码哈希复核一致。默认模型保持不变，本轮未读取最终测试预测。

| 歌曲 ID | 完整原唱秒数 | 音符数 | 参考音素 | 计划处理块 |
|---|---:|---:|---:|---:|
{rows}

未通过：

- 首次真实 8 秒冒烟在完成推理前被资源保护器中止，子进程退出码 73，原因 total_gpu_memory_limit。失败记录保留于 whole-smoke/monitor/resource-failure.json。
- 最近一次检查：整卡 {latest['total_used_bytes']/2**30:.3f} GiB，超过原定 14 GiB 整卡限制。此时没有启动新的模型进程。保留单作业最多 7 GiB 的余量后，续跑入口要求启动前整卡占用不超过 7 GiB。

未执行：

- 8/20 秒真实推理验收、32 块真实特征缓存、整曲 CUDA 梯度探针、5 个整曲监督留歌作业、独立检查点重放。
- 三个最终种子及开发评测仅在五折结果达到既定门槛后执行。当前没有新权重或新的识别质量分数。

下一步只需要释放显存，无需再次听音或提交这两项答案。已有确认已绑定到文件版本。恢复时保留失败尝试，新冒烟写入独立 attempt 目录。

剩余固定预算：5 个整曲监督作业，对照复用上轮 5 个 control 结果，连同既有 risk 分支共最多 15 个筛选作业。仍用五首训练歌曲留一歌验证；留出歌曲的所有整曲及短片段衍生数据均排除在梯度训练外。满足相对 PER 至少改善 5% 和关键音类保护门槛后，最多执行 3 个最终种子。无需重新训练已经完成的 control/risk 分支。

PowerShell 续跑命令：

```powershell
{command}
```

运行阶段保持断网、CPU 四线程、FP32、模型作业串行；资源保护仅终止自己的进程。
'''
    text_new(run/'RESULTS.md',md)
    text_new(run/'HANDOFF.md',f'''Status snapshot: awaiting_gpu_headroom. See CURRENT.json and RESULTS.md.

Answers accepted: whole-sequence-review-answers-26af9fc40abe.json SHA {verified['original_answer_sha256']}; 2 confirmed,0 pending,0 conflicts. Freedom-flower and queen-road-east complete order/repeat counts approved, exact phone timing explicitly false. No further human annotation needed for these versions. Sequence-only approval supersedes the earlier need for a successful automatic clock gate; it does not repair old timing or change reviewed short-window references.

Frozen dataset: 508.5956009070295 sec,3211 phones,1100 notes,32 planned encoder crops,25428 global emission frames. Whole feature cache NOT CREATED. New training jobs0/final seeds0. First smoke failed before completion with actual child exit73; own ProcessResourceMonitor total_gpu_memory_limit. Latest preflight {latest['total_used_bytes']/2**30:.3f}GiB at {latest['at_utc']}. No model jobs remain running. Do not claim inference, CUDA gradient probe, checkpoint replay, or model improvement passed.

CODE_FREEZE.json binds28 implementation/dependency files. Do not edit them without a new explicit version and preserving this stage. CPU regression {stats['passed']}passed,{stats['skipped']}skipped,0failures;2GPU tests explicitly skipped because CUDA_VISIBLE_DEVICES=-1,3existing environment skips. JUnit is the authoritative record; subprocess inherited console capture produced no pytest summary in command log. Focused38pass; preserved120bindings unchanged.

Resource recovery entry:
{command}

It checks 7GiB headroom, retries only documented resource-aborted smoke into whole-smoke-attempt2, runs32cached chunks and a disposable full-sequence gradient probe, then5planned held-song folds. Source8/20s first; no second model/dependency/CPU fallback. Fivefold control and risk already completed in parent sequence-routes-20260920T140400Z; reuse control, never repeat/sweep. All original optimizer settings remain frozen. Full last4 blocks continue from corresponding original full-fold parent, never an all-data checkpoint in a fold. Whole global CTC coefficient.25, one pass/available training song/epoch with deterministic child layers and matched optimizer steps; two-pass logit gradients bound activation memory. Threefinalseeds only if original5%-gain/protected-group gate passes. Development source/gates remain unchanged; no reopening final-test predictions or default deployment.

Bound parent files and historical outputs are immutable. CURRENT/RESULTS/DELIVERY here describe this completed CPU implementation/resource-blocked snapshot. After resuming, add a completion report and verification alongside the new results; do not rewrite this snapshot to imply it already included training. New code is implemented and CPU-tested but real GPU integration remains unvalidated. No browser/computer-use or subagents used.
''')
    text_new(run/'index.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>整曲监督推进状态</title><style>body{font:17px/1.75 system-ui;max-width:1000px;margin:40px auto;padding:24px;background:#f5f8f8;color:#17363b}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>已接收确认，等待 GPU 余量</h1><p>无需新增听音复核。完整状态、验证限制及续跑命令：</p><pre>'+html.escape(md)+'</pre></html>')
    files=list(run.glob('*.json'))+list(run.glob('*.md'))+[run/'index.html',run/'cpu-regression.xml',run/'focused-tests.xml',run/'run_cpu_regression.py']
    files += list((run/'review-intake').glob('*.json'))+list((run/'commands').glob('*'))+list((run/'resource-prechecks').glob('*.json'))
    files += [run/'whole-smoke/monitor/resource-failure.json']
    sources={**code,str(Path(__file__).resolve()):sha256_file(Path(__file__)),
        str(ROOT/'tools/verify_approved_whole_intake.py'):sha256_file(ROOT/'tools/verify_approved_whole_intake.py'),
        str(ROOT/'tests/test_whole_sequence_approved.py'):sha256_file(ROOT/'tests/test_whole_sequence_approved.py')}
    delivery={'snapshot_at_utc':now,'source_files':[{'path':p,'sha256':d} for p,d in sources.items()],
        'root_artifacts':[{'path':str(p.resolve()),'sha256':sha256_file(p)} for p in sorted(set(files)) if p.is_file()]}
    write_json_new(run/'DELIVERY.json',delivery)
    for row in delivery['source_files']+delivery['root_artifacts']:verify_file(Path(row['path']),row['sha256'])
    write_json_new(run/'DELIVERY_CHECK.json',{'delivery_sha256':sha256_file(run/'DELIVERY.json'),
        'files_checked':len(delivery['source_files'])+len(delivery['root_artifacts']),'all_hashes_equal':True,
        'no_quality_training_completed':True})
    print('DELIVERED',stats['passed'],'passed',stats['skipped'],'skipped; GPU integration remains blocked')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run.resolve())

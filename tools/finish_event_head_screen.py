"""One-shot post-screen workflow; waits for existing jobs, never starts training.

This survives client-side turn interruption. It may only run the already-reviewed
independent replay, CPU report, regression and verifier after all ten jobs close.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,time,subprocess,os
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.training.event_head_protocol import load_protocol


def readiness(run,spec):
    missing=[]
    for arm in spec['arms']:
        for song in spec['songs']:
            key=arm+'--'+song;exitfile=run/'commands'/(key+'.exit.json');result=run/'screen'/arm/song/'RESULT.json'
            if exitfile.exists():
                code=read_json(exitfile)['exit_code']
                if code!=0 and not (arm=='E' and song=='freedom-flower' and code==73 and result.exists() and read_json(result).get('quality_training_restarted') is False):
                    return {'ready':False,'blocked':True,'reason':key+' failed; no automatic retry'}
            if not exitfile.exists() or not result.exists():missing.append(key)
            elif not read_json(result)['complete']:return {'ready':False,'blocked':True,'reason':key+' result incomplete'}
    return {'ready':not missing,'blocked':False,'missing':missing}


def execute(run,name,command,timeout):
    review=read_json(run/'CODE_REVIEW_postprocess.json')
    if not review['approved_for_launch']:raise ValueError('postprocess review not approved')
    for path,digest in review['code_bindings'].items():verify_file(Path(path),digest)
    output=run/'postprocess';record=output/(name+'.command.json')
    if record.exists():raise ValueError('postprocess phase already attempted: '+name)
    write_json_new(record,{'command':command,'at_utc':datetime.now(timezone.utc).isoformat()});started=time.monotonic()
    code=None;error=None
    try:
        with (output/(name+'.log')).open('x',encoding='utf-8') as log:
            p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
        code=p.returncode
    except Exception as exc:error=type(exc).__name__+': '+str(exc);raise
    finally:write_json_new(output/(name+'.exit.json'),{'exit_code':code,'error':error,'elapsed_seconds':time.monotonic()-started})
    if code:raise RuntimeError(name+' failed; no automatic retry')
    print('POSTPROCESS COMPLETE',name,flush=True)


def render_final(run):
    report=read_json(run/'SCREEN_REPORT.json');verification=read_json(run/'FINAL_VERIFICATION.json')
    tests=verification['regression'];passed=tests['tests']-tests['skipped']
    status={'schema_version':'event_head_screen_final_status_v1','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'screen_complete':True,'quality_training_starts':10,'models_completed_20epochs':10,'epochs':200,
        'optimizer_updates':{'H':2140,'E':2140},'successful_quality_processes':9,'post_optimization_timeout_processes':1,
        'readonly_evaluation_recovery':1,'training_retries':0,'selected_arm':report['selected_arm'],
        'conditional_final_training_started':False,'conditional_final_status':'eligible_pending_separate_worker' if report['selected_arm'] else 'not_executed_screen_failed',
        'full_acoustic_replay_forwards':10,'fixture_replays':298,'regression_passed':passed,'regression_skipped':tests['skipped'],
        'default_replaced':False,'final_test_opened':False,'human_listening_required':False}
    write_json_new(run/'FINAL_STATUS.json',status)
    lines=['# H／E五折正式训练结果','', '完成时间（UTC）：'+status['completed_at_utc'],'',
        'H为独立线性CTC头；E为相同内容头加事件支路。主干及共享原生CTC头冻结，同一原混音、参考、采样、20epoch和更新步数。事件模型改变了序列目标及解码，不能把差异仅归因于边界参数。','',
        '| 系统 | 原唱错误/1235 | PER | 替换 | 删除 | 插入 | 送气错误/30 |', '|---|---:|---:|---:|---:|---:|---:|']
    for label,m in [('旧endpoint',report['historical_endpoint']['original'])]+[(a,report['gates'][a]['original']) for a in ['H','E']]:
        g=m['groups']['aspirated'];err=sum(g[k] for k in ('substitutions','deletions','insertions'))
        lines.append(f"| {label} | {m['edit_distance']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {err} |")
    lines+=['','## 冻结门槛与逐歌结果','']
    for arm in ['H','E']:
        g=report['gates'][arm];lines.append(f"- {arm}：{'通过' if g['passed'] else '未通过'}。失败项：{', '.join(g['failures']) or '无'}。")
    lines+=['','| 留出歌曲 | 旧endpoint | H | E |','|---|---:|---:|---:|']
    for song in read_json(run/'POLICY.json')['songs']:
        old=report['gates']['H']['per_song']['baseline'][song]['edit_distance']
        h=report['gates']['H']['per_song']['current'][song]['edit_distance'];e=report['gates']['E']['per_song']['current'][song]['edit_distance']
        lines.append(f'| {song} | {old} | {h} | {e} |')
    lines+=['','## 合成留歌保护（129窗，不是沙龙开发）','', '| 系统 | 参考音素 | 总错误 | PER |','|---|---:|---:|---:|']
    for label,m in [('旧endpoint',report['historical_endpoint']['rendered'])]+[(a,report['gates'][a]['rendered']) for a in ['H','E']]:
        lines.append(f"| {label} | {m['reference_phones']} | {m['edit_distance']} | {m['per']:.6f} |")
    lines+=['','## 超时与恢复：不隐藏失败','',
        'E自由花已完成20epoch/440更新并保存checkpoint，但旧序列化反复进行独立CPU解码/fixture核验，进程在1500秒超时退出73。不是OOM。其余作业当时停止。',
        '从同一checkpoint只读补齐评测，没有重训或增加优化步骤；原23份部分证据与恢复结果一致。只对尚未启动的四折使用重新审阅的快速序列化回调；优化器、损失、初始化、采样、epoch和1500秒上限没有改变。昂贵独立核验移至重放阶段，仍是晋级前必做项。',
        '共10次质量启动、10个完成20epoch的模型；9个质量进程正常退出，1个训练完成后的超时保留，另有1次只读评测恢复。不能写成十进程全部成功。自由花原失败进程的真实CUDA峰值未能完整落盘，其结果中的reserved是epoch观测最大值，不应冒称精确进程峰值。','',
        '## 验证与预算','',
        f'- 两臂各100epoch、2140更新，合计200epoch、4280更新；没有按留出PER挑epoch。',
        '- 10个检查点全部298个留出原唱/合成输出独立重放，另重放149个旧endpoint控制；10次完整原唱音频重放与缓存检查。',
        f'- 完整回归：{passed} passed、{tests["skipped"]} skipped、0 failed。',
        '- 全时长解析梯度引擎与原autograd/穷举/有限差分通过；FP32 CUDA打分、FP64 CPU动态规划明确记录，不裁标签/时长。',
        '- 各阶段code/data/checkpoint hash绑定。原WAV、人工VSQX、映射、默认配置保护通过；没有打开最终测试预测。',
        '- 同五歌已多轮用于研究决策，不是新的独立泛化证据。事件边界不是人工音素时长。','',
        '## 当前出口','',
        ('存在合格臂，后续三个最终种子尚未运行；按原协议继续，不能自动部署。' if report['selected_arm'] else '两臂均未通过完整门槛，因此不启动三个最终种子、不新增沙龙开发评测、不部署。无需新增人工听审。'),
        '', '细节：SCREEN_REPORT.json、FINAL_VERIFICATION.json、RECOVERY_INCIDENT.json、各fold/epoch文件和commands日志。']
    with (run/'RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[run/name for name in ['RESULTS.md','FINAL_STATUS.json','FINAL_VERIFICATION.json','SCREEN_REPORT.json','FREEZE.json','POLICY.json','PROTOCOL.md','RECOVERY_INCIDENT.json','CODE_REVIEW_training.json','CODE_REVIEW_training_v2.json','CODE_REVIEW_recovery.json','CODE_REVIEW_replay.json','regression.xml','regression-exit.json']]
    files += [Path(j['source_result_path']) for j in report['all_jobs']]
    files += [Path(read_json(Path(j['source_result_path']))['checkpoint_path']) for j in report['all_jobs']]
    files += list((run/'postprocess').glob('*.json'))
    files += [ROOT/'tools/finish_event_head_screen.py']
    write_json_new(run/'FINAL_INDEX.json',{'authoritative':True,'status':status,'artifacts':{str(p):sha256_file(p) for p in files},'nested_raw_artifacts_bound_in':'FINAL_VERIFICATION.json'})
    print('FINAL SCREEN DELIVERY WRITTEN',report['selected_arm'],flush=True)


def finish(run):
    freeze,spec,_=load_protocol(run,'CODE_REVIEW_replay.json')
    out=run/'postprocess';write_json_new(out/'START.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'wait_for_existing_training_only':True,'starts_new_quality_training':False,'full_acoustic_forwards_allowed':10,
        'script_sha256':sha256_file(Path(__file__))})
    deadline=time.monotonic()+3*3600
    while True:
        state=readiness(run,spec)
        if state['blocked']:write_json_new(out/'BLOCKED.json',state);return
        if state['ready']:break
        if time.monotonic()>deadline:write_json_new(out/'BLOCKED.json',{'reason':'existing training wait deadline; no restart'});return
        time.sleep(30)
    isolated=freeze['python'];ordinary=str(ROOT/'.venv/Scripts/python.exe')
    execute(run,'independent-replay',[isolated,'-X','utf8',str(ROOT/'tools/replay_event_head_comparison.py'),'--run',str(run)],1550)
    execute(run,'screen-report',[ordinary,'-X','utf8',str(ROOT/'tools/report_event_head_comparison.py'),'--run',str(run)],300)
    regression=[ordinary,'-X','utf8','-m','pytest','-q','--basetemp='+str(run/'regression-temp'),'--junitxml='+str(run/'regression.xml')]
    execute(run,'regression',regression,900)
    write_json_new(run/'regression-exit.json',read_json(out/'regression.exit.json'))
    execute(run,'delivery-verification',[isolated,'-X','utf8',str(ROOT/'tools/verify_event_head_delivery.py'),'--run',str(run)],2400)
    render_final(run)
    write_json_new(out/'COMPLETE.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'selected_arm':read_json(run/'SCREEN_REPORT.json')['selected_arm'],'no_automatic_deployment':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run.resolve()
    try:finish(run)
    except Exception as exc:
        failure=run/'postprocess/FAILED.json'
        if not failure.exists():write_json_new(failure,{'at_utc':datetime.now(timezone.utc).isoformat(),
            'error':type(exc).__name__+': '+str(exc),'automatic_retry':False,'new_quality_training_started':False})
        raise

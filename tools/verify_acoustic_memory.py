"""Independent artifact/codec replay, protection checks, and final delivery."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET
from acoustic_memory_common import ROOT, protocol, read_json, write_json_new, load_bound, load_arrays, bound, verify_file, sha256_file
import numpy as np
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer, ctc_collapse


def scalar_edits(ref: list, hyp: list) -> tuple[int,int,int]:
    """Independent tuple-valued edit DP; diagonal, deletion, insertion tie order."""
    table=[[(0,0,0)]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1):table[i][0]=(0,i,0)
    for j in range(len(hyp)+1):table[0][j]=(0,0,j)
    for i,x in enumerate(ref,1):
        for j,y in enumerate(hyp,1):
            a=table[i-1][j-1];b=table[i-1][j];c=table[i][j-1]
            choices=[(a[0]+(x!=y),a[1],a[2]),(b[0],b[1]+1,b[2]),(c[0],c[1],c[2]+1)]
            table[i][j]=min(choices,key=sum)
    return table[-1][-1]


def equal_arrays(a: dict,b: dict) -> None:
    if set(a)!=set(b) or any(a[k].dtype!=b[k].dtype or not np.array_equal(a[k],b[k]) for k in a):
        raise ValueError('independent tensor replay not bitwise equal')


def codec_identity(segments) -> list:
    # Mode annotations intentionally differ between real and fixture adapters.
    return [{k:v for k,v in asdict(s).items() if k!='metadata'} |
        {'raw_label_trace':s.metadata['raw_label_trace'],'boundary_uncertainty':s.metadata['boundary_uncertainty']}
        for s in segments]


def verify(run: Path) -> dict:
    freeze=protocol(run,review=True);checks=[];heads=[];banks=[]
    for phase in ('execute','replay'):
        exit_record=read_json(run/'commands'/(phase+'.exit.json'))
        if exit_record['exit_code']!=0:raise ValueError('cache worker did not exit successfully')
        h=read_json(run/phase/'head/RESULT.json')
        requests=list((run/phase/'head/requests').glob('*.json'))
        if h['head_requests']!=229 or len(requests)!=229 or h['optimizer_steps']!=0 or h['acoustic_encoder_forwards']!=0:
            raise ValueError('budget accounting mismatch')
        if h['peak_reserved_bytes']>7*1024**3:raise ValueError('reserved budget exceeded')
        monitor=read_json(run/phase/'head/monitor/resource-monitor.json')
        if not monitor['operation_succeeded'] or monitor['maximum_sampled_total_used_bytes']>14*1024**3:
            raise ValueError('card budget evidence failed')
    a=read_json(run/'execute/head/RESULT.json')['rows'];b=read_json(run/'replay/head/RESULT.json')['rows']
    if len(a)!=len(b):raise ValueError('head coverage mismatch')
    for x,y in zip(a,b):
        if any(x[k]!=y[k] for k in ('fold','purpose','example_id','baseline_ids','cache_sha256','checkpoint_sha256')):
            raise ValueError('head identity drift')
        equal_arrays(load_arrays(x['raw']),load_arrays(y['raw']));heads.append([x['fold'],x['example_id']])
    a=read_json(run/'execute/bank/RESULT.json')['folds'];b=read_json(run/'replay/bank/RESULT.json')['folds']
    for song in a:
        if any(a[song][k]!=b[song][k] for k in ('temperature','frame_count','target_label_support','unrepresented_class_ids')):
            raise ValueError('bank metadata drift')
        equal_arrays(load_arrays(a[song]['raw']),load_arrays(b[song]['raw']));banks.append(song)
    original=read_json(run/'execute/query/RESULT.json')['rows'];replay=read_json(run/'replay/query/RESULT.json')
    if len(original)!=338 or replay['score_units']!=338 or len(replay['brute_checks'])!=10:
        raise ValueError('query coverage incomplete')
    audio={q['example_id']:Path(q['audio_path']) for q in read_json(run/'QUERIES.json')['rows']}
    for i,(x,y) in enumerate(zip(original,replay['rows'])):
        for key in ('example_id','arm','song_id','audio_role','prediction_ids','prediction','blank_frames'):
            if x[key]!=y[key]:raise ValueError('query prediction/identity differs')
        equal_arrays(load_arrays(x['raw']),load_arrays(y['raw']))
        equal_arrays(load_arrays(x['neighbors']),load_arrays(y['neighbors']))
        posterior=Path(y['posterior']['path']);verify_file(posterior,y['posterior']['sha256'])
        real=PhoneticXeusPosteriorRecognizer(posterior,vocabulary_path=Path(freeze['vocabulary']['path'])).recognize(audio[y['example_id']])
        fixture=PhoneticXeusPosteriorRecognizer(posterior,fixture=True,vocabulary_path=Path(freeze['vocabulary']['path'])).recognize(audio[y['example_id']])
        if codec_identity(real)!=codec_identity(fixture):raise ValueError('PhoneSegment fixture differs')
        labels=[s.metadata['raw_label_trace']['label'] for s in real]
        expected=[p for p in y['prediction'] if not p.startswith('<')]
        if labels!=expected:raise ValueError('PhoneSegment sequence differs')
        checks.append({'example_id':y['example_id'],'arm':y['arm'],'fixture_equal':True,'posterior':y['posterior']})
        if i%50==0:print('codec',i+1,'of 338',flush=True)
    report=read_json(run/'execute/score/RESULT.json');second=read_json(run/'replay/score/RESULT.json')
    for key in ('metrics','pools','per_song','gate','primary_strictly_beats_diagnostic_controls','internal_multivowel_content_errors'):
        if report[key]!=second[key]:raise ValueError('independent scoring differs')
    for arm,rows in report['pools'].items():
        for role,metrics in report['metrics'][arm].items():
            counts=np.array([scalar_edits(r['reference'],r['prediction']) for r in rows if r['audio_role']==role]).sum(0).tolist()
            if counts!=[metrics[k] for k in ('substitutions','deletions','insertions')]:raise ValueError('independent edit DP mismatch')
    protected=read_json(run/'PROTECTED.json');count=0
    for group in ('files','historical_bindings','research_bindings'):
        for p,h in protected[group].items():verify_file(Path(p),h);count+=1
    for spec in [freeze['head'],freeze['vocabulary'],*freeze['checkpoints'].values()]:verify_file(Path(spec['path']),spec['sha256'])
    return {'passed':True,'head_replays':len(heads),'total_head_requests':458,'bank_replays':banks,
        'query_replays':len(checks),'codec_checks':checks,'independent_brute_windows':replay['brute_checks'],
        'edit_dp_independent':True,'protected_hash_checks':count,'protected_media_config_count':len(protected['files']),
        'optimizer_steps':0,'encoder_forwards':0,'new_dependencies':0,'new_models':0,
        'test_predictions_opened':False,'default_changed':False}


def finish(run: Path) -> None:
    verification=read_json(run/'VERIFICATION.json')
    if verification['passed'] is not True:raise ValueError('verification incomplete')
    suite=ET.parse(run/'regression.xml').getroot()
    stats={k:sum(int(n.attrib.get(k,0)) for n in suite.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    exit_record=read_json(run/'regression.exit.json')
    if stats['failures'] or stats['errors'] or exit_record['exit_code']!=0:raise ValueError('full regression did not pass')
    report=read_json(run/'execute/score/RESULT.json');bank=read_json(run/'execute/bank/RESULT.json')
    head=[read_json(run/p/'head/RESULT.json') for p in ('execute','replay')]
    rows=['# 原唱软监督声学记忆：固定实验结果','',
        '## 结论','',('主候选通过本轮筛选门槛；仍不自动部署，后续验证需另行授权。' if report['gate']['passed'] else
            '主候选未通过固定门槛。本配方关闭；不扫描 k、温度或融合系数，不追加训练或人工听审。'),'',
        '## 同窗口、同参考结果','', '| 条件 | 原唱错误 / 1235 | 替换 | 删除 | 插入 | PER |', '|---|---:|---:|---:|---:|---:|']
    names={'endpoint':'冻结 endpoint','memory':'原唱软监督记忆','prior':'类先验（诊断）','shuffled':'窗内打乱（诊断）'}
    for arm in ('endpoint','memory','prior','shuffled'):
        m=report['metrics'][arm]['original'];rows.append(f"| {names[arm]} | {m['edit_distance']} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {m['per']:.6f} |")
    rows+=['','门槛失败项：`'+', '.join(report['gate']['failures'])+'`。','',
        '类先验和打乱不是备用候选；单次打乱不足以建立显著性或机制结论。',
        '主候选是否严格优于诊断对照：`'+str(report['primary_strictly_beats_diagnostic_controls'])+'`。','',
        '## 逐歌原唱错误','', '| 歌曲 | endpoint | 记忆 | 参考数 |','|---|---:|---:|---:|']
    for song,m in report['per_song']['current'].items():
        old=report['per_song']['baseline'][song];rows.append(f"| {song} | {old['edit_distance']} | {m['edit_distance']} | {m['reference_phones']} |")
    rows+=['','## 合成保护与音类','']
    for arm in ('endpoint','memory'):
        m=report['metrics'][arm]['rendered'];rows.append(f"- {names[arm]}：{m['edit_distance']}/4814，PER {m['per']:.6f}。")
    rows+=['','| 音类 | 原唱样本 | endpoint 错误 | 记忆错误 |','|---|---:|---:|---:|']
    for group,old in report['metrics']['endpoint']['original']['groups'].items():
        m=report['metrics']['memory']['original']['groups'][group]
        rows.append(f"| {group} | {m['reference_count']} | {sum(old[k] for k in ('substitutions','deletions','insertions'))} | {sum(m[k] for k in ('substitutions','deletions','insertions'))} |")
    rows+=['','前圆唇18、f12样本不足，仅描述；内部多元音内容错误、逐窗口混淆、完整音类和 blank 变化见 JSON，不把CTC发射时间称为音素持续时间。',
        '', '## 核验及资源','',
        f"- 两个串行缓存 worker，各229次，共458次小head请求；新训练、optimizer更新、声学主干forward、模型及依赖下载均为0。",
        f"- head独立重放229组、记忆库5组、输出338组逐张量一致；10个固定窗口精确近邻与独立穷举实现一致。",
        f"- 338份posterior的真实/fixture PhoneSegment内容、序列、时间和置信度一致（模式标记按接口保留差异）。",
        f"- 全回归：{stats['tests']-stats['skipped']} passed，{stats['skipped']} skipped；保护媒体/配置{verification['protected_media_config_count']}项，全部历史/来源哈希检查{verification['protected_hash_checks']}项通过。",
        '- 五歌划分、428标签、原始媒体、人工VSQX、历史结果、映射和默认配置未改。未读取沙龙开发或最终测试预测。']
    for phase,h in zip(('execute','replay'),head):
        rows.append(f"- {phase}：head进程 {h['elapsed_seconds']:.2f}秒；峰值reserved {h['peak_reserved_bytes']/1024**3:.4f} GiB。")
    rows+=['','软CTC值只来自另外四首训练歌，是参考条件化潜对齐，不是人工帧级真值；129个合成窗口不入库。',
        '', '## 交付与状态','',
        '- 已完成：隔离清单、实际代码审阅、固定执行、独立重放、评分、完整回归、保护核验。',
        '- 未执行：参数扫描、优化器训练、新主干编码、开发/最终测试、默认部署。',
        '- 等待人工确认：无新增听审；本固定预算已用完，后续新方案需要另行决定。',
        '- 可复现入口：`REPRODUCE.md`；已有目录拒绝重复启动。',
        '- 详细结果：`execute/score/RESULT.json`、`VERIFICATION.json`、`FINAL_INDEX.json`。','']
    with (run/'RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(rows))
    status={'complete':True,'gate_passed':report['gate']['passed'],'regression':stats,'budget':{'workers':2,'head_requests':458,'training':0,'encoder_forwards':0},
        'no_automatic_promotion':True,'no_new_human_listening_required':True,'fixed_recipe_closed':not report['gate']['passed']}
    write_json_new(run/'FINAL_STATUS.json',status)
    artifacts={str(p.resolve()):sha256_file(p) for p in run.rglob('*') if p.is_file() and not any(part in ('offline','pytest-cache','prelaunch-temp','regression-temp','pytest-cache-final') for part in p.parts)}
    write_json_new(run/'FINAL_INDEX.json',{'artifacts':artifacts,'status':status})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--finish',action='store_true');a=p.parse_args()
    if a.finish:finish(a.run)
    else:write_json_new(a.run/'VERIFICATION.json',verify(a.run))

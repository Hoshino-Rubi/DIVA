"""Create the final reproducible research report without deploying a candidate."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,json,html,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,write_json_new,verify_file

NAMES={'endpoint':'末端适配','lora':'内部 LoRA','full':'末四层微调','baseline':'基础配方','paired':'配对一致性','unlabeled':'配对＋无标签一致性','attributes':'发音属性辅助'}
SONGS={'freedom-flower':'自由花','half-moon-night-song':'月半小夜曲','like-you':'喜欢你','promise':'约定','queen-road-east':'皇后大道東'}

def report(run:Path)->None:
    architecture=read_json(run/'ARCHITECTURE_SELECTION.json');recipe=read_json(run/'RECIPE_SELECTION.json')
    selection=read_json(run/'DEVELOPMENT_SELECTION.json');audit=read_json(run/'INDEPENDENT_AUDIT.json')
    if audit['declared_jobs']!=33:raise ValueError('report requires all declared experiments or a separate documented failure report')
    seeds=[read_json(run/'development-results'/('seed'+str(seed))/'RESULT.json') for seed in (17,29,43)]
    transfer=read_json(run/'development-transfer'/('seed'+str(selection['research_seed']))/'RESULT.json')
    old=seeds[0]['old_guarded']['metrics'];coverage=read_json(run/'COVERAGE.json')
    tree=ET.parse(run/'regression.xml');suites=[tree.getroot()] if tree.getroot().tag=='testsuite' else tree.findall('testsuite')
    counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ['tests','failures','errors','skipped']};counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
    if counts['failures'] or counts['errors']:raise ValueError('regression is not clean')
    checks=[]
    for seed in (17,29,43):
        path=run/'verification-final'/('seed'+str(seed))/'VERIFICATION.json';checks.append({'path':str(path),'sha256':sha256_file(path),'result':read_json(path)})
    real_entry=read_json(run/'free-audio-entry-check.json')
    if not real_entry['real_fixture_equal'] or not real_entry['free_audio_matches_existing_full_audio']:raise ValueError('free-audio entry not independently verified')
    summary={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'status':'development_gate_passed_pending_listening' if selection['release_qualified'] else 'experiments_complete_no_release_promotion',
        'architecture':architecture,'recipe':recipe,'development_selection':selection,'development_seeds':seeds,'selected_vocal_transfer':transfer,
        'audit':audit,'regression':counts,'final_checkpoint_verifications':checks,'free_audio_entry':real_entry,
        'challenger':read_json(run/'challenger-preflight/RESULT.json'),'default_changed':False,'test_predictions_opened':False,
        'human_listening_needed':selection['release_qualified'],'new_audio_annotations_requested':False}
    write_json_new(run/'SUMMARY.json',summary)
    lines=['# 歌唱域适配：固定实验结果','',
        '本轮完成三种架构的五折比较、三组数据/损失消融、胜出配方的三个最终种子，以及独立重放和完整回归。新模型权重下载受指定代理连接失败阻塞；没有更换下载通道或默认模型。','',
        '## 架构比较','',
        '每折只训练四首歌，留出歌的原唱、合成和无标签衍生物都不进入优化。全部以固定第20个epoch比较，没有按留出歌挑选最佳epoch。以下分数用于选择方法，不能当作额外独立测试成绩。','',
        '| 架构 | 错误／参考 | PER | 替换 | 删除 | 插入 |','|---|---:|---:|---:|---:|---:|']
    for row in architecture['arms']:
        m=row['metrics'];lines.append(f"| {NAMES[row['name']]} | {m['edit_distance']}/{m['reference_phones']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} |")
    lines+=['',f"按冻结规则选择 **{NAMES[architecture['selected_arm']]}**。末四层594与末端595仅差一个错误，不能据此宣称架构具有稳定优势。",'',
        '| 歌曲 | 末端错误 | LoRA错误 | 末四层错误 |','|---|---:|---:|---:|']
    for song,name in SONGS.items():
        values=[next(f for f in row['folds'] if f['held_song']==song)['metrics']['edit_distance'] for row in architecture['arms']]
        lines.append('| '+name+' | '+' | '.join(map(str,values))+' |')
    lines+=['','## 数据与属性消融','','| 配方 | 错误／参考 | PER | 替换 | 删除 | 插入 |','|---|---:|---:|---:|---:|---:|']
    for row in recipe['variants']:
        m=row['metrics'];lines.append(f"| {NAMES[row['name']]} | {m['edit_distance']}/{m['reference_phones']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} |")
    lines+=['',f"胜出配方：**{NAMES[recipe['selected_variant']]}**。选择只使用训练歌的分组验证，未使用《沙龙》或分离迁移成绩。",'',
        f"监督数据保持20个原唱窗口、1235个参考音素，以及129个合成窗口。27个旧映射未解决的合成窗口仍排除并记录原因。无标签分支使用{coverage['unlabeled_windows']}个窗口、{coverage['unlabeled_seconds']:.0f}秒；它们仅经过音量和区间筛选，不等于人工确认的纯歌声。",'',
        '配对分支是在原有监督目标上增加原唱/分离人声的全428标签一致性项。无标签分支进一步增加训练歌曲窗口上的特征扰动一致性，不产生硬音素伪标签。属性分支单独增加8个辅助CTC头，主音素输出仍为428标签。所有分支均完整报告，未把它们捆绑后追分。','',
        '## 固定配方的原唱开发评测','','每个最终种子从共同预训练初始化重新训练，未根据开发集早停。当前参考为214个原唱音素；PER是固定工程参考上的编辑率，不是整首原唱准确率。','',
        '| 系统 | 错误／214 | PER | 替换 | 删除 | 插入 | 发布检查 |','|---|---:|---:|---:|---:|---:|---|']
    m=old['original_dev'];lines.append(f"| 既有 guarded candidate | {m['edit_distance']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | 历史比较点 |")
    m=seeds[0]['pretrained_metrics']['original_dev'];lines.append(f"| 恢复结构的预训练模型 | {m['edit_distance']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | 起点 |")
    for row in seeds:
        m=row['metrics']['original_dev'];g=row['release_check'];state='通过' if g['passed'] else '未通过：'+', '.join(g['failures'])
        lines.append(f"| Seed {row['seed']} | {m['edit_distance']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {state} |")
    lines+=['',f"研究候选为 Seed {selection['research_seed']}；通过发布检查的种子：{selection['passing_seeds']}。要求至少两个种子通过，本轮结果为 **{'达到开发门槛，等待人工听审' if selection['release_qualified'] else '未获得发布晋级'}**。默认模型保持原状。",'',
        '## 合成保护与分离迁移','','| 系统 | 合成开发错误／679 | 合成PER |','|---|---:|---:|']
    m=old['rendered_dev'];lines.append(f"| 既有 guarded candidate | {m['edit_distance']} | {m['per']:.6f} |")
    for row in seeds:
        m=row['metrics']['rendered_dev'];lines.append(f"| Seed {row['seed']} | {m['edit_distance']} | {m['per']:.6f} |")
    chosen=next(r for r in seeds if r['seed']==selection['research_seed']);vocal=transfer['metrics']['vocal_dev'];mixed=chosen['metrics']['original_dev']
    lines+=['',f"在原混音开发指标选定候选后，才重放它的分离人声：同一214参考下，原混音{mixed['edit_distance']}个错误，分离人声{vocal['edit_distance']}个错误。没有用分离结果重新选种子。逐窗口、关键音类及完整混淆计数均在SUMMARY.json与development-results中。",'',
        '## 验证与限制','',f"独立审计核对{audit['declared_jobs']}个已完成训练作业，复算{audit['fold_distance_checks']}个留出窗口及{audit['development_distance_checks']}项开发比较。每个最终检查点在新进程重放，含8秒真实/fixture冒烟及完整窗口的原始音频/缓存一致性检查。显式实验推理入口同时支持真实与fixture模式。",'',
        f"完整回归：**{counts['passed']} passed、{counts['skipped']} skipped**。所有作业受CPU四线程、单进程7GiB/整卡14GiB限制，模型作业串行、离线运行。前十五层始终冻结，缓存绑定输入区间、音频、结构与权重身份。",'',
        f"历史文件{audit['historical_files_unchanged']}项及本轮源音频/参考哈希复核通过。原始WAV、人工VSQX、历史结果和默认模型未覆盖；本轮没有打开两首历史最终测试的预测。",'',
        '《沙龙》和本轮训练歌验证都已用于研究决策，不能描述成全新独立泛化验证。CTC时间仍只是发射位置，不是经确认的音素持续时间。现有逐音素人工校准与片段对应确认继续有效；IPA映射和新增属性是其工程派生表示，没有要求重标或用预测覆盖人工参考。','',
        '## 剩余外部条件','',
        'POWSM预检在指定127.0.0.1:10808代理处被拒绝连接，ESPnet2也尚未安装。没有下载新权重或扩展依赖。异构模型的真实基线尚未执行；要继续该分支，需要恢复指定下载通道，再锁定版本、权重许可、哈希和隔离环境。ZIPA权重许可仍需单独核验。','',
        '## 复现入口','',
        '完整命令与退出状态见commands。训练流程由tools/run_singing_domain_stages.py串行驱动，数据与预算见POLICY.json和FREEZE.json。显式推理使用训练隔离环境运行tools/infer_singing_domain.py；真实模式给出--run、--job、--audio、--start、--duration、--output，fixture模式使用--fixture。','']
    content='\n'.join(lines);(run/'RESULTS.md').write_text(content,encoding='utf-8')
    table=''.join('<tr><td>'+html.escape(NAMES[r['name']])+'</td><td>'+str(r['metrics']['edit_distance'])+'/1235</td><td>'+f"{r['per']:.4f}"+'</td></tr>' for r in recipe['variants'])
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>歌唱域适配结果</title>
<style>body{{font:18px/1.7 system-ui;color:#183a40;background:#eef4f5;max-width:1080px;margin:36px auto;padding:20px}}section,details{{background:white;border-radius:14px;padding:24px;margin:18px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;font:16px/1.7 system-ui}}a{{color:#076873}}</style>
<h1>歌唱域适配结果</h1><p>{html.escape(summary['status'])}</p><section><p>固定架构：{NAMES[recipe['selected_arm']]}；固定配方：{NAMES[recipe['selected_variant']]}。默认模型未替换。</p><table><tr><th>配方</th><th>跨歌错误／参考</th><th>PER</th></tr>{table}</table></section>
<section><a href="RESULTS.md">完整报告</a> · <a href="SUMMARY.json">全部结果</a> · <a href="INDEPENDENT_AUDIT.json">独立审计</a> · <a href="DEVELOPMENT_SELECTION.json">开发比较</a></section>
<details><summary>展开完整说明</summary><pre>{html.escape(content)}</pre></details></html>'''
    (run/'index.html').write_text(body,encoding='utf-8')
    (run/'CURRENT.json').write_text(json.dumps({'status':summary['status'],'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'active_model_job':False,'selected_arm':recipe['selected_arm'],'selected_variant':recipe['selected_variant'],'research_seed':selection['research_seed'],
        'release_qualified':selection['release_qualified'],'new_human_annotations_requested':False,'challenger_status':'blocked_proxy'},indent=2),encoding='utf-8')
    (run/'HANDOFF.md').write_text('# Completed local singing-domain experiments\n\nRead RESULTS.md, SUMMARY.json, CURRENT.json and INDEPENDENT_AUDIT.json.\n\n33declared jobs completed; fivefold architecture selection then ablations then3finalseeds. Existing default/sourcefiles/history preserved. No final-test predictions opened. Heterogeneous challenger remains blocked by the specified proxy and missing isolated dependencies. No new annotation requested.\n\nOptional real/fixture inference: tools/infer_singing_domain.py in the isolated Python3.12runtime. Checkpoints remain under final-training/seed{17,29,43}; no automatic deployment.\n',encoding='utf-8')
    sources=[p for p in (ROOT/'src/any2xsampa').rglob('singing_domain*.py')]+[ROOT/'src/any2xsampa/config/singing_domain_v1.json',ROOT/'src/any2xsampa/config/singing_phone_attributes_v1.json']
    sources+=list((ROOT/'tools').glob('*singing_domain*.py'))+[ROOT/'tools/cache_singing_development.py',ROOT/'tools/preflight_domain_challenger.py',ROOT/'tools/check_singing_inference_entry.py',ROOT/'tests/test_singing_domain.py']
    files=[{'path':str(p.resolve()),'sha256':sha256_file(p)} for p in sorted(set(sources))]
    artifacts=[{'path':str(p.resolve()),'sha256':sha256_file(p)} for p in run.iterdir() if p.is_file() and p.name!='DELIVERY.json']
    write_json_new(run/'DELIVERY.json',{'source_files':files,'root_artifacts':artifacts,'checkpoint_hashes_in':'FINAL_TRAINING.json','default_replacement':False})
    print(summary['status'],'seed',selection['research_seed'],'regression',counts)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run.resolve())

"""Deliver verified input results and explicitly account for invalid training attempts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new
from any2xsampa.training.input_supervision import load_manifest
from any2xsampa.training.input_supervision_training import require_screen_budget


def finish(run: Path) -> None:
    freeze, manifest, index = load_manifest(run/'ready-final')
    incident = read_json(run/'INCIDENT.json')
    probe = Path(freeze['probe_run'])
    measured = read_json(probe/'RESULT.json')
    evidence = read_json(run/'input-replay/VERIFICATION.json')
    runtime = read_json(run/'runtime-verification/VERIFICATION.json')
    checkpoints = read_json(run/'checkpoint-replay/VERIFICATION.json')
    schedule = read_json(run/'SCHEDULE_AND_PRESERVATION.json')
    regression_exit = read_json(run/'final-regression-exit.json')
    if regression_exit['exit_code'] != 0:
        raise ValueError('regression not complete/passing')
    suites = ET.parse(run/'final-regression.xml').getroot()
    if suites.tag == 'testsuites': suites = list(suites)
    else: suites = [suites]
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    tests['passed'] = tests['tests']-tests['failures']-tests['errors']-tests['skipped']
    if tests['failures'] or tests['errors']: raise ValueError('regression failed')
    try: require_screen_budget(run/'ready-final')
    except RuntimeError as exc: blocked = str(exc)
    else: raise ValueError('unexpected extra quality-training authorization')
    metrics = measured['condition_metrics']
    brief = {c: {k: m[k] for k in ('reference_phones','edit_distance','per','substitutions','deletions','insertions','groups')} for c,m in metrics.items()}
    artifacts = [probe/'RESULT.json', run/'INCIDENT.json', run/'input-replay/VERIFICATION.json',
                 run/'runtime-verification/VERIFICATION.json', run/'checkpoint-replay/VERIFICATION.json',
                 run/'SCHEDULE_AND_PRESERVATION.json', run/'final-regression.xml', run/'ready-final/FREEZE.json']
    summary = {
        'schema_version': 'input_supervision_delivery_v2', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'stages': {'fixed_input_comparison':'completed_verified', 'v_v1_training':'invalid_schedule_diagnostic_only',
                   'mv_training':'not_executed', 'corrected_training':'implementation_verified_waiting_additional_budget',
                   'final_seeds_development':'not_executed_gate_not_reached'},
        'input_metrics': brief, 'per_song': measured['per_song'],
        'macro_song_per': {c:sum(m['per'] for m in songs.values())/5 for c,songs in measured['per_song'].items()},
        'relative_improvement_vs_mixed': {c:(metrics['x']['edit_distance']-metrics[c]['edit_distance'])/metrics['x']['edit_distance'] for c in ('v','h')},
        'training_attempts': incident['attempts'],
        'completed_checkpoint_count': len(checkpoints['checkpoints']), 'valid_new_training_results':0,
        'total_completed_epochs_across_attempts':sum(a['epochs_saved'] for a in incident['attempts']),
        'budget':freeze['budget'], 'budget_block':blocked, 'additional_starts_needed':9,
        'verification': {'saved_logit_checks':evidence['saved_logit_checks'], 'waveform_checks':len(evidence['waveform_checks']),
                         'schedule_epoch_checks':schedule['epoch_checks'], 'real_inference_smoke_seconds':[8,20],
                         'initial_logit_errors':runtime['initial_logit_errors'], 'unscored_optimizer_steps':1,
                         'runtime_peak_reserved_gib':runtime['peak_reserved_bytes']/1024**3,
                         'checkpoint_replays':len(checkpoints['checkpoints']), 'regression':tests},
        'preservation':schedule, 'artifacts':{str(p):sha256_file(p) for p in artifacts},
        'default_model_replaced':False,'new_final_test_predictions':False,'new_human_listening_required':False,
    }
    write_json_new(run/'DELIVERY.json', summary)
    names={'x':'原混音','v':'分离人声','h':'50% 混音 + 50% 人声'}
    lines=['# 输入对照与训练修复交付（2026-09-21）','',
           '阶段 A 已完成且可复现。阶段 B 没有完成有效的 V/MV 受控比较：我的首版训练器改变了旧控制的源窗口顺序与随机数消耗，且重复启动浪费了预算。这些结果不用于晋级；该问题与此前的显存超限是两件事。','',
           '## 有效输入对照','',
           '固定五首训练歌、20 个留歌窗口、1235 个音素。每首使用未训练过该歌的旧末四层检查点；只改变音频输入。原混音重放的 20/20 音素序列与旧结果一致。','',
           '| 输入 | PER | 总错 | 替换 | 删除 | 插入 | 送气错误 / 30 |','|---|---:|---:|---:|---:|---:|---:|']
    for c,m in brief.items():
        g=m['groups']['aspirated']; e=g['substitutions']+g['deletions']+g['insertions']
        lines.append(f"| {names[c]} | {m['per']:.6f} | {m['edit_distance']} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {e} |")
    lines += ['', '分离人声总 PER 相对下降 13.47%，但送气错误由 14 增至 18，不能凭总分直接采用。固定中点混合总 PER 相对下降 12.12%，送气错误为 12；依原计划保留为无需训练的输入对照，尚未进入沙龙开发检查。前圆唇只有 18 个参考，证据不足。','',
              '各歌错误数（原混音 / 人声 / 中点）：','']
    for song in measured['per_song']['x']:
        values=[measured['per_song'][c][song]['edit_distance'] for c in ('x','v','h')]
        lines.append(f'- {song}：{values[0]} / {values[1]} / {values[2]}。')
    lines += ['', '## 训练状态与我的错误','',
              '- 本轮启动了 9 次训练进程：2 次在完整 epoch 前失败（清单错误、资源超限），1 次在第 11 epoch 被我主动中止，6 次跑完 20 epoch。其中《自由花》被重复训练。没有把这些重试从记录中抹掉。',
              '- 主动中止的作业当时仍在正常写 epoch，并无已证实的卡死。此前“长时间无输出”的解释不成立。',
              '- 六份完整结果都来自首版错误顺序，无法作为只改变输入监督条件的公平对照；所有检查点已独立重放，仍仅列诊断。MV 未运行，最终三种子和开发评测未执行。',
              '- 资源失败记录为 16,477,323,264 bytes，约 15.35 GiB；此前写成 16.48 GiB 是单位错误。整卡占用本身不能证明是模型加载导致。显存阻塞随后已解除。',
              '- 第一版阶段 A 报告的进程峰值字段取了不存在的监控键，得到 0；这是未测得，不是零显存。整卡监控有效；本次重新进行的八秒／二十秒和单步验证峰值为约 %.3f GiB。' % summary['verification']['runtime_peak_reserved_gib'],
              '', '## 修复与验证','',
              '- 修复后的 V/MV 每个源窗口就地展开四个槽位，保留旧未标注行 shuffle 的随机数消耗。五折 × 两种方案 × 二十轮，共 200 轮源顺序、随机数状态与步数核对通过。',
              '- 直接复用历史混音／人声／合成缓存，标注人声参考为继承原唱参考，不增加独立人工真值计数。哈希、对应来源、CTC 帧数、音素 ID 和歌曲划分绑定。',
              '- 完成 60 份原始 logits 的独立编辑距离及 posterior/fixture 核验、20 份中点波形的逐样本核验、六检查点重放及各一份真实音频核验。',
              '- 八秒／二十秒真实推理通过；三种训练输入的初始化误差均为 0。另做一次不评分、不保存检查点的梯度与 AdamW 验证，冻结参数未改变。',
              f"- 最终回归：{tests['passed']} passed，{tests['skipped']} skipped。",
              '- 196 个历史源码文件、24 个研究证据绑定及旧训练源音频／参考绑定均核验；唯一有意修改的旧源码是线程池重复初始化检查。源 WAV、人工 VSQX、映射和默认模型未改。',
              '', '## 下一人工窗口','',
              '无需新增听音、补歌或重标。本次需要的是训练预算决定：按进程启动保守计数，原十次上限已使用九次，剩余一次不足以完成修复后的 V/MV 十折。再启动这十折需要追加九次额度；未经批准不继续。',
              '修复数据与代码的最终冻结入口是 `ready-final/FREEZE.json`。中间 `corrected-run`、`ready` 属于修复验证快照，不是续跑入口。',
              '若获批准，创建新的冻结 run 并明确记录总额度 19、既有启动 9、新剩余 10；只跑原来两种方法，不增第四方案、比例扫描或 LoRA。达到既定筛选条件才运行原计划最终三种子。', '']
    with (run/'RESULTS.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    write_json_new(run/'CURRENT.json', {'status':'waiting_user_training_budget_decision', 'report':str(run/'RESULTS.md'),
                    'ready_manifest':str(run/'ready-final/FREEZE.json'), 'input_comparison_completed':True,
                    'quality_training_validly_completed':False,'running_model_jobs':0})
    print('Delivered',run/'RESULTS.md', tests)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);finish(p.parse_args().run.resolve())

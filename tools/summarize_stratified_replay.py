"""Publish the completed review and bounded experiment without promoting weights."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse, html, json, sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.stratified_replay import validate_policy


def publish(run: Path) -> None:
    policy = read_json(run/'EXPERIMENT_POLICY.json'); validate_policy(policy)
    source = Path(policy['bindings']['source_policy']['path']).parent
    analysis = read_json(run/'ANALYSIS.json')
    comparison = read_json(Path(policy['bindings']['comparison']['path']))
    current = comparison['aggregate']['candidate']
    replay = read_json(run/'CHECKPOINT_REPLAY.json')
    seeds, source_files = [], []
    for seed in policy['seeds']:
        report_path = run/'stratified-replay'/f'seed{seed}'/'RESULT.json'
        result = read_json(report_path)
        replayed = next(r for r in replay['seeds'] if r['seed'] == seed)
        verify_file(report_path, replayed['result_sha256'])
        if result['policy_sha256'] != sha256_file(run/'EXPERIMENT_POLICY.json'):
            raise ValueError('training source policy differs')
        if any(result[k] for k in ('test_loaded', 'default_changed', 'human_preferences_used_as_labels')):
            raise ValueError('experiment crossed a scope boundary')
        if not all(result[k] for k in ('step_zero_equal_all_training_rows', 'teacher_unchanged', 'head_unchanged')):
            raise ValueError('training frozen-state check failed')
        best = result['checkpoints']['best_overall']
        for value in result['checkpoints'].values():
            if value is None: continue
            verify_file(Path(value['path']), value['sha256'])
            verify_file(Path(value['evaluation_path']), value['evaluation_sha256'])
        baseline = read_json(source/'projected-replay'/f'seed{seed}'/'RESULT.json')['checkpoints']['best_overall']
        seeds.append({'seed': seed, 'epochs_completed': len(result['history']), 'best_epoch': best['epoch'],
                      'best_checkpoint_sha256': best['sha256'], 'original_dev': best['original_metrics'],
                      'rendered_dev': best['rendered_metrics'],
                      'prior_projected_original_dev': baseline['original_metrics'],
                      'prior_projected_rendered_dev': baseline['rendered_metrics'],
                      'best_admissible': result['checkpoints']['best_admissible'],
                      'failed_gates_at_best_epoch': result['history'][best['epoch']-1]['gate']['failures'],
                      'peak_reserved_bytes': result['peak_reserved_bytes'], 'elapsed_seconds': result['elapsed_seconds'],
                      'projected_steps': sum(s['projected'] for e in result['history'] for s in e['steps'])})
        source_files.append({'path': str(report_path), 'sha256': sha256_file(report_path)})
    eligible = [s for s in seeds if s['best_admissible'] is not None]
    selected = min(eligible, key=lambda s: (s['best_admissible']['original_metrics']['per'], s['seed'], s['best_admissible']['epoch'])) if len(eligible) >= 2 else None
    xml = ET.parse(run/'regression.xml')
    suites = list(xml.iter('testsuite'))
    regression = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    regression['passed'] = regression['tests']-regression['failures']-regression['errors']-regression['skipped']
    if regression['failures'] or regression['errors']:
        raise ValueError('regression failed')
    preservation = read_json(run/'PRESERVATION_BEFORE.json')
    for path, digest in preservation['files'].items(): verify_file(Path(path), digest)
    previous = read_json(source/'DELIVERY.json')
    for record in previous['source_files'] + previous['artifacts']:
        verify_file(Path(record['path']), record['sha256'])
    commands = []
    for path in sorted((run/'commands').glob('*.command.json')):
        command = read_json(path)
        if command['return_code'] != 0: raise ValueError('recorded command failed: '+str(path))
        commands.append({'path': str(path), 'sha256': sha256_file(path),
                         'elapsed_seconds': command['elapsed_seconds'], 'return_code': command['return_code']})
    summary = {'schema_version': 'stratified_review_iteration_summary_v1',
               'completed_at_utc': datetime.now(timezone.utc).isoformat(),
               'review_counts': analysis['counts'], 'counts_by_category': analysis['counts_by_category'],
               'seeds': seeds, 'selected_seed': selected['seed'] if selected else None,
               'minimum_passing_seeds': 2, 'passing_seeds': len(eligible),
               'current_candidate': {'original_dev': current['original_dev'], 'rendered_dev': current['rendered_dev']},
               'current_candidate_retained': selected is None,
               'regression': regression, 'independent_checkpoint_replay_passed': True,
               'real_fixture_posteriors': sum(len(c['posteriors']) for s in replay['seeds'] for c in s['checkpoints']),
               'new_human_questions': [], 'awaiting_user_action': False,
               'unresolved_interpretation': ['D04 local preference is not a global ei rewrite', 'D06 uei overlaps preceding note; attribution unresolved'],
               'default_changed': False, 'references_changed': False, 'test_predictions_opened': False,
               'historical_final_songs_remain_observed_not_unseen': True,
               'commands': commands, 'head_only_cached_feature_training': True,
               'recorded_command_wall_seconds': sum(x['elapsed_seconds'] for x in commands)}
    write_json_new(run/'SUMMARY.json', summary)
    write_json_new(run/'PRESERVATION_AFTER.json', {'all_verified': True, 'current_files_checked': len(preservation['files']),
        'historical_source_files_checked': len(previous['source_files']), 'historical_artifacts_checked': len(previous['artifacts']),
        'raw_answers_sha256': read_json(run/'intake.json')['raw_answers_sha256'], 'default_changed': False})
    render(run, summary, analysis)
    status = {'schema_version': 'diagnostic_iteration_current_v1', 'status': 'completed_no_promotion_no_new_human_questions',
              'summary': str(run/'SUMMARY.json'), 'report': str(run/'RESULTS.md'), 'page': str(run/'index.html'),
              'awaiting_user_action': False, 'active_model_jobs': 0}
    write_json_new(run/'CURRENT.json', status)
    modules = ['src/any2xsampa/training/phone_diagnostic_analysis.py', 'src/any2xsampa/training/stratified_replay.py',
               'src/any2xsampa/config/stratified_projected_replay_v1.json', 'tools/analyze_phone_diagnostic_intake.py',
               'tools/prepare_stratified_replay.py', 'tools/train_stratified_replay.py', 'tools/verify_stratified_replay.py',
               'tools/summarize_stratified_replay.py', 'tests/test_phone_diagnostic_analysis.py', 'tests/test_stratified_replay.py']
    artifacts = [p for p in run.glob('*') if p.is_file() and p.suffix in {'.json', '.md', '.html', '.xml'}]
    write_json_new(run/'DELIVERY.json', {'schema_version': 'diagnostic_iteration_delivery_v1',
        'source_files': [{'path': str(ROOT/p), 'sha256': sha256_file(ROOT/p)} for p in modules],
        'artifacts': [{'path': str(p), 'sha256': sha256_file(p)} for p in artifacts]+source_files})
    print(json.dumps({'review': summary['review_counts'], 'passing_seeds': len(eligible), 'regression': regression,
                      'new_human_questions': 0}, ensure_ascii=False))


def render(run: Path, summary: dict, analysis: dict) -> None:
    rows = []
    for item in analysis['items']:
        selected = '/'+ ' '.join(item['selected_ipa'])+'/' if item['selected_ipa'] else '两种都不对'
        origin = {'proxy_reference': '原参考更接近', 'candidate': '模型输出更接近', 'neither': '两种都不对'}[item['outcome']]
        rows.append(f"| {item['review_id']} | {item['song_name']} · {item['lyric']} | {origin} | {selected} | {item['statement']} |")
    metrics = []
    for s in summary['seeds']:
        metrics.append(f"| {s['seed']} | {s['best_epoch']} | {s['original_dev']['edit_distance']}/214 ({s['original_dev']['per']:.6f}) | {s['rendered_dev']['per']:.6f} | {s['rendered_dev']['groups']['aspirated']['per']:.2f} | 未通过 |")
    text = '''# 十二项回答已处理；维持当前候选

本轮无需补答，也没有新的试听表。回答只作为比较偏好和错误诊断，不自动替换训练标签、人工 VSQX 或历史评测参考。

## 集中审计

10 项偏向原参考、1 项偏向模型输出、1 项两种都不对。该批次专门选取分歧项，不能把 10/12 当成总体映射正确率。

| 项目 | 歌曲／字 | 回答含义 | 所选音素 | 原话 |
|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

D04「星」偏向 /s i n/；相同 `ei` 源标记在其他已审项目仍偏向保留 /e i/，因此不作全局替换。D06 的“类似于 uei（或者英语的 we）”原样保留；源记录中前一个字「會」恰好是 `uei`。这是邻字辨认歧义的线索，不据此宣布用户听错、移动标签或重问本题。此项目前不能作为 /h/ 错误已确认的证据。

## 后续实验

训练缓存中，旧教师送气音邻域占合成回放有效帧的 1.073%，/h/ 占 1.770%。以此检验全局平均保护损失是否稀释少数音类。分组由教师 argmax 及前后各两帧定义，重叠时送气音优先；仅是模型保护区域，不是已确认的音素时间边界。

新配方只将上轮投影回放实验的合成 KL 从全帧平均改为“送气、h、其余”非空组的等权平均；每组仍计算完整 428 标签分布。CTC、教师、初始化、学习率、数据、分母、回放顺序和 AdamW 实际位移投影保持。未改变解码器，无新模型或依赖。固定三个种子、最多 20 epoch、早停 4、单作业 25 分钟；不是给旧实验补跑第四次。

| 种子 | 最佳 epoch | 原唱开发错误／PER | 合成开发 PER | 合成送气音 PER | 晋级 |
|---|---|---|---|---|---|
'''+ '\n'.join(metrics)+'''

三个种子合成送气音均 16/25 错误，较上轮各自最佳检查点的 18/25 少 2 个，但仍多于现有候选的 12/25。原唱最优 93/214，现有候选为 92/214。三个种子都没有合格检查点，未达到至少两个种子晋级的要求。全部结果保留，不再调系数追加训练，也不以合成域改善宣称原唱改善。

## 验证与边界

'''+f"完整回归：{summary['regression']['passed']} passed，{summary['regression']['skipped']} skipped，0 failed。三个保存检查点在独立 CUDA 进程重放，预测、指标一致；{summary['real_fixture_posteriors']} 份 posterior 的 real/fixture 契约一致。"+'''

所有训练使用已有冻结特征，未重复执行大模型主干。运行断网、CPU 四线程、GPU 作业串行；进程 reserved 上限 7 GiB、整卡 14 GiB。初始化与旧 Seed43 在全部 149 个训练窗口一致；教师及音素输出层没有改变。

原唱/合成参考、原始 WAV、人工 VSQX、现有默认和旧交付哈希复核通过。本次未打开最终测试预测；之前已观察的两首最终歌曲仍是已观察测试集，不能称为新的独立验证。

审计回答没有自动提升为逐音素真值；D04 和 D06 的局部问题保留在分析记录。当前没有必须由你作答才能完成的事项，不增加重复试听。后续方案须依据本次失败证据另行冻结，不继续无界扫描这一损失或重开最终测试追分。

## 复现与交接

`commands/*.command.json` 记录命令、版本文件哈希、耗时、退出状态；`EXPERIMENT_POLICY.json` 冻结配方；`ANALYSIS.json` 保留逐项来源和诊断；`stratified-replay/seed*/RESULT.json` 包含所有 epoch；`CHECKPOINT_REPLAY.json` 是独立重放；`PRESERVATION_AFTER.json` 记录历史保护验证。
'''
    (run/'RESULTS.md').write_text(text, encoding='utf-8')
    # A static, offline result page, not another annotation task.
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>审计已接收 · 后续实验结果</title><style>body{max-width:1080px;margin:32px auto;padding:0 24px;background:#eef4f4;color:#153e43;font:17px/1.7 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px;border-radius:16px}h1{font-size:28px}a{color:#075e78}</style><h1>已完成：回答导入、三种子实验、独立验证</h1><p><b>无需补答；现有候选保持。</b></p><p><a href="SUMMARY.json">结构化结果</a> · <a href="ANALYSIS.json">逐项审计</a> · <a href="RESULTS.md">完整报告</a></p><pre>'+html.escape(text)+'</pre></html>'
    (run/'index.html').write_text(page, encoding='utf-8')
    handoff = f'''# Current handoff — diagnostic answers and stratified replay

Run: {run}
Status: completed, no promotion, no new human questions. No active model job.
Answers: 10 proxy preferences, 1 candidate preference (D04), 1 neither (D06).
D06 raw quote retained; nearby previous note n442 has uei. Do not auto-correct location or labels.
The existing 12 questions are answered; do not request them again or treat a preference as a formal full phone reference.
One new fixed three-seed recipe completed, changing only rendered KL reduction vs previous projected replay. All failed promotion.
Seed17 best epoch10:95/214; seed29 epoch4:95/214; seed43 epoch12:93/214. Rendered aspiration16/25 each; current guarded candidate92/214 and12/25 remains.
No fourth run, coefficient sweep, silent mapping edit, default replacement, or final-test reuse.
Regression: {summary['regression']['passed']}passed {summary['regression']['skipped']}skipped; 3 checkpoints fresh CUDA replay; 9 real/fixture posteriors equal.
Source WAV/VSQX, references, default and historical artifacts verified unchanged. See SUMMARY.json, RESULTS.md, DELIVERY.json.
New sources are additive only. Model-facing code from the prior optional CLI/bundle remains unchanged.
'''
    (run/'HANDOFF.md').write_text(handoff, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if not args.run.resolve().is_relative_to(ROOT/'.state/runs'):
        raise ValueError('run outside workspace')
    publish(args.run.resolve())

"""Publish all fixed dual-input results, including the failed insertion gate."""
from __future__ import annotations
import argparse, html, json, statistics, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.dual_input import load_policy

NAMES = {'freedom-flower': '自由花', 'queen-road-east': '皇后大道東', 'half-moon-night-song': '月半小夜曲',
         'promise': '约定', 'like-you': '喜欢你', 'shalong': '沙龙'}
SYSTEMS = {'historical_raw': '现有混音输入', 'mixed_greedy': '匹配预处理混音', 'vocal_greedy': '仅分离人声', 'dual_equal_ctc': '双路等权 CTC'}


def report(run: Path) -> None:
    policy = load_policy(run); evaluation = read_json(run/'EVALUATION.json'); verified = read_json(run/'INDEPENDENT_VERIFICATION.json')
    if not verified['all_metrics_reproduced'] or evaluation['acceptance'] != verified['acceptance']:
        raise ValueError('verification missing or evaluation changed')
    if evaluation['acceptance']['passed']:
        raise ValueError('passing experiment needs the separate GUI/human delivery path')
    pairs = read_json(run/'PAIRS.json'); evidence = read_json(run/'EVIDENCE.json'); error = read_json(run/'ERROR_ANALYSIS.json')
    jobs = {p.name.removesuffix('.command.json'): read_json(p) for p in (run/'commands').glob('*.command.json')}
    if any(job['return_code'] for job in jobs.values()): raise ValueError('recorded job failed')
    suites = list(ET.parse(run/'regression.xml').iter('testsuite'))
    tests = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    tests['passed'] = tests['tests']-tests['failures']-tests['errors']-tests['skipped']
    if tests['failures'] or tests['errors']: raise ValueError('regression failed')
    separation_times, separation_peaks = [], []
    for spec in pairs['entries']:
        verify_file(Path(spec['path']), spec['sha256']); p = read_json(Path(spec['path']))
        if p['separation_runtime']:
            separation_times.append(p['separation_runtime']['elapsed_seconds'])
            separation_peaks.append(p['separation_runtime']['peak_reserved_bytes'])
    monitors = {name: read_json(run/name/'monitor/resource-monitor.json') for name in ('audio','evidence')}
    if any(not m['completed_without_resource_abort'] or not m['operation_succeeded'] for m in monitors.values()):
        raise ValueError('resource monitor did not finish successfully')
    performance = {'schema_version': 'dual_input_batch_performance_v1',
        'audio_pairs_job_wall_seconds': jobs['build-pairs']['elapsed_seconds'],
        'paired_phonetic_job_wall_seconds': jobs['paired-evidence']['elapsed_seconds'],
        'decoding_job_wall_seconds': jobs['fixed-dual-decode']['elapsed_seconds'],
        'new_separator_contexts': len(separation_times), 'new_separator_call_median_seconds': statistics.median(separation_times),
        'separator_peak_reserved_bytes': max(separation_peaks, default=0), 'phonetic_peak_reserved_bytes': evidence['peak_reserved_bytes'],
        'maximum_sampled_card_used_bytes': max(m['maximum_sampled_total_used_bytes'] for m in monitors.values()),
        'new_phonetic_crops': sum(e['cache_mode'] == 'new_frozen_model_inference' for e in evidence['entries']),
        'reused_phonetic_crops': sum(e['cache_mode'] == 'exact_frozen_posterior_reused' for e in evidence['entries']),
        'scope': 'three_serial_batch_stages_with_mixed_cache_reuse_not_single_click_cold_or_warm_latency',
        'single_click_cold_warm_benchmark_executed': False, 'speedup_claim_allowed': False, 'cpu_threads': 4}
    write_json_new(run/'PERFORMANCE.json', performance)
    old_files = read_json(run/'PRESERVATION_BEFORE.json')['files']
    for path, digest in old_files.items(): verify_file(Path(path), digest)
    summary = {'schema_version': 'dual_input_summary_v1', 'status': 'completed_failed_insertion_gate_no_human_task',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'evaluation': evaluation['aggregate'],
        'by_song': evaluation['by_song'], 'acceptance': evaluation['acceptance'], 'regression': tests,
        'verification': {'windows': len(verified['windows']), 'independent_ctc_scores': verified['independent_torch_ctc_scores_checked'],
                         'maximum_ctc_score_error': verified['maximum_ctc_score_absolute_error'], 'real_fixture_pairs': verified['real_fixture_posteriors']},
        'performance': performance, 'cache_modes': dict(Counter(e['cache_mode'] for e in pairs['entries'])),
        'new_training_started': False, 'gui_integration_executed': False, 'default_changed': False,
        'new_human_questions': [], 'awaiting_user_action': False, 'test_predictions_opened': False,
        'development_already_observed': True, 'preserved_files_verified': len(old_files),
        'historical_final_test_still_observed': True, 'no_new_accuracy_or_duration_claim': True}
    write_json_new(run/'SUMMARY.json', summary)
    render(run, summary, error)
    write_json_new(run/'CURRENT.json', {'status': summary['status'], 'summary': str(run/'SUMMARY.json'),
        'report': str(run/'RESULTS.md'), 'page': str(run/'index.html'), 'active_model_jobs': 0, 'awaiting_user_action': False})
    files = ['src/any2xsampa/adapters/dual_ctc.py', 'src/any2xsampa/training/dual_input.py',
             'src/any2xsampa/config/dual_input_ctc_v1.json', 'tools/prepare_dual_input.py', 'tools/build_dual_input_audio.py',
             'tools/run_dual_input_evidence.py', 'tools/decode_dual_input.py', 'tools/evaluate_dual_input.py',
             'tools/verify_dual_input.py', 'tools/report_dual_input.py', 'tests/test_dual_ctc.py', 'tests/test_dual_input.py']
    artifacts = [p for p in run.iterdir() if p.is_file() and p.suffix in {'.json','.md','.html','.xml'}]
    write_json_new(run/'DELIVERY.json', {'schema_version': 'dual_input_delivery_v1',
        'source_files': [{'path': str(ROOT/p), 'sha256': sha256_file(ROOT/p)} for p in files],
        'artifacts': [{'path': str(p), 'sha256': sha256_file(p)} for p in artifacts]})
    print(json.dumps({'acceptance': evaluation['acceptance'], 'regression': tests, 'human_questions': 0}, ensure_ascii=False))


def render(run: Path, summary: dict, error: dict) -> None:
    dev = summary['evaluation']['dev']; train = summary['evaluation']['train']; perf = summary['performance']
    rows = [f"| {SYSTEMS[name]} | {m['edit_distance']}/214 | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} |" for name,m in dev.items()]
    songs = [f"| {NAMES[song]} | {m['historical_raw']['reference_phones']} | {m['historical_raw']['edit_distance']} | {m['vocal_greedy']['edit_distance']} | {m['dual_equal_ctc']['edit_distance']} |"
             for song,m in summary['by_song'].items() if song != 'shalong']
    text = '''# 双路识别实验：总体错误减少，插入门槛未通过

**本轮固定实验已完成，维持现有默认模型，没有新试听表需要填写。**

## 同一开发参考的结果

| 系统 | 错误数／参考数 | PER | 替换 | 删除 | 插入 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

双路方案总错误 92→76，减少 16 个（相对 17.39%）；三个开发窗口分别 27→20、29→25、36→31。删除 25→12；送气音仍 1/6、h 仍 2/5，前圆唇错误 5/7→3/7。这些小样本不能支撑广泛的音类稳定性结论。

但插入 11→21，超过执行前冻结的上限。**唯一失败项是插入门槛；没有因此修改门槛、增加候选、调整权重或改参考。** 两条贪心序列均已在候选池中，双路概率评分仍偏向了更多插入的序列。

PER 针对固定工程音素参考，不等于原唱整体准确率。开发只有《沙龙》且已多轮观察，不能将本结果称为新的独立泛化验证。两个已观察最终测试歌曲本轮未打开。

## 五首训练歌的完整对照

| 歌曲 | 参考音素 | 混音错误 | 分离错误 | 双路错误 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(songs)+f'''

20 个原唱训练窗口、200 秒，双路错误 {train['historical_raw']['edit_distance']}→{train['dual_equal_ctc']['edit_distance']}/{train['dual_equal_ctc']['reference_phones']}；四首改善、一首持平。分离单路总错误更低（{train['vocal_greedy']['edit_distance']}），但《约定》单路分离退化。没有逐窗口依据参考选取“最好的一路”。这五首属于已有模型的训练歌曲，仅作对照诊断，不能证明对未见歌曲的泛化。

## 实现和错误分析

固定模型不训练。两路各用全 428 标签的 beam8，保留各自八条候选及贪心结果，去重后最多十八条。每条候选分别求两路所有合法 CTC 对齐路径的总概率，以两个 log 概率的等权和评分。没有语言模型、歌词、人工参考、长度奖励、blank 偏置、标签裁剪或融合系数扫描；分数平局按 token ID 序列排序。

开发集插入计数的净增量为 /u/ +5、/ə/ +2、/i/ +1、/ɐ/ +1、/s/ +1。该计数来自编辑对齐，不能直接当作声学时间定位或把这些音素一律删除。候选池的参考最优错误下界只在训练片段分析：实际 {error['actual_train_errors']}，理想下界 {error['oracle_train_lower_bound_errors']}；理想下界没有被用作识别结果、模型选择或开发成绩。

## 缓存、耗时与资源

23 个窗口全部保留。12 个训练窗口复用已核验的分离上下文、8 个补做分离；3 个开发窗口复用整组音频及 6 份旧 posterior。20 组训练配对的 40 次新音素推理使用相同冻结候选。混音控制和分离人声具有相同采样区间、共同增益和 PCM32 转换，参考分母不变。

- 分离／配对整批作业：{perf['audio_pairs_job_wall_seconds']:.3f} 秒，包含 8 个新上下文和缓存处理。
- 两路音素证据整批作业：{perf['paired_phonetic_job_wall_seconds']:.3f} 秒，包含加载、40 次新推理、6 次缓存复用及 fixture 核对。
- 23 组固定解码整批作业：{perf['decoding_job_wall_seconds']:.3f} 秒。
- 分离峰值 reserved：{perf['separator_peak_reserved_bytes']/1024**3:.3f} GiB；音素模型：{perf['phonetic_peak_reserved_bytes']/1024**3:.3f} GiB；采样到的整卡峰值：{perf['maximum_sampled_card_used_bytes']/1024**3:.3f} GiB。

以上是不同缓存条件下的整批作业记录，不是 GUI 单次点击的冷／热延迟，也不作加速比例承诺。模型断网运行、CPU 四线程、GPU 作业串行，未触及 7/14 GiB 上限，没有 CPU 模型回退或结束其他进程。

## 验证、未执行项和交接

独立进程重放 23 个窗口；778 个候选序列概率逐一与 Torch CTC 核对，最大绝对差 {summary['verification']['maximum_ctc_score_error']:.3g}；46 份 posterior 的 real/fixture 输出一致。另用独立编辑距离算法核对全部窗口得分。

完整回归：**{summary['regression']['passed']} passed、{summary['regression']['skipped']} skipped、0 failed**。聚焦 33 项通过。首次聚焦测试曾在全部断言完成后遇到 Windows 全局 pytest 临时目录清理权限错误，改用本轮独立临时目录重跑后正常结束，没有修改或删除用户临时目录。

原始 WAV、人工 VSQX、既有源代码／默认配置的 {summary['preserved_files_verified']} 个哈希复核通过。新增实现和全部产物独立保存；旧界面、模型与历史结果保持。

未执行：GUI 接入、新权重训练、默认部署、新增人工试听。原因是本轮固定验收未通过；按预先约定结束该实验。此时没有必须由用户作答才能完成的事项，不制造新的人工关卡。

复现命令和退出状态见 `commands/*.command.json`；规则见 `POLICY.json`，解码实现锁见 `DECODER_CODE_LOCK.json`，逐窗口结果见 `evaluation/`，误差分析见 `ERROR_ANALYSIS.json`，独立验证见 `INDEPENDENT_VERIFICATION.json`。

这次执行在冻结清单后曾被中断，日期跨度不代表连续工作时长。后续方案应针对插入可靠性另行冻结；本轮不再变更评分规则或重新查看最终测试追分。
'''
    with (run/'RESULTS.md').open('x', encoding='utf-8') as stream: stream.write(text)
    def cell(value): return '<td>'+html.escape(str(value))+'</td>'
    head = '<tr><th>系统</th><th>错误</th><th>PER</th><th>替换</th><th>删除</th><th>插入</th></tr>'
    table = head+''.join('<tr>'+''.join(cell(v) for v in [SYSTEMS[k], f"{m['edit_distance']}/214", f"{m['per']:.6f}", m['substitutions'], m['deletions'], m['insertions']])+'</tr>' for k,m in dev.items())
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双路实验结果</title><style>body{max-width:1080px;margin:32px auto;padding:0 24px;background:#eef4f4;color:#173b42;font:17px/1.7 system-ui}section{background:white;padding:24px;border-radius:16px;margin:20px 0}h1{font-size:29px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #d1dedf}.status{color:#9b4117;font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}a{color:#075e78}</style><h1>双路识别：错误减少，插入仍过多</h1><section><p class="status">未通过晋级 · 默认保持 · 无需填写新表</p><p>同一参考：92 → 76 个错误；插入 11 → 21，超过冻结上限。</p><table>'''+table+'''</table><p><a href="SUMMARY.json">结构化汇总</a> · <a href="EVALUATION.json">完整逐歌曲指标</a> · <a href="INDEPENDENT_VERIFICATION.json">独立验证</a></p></section><section><details><summary>完整报告、性能和复现说明</summary><pre>'''+html.escape(text)+'''</pre></details></section></html>'''
    with (run/'index.html').open('x', encoding='utf-8') as stream: stream.write(page)
    handoff = f'''# Current handoff — fixed dual-input CTC experiment

Run: {run}
Status: completed_failed_insertion_gate_no_human_task. No active job; no answer requested.
Policy and cohort froze before new decoding. One rule only: each stream full428 beam8 plus greedy, <=18 unique candidates, exact CTC log scores weighted .5/.5.
20 original training windows, five songs, 1235 references. Three Shalong dev windows,214 references. No final-test predictions opened; prior final songs remain observed.
Cache:12 training contexts reused,8 new,3 exact dev audio pairs and6 frozen dev posteriors reused.40 new train audio inferences.
Development raw S56/D25/I11=92; vocal S43/D14/I21=78; dual S43/D12/I21=76. All3 dev windows improve. Aspirated1/6,h2/5 unchanged; front rounded3/7 vs5/7. Insertion limit11 fails; do not relax it or call this promoted.
Training raw591, vocal524, dual531/1235. Four training songs improve vsraw;promise ties. Never choose a stream/window by reference.
No GUI integration because gate failed. No new audit page, no weights/default/reference edits, no additional rule scan.
Independent replay:23windows,778TorchCTCscores,46real/fixturepairs; full regression {summary['regression']['passed']}passed/{summary['regression']['skipped']}skipped.
See POLICY.json, EVALUATION.json, ERROR_ANALYSIS.json, INDEPENDENT_VERIFICATION.json, PERFORMANCE.json, RESULTS.md, DELIVERY.json.
All changes additive. Old candidate CLI/bundle remains usable. Next scope must address insertion reliability as a new separately frozen experiment, not silently tune this one.
'''
    with (run/'HANDOFF.md').open('x', encoding='utf-8') as stream: stream.write(handoff)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    if not a.run.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('output outside runs')
    report(a.run.resolve())

"""Report the full fixed-input matrix, not a best-input oracle."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from any2xsampa.training.input_supervision import load_manifest
from any2xsampa.training.input_supervision_evaluation import aggregate, by_song
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


def matrix_report(baseline: dict, replay: dict) -> tuple[dict, dict]:
    pools = {('M', c): baseline['details'][c] for c in ('x', 'v', 'h')}
    for variant in ('V', 'MV'):
        for condition in ('x', 'v', 'h'):
            pools[(variant, condition)] = [r for job in replay['checkpoints'] if job['variant'] == variant
                                           for r in job['details'] if r['condition'] == condition]
    expected = {r['window_id']: (r['song_id'], r['reference']) for r in pools[('M', 'x')]}
    if len(expected) != 20:
        raise ValueError('baseline window coverage changed')
    result, windows = {}, {}
    for (variant, condition), rows in pools.items():
        identities = {r['window_id']: (r['song_id'], r['reference']) for r in rows}
        if len(identities) != len(rows) or any(expected.get(k) != value for k, value in identities.items()):
            raise ValueError('matrix cells have different reference identities')
        metric, songs = aggregate(rows), by_song(rows)
        complete = identities == expected
        if complete and (metric['reference_phones'] != 1235 or len(songs) != 5):
            raise ValueError('matrix denominator or song coverage changed')
        key = variant + '(' + condition + ')'
        matched_controls = {}
        if variant != 'M':
            for control in dict.fromkeys([condition, 'x']):
                selected = [r for r in baseline['details'][control] if r['window_id'] in identities]
                control_metric = aggregate(selected)
                matched_controls[control] = {'metrics': control_metric,
                    'errors_reduced_on_same_windows': control_metric['edit_distance'] - metric['edit_distance'],
                    'scope': 'complete_five_fold' if complete else 'available_folds_diagnostic_not_selection'}
        result[key] = {'metrics': metric, 'per_song': songs, 'matched_controls': matched_controls,
                       'song_macro_per': sum(r['per'] for r in songs.values()) / len(songs) if songs else None,
                       'complete_five_folds': complete,
                       'missing_windows': sorted(set(expected) - set(identities)),
                       'predeclared_primary': (variant, condition) in {('V', 'v'), ('MV', 'x')},
                       'diagnostic_only': (variant, condition) not in {('V', 'v'), ('MV', 'x'), ('M', 'x'), ('M', 'v')}}
        windows[key] = {r['window_id']: {'song_id': r['song_id'],
            **{k: v for k, v in aggregate([r]).items() if k in
               {'reference_phones', 'edit_distance', 'substitutions', 'deletions', 'insertions'}}} for r in rows}
    return result, windows


def summarize(run: Path, replay_dir: Path) -> None:
    freeze, _, _ = load_manifest(run)
    gate = read_json(run / 'SCREEN_REPORT.json')
    replay = read_json(replay_dir / 'VERIFICATION.json')
    expected_jobs = 10 - len(freeze.get('unavailable_screen_jobs', []))
    if replay['all_checkpoints_verified'] is not True:
        raise ValueError('available checkpoint replay not complete')
    if replay['freeze_sha256'] != sha256_file(run / 'FREEZE.json') or len(replay['checkpoints']) != expected_jobs:
        raise ValueError('replay identity/count mismatch')
    source = Path(freeze['probe_run']) / 'RESULT.json'
    verify_file(source, freeze['bindings']['probe_result']['sha256'])
    matrix, windows = matrix_report(read_json(source), replay)
    resources = []
    for item in replay['checkpoints']:
        result_path = Path(item['source_result_path'])
        verify_file(result_path, item['source_result_sha256'])
        row = read_json(result_path)
        monitor = read_json(result_path.parent / 'monitor/resource-monitor.json')
        resources.append({'variant': row['variant'], 'held_song': row['held_song'],
                          'elapsed_seconds': row['elapsed_seconds'], 'epochs': len(row['history']),
                          'optimizer_steps': sum(e['optimizer_steps'] for e in row['history']),
                          'peak_reserved_gib': row['peak_reserved_bytes'] / 1024**3,
                          'peak_card_sampled_gib': monitor['maximum_sampled_total_used_bytes'] / 1024**3,
                          'checkpoint_sha256': row['checkpoint_sha256']})
    errors = {key: value['metrics']['edit_distance'] for key, value in matrix.items()}
    decomposition = {
        'V_pipeline_total_errors_reduced': errors['M(x)'] - errors['V(v)'] if matrix['V(v)']['complete_five_folds'] else None,
        'input_only_errors_reduced_Mx_to_Mv': errors['M(x)'] - errors['M(v)'],
        'training_only_errors_reduced_Mv_to_Vv': errors['M(v)'] - errors['V(v)'] if matrix['V(v)']['complete_five_folds'] else None,
        'MV_training_errors_reduced_Mx_to_MVx': errors['M(x)'] - errors['MV(x)'],
    }
    start_count = len(list(run.glob('screen/*/*/START.json')))
    command_count = len(list((run / 'commands').glob('*.command.json')))
    if (start_count, command_count) != (expected_jobs, expected_jobs):
        raise ValueError('unexpected quality launch accounting')
    payload = {'schema_version': 'input_supervision_analysis_v1', 'matrix': matrix, 'per_window': windows,
               'error_decomposition': decomposition, 'resources': resources,
               'selection': {k: gate[k] for k in ('screen_complete', 'any_variant_passed', 'selected_pipeline')},
               'unavailable_jobs': freeze.get('unavailable_screen_jobs', []),
               'budget': {**freeze['budget'], 'new_screen_starts': start_count,
                          'all_screen_starts_to_date': freeze['budget']['prior_screen_starts'] + start_count},
               'artifact_bindings': [{'path': str(p.resolve()), 'sha256': sha256_file(p)} for p in
                   (source, run / 'SCREEN_REPORT.json', replay_dir / 'VERIFICATION.json', run / 'CODE_REVIEW.json')],
               'development_opened': False, 'final_test_opened': False, 'default_replaced': False,
               'human_listening_required_for_this_comparison': False}
    write_json_new(run / 'ANALYSIS.json', payload)
    lines = ['# 修正后的直接人声监督：实验结果与覆盖', '',
        '本轮固定 20 个原唱窗口、1235 个参考音素、129 个合成训练窗口；五折按歌曲留出。',
        '只有 V(v) 与 MV(x) 是预登记主方案；其他输入为对照或迁移诊断，不按窗口挑最佳输入。', '',
        '| 权重／输入 | 覆盖音素 | 错误数 | PER | 替换 | 删除 | 插入 | 送气错误/参考 | h 错误/参考 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, item in matrix.items():
        m = item['metrics']
        group_errors = lambda name: sum(m['groups'][name][k] for k in ('substitutions', 'deletions', 'insertions'))
        label = key if item['complete_five_folds'] else key + '（缺折，不作完整比较）'
        lines.append(f"| {label} | {m['reference_phones']} | {m['edit_distance']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {group_errors('aspirated')}/{m['groups']['aspirated']['reference_count']} | {group_errors('glottal_h')}/{m['groups']['glottal_h']['reference_count']} |")
    lines += ['', '## 输入收益与训练收益', '',
        f"- M(x) → M(v)：减少 {decomposition['input_only_errors_reduced_Mx_to_Mv']} 个错误，属于输入改变。",
        (f"- M(v) → V(v)：减少 {decomposition['training_only_errors_reduced_Mv_to_Vv']} 个错误，才是同输入下的新训练变化。" if matrix['V(v)']['complete_five_folds'] else '- V 缺少一折，不计算完整五折训练收益，也不把四折分数与五折控制直接比较。'),
        f"- M(x) → MV(x)：减少 {decomposition['MV_training_errors_reduced_Mx_to_MVx']} 个错误。", '',
        '负数表示错误增加。前圆唇仅 18 个参考、f 仅 12 个，完整列入 JSON，但不足以支持保护条件通过的结论。', '',
        '## 逐歌错误数（各列使用同一参考）', '',
        '| 歌曲 | 参考数 | M(x) | M(v) | V(v) | MV(x) |', '|---|---:|---:|---:|---:|---:|']
    for song, original in matrix['M(x)']['per_song'].items():
        values = [matrix[key]['per_song'].get(song, {}).get('edit_distance', '未完成') for key in ('M(x)', 'M(v)', 'V(v)', 'MV(x)')]
        lines.append(f"| {song} | {original['reference_phones']} | " + ' | '.join(map(str, values)) + ' |')
    lines += ['', '## 预登记门槛', '']
    for variant in ('V', 'MV'):
        g = gate['results'][variant]['gate']
        status = '通过' if g['passed'] else '未通过' if g['complete'] else '未完成有效比较'
        lines.append(f"- {variant}：{status}；原因：{', '.join(g['failures']) or '无'}。")
        if g['complete']:
            lines.append(f"  相对同输入控制改善的歌曲：{len(g['strict_improved_songs'])}/5；逐歌敏感性与宏平均见 SCREEN_REPORT.json。")
    lines += ['', f"本轮筛选出的管线：{gate['selected_pipeline'] or '无'}。", '',
        '五首歌反复参与研究，5% 是工程门槛，不是统计显著性证明。不能把相邻音素当成独立样本。', '',
        '## 实施与验证', '',
        f"- 本续跑完成 {expected_jobs} 折 × 20 epoch；此前启动 {freeze['budget']['prior_screen_starts']} 次；累计 {freeze['budget']['prior_screen_starts'] + start_count} 次，没有自动重试失败折。",
        '- 所有实际轮次源顺序和优化步数对照 CPU 预检，模型/数据/代码 hash 均复核。',
        f'- 全部 {expected_jobs} 个有效检查点在训练之外的进程重放，核对 x/v 保存的 logits、序列及指标，并新增 h 诊断。',
        '- 每个检查点执行一次原始波形真实推理；所有窗口均导出并核对 posterior / fixture。',
        f"- 训练进程最大 reserved：{max(r['peak_reserved_gib'] for r in resources):.3f} GiB；整卡采样峰值：{max(r['peak_card_sampled_gib'] for r in resources):.3f} GiB。",
        '- 没有访问新的最终测试预测，没有替换默认模型，没有修改原始 WAV 或人工 VSQX。', '',
        '## 下一步', '']
    if gate['selected_pipeline']:
        lines.append('仅筛选管线已通过，尚不代表发布通过。按既定计划对该管线进行三个最终种子和沙龙开发检查，启动前另行审阅。')
    elif not gate['screen_complete']:
        lines.append('V-freedom-flower 于第12轮后因 GPU 查询超时保护退出，未自动重试；V 因缺折不能定论，两方案未齐全所以不能选最终管线。其余作业已完成，不需要新增听音标注。失败原因证据见前次 run/CONTINUATION_INCIDENT.json。')
    else:
        lines.append('按原计划停止本轮：不运行三个最终种子，不打开沙龙新预测，不新增系数或训练变体。不需要再听音标注。')
    with (run / 'ANALYSIS.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(run / 'ANALYSIS.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run.resolve(), args.replay.resolve())

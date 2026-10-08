"""Independent sequence/codec verification, fixed gates, regression and handoff."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
from safetensors.numpy import load_file
from speed_invariance_common import ROOT, bound, protocol, read_json, write_json_new, verify_file, sha256_file, slots, SONGS
from any2xsampa.training.speed_invariance import compare_gates
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import emission_payload, PhoneticXeusPosteriorRecognizer, ctc_collapse
from verify_acoustic_memory import scalar_edits, codec_identity


def verify_ledgers(run: Path) -> dict:
    totals = Counter(); completions = Counter(); by_purpose = Counter(); records = []
    for slot in slots():
        out = run / 'jobs' / slot
        exit_record = read_json(run / 'commands' / (slot + '.exit.json'))
        r = read_json(out / 'RESULT.json'); monitor = read_json(out / 'monitor/resource-monitor.json')
        if exit_record['exit_code'] != 0 or not r['complete'] or not monitor['operation_succeeded']:
            raise ValueError('incomplete model process')
        if r['peak_reserved_bytes'] > 7 * 1024**3 or monitor['maximum_sampled_total_used_bytes'] > 14 * 1024**3:
            raise ValueError('resource gate failed')
        if monitor['elapsed_seconds'] > 1501: raise ValueError('worker safety time exceeded')
        counts = Counter(); done = Counter()
        with (out / 'attempts.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line); kind = event['kind']
                if event['event'] == 'attempt':
                    counts[kind] += 1
                    if counts[kind] != event['number']: raise ValueError('ledger attempt gap')
                    if kind == 'head': by_purpose[event['purpose']] += 1
                elif event['event'] == 'completed':
                    done[kind] += 1
                    if counts[kind] != done[kind] or event['number'] != done[kind]: raise ValueError('ambiguous completion')
                else: raise ValueError('unknown ledger event')
        if dict(counts) != r['attempt_counts'] or counts != done: raise ValueError('budget evidence mismatch')
        totals.update(counts); completions.update(done)
        records.append({'slot': slot, 'attempts': dict(counts), 'elapsed_seconds': r['elapsed_seconds'],
            'peak_reserved_bytes': r['peak_reserved_bytes'], 'peak_card_sampled': monitor['maximum_sampled_total_used_bytes']})
    if dict(totals) != {'encoder': 55, 'head': 34206, 'optimizer': 4280}:
        raise ValueError('total fixed budget differs')
    if dict(by_purpose) != {'training': 33440, 'initial': 10, 'held': 298, 'seen_original': 160, 'held_replay': 298}:
        raise ValueError('head purpose accounting differs')
    if len(list((run / 'commands').glob('*.pid.json'))) != 13: raise ValueError('process count differs')
    return {'attempts': dict(totals), 'completions': dict(completions), 'head_purposes': dict(by_purpose),
        'processes': records, 'training_starts': 10, 'model_workers': 13, 'failed_starts': 0, 'retries': 0}


def decode_check(record: dict, reference: list[str], vocabulary: dict) -> tuple[list[str], dict]:
    verify_file(Path(record['raw']['path']), record['raw']['sha256'])
    logits = load_file(record['raw']['path'])['frame_logits']
    ids = ctc_collapse(logits.argmax(-1).tolist())
    if ids != record['ids']: raise ValueError('saved IDs differ from logits')
    inverse = {i: p for p, i in vocabulary.items()}; prediction = [inverse[i] for i in ids]
    m = phone_metrics([(reference, prediction)])
    independent = scalar_edits(reference, prediction)
    if independent != tuple(m[k] for k in ('substitutions', 'deletions', 'insertions')):
        raise ValueError('independent edit DP differs')
    return prediction, m


def verify_codec(record: dict, q: dict, freeze: dict, checkpoint: dict) -> dict:
    path = Path(record['raw']['path']); logits = load_file(str(path))['frame_logits']
    vocab = read_json(Path(freeze['vocabulary']['path']))
    payload = emission_payload(logits, vocab, input_audio=q['audio_metadata'], crop=q['crop'],
        model_metadata={**freeze['architecture']['snapshot'], 'architecture_sha256': freeze['architecture']['sha256'],
            'adapter_checkpoint': checkpoint}, geometry=q['geometry'],
        runtime={'device': 'cuda:0', 'offline': True, 'source': 'independently_replayed_cached_head', 'precision': 'FP32'})
    payload['raw_evidence'] = {'file': path.name, 'sha256': record['raw']['sha256']}
    posterior = path.with_name('posterior.json'); write_json_new(posterior, payload)
    real = PhoneticXeusPosteriorRecognizer(posterior, vocabulary_path=Path(freeze['vocabulary']['path'])).recognize(Path(q['audio_path']))
    fixture = PhoneticXeusPosteriorRecognizer(posterior, fixture=True, vocabulary_path=Path(freeze['vocabulary']['path'])).recognize(Path(q['audio_path']))
    if codec_identity(real) != codec_identity(fixture): raise ValueError('real/fixture PhoneSegment mismatch')
    expected = [p for p in payload['decoding']['collapsed_raw_labels'] if not p.startswith('<')]
    if [s.metadata['raw_label_trace']['label'] for s in real] != expected: raise ValueError('posterior IPA mismatch')
    return {'example_id': q['example_id'], 'posterior': bound(posterior), 'real_fixture_equal': True}


def score_and_verify(run: Path) -> dict:
    freeze = protocol(run, reviewed=True)
    replay = read_json(run / 'jobs/head-replay/RESULT.json')
    if not replay['complete']: raise ValueError('all outputs must be frozen before reference read')
    budgets = verify_ledgers(run)
    reference = {r['example_id']: r for r in read_json(run / 'SCORING.json')['rows']}
    queries = {q['example_id']: q for q in read_json(run / 'QUERIES.json')['rows']}
    vocabulary = read_json(Path(freeze['vocabulary']['path']))
    pools = {'C': [], 'A': []}; seen = {'C': {}, 'A': {}}; codecs = []; state_checks = []
    independent = {r['slot']: r for r in replay['rows']}
    for song in SONGS:
        for arm in ('C', 'A'):
            slot = arm + '--' + song; result = read_json(run / 'jobs' / slot / 'RESULT.json')
            verify_file(Path(result['checkpoint']['path']), result['checkpoint']['sha256'])
            verify_file(Path(result['initial_state']['path']), result['initial_state']['sha256'])
            if result['initial_state'] != freeze['initial'][song] or not result['frozen_parameters_unchanged']:
                raise ValueError('initial/frozen parameter mismatch')
            if result['checkpoint'] != independent[slot]['checkpoint']: raise ValueError('replay checkpoint identity mismatch')
            for r, rr in zip(result['held'], independent[slot]['rows'], strict=True):
                if any(r[k] != rr[k] for k in ('example_id', 'song_id', 'audio_role', 'cache_sha256', 'ids')):
                    raise ValueError('independent query identity changed')
                a, b = load_file(r['raw']['path']), load_file(rr['raw']['path'])
                if set(a) != set(b) or any(not np.array_equal(a[k], b[k]) for k in a):
                    raise ValueError('checkpoint logits do not independently reproduce')
                q = queries[r['example_id']]; ref = reference[r['example_id']]['reference']
                pred, m = decode_check(r, ref, vocabulary); pred2, m2 = decode_check(rr, ref, vocabulary)
                if pred != pred2 or m != m2: raise ValueError('independent metrics differ')
                pools[arm].append({'example_id': q['example_id'], 'song_id': song, 'audio_role': q['audio_role'],
                    'reference': ref, 'prediction': pred, 'metrics': m, 'checkpoint': result['checkpoint']})
                codecs.append({'arm': arm, **verify_codec(r, q, freeze, result['checkpoint'])})
            seen_rows = []
            for r in result['seen_original']:
                ref = reference[r['example_id']]['reference']; pred, m = decode_check(r, ref, vocabulary)
                if r['song_id'] == song: raise ValueError('seen diagnostic leaks held song')
                seen_rows.append({'example_id': r['example_id'], 'song_id': r['song_id'], 'reference': ref, 'prediction': pred, 'metrics': m})
            seen[arm][song] = {'rows': seen_rows, 'metrics': phone_metrics([(r['reference'], r['prediction']) for r in seen_rows])}
            state_checks.append({'arm': arm, 'song': song, 'initial': result['initial_state'], 'zero_step_error': result['initial_max_error'],
                'trainable_parameters': result['trainable_parameters'], 'training_frames': result['training_frames']})
    historical = read_json(run / 'HISTORICAL.json')['rows']
    gate = compare_gates(pools['A'], pools['C'], historical)
    old_lookup = {r['example_id']: r for r in historical}
    differences = [{'example_id': r['example_id'], 'role': r['audio_role'], 'song_id': r['song_id'],
        'historical_prediction': old_lookup[r['example_id']]['prediction'], 'current_prediction': r['prediction'],
        'historical_metrics': phone_metrics([(r['reference'], old_lookup[r['example_id']]['prediction'])]), 'current_metrics': r['metrics']}
        for r in pools['C'] if r['prediction'] != old_lookup[r['example_id']]['prediction']]
    return {'complete': True, 'gate': gate, 'pools': pools, 'seen_original': seen,
        'matched_control_vs_history_sequence_differences': differences, 'budget': budgets,
        'state_checks': state_checks, 'codec_checks': codecs, 'independent_head_replays': 298,
        'independent_metrics_checked': True, 'held_labels_read_only_after_all_predictions': True,
        'new_human_review_required': False, 'test_or_dev_opened': False, 'default_replaced': False}


def full_regression(run: Path) -> dict:
    command = [str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-m', 'pytest', '-q',
        '--basetemp=' + str(run / 'regression-temp'), '-o', 'cache_dir=' + str(run / 'pytest-cache'),
        '--junitxml=' + str(run / 'regression.xml')]
    write_json_new(run / 'regression.command.json', {'command': command, 'retry': False})
    start = time.monotonic()
    with (run / 'regression.log').open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    write_json_new(run / 'regression.exit.json', {'exit_code': result.returncode, 'elapsed_seconds': time.monotonic() - start})
    suite = ET.parse(run / 'regression.xml').getroot()
    stats = {k: sum(int(e.attrib.get(k, 0)) for e in suite.iter('testsuite')) for k in ('tests', 'failures', 'errors', 'skipped')}
    if result.returncode or stats['failures'] or stats['errors']: raise ValueError('full regression failed')
    return stats


def write_report(run: Path, r: dict, regression: dict, protected_count: int) -> None:
    gate = r['gate']; metrics = gate['metrics']
    lines = ['# 固定速度增强匹配对照：最终结果', '',
        ('固定门槛通过；仅为五歌、单seed筛选信号，后续验证另行授权。' if gate['passed'] else
         '固定门槛未通过，本配方关闭。不扫描速度、不自动重训或追加种子。'), '',
        '| 条件 | 原唱错误/1235 | S | D | I | PER | 合成错误/4814 |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('historical', 'C', 'A'):
        m = metrics[arm]['original']
        lines.append(f"| {arm} | {m['edit_distance']} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {m['per']:.6f} | {metrics[arm]['rendered']['edit_distance']} |")
    lines += ['', '门槛失败项：`' + ', '.join(gate['failures']) + '`。', '',
        '| 留出歌 | 历史错误 | C错误 | A错误 | 参考 |', '|---|---:|---:|---:|---:|']
    songs = gate['song_comparisons']
    for song in SONGS:
        a = songs['C']['current'][song]; c = songs['C']['baseline'][song]; h = songs['historical']['baseline'][song]
        lines.append(f"| {song} | {h['edit_distance']} | {c['edit_distance']} | {a['edit_distance']} | {a['reference_phones']} |")
    lines += ['', '## 原唱关键音类', '', '| 音类 | 参考数 | 历史PER | C PER | A PER |', '|---|---:|---:|---:|---:|']
    for group, a in metrics['A']['original']['groups'].items():
        h = metrics['historical']['original']['groups'][group]; c = metrics['C']['original']['groups'][group]
        lines.append(f"| {group} | {a['reference_count']} | {h['per']:.6f} | {c['per']:.6f} | {a['per']:.6f} |")
    lines += ['', '前圆唇18、f12样本不足，仅描述；合成音类保护和逐窗混淆完整见 RESULT.json。', '',
        '## 归因与限制', '',
        f"- 新C与历史endpoint有 {len(r['matched_control_vs_history_sequence_differences'])} 个窗口序列不同，全部保留。历史来源、调度和标签未改。新C/A使用同一显式CPU seed17初态和同实现的缓存head；旧实现加载整模型后构造tail，初始化上下文/数值路径并非保证逐位相同，不能将差异直接归因于增强。",
        '- 仅A/C配对差异可用于本次增强对照；A还须通过固定历史门槛。没有挑epoch、音频或最好歌曲。',
        '- 40个变速视图仍只来自200秒、20个原唱窗口，不是新人工数据。变速同时影响时长、音高、频谱，不能单独定位歌手/语速原因。',
        '- 所有评测输入是未变速的原混音/原合成；训练内诊断在 RESULT.json 单列，不用于选模型。五歌已反复观察，不能作为新的独立泛化证明。', '',
        '## 核验与预算', '',
        '- 55次完整编码（45主缓存＋10独立重放）、10个训练作业、4280次优化尝试全部完成；34206次缓存head请求，13个模型进程，失败/重试0。',
        '- 5个原样缓存核对、10个增强独立重放、298个检查点预测逐位重放；298份真实/fixture posterior契约及独立编辑距离复算通过。',
        f"- 本轮完整回归：{regression['tests']-regression['skipped']} passed，{regression['skipped']} skipped。保护来源/历史/默认配置哈希 {protected_count} 项不变。",
        '- CPU4线程、CUDA FP32、7GiB reserved / 14GiB整卡限制；资源峰值和每作业耗时在 RESULT.json budget.processes。运行断网、无新增模型或依赖。', '',
        '## 状态', '', '- 已完成：代码审阅、固定执行、独立重放、评分、错误分析、回归及保护核验。',
        '- 未执行：其他种子、其他速度/方法、沙龙开发、最终测试、部署。',
        '- 无新增听审；下一阶段只在用户另行授权后启动。',
        '- 本目录不可重跑；复现入口见 REPRODUCE.md（重跑需要独立授权及新目录）。', '']
    (run / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def finish(run: Path) -> None:
    r = score_and_verify(run); write_json_new(run / 'RESULT.json', r)
    stats = full_regression(run)
    f = protocol(run, reviewed=True); protected = read_json(run / 'PROTECTED.json')['files']
    for p, h in protected.items(): verify_file(Path(p), h)
    for spec in [f['head'], f['vocabulary'], *f['initial'].values()]: verify_file(Path(spec['path']), spec['sha256'])
    write_report(run, r, stats, len(protected))
    status = {'completed': True, 'gate_passed': r['gate']['passed'], 'fixed_recipe_closed': not r['gate']['passed'],
        'budget': r['budget'], 'regression': stats, 'protected_hashes_checked': len(protected),
        'awaiting_human_listening': False, 'default_model_changed': False, 'dev_test_opened': False}
    write_json_new(run / 'FINAL_STATUS.json', status)
    text = ('# 速度增强固定对照已完成\n\n执行目录：`' + str(run) + '`\n\n'
        '已完成55次编码、10个训练作业、4280次优化、独立重放和完整回归。不得重跑本目录或沿用预算。\n\n'
        + ('质量门槛通过，但不能自动追加验证或部署。' if r['gate']['passed'] else '质量门槛未通过，该固定配方关闭。')
        + '\n\n以 RESULTS.md、RESULT.json、FINAL_STATUS.json、FINAL_INDEX.json 为准。没有读取开发/最终测试、没有新人工听审。\n')
    handoff = ROOT / 'HANDOFF_SPEED_INVARIANCE_COMPLETED_2026-09-26.md'
    with handoff.open('x', encoding='utf-8') as stream: stream.write(text)
    excluded = {'offline', 'regression-temp', 'pytest-cache', 'prelaunch-temp', 'pytest-prelaunch'}
    artifacts = {str(p.resolve()): sha256_file(p) for p in run.rglob('*') if p.is_file() and not excluded.intersection(p.parts)}
    artifacts[str(handoff)] = sha256_file(handoff)
    write_json_new(run / 'FINAL_INDEX.json', {'artifacts': artifacts, 'status': status, 'created_at_utc': datetime.now(timezone.utc).isoformat()})
    for p, h in artifacts.items(): verify_file(Path(p), h)
    write_json_new(run / 'DELIVERY_VERIFIED.json', {'passed': True, 'artifacts': len(artifacts),
        'index': bound(run / 'FINAL_INDEX.json'), 'regression_this_run': stats})
    print('Completed. Gate passed:', r['gate']['passed'], flush=True)

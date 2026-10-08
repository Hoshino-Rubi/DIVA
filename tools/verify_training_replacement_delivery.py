"""Verify immutable correspondence evidence and publish the next human window."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.replacement_candidates import overlaps
from tools.build_replacement_location_review import validate_answers


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    packet=read_json(run/'candidate-packet.json')
    review=read_json(run/'location-review/review-packet.json')
    history=read_json(run/'HISTORY.json')
    if packet['packet_sha256']!=sha256_object({k:v for k,v in packet.items() if k!='packet_sha256'}):
        raise ValueError('candidate packet hash differs')
    if review['packet_sha256']!=sha256_object({k:v for k,v in review.items() if k!='packet_sha256'}):
        raise ValueError('review packet hash differs')
    if review['source_packet_sha256']!=packet['packet_sha256']:
        raise ValueError('review lineage mismatch')
    for spec in packet['source_bindings'].values():
        verify_file(Path(spec['path']),spec['sha256'])
    contexts=read_json(Path(packet['source_bindings']['contexts']['path']))['contexts']
    lookup={c['context_id']:c for c in contexts if c['split']=='train'}
    media=[]
    for item in review['items']:
        context=lookup[item['context_id']]
        if not context['start_sec']<=item['start_sec']<item['end_sec']<=context['end_sec']:
            raise ValueError('review outside existing training context')
        if any(overlaps(item,r) for r in history['retained']):
            raise ValueError('review overlaps retained candidate')
        verify_file(Path(item['alignment']['path']),item['alignment']['sha256'])
        for preview in [item['original']]+[a['preview'] for a in item['alternatives']]:
            for kind in ('audio','waveform'):
                path=run/'location-review/media'/preview[kind+'_file']
                verify_file(path,preview[kind+'_sha256']);media.append(path)
            verify_file(Path(preview['source_path']),preview['source_sha256'])
            with wave.open(str(run/'location-review/media'/preview['audio_file']),'rb') as wav:
                if abs(wav.getnframes()/wav.getframerate()-preview['duration_sec'])>1/16000:
                    raise ValueError('preview duration mismatch')
    for choice in ('none','uncertain'):
        validate_answers(review,{'schema_version':'replacement_location_answers_v1','packet_sha256':review['packet_sha256'],
            'reviewer':'verification_fixture_not_user','answers':[{'review_id':r['review_id'],'choice':choice,'comment':''} for r in review['items']]})
    # Re-render only this undelivered new page so the latest wording is included.
    encoded=json.dumps(review,ensure_ascii=False).replace('<',r'\u003c')
    (run/'location-review/index.html').write_text((ROOT/'tools/replacement_location_gui.html').read_text(encoding='utf-8').replace('__PACKET_JSON__',encoded),encoding='utf-8')
    raw_log=(run/'full-regression.log').read_bytes()
    regression=raw_log.decode('utf-16' if raw_log.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    if '822 passed, 3 skipped' not in regression:
        raise ValueError('full regression result missing')
    status={'schema_version':'training_replacement_delivery_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'user_authorized_replacement':True,'contexts_processed':15,'training_songs':5,'tested_windows':615,
        'automatic_admitted_replacements':0,'existing_retained_slots':7,'existing_rejections_preserved':13,
        'next_human_questions':len(review['items']),'question_scope':'choose_corresponding_section_only',
        'training_started':False,'new_model_or_dependency_installed':False,'test_audio_or_predictions_opened':False,
        'full_regression':{'passed':822,'skipped':3},'dom_tests_passed':3,
        'browser_visual_or_audio_listening_verified':False,'sources_reverified':len(packet['source_bindings']),
        'media_files_verified':len(media),'initial_runtime_attempt':'isolated PhoneticXeus environment lacked librosa; no install performed',
        'successful_runtime':'existing .venv; librosa 0.11.0; CPU four threads; offline audio matching only',
        'policy_not_relaxed':True}
    write_json_new(run/'FINAL_STATUS.json',status)
    lines=['# 替换候选：下一人工窗口','',
        '已绑定用户 2026-09-17 原话“允许重切替换”。仅使用现有十五个训练上下文，五首训练歌；未打开最终测试音频或预测。','',
        '## 已完成','',
        '- 分别使用原混音和已有分离人声，重新计算上下文到合成音频的局部 DTW 路径，不再只套用全曲固定偏移。未运行新模型。',
        '- 逐一检查 615 个十秒窗口；严格双路路径差异门槛下通过 0 个，没有降低阈值、伪造音素真值或为凑数启动训练。',
        '- 原七项未否决候选保留、十三项否决保留；目前仍无新增可训练窗口。',
        '- 保存全部 15 个上下文的两路前三个匹配结果，以及每个筛除窗口的原因。初筛只比较两路各自第一候选，不宣称其他路径组合也都无解。','',
        '## 为什么还需要人工','',
        '原混音与分离人声经常定位到不同的重复段落，或者局部时间路径相差超过 0.20 秒。低音频距离不能证明对应正确。',
        '因此下一步是四个段落定位问题，不是二十个新训练窗口，也不是逐音素重标。每题一段十秒原唱、三个合成备选。仅部分对应选“都不对应”；多个备选一样或无法确认选“不确定”。',
        '这四段来自只剩双路不一致问题的窗口，均不与七个保留候选重叠。月半小夜曲没有满足此送审条件的新窗口，不增加它的题目。',
        '备选只是音频对应假设，不能从分离效果或音高相似性自动获得人工资格。约定的两个备选仍含未知映射，页面明确标出；即使选择它们，也不能直接训练。','',
        '## 入口','',
        '`location-review/index.html`：离线页面，无服务器。导出 `replacement-location-answers.json` 后交回。',
        '选择段落只用于下一次局部对应修复；尚不确认首尾音素、时长或训练资格。B01–B06 不重答。','',
        '## 筛除统计（原因可重叠）','']
    lines.extend(f'- {key}: {count}' for key,count in packet['failure_counts'].items())
    lines.extend(['','## 验证','',
        f"源绑定重新验证 {len(packet['source_bindings'])} 项，媒体 {len(media)} 项。所有新问题仍位于原定十五个上下文，保留历史答案。",
        'Python 完整回归：822 passed，3 skipped。新页面 DOM 测试：3 passed，覆盖无默认答案、不重建表单、保存/恢复、导出和错误导入。',
        '未进行真实浏览器视觉或实际听音验证。没有联网、安装依赖、修改源 WAV/VSQX、历史映射、默认模型或旧答案。',
        '首次尝试的隔离识别环境缺少 librosa，失败保留在 training-replacements-20260917T211000；随后使用已有 .venv 中 librosa 0.11.0 完成 CPU 音频匹配，不是模型 CPU 回退。'])
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[run/'AUTHORIZATION.json',run/'FROZEN_POLICY.json',run/'HISTORY.json',run/'candidate-packet.json',
           run/'excluded-windows.json',run/'location-review/review-packet.json',run/'location-review/index.html',
           run/'FINAL_STATUS.json',run/'RESULTS.md',run/'full-regression.log']+media
    write_json_new(run/'DELIVERY.json',{'artifacts':{str(p.relative_to(run)):sha256_file(p) for p in files},
        'source_code':{str(p):sha256_file(ROOT/p) for p in [Path('src/any2xsampa/training/replacement_candidates.py'),
        Path('tools/build_training_replacements.py'),Path('tools/build_replacement_location_review.py'),Path('tools/replacement_location_gui.html')]}})
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':
    main()

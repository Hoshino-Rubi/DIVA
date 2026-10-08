"""Verify the completed review's lineage and measure training-only frame support."""
from __future__ import annotations
import argparse, os, re, sys
from collections import Counter
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.phone_diagnostic_analysis import summarize_answers, contiguous_note_prediction, frame_strata


def analyze(source: Path, intake: Path) -> None:
    import numpy as np
    from safetensors.numpy import load_file
    packet_path = source/'listening-review/packet.json'
    packet = read_json(packet_path)
    report = summarize_answers(packet, read_json(intake/'raw-answers.json'))
    imported = read_json(intake/'intake.json')
    verify_file(intake/'raw-answers.json', imported['raw_answers_sha256'])
    recomputed = {row['review_id']: row for row in report['items']}
    if len(imported['responses']) != len(recomputed) or any(
        old != {k: recomputed[old['review_id']][k] for k in old}
        for old in imported['responses']
    ):
        raise ValueError('imported response changed')
    policy = read_json(source/'POLICY.json')
    for key in ('policy', 'diagnosis'):
        filename = 'POLICY.json' if key == 'policy' else 'ERROR_DIAGNOSIS.json'
        verify_file(source/filename, packet[key+'_sha256'])
    manifest_path = ROOT/'.state/runs/limited-data-20260906T050251/reference/reference-manifest.json'
    verify_file(manifest_path, '120b45d2970e98d0bccba61d827004b72d50e531b739cfc52ac9f19bc10751c5')
    songs = {s['song_id']: s for s in read_json(manifest_path)['songs'] if s['split'] == 'train'}
    cache = read_json(source/'HEAD_CACHE.json')
    if cache['policy_sha256'] != sha256_file(source/'POLICY.json'):
        raise ValueError('head cache source policy changed')
    bindings = []
    for item in packet['items']:
        row = recomputed[item['review_id']]
        song = songs[item['song_id']]
        verify_file(Path(song['human_vsqx']['path']), item['source_vsqx_sha256'])
        entry = next(e for e in cache['entries'] if e['example_id'] == item['source_example_id'] and e['cohort'] == 'original_train')
        path = source/'examples'/(entry['cache_id']+'.json')
        verify_file(path, item['source_example_sha256'])
        example = read_json(path)
        extracted = contiguous_note_prediction(example, item['note_id'], 'candidate')
        expected = {origin: item['choices'][choice] for choice, origin in item['choice_sources'].items()}
        if extracted['prediction'] != expected['candidate'] or extracted['reference'] != expected['proxy_reference']:
            raise ValueError('displayed choices differ from complete note hypothesis')
        if example['audio_role'] != 'original' or example['song_id'] not in songs or example['split'] != 'train':
            raise ValueError('review source leakage')
        if example['audio_sha256'] != item['source_audio_sha256'] or any(example[k] != item[k] for k in ('start_sec', 'end_sec')):
            raise ValueError('review audio interval binding changed')
        row['systems_on_same_edit_alignment'] = {s: contiguous_note_prediction(example, item['note_id'], s) for s in ('pretrained', 'old_seed43', 'student', 'candidate')}
        nearby = {x['note_id'] for x in item['context']}
        statement_tokens = set(re.findall(r'[A-Za-z0-9_]+', row['statement']))
        row['verbatim_token_context_matches'] = [
            {'note_id': note['note_id'], 'lyric': note['raw_lyric'], 'raw_tokens': note['raw_v4c_tokens'],
             'status': 'possible_context_ambiguity_only_not_a_location_correction'}
            for note in song['notes'] if note['note_id'] in nearby and statement_tokens.intersection(note['raw_v4c_tokens'])]
        bindings.append({'path': str(path), 'sha256': item['source_example_sha256']})
    vocab_spec = policy['bindings']['vocabulary']
    verify_file(Path(vocab_spec['path']), vocab_spec['sha256'])
    vocab = read_json(Path(vocab_spec['path']))
    labels = [p for p, i in sorted(vocab.items(), key=lambda x: x[1])]
    stats = {role: {'frames': 0, 'strata': Counter(), 'windows_with_stratum': Counter(), 'kl_sum': Counter(), 'windows': 0}
             for role in ('original_train', 'rendered_train')}
    for entry in cache['entries']:
        if entry['cohort'] not in stats:
            continue
        if entry['split'] != 'train' or entry['song_id'] not in songs:
            raise ValueError('support analysis contains development/test data')
        verify_file(Path(entry['evidence_path']), entry['evidence_sha256'])
        tensors = load_file(entry['evidence_path'])
        teacher, student = tensors['old_seed43'], tensors['student']
        masks = frame_strata(np, teacher, labels)
        def log_softmax(x):
            v = x.astype(np.float64)/2
            v -= v.max(-1, keepdims=True)
            return v - np.log(np.exp(v).sum(-1, keepdims=True))
        lp, lq = log_softmax(teacher), log_softmax(student)
        kl = (np.exp(lp)*(lp-lq)).sum(-1)*4
        result = stats[entry['cohort']]
        result['frames'] += len(teacher); result['windows'] += 1
        for name, mask in masks.items():
            result['strata'][name] += int(mask.sum())
            result['windows_with_stratum'][name] += int(mask.any())
            result['kl_sum'][name] += float(kl[mask].sum())
    for value in stats.values():
        value['frame_fraction'] = {k: v/value['frames'] for k, v in value['strata'].items()}
        value['global_mean_kl_contribution'] = {k: v/value['frames'] for k, v in value['kl_sum'].items()}
        value['stratum_mean_kl'] = {k: value['kl_sum'][k]/n if n else None for k, n in value['strata'].items()}
    if [stats[x]['windows'] for x in stats] != [20, 129]:
        raise ValueError('training-only support denominator changed')
    report.update(source_bindings=bindings, source_packet={'path': str(packet_path), 'sha256': sha256_file(packet_path)},
                  source_intake_sha256=sha256_file(intake/'intake.json'), teacher_support=stats,
                  test_predictions_loaded=False, development_logits_used_for_support_analysis=False)
    write_json_new(intake/'ANALYSIS.json', report)
    print(report['counts'])
    for role, value in stats.items():
        print(role, value['frames'], value['frame_fraction'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--intake', type=Path, required=True)
    args = parser.parse_args()
    if not args.intake.resolve().is_relative_to(ROOT/'.state/runs'):
        raise ValueError('output outside runs')
    analyze(args.source.resolve(), args.intake.resolve())

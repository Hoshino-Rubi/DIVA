"""Freeze the one dual-input experiment and its audio-only requests."""
from __future__ import annotations
import argparse, sys, wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.decoder_experiment import TRAIN_SONGS

QUALITY = ROOT/'.state/runs/quality-first8h-20260907T010738'
DECODER = ROOT/'.state/runs/decoder-usability-20260920T011000'
GUARD = ROOT/'.state/runs/guarded-retention-20260919T041500'


def prepare(output: Path) -> None:
    paths = {'config': ROOT/'src/any2xsampa/config/dual_input_ctc_v1.json',
             'source_policy': DECODER/'POLICY.json', 'raw_head_cache': DECODER/'HEAD_CACHE.json',
             'scoring_references': DECODER/'SCORING_REFERENCES.json', 'raw_comparison': DECODER/'DECODER_COMPARISON.json',
             'candidate': GUARD/'CANDIDATE_FREEZE.json', 'guarded_transfer': GUARD/'transfer/TRANSFER.json',
             'original_manifest': ROOT/'.state/runs/original-adaptation-20260919T012000/original-training-manifest.json',
             'separator_assets': QUALITY/'assets/assets-lock.json', 'separator_start': QUALITY/'execution-start.json',
             'separation_index': QUALITY/'separation-contexts/index.json', 'old_pairs': QUALITY/'paired-inputs-v1/paired-inputs.json',
             'paired_phonetic_index': QUALITY/'paired-phonetic-v1/index.json',
             'auxiliary_environment': QUALITY/'aux-environment.json'}
    source = read_json(paths['source_policy'])
    for binding in source['bindings'].values(): verify_file(Path(binding['path']), binding['sha256'])
    cache = read_json(paths['raw_head_cache'])
    if cache['policy_sha256'] != sha256_file(paths['source_policy']): raise ValueError('raw cache policy mismatch')
    old_pairs = read_json(paths['old_pairs'])['pairs']; separation = read_json(paths['separation_index'])
    records = []
    for entry in separation['entries']:
        verify_file(Path(entry['record_path']), entry['record_sha256'])
        record = read_json(Path(entry['record_path']))
        for role in ('mixed', 'vocal'): verify_file(Path(record[role]['path']), record[role]['sha256'])
        records.append((entry, record))
    requests, scores, sources = [], [], {}
    for entry in cache['entries']:
        if entry['cohort'] not in {'original_train', 'original_dev'}: continue
        row = read_json(DECODER/'examples'/(entry['cache_id']+'.json'))
        if (row['split'] == 'train' and row['song_id'] not in TRAIN_SONGS) or (row['split'] == 'dev' and row['song_id'] != 'shalong'):
            raise ValueError('song leakage')
        path = Path(row['audio_path']); verify_file(path, row['audio_sha256'])
        with wave.open(str(path), 'rb') as audio:
            rate = audio.getframerate(); duration = audio.getnframes()/rate
        start, end = row['start_sec'], row['end_sec']
        first, last = round(start*rate), round(end*rate)
        begin, finish = max(0., start-6), min(duration, end+6)
        if not 8 <= end-start <= 20 or finish-begin > 30.000001: raise ValueError('crop/context too long')
        available = [(e, r) for e, r in records if r['clock']['source']['sha256'] == row['audio_sha256']
            and r['clock']['effective_original_start_sec'] <= begin+1/44100
            and r['clock']['effective_original_end_sec'] >= finish-1/44100]
        chosen = min(available, key=lambda value: (
            abs((value[1]['clock']['effective_original_start_sec']+value[1]['clock']['effective_original_end_sec'])/2-(start+end)/2),
            value[0]['context']['context_id'])) if available else None
        exact = [p for p in old_pairs if p['context']['audio_sha256'] == row['audio_sha256']
                 and abs(p['context']['old_start_sec']-start) < 1e-8 and abs(p['context']['old_end_sec']-end) < 1e-8]
        request = {k: row[k] for k in ('example_id', 'song_id', 'split', 'audio_path', 'audio_sha256', 'start_sec', 'end_sec')}
        request.update(source_rate=rate, source_start_sample=first, source_end_sample_exclusive=last,
                       context_start_sec=begin, context_end_sec=finish,
                       reuse_separation={'path': chosen[0]['record_path'], 'sha256': chosen[0]['record_sha256']} if chosen else None,
                       reuse_exact_pair=exact[0] if exact else None,
                       contains_reference_context=False)
        requests.append(request)
        scores.append({k: row[k] for k in ('example_id', 'song_id', 'split', 'reference', 'source_phones', 'cache_id')})
        sources[str(path)] = row['audio_sha256']
    if Counter(r['split'] for r in requests) != {'train': 20, 'dev': 3} or sum(len(r['reference']) for r in scores if r['split'] == 'dev') != 214:
        raise ValueError('fixed cohort changed')
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output/'AUDIO_REQUESTS.json', {'schema_version': 'dual_input_audio_requests_v1', 'requests': requests, 'contains_reference_context': False, 'test_allowed': False})
    write_json_new(output/'SCORING_REFERENCES.json', {'schema_version': 'dual_input_scoring_references_v1', 'rows': scores, 'decoding_conditioning_allowed': False})
    paths.update(audio_requests=output/'AUDIO_REQUESTS.json', references=output/'SCORING_REFERENCES.json')
    policy = read_json(paths['config']); policy.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        quality_asset_run=str(QUALITY), phonetic_snapshot=str(ROOT/'.state/runs/limited-data-20260906T050251/model'),
        bindings={k: {'path': str(p), 'sha256': sha256_file(p)} for k,p in paths.items()})
    policy['policy_sha256'] = sha256_object(policy); write_json_new(output/'POLICY.json', policy)
    before = {}
    for folder in ('src', 'tools'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.js', '.ps1', '.html', '.css'}:
                before[str(path)] = sha256_file(path)
    before.update(sources)
    manifest = read_json(ROOT/'.state/runs/limited-data-20260906T050251/reference/reference-manifest.json')
    for song in manifest['songs']:
        if song['split'] not in {'train', 'dev'}: continue
        binding = song['human_vsqx']; verify_file(Path(binding['path']), binding['sha256']); before[binding['path']] = binding['sha256']
    write_json_new(output/'PRESERVATION_BEFORE.json', {'files': before, 'source_audio_count': len(sources), 'test_audio_opened': False})
    print({'windows': len(requests), 'cached_contexts': sum(r['reuse_separation'] is not None for r in requests),
           'exact_cached_pairs': sum(r['reuse_exact_pair'] is not None for r in requests), 'new_contexts': sum(r['reuse_separation'] is None for r in requests)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('output outside runs')
    prepare(a.output.resolve())

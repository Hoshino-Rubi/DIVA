"""Freeze the authorized zero-training cohort before any new acoustic inference."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
from input_native_probe_common import ROOT, SONGS, first_per_song, check_midpoint
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.dual_input import validate_pair

RUNS = ROOT / '.state/runs'
PROBE = RUNS / 'input-condition-probe-20260921T20260921T195018'
DUAL = RUNS / 'dual-input-ctc-20260920T073445032166Z'
RESEARCH = RUNS / 'deep-research-20260923T145957Z'


def prepare(run: Path) -> None:
    import numpy as np
    import soundfile as sf
    if run.exists() or not run.is_relative_to(RUNS):
        raise ValueError('new run directory required')
    bindings = {}
    def bind(path: Path, expected: str | None = None) -> dict:
        path = path.resolve()
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError('source hash mismatch: ' + str(path))
        bindings[str(path)] = actual
        return {'path': str(path), 'sha256': actual}
    for p, h in read_json(RESEARCH / 'RESEARCH_COMPLETE.json')['artifacts'].items():
        bind(Path(p), h)
    bind(ROOT / 'HANDOFF_RESEARCH_NEXT_2026-09-24.md')
    bundle_path = RUNS / 'decoder-usability-20260920T011000/candidate-bundle-v1.json'
    bind(bundle_path); bundle = read_json(bundle_path)
    for name in ('candidate', 'snapshot_lock', 'python', 'vocabulary'):
        bind(Path(bundle[name]['path']), bundle[name]['sha256'])
    candidate = read_json(Path(bundle['candidate']['path']))
    for name in ('teacher_checkpoint', 'student_checkpoint', 'policy'):
        bind(Path(candidate[name]['path']), candidate[name]['sha256'])
    for base, names in [(PROBE, ('CONDITIONS.json', 'INPUT_REQUESTS.json', 'CACHE.json', 'EVALUATION_REFERENCES.json', 'RESULT.json', 'SOURCE_BINDINGS.json')),
                        (DUAL, ('AUDIO_REQUESTS.json', 'PAIRS.json', 'EVIDENCE.json', 'EVALUATION.json', 'SCORING_REFERENCES.json', 'POLICY.json'))]:
        for name in names:
            bind(base / name)
    mv = RUNS / 'input-supervision-20260923T140519Z-remaining'
    for name in ('FREEZE.json', 'CACHE_BINDINGS.json', 'independent-replay/VERIFICATION.json'):
        bind(mv / name)
    old_evidence = {(r['example_id'], {'mixed': 'x', 'vocal': 'v'}[r['condition']]): r
                    for r in read_json(DUAL / 'EVIDENCE.json')['entries']}
    refs = read_json(PROBE / 'EVALUATION_REFERENCES.json')['rows']
    if len(refs) != 20 or sum(len(r['target_ipa']) for r in refs) != 1235:
        raise ValueError('frozen train reference changed')
    first = first_per_song(refs)
    requests = {r['example_id']: r for r in read_json(PROBE / 'INPUT_REQUESTS.json')['rows']}
    originals = {r['example_id']: r for r in read_json(DUAL / 'AUDIO_REQUESTS.json')['requests']}
    conditions = read_json(PROBE / 'CONDITIONS.json')['rows']
    cache = {r['condition_id']: r for r in read_json(PROBE / 'CACHE.json')['entries']}
    for r in conditions:
        bind(Path(r['audio_path']), r['audio_sha256'])
        bind(Path(r['pair_path']), r['pair_sha256'])
    waveform = []
    for ref in refs:
        wid = ref['window_id']
        cr = next(r for r in conditions if r['window_id'] == wid)
        pair = read_json(Path(cr['pair_path']))
        validate_pair(originals[wid], pair)
        if pair['song_id'] != ref['song_id'] or pair['split'] != 'train':
            raise ValueError('paired song/split mismatch')
        if pair['source_audio_sha256'] != ref['source_audio_sha256'] or abs(pair['effective_original_start_sec'] - ref['source_start_sec']) > 1/44100:
            raise ValueError('paired reference clock mismatch')
        bind(Path(pair['source_audio_path']), pair['source_audio_sha256'])
        arrays = []
        for c in ('x', 'v', 'h'):
            req = requests[wid + '--' + c]
            array, rate = sf.read(req['audio_path'], dtype='float32', always_2d=True)
            if rate != 44100 or array.shape != (441000, 2):
                raise ValueError('ten-second stereo pair required')
            arrays.append(array)
            entry = cache[wid + '--' + c]
            bind(Path(entry['path']), entry['sha256'])
            if entry['audio_sha256'] != req['audio_sha256'] or entry['crop']['source_start_sample'] != 0:
                raise ValueError('P cache input mismatch')
        waveform.append({'window_id': wid, **check_midpoint(*arrays)})
    run.mkdir(parents=True)
    tasks = {k: [] for k in ('A0', 'A1', 'B0', 'B1', 'AR', 'BR')}
    for wid in first:
        for c in ('x', 'v'):
            old = old_evidence[wid, c]['artifact']
            for field in ('posterior', 'evidence'):
                bind(Path(old[field + '_path']), old[field + '_sha256'])
            pp = read_json(Path(old['posterior_path']))
            req = requests[wid + '--' + c]
            if pp['producer']['input_audio']['sha256'] != req['audio_sha256']:
                raise ValueError('G baseline audio identity mismatch')
            tasks['A0'].append({'id': wid + '--' + c, 'request': req, 'mode': 'G',
                               'compare': {'path': old['evidence_path'], 'sha256': old['evidence_sha256'], 'key': 'frame_logits'},
                               'expected_crop': pp['producer']['crop']})
    for ref in refs:
        wid = ref['window_id']
        task = {'id': wid + '--h', 'request': requests[wid + '--h'], 'mode': 'G'}
        tasks['A1'].append(task)
        if wid in first:
            tasks['AR'].append({**task, 'replay_of': 'A1'})
        for c in ('x', 'v', 'h'):
            entry = cache[wid + '--' + c]
            task = {'id': wid + '--' + c, 'request': requests[wid + '--' + c], 'mode': 'P', 'capture': True,
                    'compare': {'path': entry['path'], 'sha256': entry['sha256'], 'key': 'frame_logits'},
                    'expected_crop': entry['crop'], 'expected_architecture': entry['architecture_sha256']}
            tasks['B1'].append(task)
            if c == 'h' and wid in first:
                tasks['BR'].append({**task, 'replay_of': 'B1'})
    origin = originals[first[0]]
    start = max(origin['context_start_sec'], origin['start_sec'] - 5)
    if start + 20 > origin['context_end_sec']:
        raise ValueError('fixed B0 smoke exceeds existing training context')
    for seconds in (8, 20):
        for capture in (False, True):
            ident = f'smoke-{seconds}-' + ('observed' if capture else 'ordinary')
            req = {k: origin[k] for k in ('song_id', 'split', 'audio_path', 'audio_sha256')}
            req.update(example_id=ident, audio_role='original', start_sec=start, end_sec=start + seconds)
            tasks['B0'].append({'id': ident, 'request': req, 'mode': 'P', 'capture': capture,
                                'compare_smoke': f'smoke-{seconds}-ordinary' if capture else None})
    for stage, rows in tasks.items():
        file = run / 'tasks' / (stage + '.json')
        write_json_new(file, {'stage': stage, 'tasks': rows, 'labels_received': False})
        bind(file)
    bindfile = run / 'REFERENCES.json'
    write_json_new(bindfile, {'rows': refs, 'first_per_song': first, 'decoding_conditioning_allowed': False}); bind(bindfile)
    for item in read_json(DUAL / 'EVALUATION.json')['windows']:
        bind(Path(item['path']), item['sha256'])
    write_json_new(run / 'PREPARATION.json', {'waveform_checks': waveform, 'first_per_song': first,
        'research_hashes_verified': 12, 'new_model_forward_calls': 0, 'training_starts': 0,
        'B0_original_context': {'window_id': first[0], 'start_sec': start, 'end_sec': start + 20},
        'G_runtime_note': 'Legacy effective architecture retained. New reviewed worker; historical bundle/code locks never overwritten.'})
    write_json_new(run / 'FREEZE.json', {'schema_version': 'input_native_probe_freeze_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'bindings': bindings,
        'model_snapshot': bundle['snapshot'], 'vocabulary': bundle['vocabulary'], 'candidate': bundle['candidate'],
        'python': bundle['python'], 'probe_run': str(PROBE), 'dual_run': str(DUAL), 'mv_run': str(mv),
        'first_per_song': first, 'training_starts_allowed': 0, 'full_forward_limit': 111,
        'new_weights': False, 'new_dependencies': False, 'automatic_retry': False, 'default_replacement': False,
        'stage_limits': {'A0':10,'A1':20,'A2':4,'B0':4,'B1':60,'AR':5,'ADR':3,'BR':5}})
    print(str(run), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    prepare(p.parse_args().run.resolve())

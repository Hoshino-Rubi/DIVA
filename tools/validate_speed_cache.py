"""CPU-only acceptance of all derived sources and independent encoder replay."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from speed_invariance_common import read_json, write_json_new, verify_file, sha256_file, bound, protocol, SONGS
from any2xsampa.training.speed_invariance import pcm_hash, RATIOS
from any2xsampa.training.singing_domain_losses import required_frames
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def accept(run: Path) -> dict:
    freeze = protocol(run, reviewed=True)
    records = {}
    for slot in ('encode', 'encode-replay'):
        if read_json(run / 'commands' / (slot + '.exit.json'))['exit_code'] != 0:
            raise ValueError('encoder process did not complete')
        result = read_json(run / 'jobs' / slot / 'RESULT.json')
        if not result['complete'] or result['encoder_forwards'] != (45 if slot == 'encode' else 10):
            raise ValueError('encoder coverage incomplete')
        records[slot] = result['rows']
    source = {(r['source_query']['example_id'], r['view']): r for r in records['encode']}
    if len(source) != 45: raise ValueError('duplicate encoder request')
    targets = {}; reference_hashes = {}
    for song in SONGS:
        for row in read_json(run / 'folds' / song / 'TRAINING.json')['rows']:
            key = row['query']['example_id']
            if key in targets and targets[key] != row['target_ids']: raise ValueError('fold reference changed')
            targets[key] = row['target_ids']
            if key in reference_hashes and reference_hashes[key] != row['reference_sha256']:
                raise ValueError('reference provenance differs across folds')
            reference_hashes[key] = row['reference_sha256']
    details = []; supervision = []; bindings = [bound(run / 'jobs' / s / 'RESULT.json') for s in records]
    for record in records['encode']:
        q, t = record['source_query'], record['transform']; up, down = RATIOS[record['view']]
        if record['architecture_sha256'] != freeze['architecture']['sha256'] or q['architecture_sha256'] != record['architecture_sha256']:
            raise ValueError('encoder model identity mismatch')
        for spec in (record['pcm'], record['raw']): verify_file(Path(spec['path']), spec['sha256']); bindings.append(spec)
        verify_file(Path(q['audio_path']), q['audio_sha256'])
        wave = np.load(record['pcm']['path'], allow_pickle=False)
        if (wave.dtype != np.float32 or wave.ndim != 1 or not np.isfinite(wave).all() or
            len(wave) != (t['original_samples'] * up + down - 1) // down or
            pcm_hash(wave) != t['derived_pcm_sha256'] or t['source_crop'] != q['crop'] or
            t['source_audio_sha256'] != q['audio_sha256'] or t['source_song_id'] != q['song_id']):
            raise ValueError('source/PCM/sample offset mismatch')
        arrays = load_file(record['raw']['path'])
        if set(arrays) != {'encoder_features', 'frame_logits'}: raise ValueError('unexpected encoder outputs')
        for name, width in (('encoder_features', 1024), ('frame_logits', 428)):
            a = arrays[name]
            if a.shape != (record['frames'], width) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('invalid encoded finite tensors')
        if required_frames(targets[q['example_id']]) > record['frames']:
            raise ValueError('full reference no longer CTC feasible; no dropping phones permitted')
        if record['view'] == 'raw' and not record['old_raw_cache_check']['greedy_equal']:
            raise ValueError('raw cache check missing')
        if record['view'] != 'raw':
            supervision.append({'source_example_id': q['example_id'], 'view': record['view'],
                'source_reference_sha256': reference_hashes[q['example_id']], 'cache': record['raw'],
                'pcm': record['pcm'], 'full_sequence_preserved': True, 'new_human_confirmation': False})
    for replay in records['encode-replay']:
        key = (replay['source_query']['example_id'], replay['view']); expected = source[key]
        if replay['transform'] != expected['transform'] or replay['source_query'] != expected['source_query']:
            raise ValueError('independent waveform/source identity mismatch')
        for spec in (replay['pcm'], replay['raw']): verify_file(Path(spec['path']), spec['sha256']); bindings.append(spec)
        left, right = load_file(expected['raw']['path']), load_file(replay['raw']['path'])
        errors = {k: float(np.max(np.abs(left[k] - right[k]))) for k in left}
        bitwise = {k: np.array_equal(left[k], right[k]) for k in left}
        if max(errors.values()) > 2e-5 or ctc_collapse(left['frame_logits'].argmax(-1).tolist()) != ctc_collapse(right['frame_logits'].argmax(-1).tolist()):
            raise ValueError('independent encoder replay outside frozen tolerance')
        if not np.array_equal(np.load(expected['pcm']['path']), np.load(replay['pcm']['path'])):
            raise ValueError('independent PCM not exact')
        details.append({'key': list(key), 'max_abs': errors, 'bitwise': bitwise, 'pcm_equal': True, 'greedy_equal': True})
    result = {'passed': True, 'encoder_forwards': 55, 'raw_checks': 5, 'derived_ctc_windows': 40,
        'unique_original_seconds': 200, 'reference_sequences_unchanged': True, 'independent_replay': details,
        'derived_supervision_bindings': supervision,
        'bindings': bindings, 'code_review_sha256': sha256_file(run / 'CODE_REVIEW.json'),
        'derived_supervision_not_new_human_truth': True}
    write_json_new(run / 'CACHE_ACCEPTED.json', result)
    return result

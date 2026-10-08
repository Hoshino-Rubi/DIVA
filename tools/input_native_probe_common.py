"""Zero-training probe invariants; no model loading at import time."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.singing_domain import validate_audio_row
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse

SONGS = ('freedom-flower', 'half-moon-night-song', 'like-you', 'promise', 'queen-road-east')
LIMITS = {'A0': 10, 'A1': 20, 'A2': 4, 'B0': 4, 'B1': 60, 'AR': 5, 'ADR': 3, 'BR': 5}
REQUEST_KEYS = {'example_id', 'song_id', 'split', 'audio_role', 'audio_path', 'audio_sha256', 'start_sec', 'end_sec'}


def prediction(values, vocabulary: dict) -> list[str]:
    import numpy as np
    if values.ndim != 2 or values.shape[1] != 428 or not len(values) or not np.isfinite(values).all():
        raise ValueError('expected finite nonempty [frames,428] logits')
    labels = {i: p for p, i in vocabulary.items()}
    return [labels[i] for i in ctc_collapse(values.argmax(-1).tolist())]


def compare_logits(actual, expected, vocabulary: dict) -> dict:
    import numpy as np
    if actual.shape != expected.shape:
        raise ValueError('logit shape mismatch')
    a, b = prediction(actual, vocabulary), prediction(expected, vocabulary)
    error = float(np.max(np.abs(actual.astype('float64') - expected.astype('float64'))))
    if error > 2e-5 or a != b:
        raise ValueError(f'logit/sequence mismatch: max_abs={error}; sequence_equal={a == b}')
    return {'maximum_absolute_error': error, 'bitwise_equal': bool(np.array_equal(actual, expected)), 'sequence_equal': True}


def check_midpoint(x, v, h) -> dict:
    import numpy as np
    if x.shape != v.shape or x.shape != h.shape or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('paired waveform geometry mismatch')
    if any(not np.isfinite(a).all() or np.max(np.abs(a)) >= 1 for a in (x, v, h)):
        raise ValueError('nonfinite or clipped paired waveform')
    expected = ((x.astype('float64') + v.astype('float64')) * .5).astype('float32')
    error = float(np.max(np.abs(h - expected)))
    if error > 2**-30:
        raise ValueError('fixed midpoint formula mismatch')
    return {'maximum_pcm_error': error, 'frames': len(h), 'peak_abs': float(np.max(np.abs(h)))}


def first_per_song(rows: list[dict]) -> list[str]:
    if {r['song_id'] for r in rows} != set(SONGS):
        raise ValueError('five training songs required')
    return [min((r for r in rows if r['song_id'] == song),
                key=lambda r: (r['source_start_sec'], r['window_id']))['window_id'] for song in SONGS]


def parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


@contextmanager
def native_observer(model, *, capture: bool):
    """Observe native CTC linear calls; never request alternate encoder outputs."""
    if list(model.encoder.interctc_layer_idx) != [4, 8, 12] or not model.encoder.interctc_use_conditioning:
        raise ValueError('native observer requires restored self-conditioning')
    result = {'condition_calls': 0, 'outputs': []}
    def conditioning(module, args, output):
        result['condition_calls'] += 1
    def ctc(module, args, output):
        result['outputs'].append(output.detach().cpu().clone())
    handles = [model.encoder.conditioning_layer.register_forward_hook(conditioning)]
    if capture:
        handles.append(model.ctc.ctc_lo.register_forward_hook(ctc))
    try:
        yield result
    finally:
        for handle in handles:
            handle.remove()


def native_outputs(observed: dict, final, vocabulary: dict) -> dict:
    if observed['condition_calls'] != 3 or len(observed['outputs']) != 4:
        raise ValueError('expected three native conditioning calls plus final CTC call')
    outputs = observed['outputs']
    if any(v.ndim != (3 if i < 3 else 2) for i, v in enumerate(outputs)):
        raise ValueError('unexpected native CTC tensor geometry')
    result = {str(layer): (v[0] if i < 3 else v).numpy() for i, (layer, v) in enumerate(zip((4, 8, 12, 19), outputs))}
    for v in result.values():
        if v.shape != final.shape:
            raise ValueError('native outlet has different frame geometry')
        prediction(v, vocabulary)
    compare_logits(result['19'], final, vocabulary)
    return result


def aggregate(rows: list[dict]) -> dict:
    return phone_metrics([(r['reference'], r['prediction']) for r in rows])


def a1_gate(rows: list[dict], baseline: list[dict]) -> dict:
    if {r['window_id']: r['reference'] for r in rows} != {r['window_id']: r['reference'] for r in baseline}:
        raise ValueError('different scoring references')
    new, old = aggregate(rows), aggregate(baseline)
    if len(rows) != 20 or new['reference_phones'] != 1235 or old['edit_distance'] != 591:
        raise ValueError('A1 frozen cohort/control mismatch')
    by_song = lambda rr: {s: aggregate([r for r in rr if r['song_id'] == s]) for s in SONGS}
    ns, os = by_song(rows), by_song(baseline)
    improved = [s for s in SONGS if ns[s]['edit_distance'] < os[s]['edit_distance']]
    failures = []
    for key, limit in [('edit_distance', 561), ('insertions', 35), ('deletions', 273)]:
        if new[key] > limit:
            failures.append(key + '_over_limit')
    if len(improved) < 3:
        failures.append('fewer_than_three_songs_improve')
    if sum(m['per'] for m in ns.values()) > sum(m['per'] for m in os.values()):
        failures.append('song_macro_regression')
    for name, group in old['groups'].items():
        if group['reference_count'] != new['groups'][name]['reference_count']:
            raise ValueError('phone group reference changed')
        if group['reference_count'] >= 20 and new['groups'][name]['per'] > group['per'] + .03:
            failures.append(name + '_regression')
    return {'passed': not failures, 'failures': failures, 'metrics': new, 'baseline': old,
            'per_song': ns, 'baseline_per_song': os, 'improved_songs': improved,
            'training_internal_not_holdout': True, 'automatic_deployment': False}


def verify_review(run: Path) -> tuple[dict, dict]:
    freeze = read_json(run / 'FREEZE.json')
    review = read_json(run / 'CODE_REVIEW.json')
    if review.get('approved_for_launch') is not True or review['freeze_sha256'] != sha256_file(run / 'FREEZE.json'):
        raise ValueError('missing or stale launch review')
    for path, digest in review['code_bindings'].items():
        verify_file(Path(path), digest)
    evidence = review['verification']
    verify_file(Path(evidence['path']), evidence['sha256'])
    if read_json(Path(evidence['path']))['passed'] is not True:
        raise ValueError('fixture verification did not pass')
    for path, digest in freeze['bindings'].items():
        verify_file(Path(path), digest)
    return freeze, review


def check_stage(run: Path, stage: str, tasks: list[dict], *, worker: bool = False) -> None:
    if stage not in LIMITS or len(tasks) > LIMITS[stage] or not tasks:
        raise ValueError('stage budget invalid')
    for row in tasks:
        validate_audio_row(row['request'], allow_dev=stage in {'A2', 'ADR'})
        if stage not in {'A2', 'ADR'} and row['request']['split'] != 'train':
            raise ValueError('development/test leakage')
    prerequisites = {'A1': ('A0/RESULT.json', 'passed'), 'A2': ('A1_GATE.json', 'passed'),
                     'B1': ('B0/RESULT.json', 'passed'), 'AR': ('A1/RESULT.json', 'passed'),
                     'ADR': ('A2/RESULT.json', 'passed'), 'BR': ('B1/RESULT.json', 'passed')}
    if stage in prerequisites:
        file, field = prerequisites[stage]
        if read_json(run / file).get(field) is not True:
            raise ValueError('prerequisite did not pass')
    command_exists = (run / 'commands' / (stage + '.command.json')).exists()
    if (run / stage / 'START.json').exists() or (command_exists and not worker):
        raise ValueError('stage already launched; automatic retry forbidden')
    if worker and not command_exists:
        raise ValueError('worker requires accounted launcher command')
    starts = list(run.glob('*/forwards/*/START.json'))
    if len(starts) + len(tasks) > 111:
        raise ValueError('complete acoustic forward budget exhausted')

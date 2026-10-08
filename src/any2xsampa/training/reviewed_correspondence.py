"""Review-history guards and section-constrained audio localization diagnostics.

Human note ranges constrain sections, never exact original phone timestamps.
No prediction, label mapping, or PER participates in choosing an audio path.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from any2xsampa.training.limited_data_artifacts import finite_number, sha256_object
from any2xsampa.training.limited_data_reference import SONG_SPLIT
from any2xsampa.training.replacement_candidates import overlaps, path_time


def pair_identity(item: dict, choice: dict) -> str:
    """Ignore display IDs/rank: renaming a rejected question cannot revive it."""
    return sha256_object({
        'song_id': item['song_id'],
        'audio_sha256': item['original_audio']['sha256'],
        'original': [round(item[k], 6) for k in ('start_sec', 'end_sec')],
        'rendered': [round(choice[k], 6) for k in ('start_sec', 'end_sec')],
        'notes': [n['note_id'] for n in choice['catalog']],
    })


def batch_history(packet: dict, answers: dict) -> list[dict]:
    from tools.batch_training_review import validate_answers

    validated = validate_answers(packet, answers)
    records = []
    for item in packet['items']:
        answer = validated.get(item['review_id'])
        records.append({
            'review_id': item['review_id'], 'song_id': item['song_id'],
            'start_sec': item['start_sec'], 'end_sec': item['end_sec'],
            'audio_sha256': item['original_audio']['sha256'],
            'status': answer['status'] if answer else 'unanswered',
            'packet_sha256': packet['packet_sha256'],
            'pair_ids': [pair_identity(item, c) for c in item['alternatives']],
            'raw_answer': deepcopy(answer),
        })
    return records


def history_failures(item: dict, history: list[dict]) -> list[dict]:
    """Protect all previously asked audio, not only accepted references."""
    pairs = {pair_identity(item, c) for c in item['alternatives']}
    failures = []
    for old in history:
        if old['song_id'] != item['song_id']:
            continue
        if old['audio_sha256'] != item['original_audio']['sha256']:
            raise ValueError('review history audio identity mismatch')
        if pairs.intersection(old['pair_ids']) or overlaps(item, old):
            failures.append({
                'review_id': old['review_id'], 'status': old['status'],
                'reason': 'repeated_review_pair' if pairs.intersection(old['pair_ids'])
                else 'overlaps_previously_reviewed_audio',
            })
    return failures


def section_distance(path: dict, section: dict) -> float:
    """Midpoint must lie within the reviewed section, not a made-up phone time."""
    t = (section['start_sec'] + section['end_sec']) / 2
    predicted = path_time(path, t)
    return max(section['rendered_start_sec'] - predicted,
               predicted - section['rendered_end_sec'], 0.0)


def dual_path_check(mixed: dict, vocal: dict, start: float, end: float,
                    maximum_difference: float = 0.2) -> dict:
    import numpy as np

    start = finite_number(start, 'start', minimum=0)
    end = finite_number(end, 'end', minimum=0)
    maximum_difference = finite_number(maximum_difference, 'threshold', minimum=0)
    if end <= start:
        raise ValueError('invalid window')
    # Check all piecewise-linear knots as well as a fixed grid. Endpoints alone
    # can hide a path that wanders into the wrong phrase inside the window.
    times = sorted({start, end, *np.arange(start, end, .5),
                    *(a['original_sec'] for p in (mixed, vocal)
                      for a in p['path_anchors'] if start < a['original_sec'] < end)})
    delta = max(abs(path_time(mixed, t) - path_time(vocal, t)) for t in times)
    return {'maximum_disagreement_sec': delta,
            'passes': delta <= maximum_difference, 'samples_checked': len(times)}


def candidate_review_gate(item: dict, history: list[dict], alignment: dict,
                          sections: list[dict]) -> dict:
    """New review admission requires fresh audio and corroborated localization."""
    failures = history_failures(item, history)
    context = alignment['context']
    if (SONG_SPLIT.get(item['song_id']) != 'train'
            or context['song_id'] != item['song_id'] or context['split'] != 'train'
            or context['audio_sha256'] != item['original_audio']['sha256']
            or not context['start_sec'] <= item['start_sec'] < item['end_sec'] <= context['end_sec']):
        raise ValueError('candidate/context identity or clock mismatch')
    anchors = [s for s in sections if s['song_id'] == item['song_id'] and
               context['start_sec'] <= (s['start_sec'] + s['end_sec']) / 2 <= context['end_sec']]
    choices = []
    for choice in item['alternatives']:
        role = choice['source_role']
        if role not in ('mixed', 'vocal'):
            raise ValueError('audio path source required')
        own = next(p for p in alignment['paths'][role]['candidates'] if p['rank'] == choice['path_rank'])
        if any(abs(path_time(own, item[side + '_sec']) - choice[side + '_sec']) > 1e-6
               for side in ('start', 'end')):
            raise ValueError('choice does not match its claimed audio path')
        other = 'vocal' if role == 'mixed' else 'mixed'
        evidence = []
        for peer in alignment['paths'][other]['candidates']:
            check = dual_path_check(own, peer, item['start_sec'], item['end_sec'])
            conflicts = [s['example_id'] for s in anchors
                         if section_distance(own, s) > 0 or section_distance(peer, s) > 0]
            check.update(peer_rank=peer['rank'], anchor_conflicts=conflicts)
            check['passes'] = (check['passes'] and bool(anchors) and not conflicts and
                               min(1 - own['mean_path_cost'], 1 - peer['mean_path_cost']) >= .45)
            evidence.append(check)
        choices.append({'choice_id': choice['choice_id'], 'cross_input_evidence': evidence,
                        'supported': any(c['passes'] for c in evidence) and
                        choice.get('mapping_complete') is True})
    return {'history_failures': failures, 'local_anchor_ids': [s['example_id'] for s in anchors],
            'choices': choices, 'review_admissible': not failures and any(c['supported'] for c in choices),
            'human_confirmation_required': True, 'training_eligible': False}


def descriptor_cost(original: Any, rendered: Any) -> Any:
    import numpy as np

    x, y = np.asarray(original), np.asarray(rendered)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or min(x.shape[1], y.shape[1]) < 2:
        raise ValueError('invalid descriptor shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('non-finite descriptor')
    x = x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=0, keepdims=True), 1e-12)
    return np.maximum(0, 1 - x.T @ y).astype(np.float64)


def constrained_path(cost: Any, sections: list[dict], frame_sec: float,
                     *, held_out_id: str | None = None, query_start_sec: float = 0.0) -> dict:
    """Monotone song path using other reviewed sections as coarse constraints.

The held-out section is completely removed from the search mask. This permits
an honest location check against already-reviewed data without asking again.
"""
    import librosa
    import numpy as np

    frame_sec = finite_number(frame_sec, 'frame_sec', minimum=0)
    query_start_sec = finite_number(query_start_sec, 'query_start_sec', minimum=0)
    base = np.asarray(cost, dtype=np.float64)
    if frame_sec <= 0 or base.ndim != 2 or min(base.shape) < 2 or not np.isfinite(base).all():
        raise ValueError('invalid cost/timebase')
    work = base.copy()
    used = []
    for section in sections:
        if section['example_id'] == held_out_id:
            continue
        start = finite_number(section['start_sec'], 'section start', minimum=0)
        end = finite_number(section['end_sec'], 'section end', minimum=0)
        lo = finite_number(section['rendered_start_sec'], 'rendered start', minimum=0)
        hi = finite_number(section['rendered_end_sec'], 'rendered end', minimum=0)
        if end <= start or hi <= lo:
            raise ValueError('invalid reviewed section')
        center = round(((start + end) / 2 - query_start_sec) / frame_sec)
        left, right = int(np.ceil(lo / frame_sec)), int(np.floor(hi / frame_sec)) + 1
        if not 1 <= center < base.shape[0] - 1 or not 0 <= left < right <= base.shape[1]:
            raise ValueError('reviewed section outside descriptor clock')
        # The largest DTW step consumes two original frames. Three masked rows
        # prevent it jumping over a one-row constraint.
        work[center - 1:center + 2, :left] = np.inf
        work[center - 1:center + 2, right:] = np.inf
        used.append(section['example_id'])
    steps = np.array([[1, 1], [1, 2], [2, 1]], dtype=np.int64)
    accumulated, backtrack = librosa.sequence.dtw(
        C=work, step_sizes_sigma=steps, weights_mul=np.array([1., 1., 2.]),
        subseq=True, backtrack=False, return_steps=True)
    endpoint = int(np.argmin(accumulated[-1]))
    if not np.isfinite(accumulated[-1, endpoint]):
        raise ValueError('reviewed section constraints have no feasible audio path')
    path = librosa.sequence.dtw_backtracking(backtrack, step_sizes_sigma=steps,
                                          subseq=True, start=endpoint)[::-1]
    return {
        'path_anchors': [{'original_sec': float(query_start_sec + i * frame_sec),
                          'rendered_sec': float(j * frame_sec)} for i, j in path],
        'mean_path_cost': float(base[path[:, 0], path[:, 1]].mean()),
        'constraint_example_ids': used, 'held_out_id': held_out_id,
        'human_reviewed': False, 'training_eligible': False,
        'scope': 'section_localization_diagnostic_not_phone_timing',
    }

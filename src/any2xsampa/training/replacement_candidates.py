"""Audio-path proposals with immutable human rejections and no phone decoding."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any


def overlaps(a: dict, b: dict) -> bool:
    return (a['song_id'] == b['song_id'] and
            a['start_sec'] < b['end_sec'] - 1e-7 and
            b['start_sec'] < a['end_sec'] - 1e-7)


def path_time(candidate: dict, time: float) -> float:
    import numpy as np
    anchors = candidate['path_anchors']
    x = np.array([a['original_sec'] for a in anchors])
    y = np.array([a['rendered_sec'] for a in anchors])
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid path anchors')
    if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
        raise ValueError('nonmonotonic path')
    unique = np.unique(x)
    medians = np.array([np.median(y[x == t]) for t in unique])
    if time < unique[0] or time > unique[-1]:
        raise ValueError('window outside supported path; no extrapolation')
    return float(np.interp(time, unique, medians))


def propose_windows(context: dict, song: dict, paths: dict, retained: list[dict],
                    rejected: list[dict], policy: dict) -> tuple[list[dict], list[dict]]:
    if context['split'] != 'train' or song['split'] != 'train' or context['song_id'] != song['song_id']:
        raise ValueError('only matching training sources are permitted')
    # Sliding windows repeatedly query the same timestamps within one path pair.
    @lru_cache(maxsize=512)
    def mapped(role: str, time: float) -> float:
        return path_time(paths[role], time)

    pool, failures = [], []
    for index in range(round((context['end_sec'] - context['start_sec'] - policy['window_sec']) / policy['step_sec']) + 1):
        start = round(context['start_sec'] + index * policy['step_sec'], 6)
        end = round(start + policy['window_sec'], 6)
        row = dict(song_id=song['song_id'], split='train', context_id=context['context_id'],
                   start_sec=start, end_sec=end, example_id=context['example_id'])
        reasons = []
        if any(overlaps(row, r) for r in retained):
            reasons.append('overlaps_retained_human_candidate')
        try:
            # Check the entire 10s path, not just its endpoints.
            samples = [start + i * policy['step_sec'] for i in range(round(policy['window_sec'] / policy['step_sec']) + 1)]
            clocks = {k: [mapped(k, t) for t in samples] for k in paths}
            disagreement = max(abs(a - b) for a, b in zip(clocks['mixed'], clocks['vocal']))
            if disagreement > policy['maximum_dual_path_disagreement_sec']:
                reasons.append('mixed_vocal_paths_disagree')
            if min(1 - p['mean_path_cost'] for p in paths.values()) < policy['minimum_similarity']:
                reasons.append('low_audio_similarity')
            rs, re = clocks['vocal'][0], clocks['vocal'][-1]
            if not policy['minimum_rendered_duration_sec'] <= re - rs <= policy['maximum_rendered_duration_sec']:
                reasons.append('implausible_local_time_scale')
            notes = [n for n in song['notes'] if n['start_sec'] < re and n['end_sec'] > rs]
            if not notes or any(n['mapping_status'] != 'resolved_proxy' or not n['model_target_ipa'] for n in notes):
                reasons.append('whole_window_unresolved_mapping')
            old_ids = []
            for old in rejected:
                if not overlaps(row, old):
                    continue
                old_ids.append(old['review_id'])
                midpoint = (max(start, old['start_sec']) + min(end, old['end_sec'])) / 2
                old_time = old['rendered_start_sec'] + midpoint - old['start_sec']
                if abs(mapped('vocal', midpoint) - old_time) < policy['minimum_changed_correspondence_sec']:
                    reasons.append('repeats_rejected_correspondence')
            row.update(rendered_start_sec=rs, rendered_end_sec=re,
                       dual_path_disagreement_sec=disagreement,
                       notes=[{k: n[k] for k in ('note_id', 'raw_lyric', 'model_target_ipa', 'mapping_status')} for n in notes],
                       overlaps_rejected_ids=old_ids,
                       center_distance_sec=abs((start+end)/2-(context['old_start_sec']+context['old_end_sec'])/2),
                       human_status='pending', training_eligible=False)
        except ValueError as error:
            reasons.append(str(error))
        if reasons:
            failures.append({**row, 'reasons': sorted(set(reasons))})
        else:
            pool.append(row)
    return pool, failures


def select_candidates(pool: list[dict], retained: list[dict], song_order: list[str], limit: int) -> list[dict]:
    counts = Counter(r['song_id'] for r in retained)
    contexts = Counter(r.get('context_id') for r in retained)
    order = {s: i for i, s in enumerate(song_order)}
    selected = []
    remaining = list(pool)
    while remaining and len(selected) < limit:
        row = min(remaining, key=lambda r: (counts[r['song_id']], contexts[r['context_id']],
                  r['center_distance_sec'], order[r['song_id']], r['start_sec']))
        selected.append(row)
        counts[row['song_id']] += 1
        contexts[row['context_id']] += 1
        remaining = [r for r in remaining if not overlaps(row, r)]
    return selected

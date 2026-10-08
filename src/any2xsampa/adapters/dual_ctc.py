"""Audio-only two-stream CTC candidates and exact sequence rescoring."""
from __future__ import annotations
import math
from typing import Any, Sequence
from any2xsampa.adapters.ctc_prefix import _log_probs, PrefixResult
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def prefix_nbest(logits: Any, *, beam_width: int = 8, blank_id: int = 0) -> list[PrefixResult]:
    """Expand every label; return all surviving prefixes, without token pruning."""
    import numpy as np
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or not 1 <= beam_width <= 4096:
        raise ValueError('beam_width must be an integer in [1, 4096]')
    probs = _log_probs(logits)
    if isinstance(blank_id, bool) or not isinstance(blank_id, int) or not 0 <= blank_id < probs.shape[1]:
        raise ValueError('invalid blank ID')
    prefixes: list[tuple[int, ...]] = [()]
    pb, pn = np.array([0.]), np.array([-np.inf]); labels = probs.shape[1]
    for frame in probs:
        total = np.logaddexp(pb, pn); blank = total+frame[blank_id]
        nonblank = np.full(len(prefixes), -np.inf)
        extensions = total[:, None]+frame[None, :]; extensions[:, blank_id] = -np.inf
        lookup = {prefix: i for i, prefix in enumerate(prefixes)}
        for i, prefix in enumerate(prefixes):
            if prefix:
                last = prefix[-1]; nonblank[i] = pn[i]+frame[last]
                extensions[i, last] = pb[i]+frame[last]
        for i, prefix in enumerate(prefixes):
            if prefix and prefix[:-1] in lookup:
                parent = lookup[prefix[:-1]]
                nonblank[i] = np.logaddexp(nonblank[i], extensions[parent, prefix[-1]])
                extensions[parent, prefix[-1]] = -np.inf
        scores = np.concatenate((np.logaddexp(blank, nonblank), extensions.ravel()))
        finite = np.flatnonzero(np.isfinite(scores)); keep = min(beam_width, len(finite))
        if not keep: raise ValueError('CTC search lost all finite paths')
        cutoff = np.partition(scores[finite], len(finite)-keep)[len(finite)-keep]
        eligible = finite[scores[finite] >= cutoff]; size = len(prefixes)
        def label(index: int) -> tuple[int, ...]:
            if index < size: return prefixes[index]
            parent, token = divmod(index-size, labels)
            return prefixes[parent]+(token,)
        ranked = sorted((int(i) for i in eligible), key=lambda i: (-float(scores[i]), label(i)))[:keep]
        next_prefixes = [label(i) for i in ranked]
        pb = np.array([blank[i] if i < size else -np.inf for i in ranked])
        pn = np.array([nonblank[i] if i < size else scores[i] for i in ranked])
        prefixes = next_prefixes
    return [PrefixResult(prefix, float(np.logaddexp(pb[i], pn[i])), beam_width, len(probs), labels)
            for i, prefix in enumerate(prefixes)]


def _sequence_log_probability(probs: Any, ids: Sequence[int], blank_id: int) -> float:
    import numpy as np
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < probs.shape[1] or i == blank_id for i in ids):
        raise ValueError('sequence contains blank or invalid token IDs')
    if not ids: return float(probs[:, blank_id].sum())
    required = len(ids)+sum(a == b for a, b in zip(ids, ids[1:]))
    if required > len(probs): return -math.inf
    states = np.full(2*len(ids)+1, blank_id, dtype=int); states[1::2] = ids
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    alpha = np.full(len(states), -np.inf); alpha[0] = 0.
    for frame in probs:
        one = np.concatenate(([-np.inf], alpha[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], alpha[:-2]))
        two[~skip] = -np.inf
        alpha = np.logaddexp(np.logaddexp(alpha, one), two)+frame[states]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def sequence_log_probability(logits: Any, ids: Sequence[int], *, blank_id: int = 0) -> float:
    probs = _log_probs(logits)
    if isinstance(blank_id, bool) or not isinstance(blank_id, int) or not 0 <= blank_id < probs.shape[1]:
        raise ValueError('invalid blank ID')
    return _sequence_log_probability(probs, tuple(ids), blank_id)


def dual_decode(mixed_logits: Any, vocal_logits: Any) -> dict:
    """One fixed rule; no references, lexicon, tunable weighting or frame fusion."""
    import numpy as np
    logits = {'mixed': np.asarray(mixed_logits), 'vocal': np.asarray(vocal_logits)}
    probs = {k: _log_probs(v) for k, v in logits.items()}
    if any(v.shape[1] != 428 for v in probs.values()):
        raise ValueError('dual decoder requires all 428 labels')
    candidates: dict[tuple[int, ...], list[str]] = {}
    greedy, beams = {}, {}
    for role, values in logits.items():
        greedy[role] = ctc_collapse(values.argmax(-1).tolist())
        nbest = prefix_nbest(values, beam_width=8)
        beams[role] = [{'ids': list(row.ids), 'search_log_probability': row.log_probability} for row in nbest]
        for row in nbest: candidates.setdefault(row.ids, []).append(role+'_beam')
        candidates.setdefault(tuple(greedy[role]), []).append(role+'_greedy')
    if len(candidates) > 18: raise RuntimeError('candidate budget exceeded')
    scored = []
    for ids, origins in sorted(candidates.items()):
        per_stream = {role: _sequence_log_probability(value, ids, 0) for role, value in probs.items()}
        score = .5*per_stream['mixed']+.5*per_stream['vocal']; finite = math.isfinite(score)
        scored.append({'ids': list(ids), 'origins': origins, 'exact_log_probabilities':
                       {k: v if math.isfinite(v) else None for k, v in per_stream.items()},
                       'joint_log_score': score if finite else None, 'both_streams_feasible': finite})
    ranked = sorted((r for r in scored if r['both_streams_feasible']), key=lambda r: (-r['joint_log_score'], tuple(r['ids'])))
    if not ranked: raise ValueError('no candidate has a feasible path in both streams')
    return {'schema_version': 'dual_ctc_decode_v1', 'selected_ids': ranked[0]['ids'],
            'selected_joint_log_score': ranked[0]['joint_log_score'], 'candidates': scored, 'beams': beams,
            'greedy': greedy, 'frames': {k: len(v) for k,v in probs.items()},
            'reference_received': False, 'score_is_calibrated_confidence': False,
            'framewise_alignment_or_averaging_used': False, 'full_label_count': 428}

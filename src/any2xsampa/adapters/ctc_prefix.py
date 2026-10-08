"""Bounded, acoustic-only CTC prefix search; raw posterior evidence is unchanged."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class PrefixResult:
    ids: tuple[int, ...]
    log_probability: float
    beam_width: int
    frames: int
    labels: int


def _log_probs(logits: Any) -> Any:
    import numpy as np
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or values.shape[1] < 2:
        raise ValueError('expected nonempty [frames, labels] logits')
    if not np.isfinite(values).all():
        raise ValueError('non-finite CTC logits')
    values = values - values.max(axis=1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=1, keepdims=True))


def prefix_beam_decode(logits: Any, *, beam_width: int, blank_id: int = 0) -> PrefixResult:
    """Sum blank/nonblank paths, then prune prefixes; expand EVERY label.

    No language model, token pruning, length reward, reference or blank bias.
    Ties use lexical token-ID order. Width pruning is an approximation to
    full-sequence MAP; width=1 need not equal greedy path collapse.
    """
    import numpy as np
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or not 1 <= beam_width <= 4096:
        raise ValueError('beam_width must be an integer in [1, 4096]')
    probs = _log_probs(logits)
    if isinstance(blank_id, bool) or not isinstance(blank_id, int) or not 0 <= blank_id < probs.shape[1]:
        raise ValueError('invalid blank ID')
    prefixes: list[tuple[int, ...]] = [()]
    pb, pn = np.array([0.]), np.array([-np.inf])
    labels = probs.shape[1]
    for frame in probs:
        total = np.logaddexp(pb, pn)
        blank = total + frame[blank_id]
        nonblank = np.full(len(prefixes), -np.inf)
        extensions = total[:, None] + frame[None, :]
        extensions[:, blank_id] = -np.inf
        lookup = {prefix: i for i, prefix in enumerate(prefixes)}
        for i, prefix in enumerate(prefixes):
            if prefix:
                last = prefix[-1]
                nonblank[i] = pn[i] + frame[last]
                # A repeated phone becomes a NEW phone only after a blank.
                extensions[i, last] = pb[i] + frame[last]
        for i, prefix in enumerate(prefixes):
            if prefix and prefix[:-1] in lookup:
                parent = lookup[prefix[:-1]]
                nonblank[i] = np.logaddexp(nonblank[i], extensions[parent, prefix[-1]])
                extensions[parent, prefix[-1]] = -np.inf
        scores = np.concatenate((np.logaddexp(blank, nonblank), extensions.ravel()))
        finite = np.flatnonzero(np.isfinite(scores))
        keep = min(beam_width, len(finite))
        if not keep:
            raise ValueError('CTC search lost all finite paths')
        cutoff = np.partition(scores[finite], len(finite) - keep)[len(finite) - keep]
        eligible = finite[scores[finite] >= cutoff]
        size = len(prefixes)
        def label(index: int) -> tuple[int, ...]:
            if index < size:
                return prefixes[index]
            parent, token = divmod(index - size, labels)
            return prefixes[parent] + (token,)
        ranked = sorted((int(i) for i in eligible), key=lambda i: (-float(scores[i]), label(i)))[:keep]
        next_prefixes = [label(i) for i in ranked]
        pb = np.array([blank[i] if i < size else -np.inf for i in ranked])
        pn = np.array([nonblank[i] if i < size else scores[i] for i in ranked])
        prefixes = next_prefixes
    score = float(np.logaddexp(pb[0], pn[0]))
    if not math.isfinite(score):
        raise ValueError('non-finite final CTC score')
    return PrefixResult(prefixes[0], score, beam_width, len(probs), labels)

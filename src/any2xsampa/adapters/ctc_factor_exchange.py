"""Reference-free, fixed blank/conditional-label posterior exchange."""
from __future__ import annotations

from typing import Any


def _values(logits: Any) -> Any:
    import numpy as np
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 428 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError('expected finite nonempty [frames,428] logits')
    return values


def _lse(values: Any) -> Any:
    import numpy as np
    maximum = values.max(axis=-1, keepdims=True)
    return maximum + np.log(np.exp(values - maximum).sum(axis=-1, keepdims=True))


def log_probabilities(logits: Any) -> Any:
    values = _values(logits)
    return values - _lse(values)


def exchange_factors(blank_logits: Any, label_logits: Any) -> Any:
    """All 428 labels preserved; no thresholds, shifts, reference or model call."""
    import numpy as np
    blank, labels = _values(blank_logits), _values(label_logits)
    if blank.shape != labels.shape:
        raise ValueError('factor sources must have identical frame/label geometry')
    normalizer = _lse(blank)
    blank_lp = blank[:, :1] - normalizer
    nonblank_lp = _lse(blank[:, 1:]) - normalizer
    conditional_lp = labels[:, 1:] - _lse(labels[:, 1:])
    result = np.concatenate([blank_lp, nonblank_lp + conditional_lp], axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('non-finite exchanged posterior')
    return result


def reconstruction_check(original: Any, reconstructed: Any) -> dict:
    import numpy as np
    from any2xsampa.adapters.phoneticxeus import ctc_collapse
    a, b = _values(original), _values(reconstructed)
    if a.shape != b.shape:
        raise ValueError('reconstruction geometry changed')
    probability_error = float(np.max(np.abs(np.exp(log_probabilities(a)) - np.exp(log_probabilities(b)))))
    same = ctc_collapse(a.argmax(-1).tolist()) == ctc_collapse(b.argmax(-1).tolist())
    if probability_error > 2e-6 or not same:
        raise ValueError('self-factor reconstruction changed probability or sequence')
    return {'maximum_probability_error': probability_error, 'collapsed_sequence_equal': same}


def diagnostics(mixed: Any, vocal: Any, exchanged: Any) -> dict:
    import numpy as np
    x, v, h = (_values(a) for a in (mixed, vocal, exchanged))
    if x.shape != v.shape or x.shape != h.shape:
        raise ValueError('diagnostic frame geometry mismatch')
    px, pv, ph = (np.exp(log_probabilities(a)) for a in (x, v, h))
    ix, iv, ih = (a.argmax(-1) for a in (x, v, h))
    offsets = []
    for t in np.flatnonzero(ix):
        # Purely acoustic diagnostic, not alignment or a decoder time warp.
        nearby = np.flatnonzero(iv[max(0, t-5):min(len(iv), t+6)] == ix[t])
        if len(nearby):
            choices = nearby + max(0, t-5) - t
            offsets.append(int(min(choices, key=lambda d: (abs(d), d))))
    return {'frames': len(x), 'maximum_blank_probability_error': float(np.max(np.abs(px[:, 0] - ph[:, 0]))),
            'blank_argmax_changed_frames': int(((ix == 0) != (ih == 0)).sum()),
            'source_blank_argmax_disagreement_frames': int(((ix == 0) != (iv == 0)).sum()),
            'vocal_blank_probability_ge_0999_frames': int((pv[:, 0] >= .999).sum()),
            'mixed_nonblank_frames': int((ix != 0).sum()),
            'nearby_same_label_frames': len(offsets),
            'nearest_same_label_offsets_within_5_frames': {str(d): offsets.count(d) for d in range(-5, 6)},
            'frame_offsets_are_not_reference_boundaries': True, 'no_alignment_or_filter_applied': True}

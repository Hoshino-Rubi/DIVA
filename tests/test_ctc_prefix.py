from collections import defaultdict
from itertools import product
import math
import numpy as np
import pytest
from any2xsampa.adapters.ctc_prefix import prefix_beam_decode
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def exhaustive(values, blank=0):
    p = np.exp(values - values.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    scores = defaultdict(float)
    for path in product(range(p.shape[1]), repeat=p.shape[0]):
        key = tuple(ctc_collapse(list(path), blank))
        scores[key] += math.prod(p[t, v] for t, v in enumerate(path))
    best = min(scores, key=lambda s: (-scores[s], s))
    return best, math.log(scores[best])


@pytest.mark.parametrize('seed', range(8))
def test_full_beam_matches_exhaustive_ctc_path_sum(seed):
    values = np.random.default_rng(seed).normal(size=(5, 3))
    result = prefix_beam_decode(values, beam_width=4096)
    best, score = exhaustive(values)
    assert result.ids == best
    assert result.log_probability == pytest.approx(score, abs=1e-12)


def test_repeated_phone_requires_blank():
    for values, expected in [([[0, 9], [0, 9]], (1,)), ([[0, 9], [9, 0], [0, 9]], (1, 1))]:
        assert prefix_beam_decode(values, beam_width=8).ids == expected


def test_empty_output_is_valid_and_offset_invariant():
    values = np.array([[90., 0., -5.], [90., 0., -5.]])
    a = prefix_beam_decode(values, beam_width=8)
    b = prefix_beam_decode(values + 10000, beam_width=8)
    assert a.ids == b.ids == ()
    assert a.log_probability == pytest.approx(b.log_probability)


def test_nonzero_blank_and_tie_are_deterministic():
    values = np.zeros((3, 3))
    best, score = exhaustive(values, 2)
    result = prefix_beam_decode(values, beam_width=4096, blank_id=2)
    assert result.ids == best
    assert result.log_probability == pytest.approx(score)


def test_428_labels_all_participate_without_inventory_reduction():
    values = np.full((3, 428), -50.)
    values[:, 427] = 1.
    assert prefix_beam_decode(values, beam_width=8).ids == (427,)


@pytest.mark.parametrize('values', [[], [[float('nan'), 0]], [[0, float('inf')]], [[0]], [0, 1]])
def test_invalid_logits(values):
    with pytest.raises(ValueError):
        prefix_beam_decode(values, beam_width=8)


@pytest.mark.parametrize('kwargs', [{'beam_width': 0}, {'beam_width': True}, {'beam_width': 8, 'blank_id': 2}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        prefix_beam_decode([[0, 1]], **kwargs)

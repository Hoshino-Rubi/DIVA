from collections import defaultdict
from itertools import product
import math
import numpy as np
import pytest
from any2xsampa.adapters.ctc_prefix import prefix_beam_decode
from any2xsampa.adapters.dual_ctc import prefix_nbest, sequence_log_probability, dual_decode
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def exhaustive(logits):
    p = np.exp(logits-logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    result = defaultdict(float)
    for path in product(range(p.shape[1]), repeat=len(p)):
        result[tuple(ctc_collapse(list(path)))] += math.prod(p[t,k] for t,k in enumerate(path))
    return result


@pytest.mark.parametrize('seed', range(5))
def test_exact_scores_and_unpruned_nbest_match_enumerated_paths(seed):
    logits = np.random.default_rng(seed).normal(size=(4,3)); gold = exhaustive(logits)
    ranked = prefix_nbest(logits, beam_width=256)
    assert [r.ids for r in ranked] == sorted(gold, key=lambda seq: (-gold[seq], seq))
    for row in ranked:
        assert row.log_probability == pytest.approx(math.log(gold[row.ids]), abs=1e-12)
        assert sequence_log_probability(logits, row.ids) == pytest.approx(row.log_probability, abs=1e-12)


@pytest.mark.parametrize('seed', range(4))
def test_top_nbest_matches_historical_full_inventory_beam(seed):
    logits = np.random.default_rng(seed).normal(size=(17,428))
    assert prefix_nbest(logits, beam_width=8)[0] == prefix_beam_decode(logits, beam_width=8)


def test_repeated_tokens_need_blank_and_empty_is_valid():
    logits = np.zeros((2,3))
    assert sequence_log_probability(logits, [1,1]) == -math.inf
    assert sequence_log_probability(logits, []) == pytest.approx(2*math.log(1/3))
    assert math.isfinite(sequence_log_probability(np.zeros((3,3)), [1,1]))


def test_sequence_score_does_not_require_same_peak_frame():
    a = np.full((8,428), -12.); a[:,0] = 5; a[1,7] = 12; a[5,9] = 12
    b = np.full((8,428), -12.); b[:,0] = 5; b[2,7] = 12; b[6,9] = 12
    result = dual_decode(a,b)
    assert result['selected_ids'] == [7,9]
    assert len(result['candidates']) <= 18 and not result['reference_received']
    for role, ids in result['greedy'].items():
        assert any(row['ids'] == ids and role+'_greedy' in row['origins'] for row in result['candidates'])


def test_uniform_nbest_ties_are_deterministic():
    a = prefix_nbest(np.zeros((1,10)), beam_width=8)
    assert [r.ids for r in a] == [()]+[(i,) for i in range(1,8)]


@pytest.mark.parametrize('kind', ['nan', 'infinity', 'empty', 'wrong_labels'])
def test_bad_logits_rejected(kind):
    a = np.zeros((3,428)); b = a.copy()
    if kind == 'nan': a[0,0] = np.nan
    if kind == 'infinity': a[0,0] = np.inf
    if kind == 'empty': a = a[:0]
    if kind == 'wrong_labels': a = a[:,:427]
    with pytest.raises(ValueError): dual_decode(a,b)


@pytest.mark.parametrize('ids', [[0], [True], [-1], [3]])
def test_invalid_target_ids_rejected(ids):
    with pytest.raises(ValueError): sequence_log_probability(np.zeros((3,3)), ids)

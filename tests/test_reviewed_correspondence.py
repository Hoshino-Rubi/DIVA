from copy import deepcopy

import numpy as np
import pytest

from any2xsampa.training.reviewed_correspondence import (
    batch_history, history_failures, dual_path_check, descriptor_cost, constrained_path, candidate_review_gate,
)
from test_batch_training_review import fixture, signed


def reviewed_fixture(status='rejected'):
    packet, answers, _ = fixture()
    item = packet['items'][0]
    item['original_audio'] = {'sha256': 'b' * 64}
    item['alternatives'][0].update(start_sec=20, end_sec=30)
    signed(packet)
    answers['packet_sha256'] = packet['packet_sha256']
    if status != 'confirmed':
        answers['answers'][0].update(status=status, choice=None, sequence_confirmed=False, edges={})
    return packet, answers


@pytest.mark.parametrize('status', ['confirmed', 'rejected', 'uncertain'])
def test_all_previously_reviewed_audio_is_protected(status):
    packet, answers = reviewed_fixture(status)
    history = batch_history(packet, answers)
    renamed = deepcopy(packet['items'][0])
    renamed['review_id'] = 'N07'
    renamed['alternatives'][0]['choice_id'] = 'B'
    assert history_failures(renamed, history)[0]['reason'] == 'repeated_review_pair'
    renamed['alternatives'][0]['start_sec'] += 1
    assert history_failures(renamed, history)[0]['reason'] == 'overlaps_previously_reviewed_audio'
    renamed.update(start_sec=10, end_sec=20)
    assert not history_failures(renamed, history)


def test_history_rejects_wrong_packet_and_source():
    packet, answers = reviewed_fixture()
    history = batch_history(packet, answers)
    packet['items'][0]['original_audio']['sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='audio identity'):
        history_failures(packet['items'][0], history)
    with pytest.raises(ValueError, match='hash'):
        batch_history(packet, answers)


def test_unanswered_review_is_not_silently_reissued():
    packet, answers = reviewed_fixture()
    answers['answers'] = []
    history = batch_history(packet, answers)
    assert history[0]['status'] == 'unanswered'
    assert history_failures(packet['items'][0], history)


def test_dual_input_gate_checks_interior_knots_not_just_endpoints():
    mixed = {'path_anchors': [{'original_sec': t, 'rendered_sec': t} for t in (0, 1, 2, 3)]}
    vocal = deepcopy(mixed)
    vocal['path_anchors'][1]['rendered_sec'] += .4
    report = dual_path_check(mixed, vocal, 0, 3)
    assert not report['passes']
    assert report['maximum_disagreement_sec'] == pytest.approx(.4)
    with pytest.raises(ValueError, match='outside'):
        dual_path_check(mixed, vocal, 0, 4)


def test_audio_cost_rejects_nan():
    with pytest.raises(ValueError, match='non-finite'):
        descriptor_cost(np.full((2, 3), np.nan), np.ones((2, 4)))


def test_held_out_section_cannot_influence_constrained_path():
    rng = np.random.default_rng(17)
    query = rng.random((12, 80))
    reference = np.concatenate([rng.random((12, 20)), query, rng.random((12, 20))], axis=1)
    cost = descriptor_cost(query, reference)
    sections = [
        dict(example_id='anchor', start_sec=1, end_sec=3, rendered_start_sec=3, rendered_end_sec=5),
        dict(example_id='target', start_sec=4, end_sec=6, rendered_start_sec=6, rendered_end_sec=8),
    ]
    first = constrained_path(cost, sections, .1, held_out_id='target')
    sections[1].update(start_sec=0, end_sec=1, rendered_start_sec=10, rendered_end_sec=11)
    second = constrained_path(cost, sections, .1, held_out_id='target')
    assert first == second
    assert first['constraint_example_ids'] == ['anchor']
    assert first['training_eligible'] is False


def test_infeasible_constraints_fail_instead_of_falling_back():
    cost = np.ones((100, 100))
    sections = [
        dict(example_id='a', start_sec=1, end_sec=3, rendered_start_sec=8, rendered_end_sec=9),
        dict(example_id='b', start_sec=6, end_sec=8, rendered_start_sec=1, rendered_end_sec=2),
    ]
    with pytest.raises(ValueError, match='feasible'):
        constrained_path(cost, sections, .1)


def test_review_gate_requires_current_history_anchor_and_two_inputs():
    packet, answers = reviewed_fixture()
    item = packet['items'][0]
    item['alternatives'][0].update(source_role='mixed', path_rank=1, mapping_complete=True)
    path = {'rank': 1, 'mean_path_cost': .1, 'path_anchors': [
        {'original_sec': t, 'rendered_sec': t + 20} for t in (0, 5, 10)]}
    alignment = {'context': {'song_id': 'promise', 'split': 'train', 'audio_sha256': 'b' * 64,
                            'start_sec': 0, 'end_sec': 10},
                 'paths': {role: {'candidates': [deepcopy(path)]} for role in ('mixed', 'vocal')}}
    sections = [dict(example_id='old-anchor', song_id='promise', start_sec=0, end_sec=10,
                     rendered_start_sec=20, rendered_end_sec=30)]
    result = candidate_review_gate(item, [], alignment, sections)
    assert result['review_admissible'] and not result['training_eligible']
    assert not candidate_review_gate(item, [], alignment, [])['review_admissible']
    signed(packet)
    answers['packet_sha256'] = packet['packet_sha256']
    history = batch_history(packet, answers)
    assert not candidate_review_gate(item, history, alignment, sections)['review_admissible']
    alignment['paths']['vocal']['candidates'][0]['path_anchors'][1]['rendered_sec'] += .5
    assert not candidate_review_gate(item, [], alignment, sections)['review_admissible']


def test_nonzero_query_clock_and_retired_generator(tmp_path):
    path = constrained_path(np.eye(30), [], .1, query_start_sec=80)
    assert path['path_anchors'][0]['original_sec'] == 80
    from tools.build_final_training_window_followup import build
    output = tmp_path / 'no-new-review'
    with pytest.raises(RuntimeError, match='Retired'):
        build(output)
    assert not output.exists()

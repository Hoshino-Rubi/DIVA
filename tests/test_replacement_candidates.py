from copy import deepcopy
import pytest
from any2xsampa.training.replacement_candidates import path_time, overlaps, propose_windows, select_candidates
from any2xsampa.training.limited_data_artifacts import read_json
from pathlib import Path


def fixtures():
    policy = read_json(Path('src/any2xsampa/config/training_replacement_policy_v1.json'))
    context = dict(song_id='song', split='train', context_id='ctx', example_id='clip',
                   start_sec=0, end_sec=30, old_start_sec=6, old_end_sec=24)
    notes = [dict(note_id=str(i), raw_lyric='x', model_target_ipa=['a'], mapping_status='resolved_proxy',
                  start_sec=i, end_sec=i+1) for i in range(40)]
    song = dict(song_id='song', split='train', notes=notes)
    path = {'mean_path_cost': .1, 'path_anchors': [{'original_sec': i, 'rendered_sec': i+1} for i in range(31)]}
    return context, song, {'mixed': deepcopy(path), 'vocal': deepcopy(path)}, policy


def test_path_never_extrapolates_or_accepts_reversed_time():
    _, _, paths, _ = fixtures()
    assert path_time(paths['mixed'], 3.5) == 4.5
    with pytest.raises(ValueError):
        path_time(paths['mixed'], -1)
    paths['mixed']['path_anchors'][3]['rendered_sec'] = 0
    with pytest.raises(ValueError):
        path_time(paths['mixed'], 3.5)


def test_retained_pool_and_rejected_correspondences_remain_excluded():
    c, s, paths, p = fixtures()
    old = dict(song_id='song', start_sec=10, end_sec=20, rendered_start_sec=11, review_id='C01')
    accepted, excluded = propose_windows(c, s, paths, [], [old], p)
    assert accepted and all(not overlaps(a, old) for a in accepted)
    assert any('repeats_rejected_correspondence' in r['reasons'] for r in excluded)
    accepted, excluded = propose_windows(c, s, paths, [old], [], p)
    assert all(not overlaps(a, old) for a in accepted)


def test_path_disagreement_inside_window_is_not_hidden_by_matching_endpoints():
    c, s, paths, p = fixtures()
    paths['vocal']['path_anchors'][15]['rendered_sec'] += .5
    accepted, excluded = propose_windows(c, s, paths, [], [], p)
    assert any('mixed_vocal_paths_disagree' in r['reasons'] for r in excluded)


def test_sealed_inputs_and_unresolved_windows_fail_closed():
    c, s, paths, p = fixtures()
    s['split'] = 'test'
    with pytest.raises(ValueError):
        propose_windows(c, s, paths, [], [], p)
    s['split'] = 'train'
    for n in s['notes']:
        n['mapping_status'] = 'unresolved'
    accepted, excluded = propose_windows(c, s, paths, [], [], p)
    assert not accepted and all('whole_window_unresolved_mapping' in r['reasons'] for r in excluded)


def test_selection_is_bounded_deterministic_and_nonoverlapping():
    c, s, paths, p = fixtures()
    pool, _ = propose_windows(c, s, paths, [], [], p)
    selected = select_candidates(pool, [], ['song'], 2)
    assert len(selected) == 2
    assert not overlaps(*selected)
    assert selected == select_candidates(list(reversed(pool)), [], ['song'], 2)
    assert all(r['training_eligible'] is False for r in selected)

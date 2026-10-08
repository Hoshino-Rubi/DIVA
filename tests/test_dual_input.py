import copy
import json
from pathlib import Path
import pytest
from any2xsampa.training.dual_input import audio_requests, validate_pair, acceptance
from any2xsampa.training.limited_data_artifacts import sha256_file


def requests():
    songs = ['freedom-flower', 'queen-road-east', 'half-moon-night-song', 'promise', 'like-you']
    rows = []
    for i in range(23):
        rows.append({'example_id': f'clip{i}', 'song_id': songs[i%5] if i < 20 else 'shalong',
            'split': 'train' if i < 20 else 'dev', 'audio_path': 'source.wav', 'audio_sha256': 'a'*64,
            'start_sec': 10., 'end_sec': 20., 'source_rate': 44100, 'source_start_sample': 441000,
            'source_end_sample_exclusive': 882000, 'context_start_sec': 4., 'context_end_sec': 26.,
            'reuse_separation': None, 'reuse_exact_pair': None, 'contains_reference_context': False})
    return {'schema_version': 'dual_input_audio_requests_v1', 'requests': rows, 'contains_reference_context': False, 'test_allowed': False}


def test_complete_cohort_accepted():
    assert len(audio_requests(requests())) == 23


@pytest.mark.parametrize('bad', ['test', 'dev_in_train', 'reference', 'sample_shift', 'duplicate', 'missing'])
def test_invalid_cohort_fails(bad):
    data = requests()
    if bad == 'test': data['requests'][0]['song_id'] = 'king-of-karaoke'
    if bad == 'dev_in_train': data['requests'][0]['song_id'] = 'shalong'
    if bad == 'reference': data['requests'][0]['phones'] = ['i']
    if bad == 'sample_shift': data['requests'][0]['source_start_sample'] += 1
    if bad == 'duplicate': data['requests'][1] = data['requests'][0]
    if bad == 'missing': data['requests'].pop()
    with pytest.raises(ValueError): audio_requests(data)


def pair(tmp_path):
    path = tmp_path/'audio.wav'; path.write_bytes(b'fixed fixture bytes')
    audio = {'path': str(path), 'sha256': sha256_file(path), 'sample_rate': 44100, 'channels': 2, 'subtype': 'PCM_32', 'samples': 441000}
    return {'example_id': 'clip0', 'source_audio_sha256': 'a'*64, 'source_start_sample': 441000,
            'source_end_sample_exclusive': 882000, 'effective_original_start_sec': 10., 'effective_original_end_sec': 20.,
            'audio': {'mixed': dict(audio), 'vocal': dict(audio)}, 'contains_reference_context': False,
            'transform': {'same_gain_for_both_conditions': True, 'clipping_performed': False}}


@pytest.mark.parametrize('bad', ['hash', 'offset', 'length', 'gain', 'clipping'])
def test_pair_rejects_clock_hash_or_gain_changes(tmp_path, bad):
    data = pair(tmp_path); request = requests()['requests'][0]; validate_pair(request, data)
    if bad == 'hash': data['audio']['vocal']['sha256'] = '0'*64
    if bad == 'offset': data['effective_original_start_sec'] += .01
    if bad == 'length': data['audio']['mixed']['samples'] -= 1
    if bad == 'gain': data['transform']['same_gain_for_both_conditions'] = False
    if bad == 'clipping': data['transform']['clipping_performed'] = True
    with pytest.raises(ValueError): validate_pair(request, data)


def test_per_improvement_cannot_hide_insertions_or_small_group_regression():
    rules = json.loads((Path(__file__).parents[1]/'src/any2xsampa/config/dual_input_ctc_v1.json').read_text())['acceptance']
    row = {'reference_phones': 214, 'sequences': 3, 'edit_distance': 78, 'deletions': 14, 'insertions': 21,
           'groups': {'aspirated': {'reference_count': 6, 'substitutions': 1, 'deletions': 0, 'insertions': 2, 'per': .5},
                      'glottal_h': {'reference_count': 5, 'substitutions': 1, 'deletions': 1, 'insertions': 0, 'per': .4}}}
    base = copy.deepcopy(row); base['edit_distance'] = 92
    result = acceptance(row, base, rules)
    assert not result['passed']
    assert result['failures'] == ['insertions_over_limit', 'aspirated_error_increase']
    row['insertions'] = 11; row['groups']['aspirated'].update(insertions=0, per=1/6)
    assert acceptance(row, base, rules)['passed']

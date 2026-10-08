import numpy as np
import pytest
from any2xsampa.training.phone_diagnostic_analysis import contiguous_note_prediction, frame_strata, summarize_answers
from any2xsampa.training.limited_data_artifacts import sha256_object


def test_internal_model_insertions_are_retained():
    example = {'source_phones': [{'note_id': 'n1'}, {'note_id': 'n1'}],
               'reference': ['i', 'n'], 'predictions': {'candidate': ['i', 'a', 'n']},
               'edit_operations': {'candidate': [
                   {'reference_index': 0, 'prediction_index': 0},
                   {'reference_index': None, 'prediction_index': 1},
                   {'reference_index': 1, 'prediction_index': 2}]}}
    result = contiguous_note_prediction(example, 'n1', 'candidate')
    assert result['prediction'] == ['i', 'a', 'n']
    assert not result['acoustic_boundaries_confirmed']


def test_noncontiguous_note_rejected():
    example = {'source_phones': [{'note_id': n} for n in ['n1', 'n2', 'n1']]}
    with pytest.raises(ValueError, match='noncontiguous'):
        contiguous_note_prediction(example, 'n1', 'candidate')


def test_support_masks_partition_full_audio_without_wraparound():
    labels = [str(i) for i in range(428)]; labels[1] = 'kʰ'; labels[2] = 'h'
    logits = np.zeros((10, 428)); logits[0, 1] = 1; logits[4, 2] = 1
    masks = frame_strata(np, logits, labels)
    assert np.all(sum(mask.astype(int) for mask in masks.values()) == 1)
    assert list(np.flatnonzero(masks['aspirated'])) == [0, 1, 2]
    assert list(np.flatnonzero(masks['glottal_h'])) == [3, 4, 5, 6]
    assert masks['other'][-1]


@pytest.mark.parametrize('shape', [(0, 428), (10, 427)])
def test_invalid_frame_shape_rejected(shape):
    with pytest.raises(ValueError): frame_strata(np, np.zeros(shape), ['x']*428)


def test_mixed_preferences_do_not_become_global_mapping_or_accuracy():
    packet = {'schema_version': 'phone_diagnostic_review_packet_v1', 'items': []}
    for i in (1, 2, 3):
        packet['items'].append({'review_id': f'D{i:02}', 'note_id': f'promise-n00{i}',
            'song_id': 'promise', 'song_name': '约定', 'lyric': '字', 'raw_v4c_tokens': ['ei'],
            'category': 'vowel_proxy_conflict', 'reference_update_allowed': False,
            'choices': {'a': ['e', 'i'], 'b': ['i']},
            'choice_sources': {'a': 'proxy_reference', 'b': 'candidate'}})
    packet['packet_sha256'] = sha256_object(packet)
    answers = {'schema_version': 'phone_diagnostic_review_answers_v1',
        'packet_sha256': packet['packet_sha256'], 'scope': 'diagnostic_only_no_reference_or_training_update',
        'responses': [{'review_id': 'D03', 'decision': 'neither', 'statement': '类似于 uei'},
                      {'review_id': 'D01', 'decision': 'a', 'statement': ''},
                      {'review_id': 'D02', 'decision': 'b', 'statement': ''}]}
    report = summarize_answers(packet, answers)
    assert report['counts'] == {'neither': 1, 'proxy_reference': 1, 'candidate': 1}
    assert report['items'][0]['statement'] == '类似于 uei'
    assert not report['formal_reference_created'] and not report['global_mapping_update_allowed']
    assert not report['preference_accuracy_claim_allowed']

from copy import deepcopy

import pytest

from any2xsampa.training.limited_data_artifacts import sha256_object
from tools.import_replacement_location_answers import (
    ANSWER_SCHEMA, PACKET_SCHEMA, edge_notes, intake, validate_edge_answers, validate_edge_packet,
)


def signed(value):
    value['packet_sha256'] = sha256_object({k: v for k, v in value.items() if k != 'packet_sha256'})
    return value


def location_fixture():
    items = [{'review_id': f'L{i:02}', 'split': 'train', 'song_id': 'queen-road-east',
              'context_id': 'c', 'start_sec': i*20, 'end_sec': i*20+10,
              'alternatives': [{'choice_id': 'A', 'mapping_complete': True}]} for i in range(1, 5)]
    packet = signed({'schema_version': 'replacement_location_review_v1', 'items': items})
    payload = {'schema_version': 'replacement_location_answers_v1', 'packet_sha256': packet['packet_sha256'],
               'scope': 'correspondence_only', 'training_started': False, 'reviewer': 'reviewer',
               'answers': [{'review_id': f'L{i:02}', 'choice': choice, 'comment': comment}
                           for i, choice, comment in [(1, 'none', 'between A and B'), (2, 'A', ''),
                                                       (3, 'A', ''), (4, 'uncertain', 'A fits better')]]}
    return packet, payload


def test_actual_choice_pattern_does_not_promote_comments_or_phone_truth():
    packet, payload = location_fixture()
    before = deepcopy(payload)
    result = intake(packet, payload)
    assert [r['status'] for r in result['rows']] == ['rejected', 'correspondence_confirmed', 'correspondence_confirmed', 'uncertain']
    assert result['confirmed_correspondence_count'] == 2
    assert result['rows'][3]['selected_alternative'] is None
    assert result['rows'][0]['raw_answer']['comment'] == 'between A and B'
    assert all(not r['training_eligible'] and not r['phone_reference_confirmed'] for r in result['rows'])
    assert payload == before


def test_partial_response_keeps_other_items_unanswered():
    packet, payload = location_fixture()
    payload['answers'] = payload['answers'][1:2]
    assert [r['status'] for r in intake(packet, payload)['rows']] == ['unanswered', 'correspondence_confirmed', 'unanswered', 'unanswered']


@pytest.mark.parametrize('field,value', [('scope', 'phone_truth'), ('training_started', True), ('reviewer', '')])
def test_rejects_expanded_scope(field, value):
    packet, payload = location_fixture()
    payload[field] = value
    with pytest.raises(ValueError):
        intake(packet, payload)


def test_non_training_and_packet_tampering_rejected():
    packet, payload = location_fixture()
    packet['items'][0]['split'] = 'test'
    with pytest.raises(ValueError, match='hash'):
        intake(packet, payload)
    signed(packet)
    payload['packet_sha256'] = packet['packet_sha256']
    with pytest.raises(ValueError, match='training'):
        intake(packet, payload)


def edge_fixture():
    note = {'note_id': 'n1', 'lyric': 'x', 'phones': ['i', 'n'], 'mapping_status': 'resolved_proxy'}
    clip = {'example_id': 'clip', 'song_id': 'promise', 'split': 'train', 'start_sec': 10,
            'end_sec': 20, 'training_eligible': False, 'notes': [note]}
    items = [{'review_id': side, 'example_id': 'clip', 'side': side, 'original_cut_sec': clip[side+'_sec'],
              'cut_in_preview_sec': 1.25, 'duration_sec': 2.5, 'notes': [note]} for side in ('start', 'end')]
    return signed({'schema_version': PACKET_SCHEMA, 'clips': [clip], 'items': items})


@pytest.mark.parametrize('kind', ['split', 'clock', 'note', 'duplicate', 'missing'])
def test_edge_packet_guards(kind):
    packet = edge_fixture()
    validate_edge_packet(packet)
    if kind == 'split': packet['clips'][0]['song_id'] = 'king-of-karaoke'
    elif kind == 'clock': packet['items'][0]['original_cut_sec'] += .1
    elif kind == 'note': packet['items'][0]['notes'] = []
    elif kind == 'duplicate': packet['items'].append(deepcopy(packet['items'][0]))
    else: packet['items'].pop()
    signed(packet)
    with pytest.raises(ValueError):
        validate_edge_packet(packet)


def test_absent_at_end_preserved_and_dev_envelope_rejected():
    packet = edge_fixture()
    answer = {'review_id': 'end', 'status': 'confirmed', 'observation_schema': 'phone_audibility_v2',
              'note_id': 'n1', 'phones': [{'phone_index': 0, 'edge_status': 'absent'}, {'phone_index': 1, 'edge_status': 'full'}],
              'statement': 'initial phone not audible', 'audible_identity_confirmed': True}
    payload = {'schema_version': ANSWER_SCHEMA, 'packet_sha256': packet['packet_sha256'], 'reviewer': 'r', 'answers': [answer]}
    assert validate_edge_answers(packet, payload)['end']['phones'][0]['edge_status'] == 'absent'
    payload['schema_version'] = 'original_edge_review_answers_v2'
    with pytest.raises(ValueError, match='binding'):
        validate_edge_answers(packet, payload)


def test_neighbors_keep_unresolved_notes_not_just_easy_phones():
    song = {'notes': [{'note_id': f'n{i}', 'raw_lyric': 'x', 'start_sec': i, 'end_sec': i+1,
                       'model_target_ipa': [] if i == 2 else ['n'],
                       'mapping_status': 'unresolved' if i == 2 else 'resolved_proxy'} for i in range(5)]}
    notes = edge_notes(song, 2.5)
    assert [n['note_id'] for n in notes] == ['n1', 'n2', 'n3']
    assert notes[1]['mapping_status'] == 'unresolved'

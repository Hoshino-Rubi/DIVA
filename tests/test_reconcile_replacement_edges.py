from copy import deepcopy

import pytest

from test_replacement_location_intake import edge_fixture, signed
from tools.reconcile_replacement_edges import build_reference


def fixture():
    packet = edge_fixture()
    notes = [{'note_id': f'n{i}', 'lyric': str(i), 'phones': ['i', 'n'], 'mapping_status': 'resolved_proxy'} for i in range(3)]
    packet['clips'][0].update(notes=notes, source_sequence_approval_bound=True, audio_path='fixture.wav',
        audio_sha256='1'*64, source_vsqx_sha256='2'*64, reference_manifest_sha256='3'*64)
    packet['items'][0]['notes'] = notes[:1]
    packet['items'][1]['notes'] = notes[-1:]
    signed(packet)
    answers = [{'review_id': side, 'status': 'confirmed', 'observation_schema': 'phone_audibility_v2',
        'note_id': nid, 'phones': [{'phone_index': 0, 'edge_status': 'absent'}, {'phone_index': 1, 'edge_status': 'partial' if side == 'start' else 'full'}],
        'statement': 'heard edges', 'audible_identity_confirmed': True} for side, nid in [('start', 'n0'), ('end', 'n2')]]
    payload = {'schema_version': 'replacement_edge_review_answers_v1', 'packet_sha256': packet['packet_sha256'],
               'reviewer': 'fixture', 'answers': answers}
    return packet, payload


def test_absence_and_partial_preserved_without_guessing_cut_direction():
    packet, payload = fixture()
    original = deepcopy(payload)
    result = build_reference(packet, payload)
    row = result['examples'][0]
    assert row['status'] == 'human_confirmed'
    assert row['source_phone_count'] == 6 and row['reference_phone_count'] == 4
    assert row['partial_phone_count'] == 1
    assert [(p['note_id'], p['phone_index']) for p in row['omitted_phones']] == [('n0', 0), ('n2', 0)]
    assert [p['phone'] for p in row['phones']] == ['n', 'i', 'n', 'n']
    assert row['training_eligible'] is False
    assert payload == original


def test_confirmed_envelope_does_not_override_uncertain_phone():
    packet, payload = fixture()
    payload['answers'][1]['phones'][0]['edge_status'] = 'uncertain'
    result = build_reference(packet, payload)
    row = result['examples'][0]
    assert row['status'] == 'pending' and row['phones'] == [] and row['source_phones'] == []
    assert row['blocking_reasons'] == ['edge_has_uncertain_phones']
    assert result['coverage']['confirmed_windows'] == 0
    assert result['unresolved_edge_phones'][0]['phone_index'] == 0


@pytest.mark.parametrize('kind,reason', [
    ('unreviewed', 'edge_has_unreviewed_phones'),
    ('missing', 'incomplete_confirmed_edge_pair'),
    ('approval', 'source_sequence_approval_not_bound'),
    ('all_absent', 'selected_edge_note_has_no_audible_phone'),
    ('mapping', 'whole selected window contains unresolved mapping'),
    ('reverse', 'selected start note follows end note'),
])
def test_fail_closed_window_conditions(kind, reason):
    packet, payload = fixture()
    if kind == 'unreviewed': payload['answers'][0]['phones'].pop()
    elif kind == 'missing': payload['answers'].pop()
    elif kind == 'approval': packet['clips'][0]['source_sequence_approval_bound'] = False
    elif kind == 'all_absent': payload['answers'][0]['phones'][1]['edge_status'] = 'absent'
    elif kind == 'mapping': packet['clips'][0]['notes'][1]['mapping_status'] = 'unresolved'
    else:
        for item in packet['items']: item['notes'] = packet['clips'][0]['notes']
        payload['answers'][0]['note_id'], payload['answers'][1]['note_id'] = 'n2', 'n0'
    signed(packet)
    payload['packet_sha256'] = packet['packet_sha256']
    row = build_reference(packet, payload)['examples'][0]
    assert row['status'] == 'pending' and row['phones'] == []
    assert row['blocking_reasons'] == [reason]


def test_unknown_note_outside_human_selected_range_not_deleted_from_evidence():
    packet, payload = fixture()
    outside = {'note_id': 'outside', 'lyric': '?', 'phones': [], 'mapping_status': 'unresolved'}
    packet['clips'][0]['notes'].insert(0, outside)
    signed(packet)
    payload['packet_sha256'] = packet['packet_sha256']
    row = build_reference(packet, payload)['examples'][0]
    assert row['status'] == 'human_confirmed'
    assert row['catalog_outside_confirmed_note_range'] == ['outside']
    assert row['note_ids'] == ['n0', 'n1', 'n2']


def test_rejected_correspondence_has_no_reference():
    packet, payload = fixture()
    payload['answers'][1].update(status='wrong_correspondence', note_id=None, phones=[], audible_identity_confirmed=False)
    row = build_reference(packet, payload)['examples'][0]
    assert row['status'] == 'rejected' and row['phones'] == []

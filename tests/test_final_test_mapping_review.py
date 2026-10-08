from copy import deepcopy
import pytest
from tests.test_final_test_reference_review import fixture as original_fixture
from any2xsampa.training.limited_data_artifacts import read_json, sha256_object
from tools.final_test_mapping_review import (CONFIG, SCHEMA, ANSWERS, CONSENT, audit_intake,
    validate_decisions, assemble_proxy_reference)


def fixture():
    source, original = original_fixture()
    base = {'tokens': {p:{'ipa':[p]} for p in ['s','a','n']},
            'model_token_aliases':{}, 'unresolved_tokens':['7','UN','iE_r','iUN','uo']}
    for song in source['songs']:
        song['song_name'] = song['song_id']
        song['catalog'] = [dict(note_id=song['song_id']+f'-n{i:03}', lyric=str(i),
            raw_v4c_tokens=['s','a','n'] if i==1 else [['7'],['UN'],['iE_r'],['iUN'],['uo']][i-2],
            phones=['s','a','n'] if i==1 else [], mapping_status='resolved_proxy' if i==1 else 'unresolved',
            unresolved_tokens=[] if i==1 else [['7'],['UN'],['iE_r'],['iUN'],['uo']][i-2]) for i in range(1,7)]
    source['packet_sha256'] = sha256_object({k:v for k,v in source.items() if k!='packet_sha256'})
    original['packet_sha256'] = source['packet_sha256']; original['unconfirmed_drafts'] = {}
    for a, item in zip(original['answers'], source['items']):
        song = next(s for s in source['songs'] if s['song_id']==item['song_id'])
        a.update(status='located', sequence_confirmed=True, edges={side:dict(note_id=song['catalog'][i]['note_id'],
            phones=[],whole_note_audibility='uncertain' if side=='start' else 'full') for side,i in [('start',0),('end',5)]})
        draft = deepcopy(a); draft['status']='confirmed'
        draft['edges']['start']['phones'] = [dict(phone_index=i,edge_status='absent' if i==0 else 'full') for i in range(3)]
        original['unconfirmed_drafts'][a['window_id']] = draft
    rules=read_json(CONFIG)['rules']
    phones=sorted({'s','a','n',*(p for r in rules for o in r['options'] for p in o['ipa'])})
    vocab={'<blank>':0,**{p:i+1 for i,p in enumerate(phones)}}
    review=dict(schema_version=SCHEMA, manual_audit=audit_intake(source,original), rules=rules,
                allowed_ipa=phones,consent_scope=CONSENT, training_allowed=False,test_predictions_opened=False)
    review['packet_sha256']=sha256_object(review)
    decisions=[dict(token=r['token'],status='approve_proxy',option_id=r['options'][0]['id'],ipa=r['options'][0]['ipa'],comment='') for r in rules]
    payload=dict(schema_version=ANSWERS,packet_sha256=review['packet_sha256'],reviewer='human',
                 decisions=decisions,consent=CONSENT,authorize_frozen_proxy_comparison=True,
                 training_started=False,test_predictions_opened=False)
    return review,payload,source,original,base,vocab


def test_known_draft_absence_recovered_without_reasking_or_inventing_unknown_edge():
    review,_,_,_,_,_=fixture();a=review['manual_audit']
    assert a['coverage']['location_confirmed_windows']==6
    assert a['coverage']['complete_reference_windows']==0
    first=a['rows'][0]['edges'];assert first['start']['phones'][0]['edge_status']=='absent'
    assert first['start']['source']=='preserved_explicitly_confirmed_draft'
    assert first['end']['phones']==[] and first['end']['whole_note_audibility']=='full'


def test_proxy_requires_consent_and_preserves_full_window_and_absence():
    review,payload,source,original,base,vocab=fixture();before=deepcopy((source,original,base))
    result=assemble_proxy_reference(review,payload,source,original,base,vocab)
    assert result['coverage']['proxy_ready_windows']==6
    assert result['coverage']['reference_phones']==72
    assert result['formal_original_ipa_accuracy_allowed'] is False
    assert result['training_allowed'] is False and result['test_predictions_opened'] is False
    for row in result['examples']:
        assert len(row['note_ids'])==6 and row['omitted_phones'][0]['phone']=='s'
        assert row['training_eligible'] is False and row['status']=='proxy_reference_ready'
        assert row['edge_evidence']['end']['evidence_scope']=='derived_proxy_units_from_whole_note_full'
    assert (source,original,base)==before


def test_all_deferred_exports_and_cannot_create_labels():
    review,payload,source,original,base,vocab=fixture()
    for d in payload['decisions']: d.update(status='defer',ipa=[],option_id=None)
    payload.update(consent=None,authorize_frozen_proxy_comparison=False)
    r=assemble_proxy_reference(review,payload,source,original,base,vocab)
    assert r['coverage']['proxy_ready_windows']==0 and r['coverage']['reference_phones']==0


def test_unresolved_internal_token_excludes_entire_window():
    review,payload,source,original,base,vocab=fixture();payload['decisions'][0].update(status='defer',ipa=[],option_id=None)
    r=assemble_proxy_reference(review,payload,source,original,base,vocab)
    assert all(not row['proxy_evaluation_eligible'] and row['phones']==[] for row in r['examples'])


def test_unknown_partial_note_never_expanded_to_full_phones():
    review,payload,source,original,base,vocab=fixture()
    original['answers'][0]['edges']['end']['whole_note_audibility']='partial'
    original['unconfirmed_drafts']['T01']['edges']['end']['whole_note_audibility']='partial'
    review['manual_audit']=audit_intake(source,original)
    review['packet_sha256']=sha256_object({k:v for k,v in review.items() if k!='packet_sha256'});payload['packet_sha256']=review['packet_sha256']
    r=assemble_proxy_reference(review,payload,source,original,base,vocab)
    assert r['coverage']['proxy_ready_windows']==5 and r['examples'][0]['phones']==[]


def test_explicit_repeated_custom_phone_units_are_not_collapsed():
    review,payload,source,original,base,vocab=fixture()
    payload['decisions'][0].update(status='custom_proxy',option_id=None,ipa=['n','n'])
    r=assemble_proxy_reference(review,payload,source,original,base,vocab)
    phones=[p['phone'] for p in r['examples'][0]['phones'] if p['note_id'].endswith('n002')]
    assert phones==['n','n']


@pytest.mark.parametrize('fault',['packet_hash','answers_hash','duplicate','option_mismatch','unsupported','blank','consent','leak','scope'])
def test_invalid_proxy_decisions_fail(fault):
    review,payload,*_=fixture()
    if fault=='packet_hash': review['packet_sha256']='bad'
    if fault=='answers_hash': payload['packet_sha256']='bad'
    if fault=='duplicate': payload['decisions'][1]=deepcopy(payload['decisions'][0])
    if fault=='option_mismatch': payload['decisions'][0]['ipa']=['n']
    if fault in {'unsupported','blank'}: payload['decisions'][0].update(status='custom_proxy',option_id=None,ipa=['<blank>' if fault=='blank' else '?'])
    if fault=='consent':payload['consent']=None
    if fault=='leak':payload['test_predictions_opened']=True
    if fault=='scope':payload['authorize_frozen_proxy_comparison']=False
    with pytest.raises(ValueError): validate_decisions(review,payload)


def test_changed_saved_draft_does_not_override_location():
    _,_,source,original,_,_=fixture();original['unconfirmed_drafts']['T01']['edges']['start']['note_id']='other'
    with pytest.raises(ValueError,match='does not match'):audit_intake(source,original)

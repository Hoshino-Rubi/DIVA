from copy import deepcopy

import pytest

from any2xsampa.training.limited_data_artifacts import sha256_object
from tools.free_location_review import assemble, validate_answers, validate_packet


def fixtures():
    notes = [dict(note_id=f'n{i}',lyric='字',phones=['a','n'],mapping_status='resolved_proxy',start_sec=i,end_sec=i+1) for i in range(8)]
    p = dict(schema_version='free_location_review_packet_v1',song_id='queen-road-east',split='train',
             maximum_accepted_windows=1,active_slots_after=20,machine_correspondence_passed=False,training_started=False,
             windows=[dict(window_id='M01-A',start_sec=10,end_sec=20),dict(window_id='M01-B',start_sec=30,end_sec=40)],
             catalog=notes,source_sequence_approval_bound=True,original_audio={'path':'fixture.wav','sha256':'a'*64},
             source_vsqx={'sha256':'b'*64},source_bindings={'reference':{'sha256':'c'*64}})
    p['packet_sha256']=sha256_object(p)
    edges={side:dict(review_id='M01-A'+suffix,status='confirmed',observation_schema='phone_audibility_v2',
                    note_id=nid,phones=[dict(phone_index=0,edge_status='full'),dict(phone_index=1,edge_status='partial')],
                    audible_identity_confirmed=True,statement='human observation')
           for side,suffix,nid in [('start','-S','n4'),('end','-E','n7')]}
    a=dict(schema_version='free_location_review_answers_v1',packet_sha256=p['packet_sha256'],reviewer='fixture',training_started=False,
           answer=dict(window_id='M01-A',status='confirmed',comment='',sequence_confirmed=True,edges=edges))
    return p,a


def resign(p,a):
    p['packet_sha256']=sha256_object({k:v for k,v in p.items() if k!='packet_sha256'})
    a['packet_sha256']=p['packet_sha256']


def test_complete_catalog_selection_produces_one_continuous_reference():
    p,a=fixtures();row=assemble(p,a)['examples'][0]
    assert row['status']=='human_confirmed'
    assert row['note_ids']==['n4','n5','n6','n7']
    assert len(row['phones'])==8 and row['partial_phone_count']==2
    assert row['start_sec']==10 and row['end_sec']==20
    assert not row['training_eligible']


@pytest.mark.parametrize('kind',['packet','window','missing_phone','reversed','assertion','phone_index','sealed','slots'])
def test_bad_free_location_answers_fail(kind):
    p,a=fixtures();edge=a['answer']['edges']['start']
    if kind=='packet': a['packet_sha256']='f'*64
    elif kind=='window': a['answer']['window_id']='M01-C'
    elif kind=='missing_phone': edge['phones'].pop()
    elif kind=='reversed': edge['note_id']='n7';a['answer']['edges']['end']['note_id']='n4'
    elif kind=='assertion':a['answer']['sequence_confirmed']=False
    elif kind=='phone_index':edge['phones'][1]['phone_index']=0
    elif kind=='sealed':p['song_id']='king-of-karaoke';resign(p,a)
    else:p['maximum_accepted_windows']=2;resign(p,a)
    with pytest.raises(ValueError):validate_answers(p,a)


def test_unknown_inside_or_uncertain_edge_excludes_whole_window():
    p,a=fixtures();p['catalog'][5]['mapping_status']='unresolved';resign(p,a)
    row=assemble(p,a)['examples'][0]
    assert row['status']=='pending' and not row['phones']
    p,a=fixtures();a['answer']['edges']['start']['phones'][0]['edge_status']='uncertain'
    row=assemble(p,a)['examples'][0]
    assert row['status']=='pending' and not row['phones']


def test_absent_and_partial_are_preserved_and_only_one_option_is_accepted():
    p,a=fixtures();a['answer']['edges']['start']['phones'][0]['edge_status']='absent'
    result=assemble(p,a);row=result['examples'][0]
    assert result['confirmed_count']==1 and len(result['examples'])==1
    assert row['omitted_phones'][0]['audibility']=='absent'
    assert row['phones'][0]['audibility']=='partial'


def test_pending_contains_no_labels_and_keeps_human_words():
    p,a=fixtures();a['answer'].update(status='pending',sequence_confirmed=False,edges={},comment='cannot locate')
    result=assemble(p,a)
    assert result['confirmed_count']==0
    assert result['examples'][0]['raw_answer']['comment']=='cannot locate'


def test_feature_intake_recognizes_new_schema(tmp_path,monkeypatch):
    from any2xsampa.training.limited_data_artifacts import write_json_new,sha256_file
    from tools.accept_reviewed_training_features import verified_reference,requests_for
    import tools.free_location_review as tool
    p,a=fixtures();result=assemble(p,a)
    pp,ap=tmp_path/'packet.json',tmp_path/'answers.json'
    write_json_new(pp,p);write_json_new(ap,a);write_json_new(tmp_path/'reviewed-reference.json',result)
    write_json_new(tmp_path/'INPUTS.json',{n:{'path':str(v),'sha256':sha256_file(v)} for n,v in [('packet',pp),('answers',ap)]})
    monkeypatch.setattr(tool,'verify_packet',lambda directory,packet:validate_packet(packet))
    rebuilt=verified_reference(tmp_path);assert rebuilt==result
    request=requests_for(rebuilt,sha256_file(tmp_path/'reviewed-reference.json'))['rows'][0]
    assert request['audio_role']=='original' and 'phones' not in request and 'catalog' not in request

from copy import deepcopy

import pytest

from any2xsampa.training.limited_data_artifacts import sha256_object
from tools.batch_training_review import validate_answers,validate_packet,assemble
from tools.build_batch_training_review import choice_for,proposals


def signed(value):
    value['packet_sha256']=sha256_object({k:v for k,v in value.items() if k!='packet_sha256'})
    return value


def fixture():
    notes=[{'note_id':f'n{i}','lyric':str(i),'phones':['n','a'],'mapping_status':'resolved_proxy'} for i in range(3)]
    choice={'choice_id':'A','catalog':notes,'notes':notes,'edge_notes':{'start':notes[:1],'end':notes[-1:]}}
    item={'review_id':'R01','song_id':'promise','split':'train','start_sec':0,'end_sec':10,'kind':'replacement',
          'training_eligible':False,'correspondence_preconfirmed':False,'alternatives':[choice]}
    packet=signed({'schema_version':'batch_training_review_packet_v1','items':[item],
                   'active_slots_after':10,'source_bindings':{'reference':{'sha256':'a'*64}}})
    edges={side:{'review_id':'R01'+suffix,'status':'confirmed','observation_schema':'phone_audibility_v2',
                 'note_id':nid,'phones':[{'phone_index':0,'edge_status':'absent'},{'phone_index':1,'edge_status':'full'}],
                 'audible_identity_confirmed':True,'statement':'heard'} for side,suffix,nid in [('start','-S','n0'),('end','-E','n2')]}
    answer={'review_id':'R01','status':'confirmed','choice':'A','sequence_confirmed':True,'edges':edges,'comment':'raw text'}
    payload={'schema_version':'batch_training_review_answers_v1','packet_sha256':packet['packet_sha256'],
             'reviewer':'fixture','training_started':False,'answers':[answer]}
    song={'original_audio':{'path':'fixture.wav','sha256':'b'*64},'human_vsqx':{'sha256':'c'*64}}
    verified={'songs':{'promise':song},'triplets':{'items':[{'song_id':'promise','human_vsqx':{'sha256':'c'*64},
                                                        'human_claims':{'phoneme_correctness':'user_asserted_correct'}}]}}
    return packet,payload,verified


def test_combined_correspondence_edges_assemble_once_no_internal_omission():
    packet,payload,verified=fixture()
    result=assemble(packet,payload,verified)
    assert result['confirmed_count']==1
    row=result['examples'][0]
    assert row['reference_phone_count']==4 and row['source_phone_count']==6
    assert row['note_ids']==['n0','n1','n2'] and len(row['omitted_phones'])==2
    assert row['training_eligible'] is False
    from tools.accept_reviewed_training_features import requests_for
    requests=requests_for(result,'f'*64)
    assert len(requests['rows'])==1 and requests['rows'][0]['example_id']=='batch-r01'
    assert 'phones' not in requests['rows'][0]


@pytest.mark.parametrize('kind',['hash','id','duplicate','choice','range','missing_phone','assertion','reviewer','schema'])
def test_bad_answer_rejected(kind):
    packet,payload,_=fixture();a=payload['answers'][0]
    if kind=='hash': payload['packet_sha256']='f'*64
    elif kind=='id': a['review_id']='R99'
    elif kind=='duplicate': payload['answers'].append(deepcopy(a))
    elif kind=='choice': a['choice']='C'
    elif kind=='range': a['edges']['start']['note_id']='n2';a['edges']['end']['note_id']='n0'
    elif kind=='missing_phone': a['edges']['start']['phones'].pop()
    elif kind=='assertion': a['sequence_confirmed']=False
    elif kind=='reviewer': payload['reviewer']=''
    else: payload['schema_version']='replacement_edge_review_answers_v1'
    with pytest.raises(ValueError): validate_answers(packet,payload)


def test_uncertain_phone_keeps_whole_window_pending():
    packet,payload,verified=fixture()
    payload['answers'][0]['edges']['end']['phones'][0]['edge_status']='uncertain'
    row=assemble(packet,payload,verified)['examples'][0]
    assert row['status']=='pending' and not row['phones']


def test_unknown_interior_is_not_dropped_to_pass():
    packet,payload,verified=fixture()
    packet['items'][0]['alternatives'][0]['catalog'][1]['mapping_status']='unresolved'
    signed(packet);payload['packet_sha256']=packet['packet_sha256']
    row=assemble(packet,payload,verified)['examples'][0]
    assert row['status']=='pending' and not row['phones']


def test_pending_or_rejected_does_not_force_phones():
    packet,payload,verified=fixture()
    a=payload['answers'][0]
    for status in ('uncertain','rejected'):
        a.update(status=status,choice=None,edges={},sequence_confirmed=False)
        assert assemble(packet,payload,verified)['confirmed_count']==0


@pytest.mark.parametrize('kind',['test','overlap','limit','catalog'])
def test_packet_guards(kind):
    packet,_,_=fixture()
    if kind=='test': packet['items'][0]['song_id']='king-of-karaoke'
    elif kind=='overlap': packet['items'].append({**deepcopy(packet['items'][0]),'review_id':'R02'})
    elif kind=='limit': packet['active_slots_after']=21
    else: packet['items'][0]['alternatives'][0]['catalog'].append(deepcopy(packet['items'][0]['alternatives'][0]['catalog'][0]))
    signed(packet)
    with pytest.raises(ValueError):validate_packet(packet)


def test_leading_score_silence_is_visible_navigation_not_invented_phone_time():
    song={'notes':[{'note_id':'n1','raw_lyric':'x','start_sec':8,'end_sec':9,'model_target_ipa':['n'],
                    'mapping_status':'resolved_proxy'}]}
    c=choice_for(song,0,10,'human_reviewed_section',0)
    assert c['edge_score_gaps']==['start'] and c['catalog'][0]['note_id']=='n1'
    assert c['training_eligible'] is False


def test_candidate_pair_rejection_and_reviewed_audio_protection():
    from test_replacement_candidates import fixtures
    context,song,paths,policy=fixtures()
    song['rendered_audio']={'duration_sec':100}
    policy={**policy,'maximum_alternatives':3}
    alignment={'paths':{role:{'candidates':[{**paths[role],'rank':1}]} for role in ('mixed','vocal')}}
    pool,_=proposals(context,song,alignment,[],[],[],policy)
    assert pool
    chosen=pool[0]
    blocked={'song_id':song['song_id'],'start_sec':context['start_sec'],'end_sec':context['end_sec']}
    assert not proposals(context,song,alignment,[blocked],[],[],policy)[0]
    old={**chosen,'rendered_start_sec':chosen['alternatives'][0]['start_sec'],'review_id':'C01'}
    new,_=proposals(context,song,alignment,[],[old],[],policy)
    assert not any(r['start_sec']==chosen['start_sec'] for r in new)
    assert all(r['machine_accepted'] is False for r in pool)

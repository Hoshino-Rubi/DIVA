from copy import deepcopy
import pytest
from tools.final_test_reference_review import validate_answers,assemble,validate_packet,SCHEMA,ANSWERS
from any2xsampa.training.limited_data_artifacts import sha256_object


def fixture():
    songs=[];items=[]
    for sid in ['ke-yi-ke-zai','king-of-karaoke']:
        notes=[dict(note_id=sid+f'-n{i:03}',lyric=str(i),phones=['a','n'],mapping_status='resolved_proxy') for i in range(1,4)]
        songs.append(dict(song_id=sid,catalog=notes,original_audio=dict(path='original.wav',sha256='a'*64),source_vsqx=dict(sha256='b'*64),source_sequence_approval_bound=True))
        for i in range(3):items.append(dict(window_id=f'T{len(items)+1:02}',source_example_id=sid+f'-original-{i+1:02}',song_id=sid,start_sec=20*i,end_sec=20*i+18))
    p=dict(schema_version=SCHEMA,songs=songs,items=items,mapping_questions=[{'token':t} for t in ['7','UN','iE_r','iUN','uo']],source_bindings={'reference':{'sha256':'c'*64},'candidate':{'sha256':'d'*64}},test_predictions_opened=False,candidate_already_frozen=True,training_allowed=False)
    p['packet_sha256']=sha256_object(p)
    a=dict(schema_version=ANSWERS,packet_sha256=p['packet_sha256'],reviewer='human',training_started=False,test_predictions_opened=False,answers=[dict(window_id=i['window_id'],status='pending',comment='',sequence_confirmed=False,edges={}) for i in items],mapping_responses=[dict(token=q['token'],status='pending',description='') for q in p['mapping_questions']])
    return p,a


def confirm_first(p,a):
    row=a['answers'][0];row.update(status='confirmed',sequence_confirmed=True)
    for side,i in [('start',0),('end',2)]:
        n=p['songs'][0]['catalog'][i]
        row['edges'][side]=dict(note_id=n['note_id'],review_id=row['window_id']+('-S' if side=='start' else '-E'),status='confirmed',observation_schema='phone_audibility_v2',audible_identity_confirmed=True,statement='heard full edge',phones=[dict(phone_index=j,edge_status='full') for j in range(2)])


def test_pending_batch_never_opens_test_or_creates_references():
    p,a=fixture();r=assemble(p,a);assert all(x['status']=='pending' and x['phones']==[] for x in r['examples']);assert r['training_allowed'] is False and r['test_predictions_opened'] is False


def test_confirmed_test_reference_remains_test_and_no_training():
    p,a=fixture();confirm_first(p,a);r=assemble(p,a);x=r['examples'][0]
    assert x['status']=='human_confirmed' and len(x['phones'])==6 and x['split']=='test' and x['training_eligible'] is False


def test_located_unknown_mapping_does_not_automatically_apply_description():
    p,a=fixture();row=a['answers'][0];row.update(status='located',sequence_confirmed=True,edges={side:dict(note_id=p['songs'][0]['catalog'][i]['note_id'],phones=[],whole_note_audibility='full') for side,i in [('start',0),('end',2)]})
    a['mapping_responses'][0].update(status='described',description='user description retained verbatim')
    r=assemble(p,a);assert r['examples'][0]['status']=='pending' and not r['mapping_descriptions_automatically_applied']


def test_unknown_interior_mapping_blocks_entire_window():
    p,a=fixture();p['songs'][0]['catalog'][1]['mapping_status']='unresolved';p['songs'][0]['catalog'][1]['phones']=[];p['packet_sha256']=sha256_object({k:v for k,v in p.items() if k!='packet_sha256'});a['packet_sha256']=p['packet_sha256'];confirm_first(p,a)
    r=assemble(p,a);assert r['examples'][0]['status']=='pending' and r['examples'][0]['phones']==[]


@pytest.mark.parametrize('fault',['hash','duplicates','reversed','mapping_duplicate','predictions'])
def test_invalid_answers_fail(fault):
    p,a=fixture();confirm_first(p,a)
    if fault=='hash':a['packet_sha256']='wrong'
    if fault=='duplicates':a['answers'][1]=deepcopy(a['answers'][0])
    if fault=='reversed':a['answers'][0]['edges']['start']['note_id']=p['songs'][0]['catalog'][2]['note_id'];a['answers'][0]['edges']['end']['note_id']=p['songs'][0]['catalog'][0]['note_id']
    if fault=='mapping_duplicate':a['mapping_responses'][1]=deepcopy(a['mapping_responses'][0])
    if fault=='predictions':a['test_predictions_opened']=True
    with pytest.raises(ValueError):validate_answers(p,a)

from copy import deepcopy

import pytest

from tools.accept_reviewed_training_features import accept_row, requests_for
from tools.prepare_training_progress_review import anchor_failures, select_retained
from tools.reconcile_replacement_edges import potential_window_count
from test_reconcile_replacement_edges import fixture
from tools.reconcile_replacement_edges import build_reference


def acceptance_fixture():
    row = {'example_id':'replacement-L03','song_id':'promise','split':'train','audio_role':'original',
        'audio_path':'fixture.wav','audio_sha256':'a'*64,'start_sec':1,'end_sec':11,'status':'human_confirmed',
        'phones':[{'phone':'x'},{'phone':'x'},{'phone':'y'}]}
    entry = {k:v for k,v in row.items() if k not in {'phones','status','audio_path'}}
    entry.update(example_id='replacement-l03',frames=5,test_outputs_sealed=False)
    vocab={'<blank>':0,'x':1,'y':2,**{f'p{i}':i for i in range(3,428)}}
    return row,entry,vocab


def test_acceptance_checks_repeated_ctc_labels_without_training():
    row,entry,vocab=acceptance_fixture()
    result=accept_row(row,entry,vocab,(5,1024),(5,428))
    assert result['ctc_minimum_frames']==4 and result['dataset_training_gate_passed'] is False


@pytest.mark.parametrize('kind',['hash','clock','role','blank','unknown','frames','shape','pending'])
def test_feature_acceptance_fails_closed(kind):
    row,entry,vocab=acceptance_fixture()
    shape=(5,1024)
    if kind=='hash': entry['audio_sha256']='b'*64
    elif kind=='clock': entry['start_sec']+=.1
    elif kind=='role': entry['audio_role']='rendered'
    elif kind=='blank': row['phones'][0]['phone']='<blank>'
    elif kind=='unknown': row['phones'][0]['phone']='?'
    elif kind=='frames': entry['frames']=3; shape=(3,1024)
    elif kind=='shape': shape=(5,64)
    else: row['status']='pending'
    with pytest.raises(ValueError): accept_row(row,entry,vocab,shape,(entry['frames'],428))


def test_audio_worker_never_gets_phone_labels_or_pending_reference():
    packet,payload=fixture()
    reference=build_reference(packet,payload)
    reference['examples'].append({**deepcopy(reference['examples'][0]),'example_id':'pending','status':'pending'})
    requests=requests_for(reference,'f'*64)
    assert len(requests['rows'])==1
    assert 'phones' not in requests['rows'][0] and 'note_ids' not in requests['rows'][0]
    assert requests['contains_labels_or_score_context'] is False


def test_anchor_is_local_section_constraint_not_constant_offset():
    path={'path_anchors':[{'original_sec':0,'rendered_sec':10},{'original_sec':10,'rendered_sec':21}]}
    anchor={'review_id':'C11','start_sec':2,'end_sec':8,'rendered_start_sec':14,'rendered_end_sec':18}
    assert anchor_failures(path,[anchor])==[]
    assert anchor_failures(path,[{**anchor,'rendered_start_sec':100,'rendered_end_sec':110}])==['C11']
    assert anchor_failures(path,[{**anchor,'start_sec':12,'end_sec':18}])==['C11']


def test_review_prioritizes_distinct_songs_and_clean_existing_correspondence():
    rows=[{'review_id':rid,'song_id':song,'start_sec':start,'clean_correspondence':clean}
          for rid,song,start,clean in [('C03','half',58,True),('C11','freedom',194,True),
                                      ('C13','half',218,True),('C20','half',229,False)]]
    selected=select_retained(rows,[{'song_id':'promise'}],['freedom','queen','half','promise','like'],2)
    assert [r['review_id'] for r in selected]==['C11','C03']


def test_promoting_existing_candidate_does_not_count_same_window_twice():
    old={'song_id':'half','start_sec':10,'end_sec':20}
    prior={'song_id':'promise','start_sec':20,'end_sec':30}
    assert potential_window_count({'retained':[old]},[dict(old)],[prior])==2


def test_interpolation_cache_preserves_exact_proposals(monkeypatch):
    import any2xsampa.training.replacement_candidates as module
    from test_replacement_candidates import fixtures
    context,song,paths,policy=fixtures()
    calls=[]
    real=module.path_time
    def counted(candidate,t):
        calls.append((id(candidate),t))
        return real(candidate,t)
    monkeypatch.setattr(module,'path_time',counted)
    cached=module.propose_windows(context,song,paths,[],[],policy)
    assert len(calls)==len(set(calls))
    monkeypatch.setattr(module,'lru_cache',lambda **kwargs: lambda fn:fn)
    calls.clear()
    uncached=module.propose_windows(context,song,paths,[],[],policy)
    assert cached==uncached
    assert len(calls)>len(set(calls))


def test_finish_handles_rejected_rows_without_example_ids(tmp_path, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file
    import tools.accept_reviewed_training_features as module
    row, entry, vocabulary = acceptance_fixture()
    row['example_id'] = 'fixture-training'
    entry['example_id'] = row['example_id']
    reference = {'manifest_hash': 'a'*64, 'examples': [
        {'review_id': 'R03', 'status': 'rejected', 'phones': []}, row]}
    monkeypatch.setattr(module, 'verified_reference', lambda _: reference)
    monkeypatch.setattr(module, 'sha256_file', lambda _: 'b'*64)
    monkeypatch.setattr(module, 'verify_file', lambda *_: None)
    evidence = tmp_path/'evidence.safetensors'
    save_file({'encoder_features': np.zeros((5,1024),dtype=np.float32),
               'frame_logits': np.zeros((5,428),dtype=np.float32)},str(evidence))
    entry.update(evidence_path=str(evidence),evidence_sha256='b'*64,
                 posterior_path=str(tmp_path/'posterior.json'),posterior_sha256='b'*64,vocabulary_sha256='b'*64)
    requests=module.requests_for(reference,'b'*64)
    index={'requests_sha256':'b'*64,'source_dataset_hash':reference['manifest_hash'],
           'all_parameters_frozen':True,'no_labels_or_score_received':True,'entries':[entry]}
    posterior={'producer':{'crop':{'requested_start_sec':1,'requested_end_sec':11},
                           'input_audio':{'sha256':row['audio_sha256']}}}
    docs={'original-reviewed.json':requests,'index.json':index,'posterior.json':posterior}
    monkeypatch.setattr(module,'read_json',lambda p:docs[p.name])
    monkeypatch.setattr(module,'load_vocabulary',lambda _:vocabulary)
    accepted=module.finish(tmp_path,tmp_path,write=False)
    assert accepted['valid_window_count']==1
    assert accepted['rows'][0]['example_id']=='fixture-training'

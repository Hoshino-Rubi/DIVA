from __future__ import annotations
from copy import deepcopy
import time
import pytest
from any2xsampa.training.whole_sequence_supervision import (
    plan_chunks, validate_chunks, accepted_reference, backward_whole_sequence,
    deterministic_checkpointed_tail,
)

SPEC={'core_seconds':16.,'context_seconds':2.,'maximum_encoder_seconds':20.}


@pytest.mark.parametrize('rate,samples',[(44100,11269120),(48000,12146880),(16000,256001),(16000,256000)])
def test_global_grid_is_complete_without_context_duplicates(rate,samples):
    plan=plan_chunks('freedom-flower',{'sample_rate':rate,'frame_count':samples},SPEC)
    validate_chunks(plan['chunks'],plan['total_frames'])
    frames=[]
    for row in plan['chunks']:
        assert 0<row['end_sec']-row['start_sec']<=20.000001
        origin=round(row['start_sec']*50)
        frames.extend(range(origin+row['keep_start_frame'],origin+row['keep_end_frame']))
    assert frames==list(range(plan['total_frames']))
    assert plan['chunks'][0]['source_start_sample']==0
    assert plan['chunks'][-1]['source_end_sample_exclusive']==samples
    assert all(a['core_end_sec']==b['core_start_sec'] for a,b in zip(plan['chunks'],plan['chunks'][1:]))


def test_corrupt_frame_ownership_is_rejected():
    plan=plan_chunks('promise',{'sample_rate':16000,'frame_count':640000},SPEC)
    plan['chunks'][1]['global_start_frame']-=1
    with pytest.raises(ValueError,match='gap, overlap'):
        validate_chunks(plan['chunks'],plan['total_frames'])


def source_fixture():
    song={'song_id':'freedom-flower','split':'train',
        'original_audio':{'path':'original.wav','sha256':'a'*64},
        'human_vsqx':{'path':'calibrated.vsqx','sha256':'b'*64},
        'notes':[{'note_index':i,'note_id':f'freedom-flower-n{i:03d}',
            'mapping_status':'resolved_proxy','unresolved_tokens':[],
            'model_target_ipa':['i'],'model_target_ids':[1]} for i in (1,2)]}
    approval={'song_id':'freedom-flower','status':'sequence_confirmed','review_id':'S01',
        'original_answer':{'explicit_sequence_scope_confirmed':True},'phone_timing_confirmed':False,
        'mapping_extended':False,'source_bindings':{k:song[k] for k in ('original_audio','human_vsqx')}}
    return song,approval


def test_sequence_approval_preserves_repeats_without_phone_timing():
    song,approval=source_fixture();ref=accepted_reference(song,approval,{'i':1})
    assert ref['target_ids']==[1,1] and ref['minimum_ctc_frames']==3
    assert not ref['phone_timing_confirmed'] and not ref['per_chunk_targets_created']


@pytest.mark.parametrize('defect',['pending','timing','hash','unknown','blank','gap','test'])
def test_approval_and_mapping_defects_fail_closed(defect):
    song,approval=deepcopy(source_fixture())
    if defect=='pending':approval['status']='pending'
    if defect=='timing':approval['phone_timing_confirmed']=True
    if defect=='hash':approval['source_bindings']=deepcopy(approval['source_bindings']);approval['source_bindings']['original_audio']['sha256']='c'*64
    if defect=='unknown':song['notes'][0]['unresolved_tokens']=['?']
    if defect=='blank':song['notes'][0]['model_target_ids']=[0]
    if defect=='gap':song['notes'][1]['note_index']=3
    if defect=='test':song['split']='test'
    with pytest.raises(ValueError):accepted_reference(song,approval,{'i':1})


def test_streamed_gradient_matches_full_sequence_graph_and_restores_modes():
    torch=pytest.importorskip('torch')
    from any2xsampa.training.singing_domain_losses import sequence_ctc
    torch.manual_seed(17)
    net=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.5),torch.nn.Linear(5,4)).double().train()
    rows=[{'features':torch.randn(1,6,3,dtype=torch.float64),'expected_frames':6,
        'keep_start_frame':1,'keep_end_frame':5,'global_start_frame':i*4,'global_end_frame':(i+1)*4} for i in range(2)]
    target=[1,1,2];coefficient=.25
    with deterministic_checkpointed_tail(net):
        joined=torch.cat([net(r['features'])[:,1:5] for r in rows],1)
        loss=sequence_ctc(torch,joined,target)
        (coefficient*loss).backward()
    expected=[p.grad.clone() for p in net.parameters()]
    net.zero_grad(set_to_none=True)
    result=backward_whole_sequence(torch,net,rows,target,lambda r:net(r['features']),
        coefficient=coefficient,deadline=time.monotonic()+60,device='cpu')
    assert result['ctc']==pytest.approx(float(loss.detach()),abs=1e-12)
    assert result['maximum_recompute_logit_difference']==0.
    assert all(torch.allclose(p.grad,g,rtol=1e-10,atol=1e-12) for p,g in zip(net.parameters(),expected))
    assert all(m.training for m in net.modules())


def test_streamed_recompute_rejects_changed_logits_and_nonfinite_inputs():
    torch=pytest.importorskip('torch')
    net=torch.nn.Linear(2,3)
    row={'features':torch.ones(1,4,2),'expected_frames':4,'keep_start_frame':0,
         'keep_end_frame':4,'global_start_frame':0,'global_end_frame':4}
    calls=[]
    def drift(r):
        calls.append(1)
        out=net(r['features']);return out if len(calls)==1 else out+1
    with pytest.raises(ValueError,match='recomputation logits changed'):
        backward_whole_sequence(torch,net,[row],[1],drift,coefficient=.25,
            deadline=time.monotonic()+60,device='cpu')
    row['features'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='non-finite whole chunk'):
        backward_whole_sequence(torch,net,[row],[1],lambda r:net(r['features']),
            coefficient=.25,deadline=time.monotonic()+60,device='cpu')


def test_whole_source_schedule_adds_no_optimizer_steps():
    from any2xsampa.training.whole_sequence_training import whole_schedule
    songs=[{'song_id':'queen-road-east'},{'song_id':'freedom-flower'}]
    scheduled=whole_schedule(21,songs)
    assert list(scheduled)==[6,13]
    assert [s['song_id'] for s in scheduled.values()]==['freedom-flower','queen-road-east']
    assert len(scheduled)==2 and max(scheduled)<21
    with pytest.raises(ValueError):whole_schedule(1,songs)


def cache_fixture(tmp_path):
    torch=pytest.importorskip('torch')
    from safetensors.torch import save_file
    from any2xsampa.training.limited_data_artifacts import write_json_new,sha256_file
    run=tmp_path/'run';run.mkdir();audio=run/'source.wav';audio.write_bytes(b'fixture audio source')
    requests=[];entries=[];refs=[];geometry=[]
    for song in ('freedom-flower','queen-road-east'):
        request={'example_id':song+'-000','song_id':song,'split':'train','audio_role':'original',
            'audio_path':str(audio),'audio_sha256':sha256_file(audio),'start_sec':0.,'end_sec':1.}
        path=run/(song+'.safetensors');save_file({'prefix_features':torch.zeros(3,1024)},str(path))
        requests.append(request)
        entries.append({**request,'architecture_sha256':'a'*64,'cache_layer':15,'no_reference_received':True,
            'frames':3,'width':1024,'path':str(path),'sha256':sha256_file(path)})
        refs.append({k:request[k] for k in ('song_id','split','audio_role','audio_path','audio_sha256')})
        geometry.append({'song_id':song,'total_frames':3,'chunks':[{'example_id':request['example_id'],
            'expected_frames':3,'keep_start_frame':0,'keep_end_frame':3,'global_start_frame':0,'global_end_frame':3}]})
    write_json_new(run/'WHOLE_REFERENCES.json',{'songs':refs})
    write_json_new(run/'WHOLE_GEOMETRY.json',{'songs':geometry})
    write_json_new(run/'WHOLE_AUDIO_REQUESTS.json',{'rows':requests})
    write_json_new(run/'POLICY.json',{'songs':['freedom-flower','queen-road-east']})
    write_json_new(run/'review-intake/REVIEW_INTAKE.json',{'fixture':True})
    keys={'WHOLE_REFERENCES.json':'whole_references_sha256','WHOLE_GEOMETRY.json':'whole_geometry_sha256',
          'WHOLE_AUDIO_REQUESTS.json':'whole_requests_sha256','review-intake/REVIEW_INTAKE.json':'review_intake_sha256'}
    freeze={key:sha256_file(run/name) for name,key in keys.items()};write_json_new(run/'FREEZE.json',freeze)
    cache={'architecture':{'sha256':'a'*64},'reference_received':False,'all_parameters_frozen':True,
        'audio_requests_sha256':freeze['whole_requests_sha256'],'geometry_sha256':freeze['whole_geometry_sha256'],'entries':entries}
    write_json_new(run/'WHOLE_CACHE.json',cache)
    return run,torch,cache


def test_whole_cache_excludes_held_song_before_loading_features(tmp_path):
    from any2xsampa.training.whole_sequence_cache import load_whole_rows
    run,torch,_=cache_fixture(tmp_path)
    (run/'freedom-flower.safetensors').unlink()
    result=load_whole_rows(run,torch,'a'*64,'freedom-flower')
    assert [r['song_id'] for r in result]==['queen-road-east']
    with pytest.raises(ValueError,match='unknown held song'):
        load_whole_rows(run,torch,'a'*64,'king-of-karaoke')


@pytest.mark.parametrize('defect',['architecture','role','hash','label_conditioning'])
def test_whole_cache_rejects_stale_or_confused_inputs(tmp_path,defect):
    import json
    from any2xsampa.training.whole_sequence_cache import load_whole_rows
    run,torch,cache=cache_fixture(tmp_path)
    if defect=='architecture':cache['architecture']['sha256']='b'*64
    if defect=='role':cache['entries'][1]['audio_role']='rendered'
    if defect=='hash':cache['entries'][1]['sha256']='c'*64
    if defect=='label_conditioning':cache['reference_received']=True
    (run/'WHOLE_CACHE.json').write_text(json.dumps(cache),encoding='utf-8')
    with pytest.raises(ValueError):load_whole_rows(run,torch,'a'*64,'freedom-flower')


def test_whole_posterior_rejects_lyrics_before_reading_files(tmp_path):
    from any2xsampa.adapters.whole_sequence_inference import WholeSequencePosteriorRecognizer
    recognizer=WholeSequencePosteriorRecognizer(tmp_path/'not_read.json')
    with pytest.raises(ValueError,match='audio only'):
        recognizer.recognize(tmp_path/'not_read.wav',lyrics='forbidden reference')


def test_whole_screening_gate_preserves_denominators_and_protected_groups():
    import importlib.util,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'tools'))
    spec=importlib.util.spec_from_file_location('whole_screen_fixture',root/'tools/run_approved_whole_sequence.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rules={'minimum_relative_PER_gain_against_prior_folds':.05,'protected_group_min_references':20,
        'maximum_group_PER_increase':.03}
    def metric(s,d,i,asp):
        return {'reference_phones':1235,'edit_distance':s+d+i,'substitutions':s,'deletions':d,'insertions':i,
            'per':(s+d+i)/1235,'groups':{'aspirated':{'reference_count':30,'per':asp/30}}}
    prior=metric(301,241,52,14);improved=metric(280,210,50,14)
    assert module.screening_gate(improved,prior,rules)['passed']
    regression=metric(280,210,50,16)
    assert not module.screening_gate(regression,prior,rules)['passed']
    invalid=deepcopy(improved);invalid['per']=float('nan')
    with pytest.raises(ValueError,match='arithmetic'):module.screening_gate(invalid,prior,rules)
    invalid=deepcopy(improved);invalid['groups']['aspirated']['reference_count']=29
    with pytest.raises(ValueError,match='coverage'):module.screening_gate(invalid,prior,rules)


def test_whole_checkpoint_cannot_claim_held_song_training(tmp_path):
    from any2xsampa.adapters.whole_sequence_inference import verify_job
    from any2xsampa.training.limited_data_artifacts import write_json_new,sha256_file
    run=tmp_path/'run';job=run/'job';job.mkdir(parents=True)
    hashes={}
    for name,key in [('POLICY.json','policy_sha256'),('CODE_FREEZE.json','code_freeze_sha256'),
                     ('WHOLE_REFERENCES.json','whole_reference_sha256'),('WHOLE_CACHE.json','whole_cache_sha256')]:
        write_json_new(run/name,{'fixture':name});hashes[key]=sha256_file(run/name)
    checkpoint=job/'fixture.bin';checkpoint.write_bytes(b'not a live model')
    result={**hashes,'complete':True,'fixed_epoch':20,'arm':'whole_sequence','frozen_parameters_unchanged':True,
        'code_bindings':{},'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),
        'parent_checkpoint_path':str(checkpoint),'parent_checkpoint_sha256':sha256_file(checkpoint),
        'held_song':'freedom-flower','training_songs':['queen-road-east'],
        'whole_training_songs':['freedom-flower']}
    write_json_new(job/'RESULT.json',result)
    with pytest.raises(ValueError,match='leaked its held song'):verify_job(run,job)

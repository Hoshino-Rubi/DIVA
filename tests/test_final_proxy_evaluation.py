from copy import deepcopy
from pathlib import Path
import pytest
from tests.test_final_test_mapping_review import fixture as mapping_fixture
from tools.final_test_mapping_review import assemble_proxy_reference
from any2xsampa.training.final_proxy_evaluation import (validate_reference,validate_requests,verify_release,score_predictions,
    SYSTEMS,SCOPE,AUDIO_KEYS,REQUEST_SCHEMA,GATE_SCHEMA)
from any2xsampa.training.limited_data_artifacts import sha256_object,write_json_new,sha256_file,read_json
from any2xsampa.training.limited_data_reference import SPLITS
from any2xsampa.adapters.phoneticxeus import MODEL_ID,REVISION,WEIGHT_SHA256


def fixture():
    review,answers,source,original,base,vocab=mapping_fixture()
    ref=assemble_proxy_reference(review,answers,source,original,base,vocab)
    entries=[]
    for row in ref['examples']:
        entries.append({**{k:row[k] for k in AUDIO_KEYS-{'audio_path'}},
            'evidence_path':'evidence.safetensors','evidence_sha256':'a'*64,'posterior_path':'posterior.json','posterior_sha256':'b'*64,
            'frames':899,'feature_width':1024,'vocabulary_sha256':'c'*64,'model_snapshot_lock_sha256':'d'*64,
            'test_outputs_sealed':True,'created_at_utc':'2026-09-05T21:45:42+00:00'})
    index=dict(purpose='original-baselines',split_sha256=sha256_object(SPLITS),entries=entries,
        model_metadata=dict(model_id=MODEL_ID,revision=REVISION,weights_sha256=WEIGHT_SHA256,vocabulary_sha256='c'*64,snapshot_lock_sha256='d'*64))
    req=dict(schema_version=REQUEST_SCHEMA,split_sha256=sha256_object(SPLITS),contains_labels_or_score_context=False,
        rows=[dict(audio={k:r[k] for k in AUDIO_KEYS},cache=e) for r,e in zip(ref['examples'],entries)])
    return ref,index,vocab,req


def rehash(value,key='manifest_hash'):
    value[key]=sha256_object({k:v for k,v in value.items() if k!=key})


def test_six_complete_proxy_windows_keep_absent_and_repeated_references():
    ref,index,vocab,req=fixture();assert len(validate_reference(ref,index,vocab))==6
    validate_requests(req,index)
    assert len(ref['examples'][0]['omitted_phones'])==1


@pytest.mark.parametrize('fault',['reference_hash','leak','role','crop','vocabulary','frame_count','special','deleted_phone','duplicate','scope'])
def test_reference_gate_rejects_invalid_lineage(fault):
    ref,index,vocab,_=fixture();r=ref['examples'][0]
    if fault=='reference_hash':ref['manifest_hash']='bad'
    if fault=='leak':r['song_id']='shalong'
    if fault=='role':r['audio_role']='rendered'
    if fault=='crop':r['start_sec']+=.5
    if fault=='vocabulary':index['entries'][0]['vocabulary_sha256']='e'*64
    if fault=='frame_count':index['entries'][0]['frames']=1
    if fault=='special':r['phones'][0]['phone']='<blank>'
    if fault=='deleted_phone':r['phones']=r['phones'][1:];r['reference_phone_count']-=1;ref['coverage']['reference_phones']-=1
    if fault=='duplicate':ref['examples'][1]['example_id']=r['example_id']
    if fault=='scope':ref['formal_original_ipa_accuracy_allowed']=True
    if fault!='reference_hash':rehash(ref)
    with pytest.raises(ValueError):validate_reference(ref,index,vocab)


@pytest.mark.parametrize('location',['top','row','audio','cache'])
def test_inference_requests_forbid_reference_context(location):
    _,index,_,req=fixture()
    target=req if location=='top' else req['rows'][0] if location=='row' else req['rows'][0][location]
    target['lyrics']='forbidden'
    with pytest.raises(ValueError):validate_requests(req,index)


def test_pending_window_is_wholly_excluded():
    ref,index,vocab,_=fixture();r=ref['examples'][0]
    ref['coverage']['reference_phones']-=len(r['phones']);ref['coverage']['proxy_ready_windows']-=1
    r.update(status='pending',proxy_evaluation_eligible=False,phones=[]);rehash(ref)
    assert len(validate_reference(ref,index,vocab))==5


def test_score_preserves_empty_outputs_and_identical_system_denominators():
    ref,_,_,_=fixture();pred={s:{r['example_id']:[] for r in ref['examples']} for s in SYSTEMS}
    result=score_predictions(ref,pred)
    for system in result['systems'].values():
        assert system['overall']['per']==1 and system['overall']['deletions']==72
        assert system['overall']['insertions']==0
    assert result['formal_original_ipa_accuracy_claim_allowed'] is False
    pred['old_seed43'].pop(next(iter(pred['old_seed43'])))
    with pytest.raises(ValueError):score_predictions(ref,pred)


def gate_fixture(tmp_path):
    ref,index,vocab,req=fixture()
    def put(name,value):
        p=tmp_path/(name+'.json');write_json_new(p,value);return {'path':str(p),'sha256':sha256_file(p)}
    generic=put('generic',{'fixture':True})
    teacher=put('teacher',{'weights':'teacher'});student=put('student',{'weights':'student'});guard=put('guard',{'radius':2})
    candidate=put('candidate',dict(teacher_checkpoint=teacher,student_checkpoint=student,policy=guard))
    decision=put('decisions',dict(consent=SCOPE,authorize_frozen_proxy_comparison=True,reviewer='human'))
    index['head']=generic
    bindings={k:generic for k in ['reference','mapping_review','intake_inputs','original_packet','original_answers','policy','legacy_policy','worker_source']}
    bindings.update(mapping_decisions=decision,candidate=candidate,source_index=put('index',index),requests=put('requests',req),
        vocabulary=put('vocabulary',vocab),pretrained_head=generic,teacher_checkpoint=teacher,student_checkpoint=student,guard_policy=guard)
    gate=dict(schema_version=GATE_SCHEMA,bindings=bindings,systems=list(SYSTEMS),scope=SCOPE,execution_authorized=True,
        training_allowed=False,model_reselection_allowed=False,default_replacement=False,formal_original_ipa_accuracy_allowed=False,
        reference_frozen_before_predictions=True,test_predictions_opened_at_freeze=False,
        included_example_ids=[r['example_id'] for r in ref['examples']])
    rehash(gate,'gate_hash');path=tmp_path/'RELEASE.json';write_json_new(path,gate)
    return path,gate


def test_release_verifies_hashes_without_loading_torch_or_predictions(tmp_path):
    path,gate=gate_fixture(tmp_path);got,req,index=verify_release(path,sha256_file(path))
    assert got==gate and len(req['rows'])==6
    Path(gate['bindings']['reference']['path']).write_text('{}')
    with pytest.raises(ValueError,match='SHA-256'):verify_release(path,sha256_file(path))


def test_automated_fixture_cannot_release_test(tmp_path):
    path,gate=gate_fixture(tmp_path);spec=gate['bindings']['mapping_decisions'];d=read_json(Path(spec['path']));d['automated_regression_fixture']=True
    Path(spec['path']).write_text(__import__('json').dumps(d));spec['sha256']=sha256_file(Path(spec['path']))
    rehash(gate,'gate_hash');path.write_text(__import__('json').dumps(gate))
    with pytest.raises(ValueError,match='automated fixture'):verify_release(path,sha256_file(path))


def test_changed_system_list_cannot_be_added_after_freeze(tmp_path):
    path,gate=gate_fixture(tmp_path);gate['systems'].append('new_checkpoint')
    rehash(gate,'gate_hash');path.write_text(__import__('json').dumps(gate))
    with pytest.raises(ValueError,match='wrong evaluation scope'):verify_release(path,sha256_file(path))

def test_report_cannot_present_empty_reference_as_formal_accuracy(tmp_path):
    from tools.score_final_proxy_evaluation import decorate,write_reports
    ref,_,_,_=fixture();pred={s:{r['example_id']:[] for r in ref['examples']} for s in SYSTEMS}
    result=score_predictions(ref,pred)
    coverage=dict(reference_phone_count=72,included_windows=6,included_audio_seconds=108,absent_phones=6,partial_phones=0)
    decorate(result,{'coverage':coverage});write_reports(tmp_path,result)
    assert result['candidate_comparisons']['old_seed43']['relative_per_reduction']==0
    assert result['final_test_is_now_opened'] is True
    assert result['selected_after_test'] is False
    report=(tmp_path/'RESULTS.md').read_text(encoding='utf-8')
    assert '代理' in report and '| T01 |' in report and '精确原唱' in report
    html=(tmp_path/'index.html').read_text(encoding='utf-8')
    assert '1−PER' in html and '<script' not in html


def test_gpu_preflight_refuses_before_torch_or_test_evidence(tmp_path,monkeypatch):
    import sys
    from tools.run_final_proxy_evaluation import run
    import any2xsampa.adapters.phoneticxeus_runtime as runtime
    path,_=gate_fixture(tmp_path);output=tmp_path/'blocked'
    monkeypatch.setattr(runtime,'enforce_offline',lambda p:p.mkdir(parents=True))
    monkeypatch.setattr(runtime,'gpu_total_used_bytes',lambda:15*1024**3)
    def fail_cuda():raise AssertionError('CUDA must not load')
    monkeypatch.setattr(runtime,'configure_cuda',fail_cuda)
    with pytest.raises(RuntimeError,match='14 GiB'):run(path,sha256_file(path),output)
    result=read_json(output/'BLOCKED.json')
    assert result['no_model_loaded'] and result['no_test_evidence_read']
    assert not (tmp_path/'TEST_OPENING.json').exists()

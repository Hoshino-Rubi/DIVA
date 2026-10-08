"""Fixtures for the authorized zero-training input/native probe."""
from pathlib import Path
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from input_native_probe_common import (prediction, compare_logits, check_midpoint, first_per_song,
    native_observer, native_outputs, check_stage, verify_review, SONGS, LIMITS)
from any2xsampa.training.limited_data_artifacts import write_json_new, sha256_file

VOCAB = {str(i): i for i in range(428)}


def logits(ids):
    x = np.full((len(ids), 428), -10., dtype='float32')
    for i, label in enumerate(ids):
        x[i, label] = 10.
    return x


def test_ctc_blank_repeat_and_empty_sequence():
    assert prediction(logits([0, 1, 1, 0, 1, 2, 2]), VOCAB) == ['1', '1', '2']
    assert prediction(logits([0, 0]), VOCAB) == []


@pytest.mark.parametrize('x', [np.zeros((0,428)), np.zeros((2,427)), np.full((2,428),np.nan), np.full((2,428),np.inf)])
def test_invalid_evidence(x):
    with pytest.raises(ValueError): prediction(x, VOCAB)


def test_equivalence_tolerance_and_sequence_guard():
    a = logits([0, 1, 2]); b = a.copy()
    assert compare_logits(a, b, VOCAB)['bitwise_equal']
    b[0, 7] += 1e-5
    assert compare_logits(a, b, VOCAB)['sequence_equal']
    b[0, 7] += 1e-3
    with pytest.raises(ValueError): compare_logits(a, b, VOCAB)
    a = np.zeros((2,428),dtype='float32'); b=a.copy(); b[0,1]=1e-6
    with pytest.raises(ValueError): compare_logits(a,b,VOCAB)


@pytest.mark.parametrize('kind', ['shape','nonfinite','clipping','formula'])
def test_midpoint_rejects_wrong_sources(kind):
    x=np.full((10,2),.2,dtype='float32'); v=np.full_like(x,.4); h=(x+v)*.5
    assert check_midpoint(x,v,h)['frames']==10
    if kind=='shape': h=h[:9]
    if kind=='nonfinite': h[0,0]=np.nan
    if kind=='clipping': x[0,0]=1
    if kind=='formula': h[0,0]+=.01
    with pytest.raises(ValueError): check_midpoint(x,v,h)


def test_fixed_selection_ignores_scores():
    rows=[{'window_id':s+suffix,'song_id':s,'source_start_sec':time,'score':-time}
          for s in SONGS for suffix,time in [('b',3),('a',3),('c',1)]]
    assert first_per_song(rows)==[s+'c' for s in SONGS]


def request(song='freedom-flower', split='train'):
    return {'example_id':'example','song_id':song,'split':split,'audio_role':'original',
            'audio_path':'unused.wav','audio_sha256':'0'*64,'start_sec':0.,'end_sec':10.}


@pytest.mark.parametrize('stage,count', [('A0',11),('A1',21),('B0',5),('B1',61),('AR',6),('BR',6)])
def test_stage_budget(stage,count,tmp_path):
    with pytest.raises(ValueError): check_stage(tmp_path,stage,[{'request':request()}]*count)


def test_splits_and_labels_and_retry(tmp_path):
    for req in [request('final-test'), request('shalong','dev'), {**request(),'target_ipa':['i']}]:
        with pytest.raises(ValueError): check_stage(tmp_path,'A0',[{'request':req}])
    check_stage(tmp_path,'A0',[{'request':request()}])
    write_json_new(tmp_path/'A0/START.json',{})
    with pytest.raises(ValueError): check_stage(tmp_path,'A0',[{'request':request()}])


def test_conditional_stage_gate(tmp_path):
    with pytest.raises(FileNotFoundError): check_stage(tmp_path,'A1',[{'request':request()}])
    write_json_new(tmp_path/'A0/RESULT.json',{'passed':False})
    with pytest.raises(ValueError): check_stage(tmp_path,'A1',[{'request':request()}])


def test_review_invalidates_on_code_change(tmp_path):
    code=tmp_path/'code.py';code.write_text('a=1')
    evidence=tmp_path/'fixture.json';write_json_new(evidence,{'passed':True})
    freeze=tmp_path/'FREEZE.json';write_json_new(freeze,{'bindings':{}})
    write_json_new(tmp_path/'CODE_REVIEW.json',{'approved_for_launch':True,'freeze_sha256':sha256_file(freeze),
        'code_bindings':{str(code):sha256_file(code)},'verification':{'path':str(evidence),'sha256':sha256_file(evidence)}})
    verify_review(tmp_path)
    code.write_text('a=2')
    with pytest.raises(ValueError): verify_review(tmp_path)


def test_111_total_limit(tmp_path):
    for i in range(111): write_json_new(tmp_path/f'old/forwards/f{i}/START.json',{})
    with pytest.raises(ValueError): check_stage(tmp_path,'A0',[{'request':request()}])


def test_native_observer_passive_and_cleanup():
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    head=torch.nn.Linear(2,428).eval().requires_grad_(False)
    conditioning=torch.nn.Linear(428,2).eval().requires_grad_(False)
    model=SimpleNamespace(encoder=SimpleNamespace(interctc_layer_idx=[4,8,12],
        interctc_use_conditioning=True,conditioning_layer=conditioning),ctc=SimpleNamespace(ctc_lo=head))
    def forward():
        x=torch.ones((1,3,2))
        for _ in range(3): x=x+conditioning(head(x).softmax(-1))
        return head(x[0]).detach().numpy()
    with torch.inference_mode():
        with native_observer(model,capture=False) as ordinary: a=forward()
        with native_observer(model,capture=True) as observed: b=forward()
    assert ordinary['condition_calls']==3 and observed['condition_calls']==3
    assert compare_logits(a,b,VOCAB)['bitwise_equal']
    assert set(native_outputs(observed,b,VOCAB))=={'4','8','12','19'}
    assert len(head._forward_hooks)==0 and len(conditioning._forward_hooks)==0
    with pytest.raises(RuntimeError):
        with native_observer(model,capture=True): raise RuntimeError('fault')
    assert len(head._forward_hooks)==0
    observed['condition_calls']=2
    with pytest.raises(ValueError): native_outputs(observed,b,VOCAB)
    model.encoder.interctc_layer_idx=[]
    with pytest.raises(ValueError):
        with native_observer(model,capture=True): pass


def test_budget_sum_and_no_return_all_hs():
    assert sum(LIMITS.values())==111
    import inspect
    assert 'return_all_hs' not in inspect.getsource(native_observer)

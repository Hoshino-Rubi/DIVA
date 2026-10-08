from __future__ import annotations
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from any2xsampa.training.acoustic_memory import (validate_query, validate_sources, normalized_keys,
    soft_ctc_values, softmax, probabilities, exact_neighbors, memory_temperature, shuffled_values,
    memory_distribution, fuse, SONGS)
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.training.limited_data_artifacts import verify_file,sha256_file,write_json_new


def qrow():
    return {'example_id':'test','song_id':SONGS[0],'split':'train','audio_role':'original',
        'audio_path':'test.wav','audio_sha256':'a'*64,'start_sec':0.,'end_sec':10.,
        'architecture_sha256':'b'*64,'frames':499,'path':'feature.safetensors','sha256':'c'*64,
        'audio_metadata':{'sha256':'a'*64,'sample_rate':16000,'channels':1,'sample_width_bytes':2,'frame_count':160000,'duration_sec':10.},
        'crop':{'requested_start_sec':0.,'requested_end_sec':10.,'source_start_sample':0,
            'source_end_sample_exclusive':160000,'effective_start_sec':0.,'effective_end_sec':10.,
            'model_sample_rate':16000,'model_samples':160000,'channel_policy':'arithmetic_mean',
            'resample_policy':'scipy_resample_poly_v1','waveform_normalization':'none_before_model_frontend'},
        'geometry':{'frame_count':499,'stride_samples':320,'receptive_field_samples':400}}


@pytest.mark.parametrize('target',[[1],[1,1],[1,2],[1,2,1]])
def test_soft_ctc_matches_exhaustive_paths(target):
    z=np.arange(12,dtype=float).reshape(4,3)/7;p=softmax(z);g,_=soft_ctc_values(z,target)
    expected=np.zeros_like(g);mass=0.
    for path in product(range(3),repeat=4):
        if ctc_collapse(path)!=target:continue
        prob=float(np.prod([p[t,i] for t,i in enumerate(path)]));mass+=prob
        for t,i in enumerate(path):expected[t,i]+=prob
    assert np.allclose(g,expected/mass,atol=1e-12,rtol=0)
    assert np.allclose(g.sum(1),1,atol=1e-12)


def test_ctc_rejects_impossible_empty_blank_and_nonfinite():
    for target in ([],[0],[1,1,1],[True]):
        with pytest.raises(ValueError):soft_ctc_values(np.ones((2,428)),target)
    with pytest.raises(ValueError):soft_ctc_values(np.full((4,428),np.nan),[1])
    assert ctc_collapse([0,1,1,0,1,0])==[1,1]
    assert ctc_collapse([0,0])==[]


def test_all_428_and_unseen_class_probability_retained():
    p=softmax(np.zeros((4,428)));g,_=soft_ctc_values(np.ones((4,428)),[1,1])
    assert g.shape==(4,428) and (g[:,427]==0).all()
    result=fuse(p,g);assert np.allclose(result[:,427],.75*p[:,427])
    assert np.array_equal(fuse(p,g,0),p)


@pytest.mark.parametrize('field',['target_ipa','target_ids','lyrics','note_id','pitch','target_length','reference_boundaries'])
def test_query_rejects_reference_conditions(field):
    q=qrow();validate_query(q);q[field]=[]
    with pytest.raises(ValueError):validate_query(q)


@pytest.mark.parametrize('field',['audio_metadata','crop','geometry'])
def test_query_nested_conditions_rejected(field):
    q=qrow();q[field]['target_ids']=[1]
    with pytest.raises(ValueError):validate_query(q)


@pytest.mark.parametrize('failure',['hash','offset','dev','role','song','empty'])
def test_query_identity_guards(failure):
    q=qrow()
    if failure=='hash':q['audio_metadata']['sha256']='x'*64
    if failure=='offset':q['crop']['source_start_sample']=1
    if failure=='dev':q['split']='dev'
    if failure=='role':q['audio_role']='vocal'
    if failure=='song':q['song_id']='shalong'
    if failure=='empty':q['frames']=0
    with pytest.raises(ValueError):validate_query(q)


def test_source_song_and_rendered_exclusion():
    q=qrow();s={'query':q,'example_id':q['example_id'],'song_id':q['song_id'],
        'target_ids':[1],'reference_sha256':'d'*64,'reference_nature':'human_sequence'}
    validate_sources([s],SONGS[1])
    with pytest.raises(ValueError):validate_sources([s],SONGS[0])
    q['audio_role']='rendered'
    with pytest.raises(ValueError):validate_sources([s],SONGS[1])


def test_exact_neighbor_brute_force_and_ties():
    from acoustic_memory_cpu import brute_neighbors
    rng=np.random.default_rng(2);k=normalized_keys(rng.normal(size=(25,7)));q=normalized_keys(rng.normal(size=(9,7)))
    i,d=exact_neighbors(q,k,k=16,block=3,source_block=6);bi,bd=brute_neighbors(q,k,16)
    assert np.array_equal(i,bi);assert np.allclose(d,bd,atol=1e-12,rtol=0)
    keys=np.array([[1.,0.],[1.,0.],[0.,1.]])
    i,d=exact_neighbors(keys[:1],keys,k=16,source_block=1)
    assert i.tolist()==[[0,1,2]] and d.tolist()==[[0.,0.,2.]]


def test_temperature_only_cross_training_songs():
    k=np.eye(3);tau,d=memory_temperature(k,np.array(['a','b','b']))
    assert tau==2 and d.tolist()==[2,2,2]
    with pytest.raises(ValueError):memory_temperature(k,np.array(['a']*3))
    tau,_=memory_temperature(np.array([[1.,0.],[1.,0.]]),np.array(['a','b']))
    assert tau==1e-6


def test_shuffle_preserves_window_marginals_and_is_deterministic():
    values=softmax(np.arange(60).reshape(10,6)**2/70);spans=[(0,4),(4,10)]
    a,p=shuffled_values(values,spans);b,q=shuffled_values(values,spans)
    assert np.array_equal(a,b) and np.array_equal(p,q)
    for lo,hi in spans:assert np.allclose(a[lo:hi].sum(0),values[lo:hi].sum(0))
    with pytest.raises(ValueError):shuffled_values(values,[(1,10)])


def test_blank_queries_are_processed():
    base=softmax(np.array([[2.,0.,0.],[2.,0.,0.]]));values=np.array([[0.,1.,0.]])
    i,d=exact_neighbors(np.array([[1.,0.],[1.,0.]]),np.array([[1.,0.]]))
    p=fuse(base,memory_distribution(values,i,d,1.))
    assert len(i)==2 and (base.argmax(1)==0).all() and (p[:,1]>base[:,1]).all()


@pytest.mark.parametrize('array',[np.zeros((1,3)),np.ones((0,3)),np.array([[np.nan,1.]]),np.array([[np.inf,1.]])])
def test_invalid_features_fail_closed(array):
    with pytest.raises(ValueError):normalized_keys(array)


def test_invalid_numeric_and_empty_memory_fail():
    with pytest.raises(ValueError):probabilities([[.1,.2]])
    with pytest.raises(ValueError):exact_neighbors(np.ones((1,2)),np.ones((0,2)))
    with pytest.raises(ValueError):memory_distribution(np.array([[1.,0.]]),np.array([[0]]),np.array([[np.nan]]),1.)
    with pytest.raises(ValueError):fuse(np.array([[1.,0.]]),np.array([[1.,0.]]),float('nan'))


def test_hash_change_and_no_overwrite(tmp_path):
    p=tmp_path/'artifact.json';write_json_new(p,{'a':1});digest=sha256_file(p)
    with pytest.raises(FileExistsError):write_json_new(p,{'a':2})
    p.write_text('bad',encoding='utf-8')
    with pytest.raises(ValueError):verify_file(p,digest)


def test_stage_reference_separation_static():
    import ast
    source=(ROOT/'tools/acoustic_memory_cpu.py').read_text(encoding='utf-8');tree=ast.parse(source)
    query=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='query')
    text=ast.get_source_segment(source,query)
    assert "read_json(run/'SCORING.json')" not in text and "read_json(run/'SOURCES.json')" not in text
    head=(ROOT/'tools/run_acoustic_memory_head.py').read_text(encoding='utf-8')
    assert 'SCORING.json' not in head and 'SOURCES.json' not in head and '.backward(' not in head and 'torch.optim' not in head
    assert "cache['encoder_features']" in head and "cache['prefix_features']" not in head


def test_deadline_and_policy_limits():
    from acoustic_memory_common import POLICY,deadline
    assert POLICY['head_requests_per_worker']*POLICY['cache_workers']==458
    assert POLICY['new_training']==POLICY['encoder_forward']==POLICY['optimizer_steps']==0
    with pytest.raises(TimeoutError):deadline(-1)()


@pytest.mark.parametrize('ids',[[0,1,1,0,1,2,0],[0,0,0,0,0,0,0]])
def test_real_fixture_public_posterior_contract(tmp_path,ids):
    import wave
    from acoustic_memory_common import save_arrays
    from any2xsampa.adapters.audio_provenance import describe_audio_file
    from any2xsampa.adapters.phoneticxeus import emission_payload,PhoneticXeusPosteriorRecognizer
    from verify_acoustic_memory import codec_identity
    audio=tmp_path/'fixture.wav'
    with wave.open(str(audio),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(bytes(16000))
    vocab={'<blank>':0,'a':1,'b':2,**{'ipa'+str(i):i for i in range(3,428)}}
    vp=tmp_path/'vocab.json';write_json_new(vp,vocab)
    z=np.full((7,428),-8,dtype=np.float32);z[np.arange(7),ids]=8
    raw=tmp_path/'posterior.safetensors';save_arrays(raw,{'frame_logits':z})
    payload=emission_payload(z,vocab,input_audio=describe_audio_file(audio),
        crop={'effective_start_sec':0.,'effective_end_sec':.5},
        geometry={'frame_count':7,'stride_samples':320,'receptive_field_samples':400},
        model_metadata={'vocabulary_sha256':sha256_file(vp)},runtime={'fixture':True})
    payload['raw_evidence']={'file':raw.name,'sha256':sha256_file(raw)}
    pp=tmp_path/'posterior.json';write_json_new(pp,payload)
    real=PhoneticXeusPosteriorRecognizer(pp,vocabulary_path=vp).recognize(audio)
    fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True,vocabulary_path=vp).recognize(audio)
    assert codec_identity(real)==codec_identity(fixture)
    assert [s.metadata['raw_label_trace']['label'] for s in real]==[next(k for k,v in vocab.items() if v==i) for i in ctc_collapse(ids)]
    audio.write_bytes(b'changed')
    with pytest.raises((ValueError,EOFError,wave.Error)):PhoneticXeusPosteriorRecognizer(pp,vocabulary_path=vp).recognize(audio)


def test_independent_edit_tie_accounting():
    from verify_acoustic_memory import scalar_edits
    from any2xsampa.training.phone_error_metrics import phone_metrics
    sequences=[list(v) for n in range(3) for v in product(('a','b'),repeat=n)]
    for ref in sequences:
        for hyp in sequences:
            result=phone_metrics([(ref,hyp)])
            assert list(scalar_edits(ref,hyp))==[result[k] for k in ('substitutions','deletions','insertions')]

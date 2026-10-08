from __future__ import annotations
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import wave
import numpy as np
import pytest
from any2xsampa.adapters.segmental_event import EventScores,decode_events,phone_segments,SegmentalEventRecognizer
from any2xsampa.training.segmental_event import log_partitions,sequence_nll,make_event_head
from any2xsampa.training.limited_data_artifacts import sha256_file,write_json_new
from any2xsampa.adapters.audio_provenance import describe_audio_file


def lp(x):
    x=np.asarray(x,dtype='float64');return x-np.logaddexp.reduce(x,axis=-1,keepdims=True)


def enumerate_paths(q,b):
    scores=EventScores(q,b);paths=[]
    def visit(t,labels,value,arcs):
        if t==len(q):paths.append((labels,value,arcs));return
        visit(t+1,labels,value+q[t,0],arcs+[(t,t+1,0)])
        for end in range(t+1,len(q)+1):
            row=scores.segment(t,end)
            for k in range(1,q.shape[1]):visit(end,labels+[k],value+row[k-1],arcs+[(t,end,k)])
    visit(0,[],0.,[]);return paths


def test_exact_partition_target_and_viterbi_against_exhaustive_paths():
    torch=pytest.importorskip('torch');rng=np.random.default_rng(17)
    for t in (1,2,3,4):
        q=lp(rng.normal(size=(t,3)));b=rng.normal(size=t);paths=enumerate_paths(q,b)
        z=np.logaddexp.reduce([r[1] for r in paths]);decoded=decode_events(q,b)
        assert abs(decoded['log_partition']-z)<1e-10
        assert abs(decoded['best_path_score']-max(r[1] for r in paths))<1e-10
        for n in range(t+1):
            for target in itertools.product((1,2),repeat=n):
                a,c=log_partitions(torch,torch.tensor(q),torch.tensor(b),target)
                expected=np.logaddexp.reduce([r[1] for r in paths if r[0]==list(target)])
                assert abs(a.item()-z)<1e-10 and abs(c.item()-expected)<1e-10


def test_gradients_match_independent_path_enumeration():
    torch=pytest.importorskip('torch');torch.manual_seed(29)
    x=torch.randn(3,3,dtype=torch.float64,requires_grad=True);b=torch.randn(3,dtype=torch.float64,requires_grad=True)
    q=x.log_softmax(-1);target=[1,1];paths=enumerate_paths(q.detach().numpy(),b.detach().numpy())
    def value(arcs):
        values=[]
        for s,e,k in arcs:
            values.append(q[s,0] if not k else q[s:e,k].mean()+torch.nn.functional.logsigmoid(b[s])+torch.nn.functional.logsigmoid(-b[s+1:e]).sum())
        return torch.stack(values).sum()
    allvalues=torch.stack([value(arcs) for _,_,arcs in paths])
    targetvalues=torch.stack([value(arcs) for labels,_,arcs in paths if labels==target])
    brute=torch.logsumexp(allvalues,0)-torch.logsumexp(targetvalues,0)
    dynamic=sequence_nll(torch,q,b,target)
    a=torch.autograd.grad(brute,(x,b),retain_graph=True);c=torch.autograd.grad(dynamic,(x,b))
    for left,right in zip(a,c):torch.testing.assert_close(left,right,atol=1e-10,rtol=1e-10)


def test_finite_difference_gradient_and_empty_target():
    torch=pytest.importorskip('torch');torch.manual_seed(43)
    x=torch.randn(3,4,dtype=torch.float64,requires_grad=True);b=torch.randn(3,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,c:sequence_nll(torch,a.log_softmax(-1),c,[1,1]),(x,b),eps=1e-6,atol=1e-5)
    loss=sequence_nll(torch,x.log_softmax(-1),b,[]);loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(b.grad).all()


def event_fixture(ids,boundaries,k=4):
    logits=np.full((len(ids),k),-9.)
    for i,label in enumerate(ids):logits[i,label]=9.
    b=np.full(len(ids),-8.);b[list(boundaries)]=8.
    return lp(logits),b


def test_long_vowel_has_no_300ms_cap():
    q,b=event_fixture([1]*80,[0]);d=decode_events(q,b)
    assert d['event_ids']==[1]
    assert d['events'][0]['end_frame_exclusive']==80


def test_repeated_same_phone_preserves_new_event():
    q,b=event_fixture([1]*8,[0,4]);d=decode_events(q,b)
    assert d['event_ids']==[1,1]
    assert [(x['start_frame'],x['end_frame_exclusive']) for x in d['events']]==[(0,4),(4,8)]


def test_internal_vowel_transition_is_two_events():
    q,b=event_fixture([1]*4+[2]*4,[0,4]);assert decode_events(q,b)['event_ids']==[1,2]


def test_short_aspirated_event_and_final_inventory_label():
    q,b=event_fixture([427]+[2]*8,[0,1],428);assert decode_events(q,b)['event_ids']==[427,2]


def test_pure_null_noise_is_empty_sequence():
    q,b=event_fixture([0]*12,[0,4,7]);assert decode_events(q,b)['event_ids']==[]


def test_arc_marginals_against_exhaustive_path_mass():
    rng=np.random.default_rng(7);q=lp(rng.normal(size=(4,3)));b=rng.normal(size=4)
    paths=enumerate_paths(q,b);d=decode_events(q,b)
    for event in d['events']:
        key=(event['start_frame'],event['end_frame_exclusive'],event['label_id'])
        actual=sum(np.exp(v-d['log_partition']) for _,v,arcs in paths if key in arcs)
        assert abs(actual-event['arc_posterior'])<1e-12


@pytest.mark.parametrize('which',['nan','unscaled','shape','target_blank','target_long'])
def test_invalid_evidence_or_targets(which):
    torch=pytest.importorskip('torch');q=torch.zeros(3,4,dtype=torch.float64).log_softmax(-1);b=torch.zeros(3,dtype=torch.float64);target=[1]
    if which=='nan':q[0,0]=float('nan')
    if which=='unscaled':q+=1
    if which=='shape':b=b[:2]
    if which=='target_blank':target=[0]
    if which=='target_long':target=[1]*4
    with pytest.raises(ValueError):sequence_nll(torch,q,b,target)


def test_head_does_not_alias_or_update_pretrained_head():
    torch=pytest.importorskip('torch');torch.manual_seed(17)
    weight=torch.randn(428,1024);bias=torch.randn(428);a=weight.clone();c=bias.clone()
    head=make_event_head(torch,weight,bias);features=torch.randn(4,1024)
    q,b=head(features);loss=sequence_nll(torch,q,b,[1,2]);loss.backward()
    torch.testing.assert_close(weight,a);torch.testing.assert_close(bias,c)
    assert head.content.weight.data_ptr()!=weight.data_ptr()
    assert head.content.weight.grad is not None and head.event.weight.grad is not None
    assert q.shape==(4,428) and torch.count_nonzero(b)==0


def test_versioned_evidence_replay_and_hash_binding(tmp_path):
    pytest.importorskip('safetensors');from safetensors.numpy import save_file
    audio=tmp_path/'audio.wav'
    with wave.open(str(audio),'wb') as out:out.setparams((1,2,16000,3200,'NONE',''));out.writeframes(b'\0\0'*3200)
    q,b=event_fixture([1]*4+[1]*4,[0,4],428)
    raw=tmp_path/'scores.safetensors';save_file({'frame_log_probs':q.astype('float32'),'boundary_logits':b.astype('float32')},str(raw))
    vocab={'<blank>':0,**{f'p{k}':k for k in range(1,428)}};vp=tmp_path/'vocab.json';write_json_new(vp,vocab)
    d=decode_events(q,b);pp=tmp_path/'events.json';write_json_new(pp,{'schema_version':'segmental_event_evidence_v1','reference_conditioning':False,'scoring_spec':'mean_logp_start_hold_v1',
        'input_audio':describe_audio_file(audio),'scores':{'file':raw.name,'sha256':sha256_file(raw)},
        'vocabulary':{'file':vp.name,'sha256':sha256_file(vp)},'decoded_event_ids':d['event_ids'],
        'crop':{'effective_start_sec':0,'effective_end_sec':.2},'geometry':{'frame_count':8,'stride_samples':320,'receptive_field_samples':400}})
    left=SegmentalEventRecognizer(pp).recognize(audio);right=SegmentalEventRecognizer(pp,fixture=True).recognize(audio)
    assert [asdict(s) for s in left]==[asdict(s) for s in right]
    assert [s.normalized_ipa for s in left]==['p1','p1']
    assert all(s.metadata['timing_kind']=='latent_segment_support_not_ctc_or_ground_truth' for s in left)
    with pytest.raises(ValueError):SegmentalEventRecognizer(pp).recognize(audio,lyrics='not allowed')
    with raw.open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError):SegmentalEventRecognizer(pp).recognize(audio)

from copy import deepcopy
import pytest
from any2xsampa.training.singing_domain import policy,validate_audio_row,validate_cache_entry,partition,choice,architecture_identity
from any2xsampa.training.singing_domain_losses import attribute_inventory,attribute_targets,required_frames,sequence_ctc,two_view_kl

def row(song='freedom-flower',role='original',name='one'):
    return {'example_id':name,'song_id':song,'split':'train','audio_role':role,'audio_path':'fixture.wav','audio_sha256':'a'*64,'start_sec':0.,'end_sec':10.}

def test_requests_reject_reference_and_test_song():
    validate_audio_row(row())
    for change in [{'target_ids':[1]},{'song_id':'ke-yi-ke-zai'},{'split':'dev'},{'end_sec':float('nan')},{'audio_role':'pretend_original'},{'example_id':'../outside'}]:
        with pytest.raises(ValueError):validate_audio_row({**row(),**change})

def test_development_explicit_only():
    value={**row('shalong'),'split':'dev'}
    with pytest.raises(ValueError):validate_audio_row(value)
    validate_audio_row(value,allow_dev=True)

def test_cache_identity_crop_and_roles():
    source=row();entry={**source,'architecture_sha256':'arch','cache_layer':15,'frames':499,'width':1024,'no_reference_received':True}
    validate_cache_entry(entry,source,'arch')
    for change in [{'start_sec':.1},{'audio_role':'rendered'},{'audio_sha256':'b'*64},{'cache_layer':19},{'architecture_sha256':'legacy'},{'no_reference_received':False}]:
        with pytest.raises(ValueError):validate_cache_entry({**entry,**change},source,'arch')

def test_all_derivatives_follow_song_holdout():
    rows=[row(song,role,f'{song}-{role}') for song in policy()['songs'] for role in ['original','rendered','separated_vocal','unlabeled_original']]
    train,valid=partition(rows,'freedom-flower')
    assert len(train)==16 and len(valid)==1
    assert all(r['song_id']!='freedom-flower' for r in train)
    with pytest.raises(ValueError):partition(rows+[rows[0]],'freedom-flower')
    with pytest.raises(ValueError):partition(rows+[row('shalong')],None)

def test_choice_requires_all_folds_and_frozen_denominator():
    base={'complete':True,'fold_songs':policy()['songs'],'reference_phones':1235,'per':.5,'trainable_parameters':100,'name':'one'}
    assert choice([base,{**base,'name':'two','per':.4,'trainable_parameters':10000}])['name']=='two'
    for changed in [{'fold_songs':policy()['songs'][:-1]},{'reference_phones':1100},{'per':float('inf')}]:
        with pytest.raises(ValueError):choice([{**base,**changed}])

def test_architecture_cannot_reuse_legacy_configuration():
    spec=policy()['architecture'];a=architecture_identity({'weights_sha256':'one'},spec)
    assert a['sha256']!=architecture_identity({'weights_sha256':'two'},spec)['sha256']
    with pytest.raises(ValueError):architecture_identity({}, {**spec,'interctc_layers':[]})

def test_auxiliary_targets_preserve_repeats_and_unknown():
    inventory=attribute_inventory();targets=attribute_targets(['i','i','kʰ'],inventory)
    assert len(targets['height'])==3 and targets['height'][0]==targets['height'][1]
    assert targets['aspiration'][2]!=targets['aspiration'][0]
    assert attribute_targets(['unknown'],inventory) is None
    assert required_frames([1,1,2,2])==6

def test_consistency_padding_and_stop_gradient():
    torch=pytest.importorskip('torch');torch.set_num_threads(4)
    left=torch.randn(2,5,428,requires_grad=True);right=torch.randn(2,5,428,requires_grad=True)
    lengths=torch.tensor([3,5]);reference=two_view_kl(torch,left,right,lengths)
    lp=left.detach().clone();rp=right.detach().clone();lp[0,3:]=float('nan');rp[0,3:]=float('inf')
    actual=two_view_kl(torch,lp,rp,lengths)
    assert torch.allclose(reference.detach(),actual)
    reference.backward();assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()
    assert left.grad[0,3:].count_nonzero()==0 and right.grad[0,3:].count_nonzero()==0

def test_ctc_blank_repeats_and_invalid_values():
    torch=pytest.importorskip('torch');logits=torch.randn(1,5,4,requires_grad=True)
    loss=sequence_ctc(torch,logits,[1,1]);loss.backward();assert torch.isfinite(logits.grad).all()
    for ids in [[0],[1,1,1,1],[4],[]]:
        with pytest.raises(ValueError):sequence_ctc(torch,logits,ids)
    bad=logits.detach().clone();bad[0,0,0]=float('nan')
    with pytest.raises(ValueError):sequence_ctc(torch,bad,[1])

def fake_model(torch):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.attn=torch.nn.Module()
            for name in policy()['lora']['projections']:setattr(self.attn,name,torch.nn.Linear(1024,1024,bias=False))
        def forward(self,x,mask):return x+.01*sum(getattr(self.attn,n)(x) for n in policy()['lora']['projections']),mask
    model=torch.nn.Module();model.encoder=torch.nn.Module();model.encoder.encoders=torch.nn.ModuleList([torch.nn.Identity() for _ in range(15)]+[Block() for _ in range(4)])
    model.encoder.after_norm=torch.nn.LayerNorm(1024);model.ctc=torch.nn.Module();model.ctc.ctc_lo=torch.nn.Linear(1024,428)
    return model.requires_grad_(False)

@pytest.mark.parametrize('arm',['endpoint','lora','full'])
def test_tail_initialization_state_and_frozen_weights(arm):
    torch=pytest.importorskip('torch');torch.set_num_threads(4)
    from any2xsampa.adapters.singing_domain_runtime import make_tail,trainable_state,load_trainable_state
    model=fake_model(torch);x=torch.randn(1,6,1024);mask=torch.ones(1,1,6,dtype=torch.bool);h=x
    for b in list(model.encoder.encoders)[-4:]:h,mask=b(h,mask)
    expected=model.ctc.ctc_lo(model.encoder.after_norm(h))
    tail=make_tail(torch,model,arm).eval();actual,_=tail(x,torch.tensor([6]))
    assert torch.equal(expected,actual)
    before=model.ctc.ctc_lo.weight.detach().clone();actual.sum().backward()
    assert model.ctc.ctc_lo.weight.grad is None and torch.equal(before,model.ctc.ctc_lo.weight)
    state=trainable_state(tail);load_trainable_state(tail,state)
    with pytest.raises(ValueError):load_trainable_state(tail,{**state,'unexpected':torch.zeros(1)})
    assert sum(p.numel() for p in tail.parameters() if p.requires_grad)==({'endpoint':132160,'lora':262144,'full':4*4*1024*1024}[arm])

def test_tail_padding_cannot_poison_valid_predictions():
    torch=pytest.importorskip('torch');torch.set_num_threads(4)
    from any2xsampa.adapters.singing_domain_runtime import make_tail
    tail=make_tail(torch,fake_model(torch),'full').eval();x=torch.randn(2,5,1024);lengths=torch.tensor([3,5]);a,_=tail(x,lengths)
    x[0,3:]=float('nan');b,_=tail(x,lengths)
    assert torch.equal(a[0,:3],b[0,:3])

def test_posterior_requires_explicit_architecture_and_checkpoint():
    from any2xsampa.adapters.singing_domain_inference import validate_domain_posterior
    identity=architecture_identity({'weights_sha256':'a'*64},policy()['architecture'])
    model={'singing_domain_architecture':identity,'singing_domain_checkpoint':{'sha256':'b'*64,'arm':'lora','variant':'baseline','architecture_sha256':identity['sha256']}}
    payload={'producer':{'model_metadata':model}}
    assert validate_domain_posterior(payload)['arm']=='lora'
    for change in [{'architecture_sha256':'legacy'},{'sha256':'bad'},{'arm':'unknown'},{'variant':'unknown'}]:
        value=deepcopy(payload);value['producer']['model_metadata']['singing_domain_checkpoint'].update(change)
        with pytest.raises(ValueError):validate_domain_posterior(value)
    with pytest.raises(ValueError):validate_domain_posterior({'producer':{'model_metadata':{}}})

def test_release_cannot_hide_insertions_with_lower_overall_per():
    from any2xsampa.training.singing_domain_evaluation import release_check
    groups={name:{'reference_count':5,'substitutions':0,'deletions':0,'insertions':0,'per':0.} for name in ['aspirated','glottal_h','front_rounded']}
    def metric(n,s,d,i):return {'reference_phones':n,'edit_distance':s+d+i,'substitutions':s,'deletions':d,'insertions':i,'per':(s+d+i)/n,'groups':deepcopy(groups)}
    old=metric(214,56,25,11);render=metric(679,100,50,20)
    candidate=metric(214,43,12,21)
    check=release_check(candidate,render,old,render)
    assert not check['passed'] and 'original_insertions' in check['failures']
    good=metric(214,45,20,10);assert release_check(good,render,old,render)['passed']
    with pytest.raises(ValueError):release_check({**good,'per':float('nan')},render,old,render)

import numpy as np
import pytest
from any2xsampa.training.segmental_event import sequence_nll
from any2xsampa.training.segmental_event_exact import exact_sequence_nll,exact_viterbi,arc_indices
from any2xsampa.adapters.segmental_event import decode_events


@pytest.mark.parametrize('target',[[],[1],[1,1],[1,2,1]])
@pytest.mark.parametrize('tile',[1,7,8192])
def test_exact_tiling_loss_gradient_and_decode(target,tile):
    torch=pytest.importorskip('torch');torch.manual_seed(17)
    x=torch.randn(5,4,dtype=torch.float64,requires_grad=True);b=torch.randn(5,dtype=torch.float64,requires_grad=True)
    q=x.log_softmax(-1)
    plain=sequence_nll(torch,q,b,target);fast=exact_sequence_nll(torch,q,b,target,tile_arcs=tile)
    a=torch.autograd.grad(plain,(x,b),retain_graph=True);c=torch.autograd.grad(fast,(x,b),retain_graph=True)
    torch.testing.assert_close(plain,fast,atol=1e-10,rtol=1e-10)
    for i,j in zip(a,c):torch.testing.assert_close(i,j,atol=1e-10,rtol=1e-10)
    slow=decode_events(q.detach().numpy(),b.detach().numpy())
    decoded=exact_viterbi(torch,q.detach(),b.detach(),tile_arcs=tile)
    assert decoded['event_ids']==slow['event_ids']
    assert abs(decoded['best_path_score']-slow['best_path_score'])<1e-10


def test_exact_gradcheck():
    torch=pytest.importorskip('torch');torch.manual_seed(29)
    x=torch.randn(4,3,dtype=torch.float64,requires_grad=True);b=torch.randn(4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,c:exact_sequence_nll(torch,a.log_softmax(-1),c,[1,1],tile_arcs=4),(x,b),atol=1e-5)


def test_full_duration_triangle_and_repeats():
    for t in (1,2,10,80):
        s,e,d=arc_indices(t);assert len(s)==t*(t+1)//2
        assert (s[-t],e[-t],d[-t])==(0,t,t)
        assert (d>0).all()


@pytest.mark.parametrize('boundary',[-20.,0.,20.])
def test_longer_FP32_gradients_preserve_repeated_target_and_extreme_events(boundary):
    torch=pytest.importorskip('torch');torch.manual_seed(43)
    x=(torch.randn(31,12)*4).requires_grad_();b=torch.full((31,),boundary,requires_grad=True)
    q=x.log_softmax(-1);target=[1,1,2,3,3,3,4,2]
    plain=sequence_nll(torch,q,b,target);fast=exact_sequence_nll(torch,q,b,target,tile_arcs=32)
    a=torch.autograd.grad(plain,(x,b),retain_graph=True);c=torch.autograd.grad(fast,(x,b))
    torch.testing.assert_close(plain,fast,atol=.001,rtol=.0001)
    for i,j in zip(a,c):
        assert torch.isfinite(j).all()
        torch.testing.assert_close(i,j,atol=.001,rtol=.002)


def test_exact_never_infers_target_from_decoder_or_caps_long_arc():
    torch=pytest.importorskip('torch')
    x=torch.full((80,3),-9.);x[:,1]=9.;q=x.log_softmax(-1)
    b=torch.full((80,),-8.);b[0]=8.
    d=exact_viterbi(torch,q,b,tile_arcs=111)
    assert d['event_ids']==[1] and d['events'][0]['end_frame_exclusive']==80
    b[40]=8.
    d=exact_viterbi(torch,q,b,tile_arcs=111)
    assert d['event_ids']==[1,1]

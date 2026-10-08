from copy import deepcopy
import time
import pytest
from any2xsampa.training.whole_sequence_supervision_v2 import backward_whole_sequence
from any2xsampa.training.whole_sequence_supervision import deterministic_checkpointed_tail


def test_v2_first_and_second_pass_use_grad_and_match_full_gradient():
    torch=pytest.importorskip('torch')
    from any2xsampa.training.singing_domain_losses import sequence_ctc
    torch.manual_seed(17)
    net=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.5),torch.nn.Linear(5,4)).double()
    rows=[{'example_id':str(i),'features':torch.randn(1,6,3,dtype=torch.float64),'expected_frames':6,
        'keep_start_frame':1,'keep_end_frame':5,'global_start_frame':i*4,'global_end_frame':(i+1)*4} for i in range(2)]
    with deterministic_checkpointed_tail(net):
        loss=sequence_ctc(torch,torch.cat([net(r['features'])[:,1:5] for r in rows],1),[1,1,2])
        (.25*loss).backward()
    expected=[p.grad.clone() for p in net.parameters()];net.zero_grad(set_to_none=True);modes=[]
    def forward(row):
        modes.append(torch.is_grad_enabled());return net(row['features'])
    result=backward_whole_sequence(torch,net,rows,[1,1,2],forward,coefficient=.25,deadline=time.monotonic()+60,device='cpu')
    assert modes==[True,True,True,True]
    assert result['ctc']==pytest.approx(float(loss.detach()),abs=1e-12)
    assert result['maximum_recompute_logit_difference']==0.
    assert all(torch.allclose(p.grad,g,rtol=1e-10,atol=1e-12) for p,g in zip(net.parameters(),expected))


def test_v2_does_not_relax_recomputation_tolerance():
    torch=pytest.importorskip('torch');net=torch.nn.Linear(2,3)
    row={'example_id':'changed','expected_frames':4,'keep_start_frame':0,'keep_end_frame':4,
        'global_start_frame':0,'global_end_frame':4};calls=[]
    def forward(row):
        calls.append(1);return net(torch.ones(1,4,2))+(0. if len(calls)==1 else .001)
    with pytest.raises(ValueError,match='example=changed max_absolute_difference'):
        backward_whole_sequence(torch,net,[row],[1],forward,coefficient=.25,deadline=time.monotonic()+60,device='cpu')

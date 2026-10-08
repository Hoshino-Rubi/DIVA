"""Production-runtime CPU-only fixture; zero actual-data neural requests."""
from __future__ import annotations
import argparse
from pathlib import Path
from acoustic_memory_common import write_json_new
import numpy as np
import torch
from any2xsampa.training.acoustic_memory import soft_ctc_values,softmax,normalized_keys,exact_neighbors,fuse


def check(path: Path) -> None:
    torch.set_num_threads(4)
    z=np.arange(7*428,dtype=np.float64).reshape(7,428)%19/3
    target=[1,1,427];g,diag=soft_ctc_values(z,target)
    logp=torch.tensor(z).log_softmax(-1)[:,None,:]
    loss=torch.nn.functional.ctc_loss(logp,torch.tensor(target),torch.tensor([7]),torch.tensor([3]),reduction='sum')
    assert abs(float(loss)+diag['log_target_probability'])<1e-10
    p=fuse(softmax(z),g);assert p.shape==(7,428)
    k=normalized_keys(np.eye(3));i,d=exact_neighbors(k,k)
    assert i[:,0].tolist()==[0,1,2]
    write_json_new(path,{'passed':True,'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),
        'fixture_device':'cpu','real_data_head_requests':0,'cuda_context_initialized':torch.cuda.is_initialized(),
        'ctc_torch_loss_error':abs(float(loss)+diag['log_target_probability'])})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();check(a.output)

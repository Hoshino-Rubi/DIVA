"""Isolated, opt-in PhoneticXeus singing architecture and frozen-prefix cache."""
from __future__ import annotations
from pathlib import Path
from typing import Any,Mapping
import math,time
from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime,check_gpu_budget
from any2xsampa.training.limited_data_artifacts import sha256_file,write_json_new,read_json,verify_file
from any2xsampa.training.singing_domain import policy,architecture_identity,validate_audio_row

class SingingDomainRuntime:
    def __init__(self,snapshot:Path,cache_dir:Path,*,audio_cache:Any=None)->None:
        self.base=PhoneticXeusRuntime(snapshot,cache_dir=cache_dir,audio_cache=audio_cache)
        self.torch=self.base.torch;self.model=self.base.model.model;self.vocabulary=self.base.vocabulary
        self.spec=policy()['architecture']
        if len(self.model.encoder.encoders)!=self.spec['encoder_blocks']:raise ValueError('encoder depth mismatch')
        self.model.encoder.interctc_layer_idx=list(self.spec['interctc_layers'])
        if not self.model.encoder.interctc_use_conditioning:raise ValueError('upstream conditioning flag absent')
        upstream=read_json(snapshot/'snapshot-lock.json')
        entry=next(r for r in upstream['files'] if r['file']=='config_tree.log')
        verify_file(snapshot/'config_tree.log',entry['sha256'])
        self.identity=architecture_identity(self.base.identity,self.spec)
        self.identity['upstream_config_sha256']=entry['sha256']

    def encode_prefix(self,row:Mapping[str,Any],*,allow_dev:bool=False)->tuple[dict,dict]:
        validate_audio_row(row,allow_dev=allow_dev)
        captured=[];calls=[]
        def capture(module:Any,args:tuple)->None:
            if len(args)!=2 or not all(isinstance(v,self.torch.Tensor) for v in args):raise ValueError('unexpected prefix boundary')
            captured.append(args[0].detach().cpu().clone())
        h1=self.model.encoder.encoders[15].register_forward_pre_hook(capture)
        h2=self.model.encoder.conditioning_layer.register_forward_hook(lambda m,a,o:calls.append(1))
        try:features,logits,payload=self.base.encode(row)
        finally:h1.remove();h2.remove()
        if len(captured)!=1 or len(calls)!=3:raise RuntimeError('effective architecture did not execute exactly once')
        prefix=captured[0][0,:len(features)].contiguous()
        if prefix.shape!=features.shape or not self.torch.isfinite(prefix).all():raise ValueError('invalid prefix cache')
        payload['producer']['model_metadata']['singing_domain_architecture']=self.identity
        return {'prefix_features':prefix,'encoder_features':features,'frame_logits':logits},payload

    def save_prefix(self,row:Mapping[str,Any],destination:Path,*,allow_dev:bool=False)->dict:
        from safetensors.torch import save_file
        destination.mkdir(parents=True,exist_ok=False)
        before=time.monotonic();tensors,payload=self.encode_prefix(row,allow_dev=allow_dev)
        path=destination/'features.safetensors';save_file(tensors,str(path))
        entry={**row,'architecture_sha256':self.identity['sha256'],'cache_layer':15,'no_reference_received':True,
            'frames':len(tensors['frame_logits']),'width':1024,'path':str(path.resolve()),'sha256':sha256_file(path),
            'audio_metadata':payload['producer']['input_audio'],'crop':payload['producer']['crop'],
            'runtime':payload['producer']['runtime'],'geometry':payload['decoding'].get('geometry'),
            'elapsed_seconds':time.monotonic()-before}
        write_json_new(destination/'entry.json',entry)
        return entry

def make_tail(torch:Any,model:Any,arm:str,*,attribute_sizes:Mapping[str,int]|None=None)->Any:
    """Create a trainable tail from the same immutable pretrained initialization."""
    from torch.utils.checkpoint import checkpoint
    spec=policy()
    if arm not in spec['architecture_arms']:raise ValueError('unknown architecture arm')
    class LowRank(torch.nn.Module):
        def __init__(self,base:Any)->None:
            super().__init__();self.base=base
            rank=spec['lora']['rank']
            self.A=torch.nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device))
            self.B=torch.nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
            torch.nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        def forward(self,x:Any)->Any:
            return self.base(x)+spec['lora']['scale']*torch.nn.functional.linear(torch.nn.functional.linear(x,self.A),self.B)
    class Tail(torch.nn.Module):
        def __init__(self)->None:
            super().__init__();self.arm=arm
            self.blocks=torch.nn.ModuleList(list(model.encoder.encoders)[-4:])
            self.norm=model.encoder.after_norm;self.head=model.ctc.ctc_lo
            self.requires_grad_(False)
            self.residual=None
            if arm=='endpoint':
                self.residual=torch.nn.Sequential(torch.nn.Linear(1024,64),torch.nn.GELU(),torch.nn.Linear(64,1024)).to(self.head.weight.device)
                torch.nn.init.zeros_(self.residual[-1].weight);torch.nn.init.zeros_(self.residual[-1].bias)
            elif arm=='lora':
                for block in self.blocks:
                    for name in spec['lora']['projections']:setattr(block.attn,name,LowRank(getattr(block.attn,name)))
            else:self.blocks.requires_grad_(True)
            self.attribute_heads=torch.nn.ModuleDict({name:torch.nn.Linear(1024,size).to(self.head.weight.device) for name,size in (attribute_sizes or {}).items()})
            self.gradient_checkpointing=True
        def train(self,mode:bool=True)->Any:
            super().train(mode)
            if self.arm=='endpoint':self.blocks.eval()
            self.norm.eval();self.head.eval()
            return self
        def forward(self,prefix:Any,lengths:Any,*,final_features:Any=None)->tuple[Any,dict]:
            if prefix.ndim!=3 or prefix.shape[-1]!=1024:raise ValueError('tail expects [batch,frames,1024]')
            if len(lengths)!=len(prefix) or not torch.all((lengths>0)&(lengths<=prefix.shape[1])):raise ValueError('invalid tail lengths')
            valid=torch.arange(prefix.shape[1],device=prefix.device)[None,:]<lengths[:,None]
            if not torch.isfinite(prefix[valid]).all():raise ValueError('non-finite valid tail features')
            hidden=torch.where(valid[:,:,None],prefix,0.)
            mask=valid[:,None,:]
            if self.arm=='endpoint' and final_features is not None:
                if final_features.shape!=prefix.shape or not torch.isfinite(final_features[valid]).all():raise ValueError('invalid final feature cache')
                hidden=torch.where(valid[:,:,None],final_features,0.)
            else:
                for block in self.blocks:
                    if self.training and self.gradient_checkpointing and self.arm!='endpoint':
                        hidden,mask=checkpoint(block,hidden,mask,use_reentrant=False)
                    else:hidden,mask=block(hidden,mask)
                hidden=self.norm(hidden)
            if self.residual is not None:hidden=hidden+self.residual(hidden)
            return self.head(hidden),{name:head(hidden) for name,head in self.attribute_heads.items()}
    return Tail()

def trainable_state(tail:Any)->dict:
    keys={name for name,p in tail.named_parameters() if p.requires_grad}
    return {name:value.detach().cpu().contiguous().clone() for name,value in tail.state_dict().items() if name in keys}

def load_trainable_state(tail:Any,state:Mapping[str,Any])->None:
    expected={name for name,p in tail.named_parameters() if p.requires_grad}
    if set(state)!=expected:raise ValueError('checkpoint trainable key set differs from architecture')
    current=tail.state_dict()
    if any(tuple(state[k].shape)!=tuple(current[k].shape) or state[k].dtype!=current[k].dtype for k in state):raise ValueError('checkpoint tensor shape/type mismatch')
    for key,value in state.items():
        if not value.isfinite().all():raise ValueError('non-finite checkpoint')
    tail.load_state_dict(dict(state),strict=False)

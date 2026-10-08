"""Opt-in checkpoint inference and stable posterior replay for domain models."""
from __future__ import annotations
from pathlib import Path
from typing import Any,Mapping
from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneSegment
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer,emission_payload
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new,require_digest
from any2xsampa.training.singing_domain import architecture_identity,policy

def validate_domain_posterior(payload:Mapping[str,Any])->dict:
    metadata=payload.get('producer',{}).get('model_metadata',{})
    identity=metadata.get('singing_domain_architecture')
    if not isinstance(identity,dict):raise ValueError('missing explicit singing architecture')
    rebuilt=architecture_identity(identity['snapshot'],identity['architecture'])
    if rebuilt['sha256']!=identity.get('sha256'):raise ValueError('domain architecture digest mismatch')
    state=metadata.get('singing_domain_checkpoint',{})
    require_digest(state.get('sha256'),'checkpoint_sha256')
    if state.get('arm') not in policy()['architecture_arms'] or state.get('variant') not in ['baseline']+policy()['ablations']['variants']:
        raise ValueError('unsupported domain checkpoint kind')
    if state.get('architecture_sha256')!=identity['sha256']:raise ValueError('checkpoint and prefix architectures differ')
    return dict(state)

class SingingDomainPosteriorRecognizer(UniversalPhoneRecognizer):
    def __init__(self,posterior_path:Path,*,fixture:bool=False,vocabulary_path:Path|None=None)->None:
        self.path=posterior_path;self.fixture=fixture;self.vocabulary=vocabulary_path
    def recognize(self,vocal_audio_path:Path,lyrics:str|None=None)->list[PhoneSegment]:
        if lyrics is not None:raise ValueError('singing domain recognition is audio only')
        validate_domain_posterior(read_json(self.path))
        return PhoneticXeusPosteriorRecognizer(self.path,fixture=self.fixture,vocabulary_path=self.vocabulary).recognize(vocal_audio_path)

def load_domain_model(run:Path,job:Path,cache_dir:Path)->tuple[Any,Any,dict]:
    from safetensors.torch import load_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime,make_tail,load_trainable_state
    from any2xsampa.training.singing_domain_losses import attribute_inventory
    result=read_json(job/'RESULT.json');freeze=read_json(run/'FREEZE.json')
    if not result.get('complete') or result['fixed_epoch']!=20:raise ValueError('checkpoint is not a completed fixed-budget run')
    verify_file(run/'POLICY.json',result['policy_sha256']);verify_file(run/'CACHE.json',result['cache_sha256'])
    for p,digest in result['code_bindings'].items():verify_file(Path(p),digest)
    cp=Path(result['checkpoint_path']);verify_file(cp,result['checkpoint_sha256'])
    rt=SingingDomainRuntime(Path(freeze['model_snapshot']),cache_dir)
    if result['architecture']['sha256']!=rt.identity['sha256']:raise ValueError('checkpoint architecture differs from runtime')
    tail=make_tail(rt.torch,rt.model,result['arm'],attribute_sizes=attribute_inventory()['sizes'] if result['variant']=='attributes' else None)
    load_trainable_state(tail,load_file(str(cp),device='cpu'));tail.eval().requires_grad_(False)
    return rt,tail,result

def save_domain_posterior(rt:Any,tail:Any,checkpoint:dict,row:dict,destination:Path,*,allow_dev:bool=False,free_audio:bool=False)->dict:
    from safetensors.torch import save_file
    destination.mkdir(parents=True,exist_ok=False)
    torch=rt.torch
    if free_audio:
        from any2xsampa.adapters.candidate_inference import validate_request
        validate_request({'schema_version':'candidate_audio_request_v1','audio_path':row['audio_path'],'audio_sha256':row['audio_sha256'],
            'start_sec':row['start_sec'],'end_sec':row['end_sec'],'contains_labels_or_score_context':False})
        if set(row)!={'example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec'}:raise ValueError('free inference accepts audio only')
        features,logits,base=rt.base.encode(row)
        if tail.arm=='endpoint':
            with torch.no_grad():
                hidden=features.to('cuda');logits=tail.head(hidden+tail.residual(hidden)).float().cpu().contiguous()
    else:
        tensors,base=rt.encode_prefix(row,allow_dev=allow_dev)
        with torch.no_grad():
            prefix=tensors['prefix_features'].unsqueeze(0).cuda()
            final=tensors['encoder_features'].unsqueeze(0).cuda() if tail.arm=='endpoint' else None
            values,_=tail(prefix,torch.tensor([prefix.shape[1]],device='cuda'),final_features=final)
        logits=values[0].float().cpu().contiguous()
    metadata={**base['producer']['model_metadata'],'singing_domain_architecture':rt.identity,'singing_domain_checkpoint':{
        'arm':checkpoint['arm'],'variant':checkpoint['variant'],'seed':checkpoint['seed'],
        'sha256':checkpoint['checkpoint_sha256'],'architecture_sha256':rt.identity['sha256']}}
    payload=emission_payload(logits.numpy(),rt.vocabulary,input_audio=base['producer']['input_audio'],crop=base['producer']['crop'],
        model_metadata=metadata,geometry=base['decoding']['geometry'],runtime=base['producer']['runtime'])
    raw=destination/'logits.safetensors';save_file({'frame_logits':logits},str(raw))
    vocab=destination/'vocabulary.json'
    # Preserve upstream vocabulary bytes and digest, not a reserialized representation.
    original=rt.base.snapshot/'src/model/xeusphoneme/resources/ipa_vocab.json'
    with vocab.open('xb') as stream:stream.write(original.read_bytes())
    payload['raw_evidence']={'file':raw.name,'sha256':sha256_file(raw),'logit_shape':list(logits.shape),'dtype':'float32',
        'vocabulary_file':vocab.name,'vocabulary_sha256':sha256_file(vocab)}
    validate_domain_posterior(payload)
    pp=destination/'posterior.json';write_json_new(pp,payload)
    return {'example_id':row['example_id'],'posterior_path':str(pp.resolve()),'posterior_sha256':sha256_file(pp),
        'logits_path':str(raw.resolve()),'logits_sha256':sha256_file(raw),'prediction_ids':payload['decoding']['collapsed_raw_ids']}

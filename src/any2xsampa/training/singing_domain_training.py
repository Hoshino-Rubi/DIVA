"""Fixed-epoch song-grouped experiments on an explicit, frozen prefix cache."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from datetime import datetime,timezone
import random,time,math
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.training.singing_domain import load_manifest,partition,validate_cache_entry
from any2xsampa.training.singing_domain_losses import sequence_ctc,two_view_kl,attribute_inventory,auxiliary_ctc
from any2xsampa.training.protected_adaptation import baseline_kl
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse

def load_rows(run:Path,torch:Any)->tuple[list[dict],dict]:
    from safetensors.torch import load_file
    _,references=load_manifest(run);index=read_json(run/'CACHE.json')
    verify_file(run/'AUDIO_REQUESTS.json',index['audio_requests_sha256'])
    if not index['all_parameters_frozen'] or index['reference_received']:raise ValueError('invalid acoustic cache origin')
    requests={r['example_id']:r for r in read_json(run/'AUDIO_REQUESTS.json')['rows']}
    labels={r['example_id']:r for r in references['rows']}
    if len(index['entries'])!=len(requests) or {r['example_id'] for r in index['entries']}!=set(requests):raise ValueError('incomplete cache coverage')
    rows=[]
    for entry in index['entries']:
        validate_cache_entry(entry,requests[entry['example_id']],index['architecture']['sha256'])
        verify_file(Path(entry['path']),entry['sha256'])
        values=load_file(entry['path'],device='cpu');frames=entry['frames']
        for name,width in [('prefix_features',1024),('encoder_features',1024),('frame_logits',428)]:
            if values[name].shape!=(frames,width) or values[name].dtype!=torch.float32 or not torch.isfinite(values[name]).all():raise ValueError('invalid cached tensor')
        rows.append({**entry,**labels.get(entry['example_id'],{}),**values})
    return rows,index

def predict(torch:Any,tail:Any,row:dict,*,dropout:float=0.)->tuple[Any,dict]:
    prefix=row['prefix_features'].unsqueeze(0).to('cuda')
    final=row['encoder_features'].unsqueeze(0).to('cuda') if tail.arm=='endpoint' and not dropout else None
    if dropout:prefix=torch.nn.functional.dropout(prefix,p=dropout,training=True)
    lengths=torch.tensor([prefix.shape[1]],device='cuda',dtype=torch.long)
    return tail(prefix,lengths,final_features=final)

def evaluate(torch:Any,tail:Any,rows:list[dict],labels:list[str])->dict:
    pairs=[];details=[];tail.eval()
    with torch.no_grad():
        for row in rows:
            logits,_=predict(torch,tail,row)
            ids=ctc_collapse(logits[0].argmax(-1).cpu().tolist());pred=[labels[i] for i in ids]
            pairs.append((row['target_ipa'],pred))
            details.append({'example_id':row['example_id'],'song_id':row['song_id'],'reference':row['target_ipa'],
                'prediction':pred,'metrics':phone_metrics([pairs[-1]])})
    return {'metrics':phone_metrics(pairs),'examples':details}

def supervised_loss(torch:Any,tail:Any,row:dict,lookup:dict,variant:str,spec:dict,attributes:dict)->tuple[Any,dict]:
    logits,aux=predict(torch,tail,row)
    ctc=sequence_ctc(torch,logits,row['target_ids'])
    weight=spec['training']['rendered_loss_weight'] if row['audio_role']=='rendered' else 1.
    lengths=torch.tensor([logits.shape[1]],device='cuda')
    prior=row['frame_logits'].unsqueeze(0).to('cuda')
    kl=baseline_kl(torch,logits,prior,lengths,temperature=spec['training']['temperature'])
    loss=weight*ctc+spec['training']['baseline_kl_weight']*kl
    diagnostics={'ctc':float(ctc.detach()),'prior_kl':float(kl.detach()),'paired_kl':0.,'attribute_ctc':0.,'aux_excluded':[]}
    if variant in {'paired','unlabeled'} and row['audio_role']=='original':
        mate=lookup[row['paired_example_id']]
        if mate['song_id']!=row['song_id'] or mate['frames']!=row['frames']:raise ValueError('paired source/time mismatch')
        right,_=predict(torch,tail,mate)
        consistency=two_view_kl(torch,logits,right,lengths,spec['training']['temperature'])
        loss=loss+spec['ablations']['paired_kl_weight']*consistency
        diagnostics['paired_kl']=float(consistency.detach())
    if variant=='attributes':
        attribute_loss,excluded=auxiliary_ctc(torch,aux,row['target_ipa'],attributes)
        loss=loss+spec['ablations']['attribute_ctc_weight']*attribute_loss
        diagnostics.update(attribute_ctc=float(attribute_loss.detach()),aux_excluded=excluded)
    if not torch.isfinite(loss):raise ValueError('non-finite supervised objective')
    return loss,diagnostics

def train_epoch(torch:Any,tail:Any,train:list[dict],variant:str,spec:dict,attributes:dict,optimizer:Any,rng:Any,deadline:float)->dict:
    tail.train();lookup={r['example_id']:r for r in train}
    labelled=[r for r in train if r['audio_role'] in {'original','rendered'}]
    ordered=[]
    for row in labelled:ordered.extend([row]*spec['training']['original_repeats'] if row['audio_role']=='original' else [row])
    rng.shuffle(ordered)
    unlabeled=[r for r in train if r['audio_role']=='unlabeled_original'];rng.shuffle(unlabeled)
    micro=spec['training']['gradient_accumulation'];params=[p for p in tail.parameters() if p.requires_grad]
    totals={'ctc':0.,'prior_kl':0.,'paired_kl':0.,'attribute_ctc':0.,'unlabeled_kl':0.};excluded={};steps=0
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
    for offset in range(0,len(ordered),micro):
        if time.monotonic()>deadline:raise TimeoutError('fixed job budget exceeded; no partial-epoch selection')
        optimizer.zero_grad(set_to_none=True);batch=ordered[offset:offset+micro]
        for row in batch:
            loss,diagnostics=supervised_loss(torch,tail,row,lookup,variant,spec,attributes)
            (loss/len(batch)).backward()
            for key in ('ctc','prior_kl','paired_kl','attribute_ctc'):totals[key]+=diagnostics[key]
            for reason in diagnostics['aux_excluded']:excluded[reason]=excluded.get(reason,0)+1
        if variant=='unlabeled':
            if not unlabeled:raise ValueError('no training-song unlabeled windows')
            row=unlabeled[steps%len(unlabeled)]
            left,_=predict(torch,tail,row);right,_=predict(torch,tail,row,dropout=spec['ablations']['feature_dropout'])
            lengths=torch.tensor([left.shape[1]],device='cuda')
            loss=two_view_kl(torch,left,right,lengths,spec['training']['temperature'])
            if not torch.isfinite(loss):raise ValueError('non-finite unlabeled consistency')
            (spec['ablations']['unlabeled_kl_weight']*loss).backward();totals['unlabeled_kl']+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params,spec['training']['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();steps+=1
        if torch.cuda.memory_reserved()>7*1024**3:raise RuntimeError('own GPU reserved budget exceeded')
    return {'optimizer_steps':steps,'labelled_microsteps':len(ordered),
            'means':{k:v/(steps if k=='unlabeled_kl' else len(ordered)) for k,v in totals.items()},'attribute_exclusions':excluded}

def train_job(run:Path,arm:str,variant:str,held_song:str|None,seed:int,output:Path)->None:
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime,make_tail,trainable_state
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from safetensors.torch import save_file
    spec,_=load_manifest(run);freeze=read_json(run/'FREEZE.json')
    if variant not in ['baseline']+spec['ablations']['variants']:raise ValueError('unplanned variant')
    if seed not in spec['training']['final_seeds'] or (held_song is not None and seed!=spec['training']['screen_seed']):raise ValueError('unplanned seed')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=spec['training']['maximum_job_seconds']):
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),output/'runtime-cache');torch=rt.torch
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        rows,index=load_rows(run,torch)
        if index['architecture']['sha256']!=rt.identity['sha256']:raise ValueError('cache architecture does not match live model')
        train,validation=partition(rows,held_song)
        attrs=attribute_inventory()
        tail=make_tail(torch,rt.model,arm,attribute_sizes=attrs['sizes'] if variant=='attributes' else None)
        # The lower blocks never run during optimization; keep their immutable weights off GPU.
        rt.model.frontend.cpu();rt.model.preencoder.cpu()
        for block in list(rt.model.encoder.encoders)[:15]:block.cpu()
        rt.model.encoder.conditioning_layer.cpu()
        labels=[p for p,i in sorted(rt.vocabulary.items(),key=lambda x:x[1])]
        probe=next(r for r in train if r['audio_role']=='original');tail.eval()
        with torch.no_grad():initial,_=predict(torch,tail,probe)
        initial_error=float((initial[0].cpu()-probe['frame_logits']).abs().max())
        if initial_error>2e-5:raise ValueError(f'non-equivalent pretrained initialization: {initial_error}')
        frozen=[p for p in rt.model.parameters() if not p.requires_grad]+[p for p in tail.parameters() if not p.requires_grad]
        versions=[p._version for p in frozen]
        trainable=[p for p in tail.parameters() if p.requires_grad]
        groups=[{'params':[p for n,p in tail.named_parameters() if p.requires_grad and not n.startswith('attribute_heads.')], 'lr':spec['learning_rates'][arm]}]
        auxiliary=[p for p in tail.attribute_heads.parameters() if p.requires_grad]
        if auxiliary:groups.append({'params':auxiliary,'lr':spec['ablations']['attribute_learning_rate']})
        optimizer=torch.optim.AdamW(groups,weight_decay=spec['training']['weight_decay'],foreach=False)
        baseline=evaluate(torch,tail,validation,labels) if validation else None
        counts={role:sum(r['audio_role']==role for r in train) for role in ('original','rendered','separated_vocal','unlabeled_original')}
        start={'arm':arm,'variant':variant,'held_song':held_song,'seed':seed,'training_counts':counts,
            'training_example_ids':[r['example_id'] for r in train],'training_songs':sorted({r['song_id'] for r in train}),
            'validation_ids':[r['example_id'] for r in validation],
            'policy_sha256':sha256_file(run/'POLICY.json'),'cache_sha256':sha256_file(run/'CACHE.json'),
            'architecture':rt.identity,'initial_maximum_logit_error':initial_error,
            'trainable_parameters':sum(p.numel() for p in trainable),'baseline_validation':baseline}
        start['code_bindings']={str(p.resolve()):sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('singing_domain_losses.py'),
            Path(__file__).parents[1]/'adapters/singing_domain_runtime.py',Path(__file__).parents[1]/'config/singing_phone_attributes_v1.json']}
        write_json_new(output/'START.json',start)
        rng=random.Random(seed);history=[];torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,spec['training']['epochs']+1):
            trained=train_epoch(torch,tail,train,variant,spec,attrs,optimizer,rng,started+spec['training']['maximum_job_seconds'])
            evaluated=evaluate(torch,tail,validation,labels) if validation and epoch in {5,10,20} else None
            event={'epoch':epoch,**trained,'validation_metrics':evaluated['metrics'] if evaluated else None,
                'elapsed_seconds':time.monotonic()-started,'memory':check_gpu_budget(torch)}
            history.append(event);write_json_new(output/f'epoch-{epoch:02d}.json',event)
            print(arm,variant,held_song or 'all',seed,'epoch',epoch,'ctc',round(trained['means']['ctc'],4),
                  'per',round(evaluated['metrics']['per'],5) if evaluated else '-',flush=True)
        if versions!=[p._version for p in frozen]:raise RuntimeError('frozen parameters changed')
        final=evaluate(torch,tail,validation,labels) if validation else None
        path=output/'final.safetensors';save_file(trainable_state(tail),str(path))
        result={**start,'complete':True,'fixed_epoch':spec['training']['selection_epoch'],'history':history,
            'final_validation':final,'checkpoint_path':str(path.resolve()),'checkpoint_sha256':sha256_file(path),
            'frozen_parameters_unchanged':True,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'elapsed_seconds':time.monotonic()-started,'default_replacement':False,'test_predictions_opened':False}
        write_json_new(output/'RESULT.json',result)

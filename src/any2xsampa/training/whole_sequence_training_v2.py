"""Matched whole-sequence continuation with unchanged short-window supervision."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from datetime import datetime,timezone
import random,time,math
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from any2xsampa.training.singing_domain_training import load_rows,predict,evaluate
from any2xsampa.training.singing_domain import partition
from any2xsampa.training.singing_domain_losses import sequence_ctc
from any2xsampa.training.protected_adaptation import baseline_kl
from any2xsampa.training.whole_sequence_cache import load_whole_rows
from any2xsampa.training.whole_sequence_supervision_v2 import backward_whole_sequence


def whole_schedule(total_steps:int,songs:list[dict])->dict[int,dict]:
    if total_steps<len(songs) or not songs or len({s['song_id'] for s in songs})!=len(songs):
        raise ValueError('invalid whole-source optimizer placement')
    ordered=sorted(songs,key=lambda r:r['song_id'])
    indices=[max(0,(i+1)*total_steps//(len(ordered)+1)-1) for i in range(len(ordered))]
    if len(set(indices))!=len(ordered):raise ValueError('whole-source optimizer placement collides')
    return dict(zip(indices,ordered))


def train_epoch(torch:Any,tail:Any,train:list[dict],whole:list[dict],spec:dict,
                optimizer:Any,rng:Any,deadline:float)->dict:
    ordered=[]
    for row in train:
        if row['audio_role'] in {'original','rendered'}:
            ordered.extend([row]*(spec['training']['original_repeats'] if row['audio_role']=='original' else 1))
    rng.shuffle(ordered);tail.train()
    micro=spec['training']['gradient_accumulation'];schedule=whole_schedule(math.ceil(len(ordered)/micro),whole)
    parameters=[p for p in tail.parameters() if p.requires_grad]
    totals={'ctc':0.,'prior_kl':0.};events=[];steps=0
    for offset in range(0,len(ordered),micro):
        if time.monotonic()>deadline:raise TimeoutError('whole continuation exceeded fixed job budget')
        optimizer.zero_grad(set_to_none=True);batch=ordered[offset:offset+micro]
        for row in batch:
            logits,_=predict(torch,tail,row)
            ctc=sequence_ctc(torch,logits,row['target_ids'])
            kl=baseline_kl(torch,logits,row['frame_logits'].unsqueeze(0).cuda(),
                torch.tensor([logits.shape[1]],device='cuda'),temperature=spec['training']['temperature'])
            weight=spec['training']['rendered_loss_weight'] if row['audio_role']=='rendered' else 1.
            loss=weight*ctc+spec['training']['pretrained_kl_weight']*kl
            if not torch.isfinite(loss):raise ValueError('non-finite short-window objective')
            (loss/len(batch)).backward();totals['ctc']+=float(ctc.detach());totals['prior_kl']+=float(kl.detach())
        if steps in schedule:
            song=schedule[steps]
            result=backward_whole_sequence(torch,tail,song['chunks'],song['target_ids'],
                lambda row:predict(torch,tail,row)[0],coefficient=spec['whole_sequence']['coefficient'],
                deadline=deadline,device='cuda')
            events.append({'song_id':song['song_id'],'optimizer_step':steps+1,**result})
        torch.nn.utils.clip_grad_norm_(parameters,spec['training']['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();steps+=1
        if torch.cuda.memory_reserved()>7*2**30:raise RuntimeError('whole training exceeded own GPU budget')
    return {'optimizer_steps':steps,'supervised_microsteps':len(ordered),
        'means':{k:v/len(ordered) for k,v in totals.items()},'whole_sequences':events,
        'whole_extra_compute_seconds':sum(e['elapsed_seconds'] for e in events)}


def train_job(run:Path,held_song:str|None,seed:int,output:Path)->None:
    from safetensors.torch import load_file,save_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime,make_tail,load_trainable_state,trainable_state
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
    spec,freeze=read_json(run/'POLICY.json'),read_json(run/'FREEZE.json')
    verify_file(run/'POLICY.json',freeze['policy_sha256'])
    code=read_json(run/'CODE_FREEZE.json')
    addendum=read_json(run/'NUMERICAL_FIX_FREEZE.json')
    code={'files':{**code['files'],**addendum['files']}}
    for path,digest in code['files'].items():verify_file(Path(path),digest)
    if held_song is not None and (held_song not in spec['songs'] or seed!=17):raise ValueError('unplanned whole fold')
    if seed not in spec['training']['final_seeds']:raise ValueError('unplanned whole seed')
    if held_song is None and not read_json(run/'ROUTE_SELECTION.json')['passed']:
        raise ValueError('final seed requires passed training-song cross-validation')
    probe=read_json(run/'whole-cuda-probe-v2/RESULT.json')
    if not probe['gradients_finite'] or not probe['frozen_parameters_unchanged']:
        raise ValueError('real whole-gradient resource probe did not pass')
    parent_run=Path(freeze['parents']['singing'])
    parent_job=parent_run/'architecture/full'/held_song if held_song else parent_run/'final-training'/('seed'+str(seed))
    parent=read_json(parent_job/'RESULT.json')
    if not parent['complete'] or (parent['arm'],parent['variant'],parent['held_song'],parent['seed'])!=('full','baseline',held_song,seed):
        raise ValueError('whole initialization checkpoint mismatch')
    if held_song is not None and held_song in parent['training_songs']:raise ValueError('parent checkpoint leaks held song')
    for p,digest in parent['code_bindings'].items():verify_file(Path(p),digest)
    verify_file(Path(parent['checkpoint_path']),parent['checkpoint_sha256'])
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=spec['training']['maximum_job_seconds']):
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),output/'runtime-cache');torch=rt.torch
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        rows,index=load_rows(parent_run,torch)
        if index['architecture']['sha256']!=rt.identity['sha256'] or parent['architecture']['sha256']!=rt.identity['sha256']:
            raise ValueError('whole architecture/prefix mismatch')
        train,validation=partition(rows,held_song)
        whole=load_whole_rows(run,torch,rt.identity['sha256'],held_song)
        if any(s['song_id']==held_song for s in whole):raise ValueError('held whole sequence leaked into training')
        tail=make_tail(torch,rt.model,'full');load_trainable_state(tail,load_file(parent['checkpoint_path'],device='cpu'))
        rt.model.frontend.cpu();rt.model.preencoder.cpu()
        for block in list(rt.model.encoder.encoders)[:15]:block.cpu()
        rt.model.encoder.conditioning_layer.cpu()
        labels=[p for p,i in sorted(rt.vocabulary.items(),key=lambda x:x[1])]
        initial=evaluate(torch,tail,validation,labels) if validation else None
        if initial is not None and initial!=parent['final_validation']:
            raise ValueError('whole continuation step zero differs from parent checkpoint')
        frozen=[p for p in rt.model.parameters() if not p.requires_grad];versions=[p._version for p in frozen]
        parameters=[p for p in tail.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=spec['training']['learning_rate'],weight_decay=spec['training']['weight_decay'],foreach=False)
        record={'arm':'whole_sequence','held_song':held_song,'seed':seed,'architecture':rt.identity,
            'parent_checkpoint_path':parent['checkpoint_path'],'parent_checkpoint_sha256':parent['checkpoint_sha256'],
            'parent_result_sha256':sha256_file(parent_job/'RESULT.json'),'initial_validation':initial,
            'initialization':'continued full tail; not zero initialization',
            'training_example_ids':[r['example_id'] for r in train],
            'training_songs':sorted({r['song_id'] for r in train}),
            'whole_training_songs':[r['song_id'] for r in whole],
            'whole_reference_sha256':freeze['whole_references_sha256'],'whole_cache_sha256':sha256_file(run/'WHOLE_CACHE.json'),
            'policy_sha256':sha256_file(run/'POLICY.json'),'cache_sha256':sha256_file(parent_run/'CACHE.json'),
            'implementation_addendum_path':str(run/'NUMERICAL_FIX_FREEZE.json'),
            'implementation_addendum_sha256':sha256_file(run/'NUMERICAL_FIX_FREEZE.json'),
            'implementation_version':'whole_sequence_two_grad_passes_v2',
            'code_bindings':code['files'],'code_freeze_sha256':sha256_file(run/'CODE_FREEZE.json'),
            'trainable_parameters':sum(p.numel() for p in parameters),'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'default_replacement':False,'test_predictions_opened':False}
        write_json_new(output/'START.json',record);history=[];rng=random.Random(seed)
        torch.cuda.reset_peak_memory_stats();deadline=start+spec['training']['maximum_job_seconds']
        for epoch in range(1,spec['training']['epochs']+1):
            result=train_epoch(torch,tail,train,whole,spec,optimizer,rng,deadline)
            scored=evaluate(torch,tail,validation,labels) if validation and epoch in {5,10,20} else None
            event={'epoch':epoch,**result,'validation':scored['metrics'] if scored else None,
                'elapsed_seconds':time.monotonic()-start,'memory':check_gpu_budget(torch)}
            write_json_new(output/f'epoch-{epoch:02d}.json',event);history.append(event)
            print('WHOLE',held_song or 'all',seed,epoch,'shortCTC',round(result['means']['ctc'],4),
                'wholeCTC',[round(e['ctc'],4) for e in result['whole_sequences']],
                'PER',round(scored['metrics']['per'],5) if scored else '-',flush=True)
        if versions!=[p._version for p in frozen]:raise RuntimeError('whole training changed frozen parameters')
        final=evaluate(torch,tail,validation,labels) if validation else None
        checkpoint=output/'final.safetensors';save_file(trainable_state(tail),str(checkpoint))
        steps=sum(e['optimizer_steps'] for e in history)
        if held_song:
            control=read_json(Path(freeze['parents']['routes'])/'screen/control'/held_song/'RESULT.json')
            if control['optimizer_steps']!=steps or control['training_example_ids']!=record['training_example_ids'] or control['parent_checkpoint_sha256']!=record['parent_checkpoint_sha256']:
                raise ValueError('whole route differs from matched control budget/data/initialization')
        write_json_new(output/'RESULT.json',{**record,'complete':True,'fixed_epoch':20,'history':history,
            'final_validation':final,'checkpoint_path':str(checkpoint.resolve()),'checkpoint_sha256':sha256_file(checkpoint),
            'optimizer_steps':steps,'frozen_parameters_unchanged':True,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-start,
            'whole_extra_compute_seconds':sum(e['whole_extra_compute_seconds'] for e in history)})

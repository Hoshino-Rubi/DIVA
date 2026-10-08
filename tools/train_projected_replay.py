"""One frozen three-seed projected replay recipe; no test or decoder tuning."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,random,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.training.decoder_experiment import validate_policy as validate_source, promotion_gate
from any2xsampa.training.projected_replay import project_optimizer_step


def epoch(torch,adapter,data,cycle,order,optimizer,deadline):
    from any2xsampa.training.retention_adaptation import _batch,feature_batch,replay_kl,baseline_kl
    head=data['head'];parameters=list(adapter.parameters());events=[]
    for offset in range(0,len(order),8):
        if time.monotonic()>deadline:return {'complete':False,'steps':events}
        adapter.train();rows=order[offset:offset+8];f,lengths,targets,ts=_batch(rows,torch)
        _,_,teachers=feature_batch(torch,rows)
        replay=cycle.take(8);rf,rl,rt=feature_batch(torch,replay)
        sequences=[data['rendered_targets'][r['example_id']]['model_target_ids'] for r in replay]
        ids=torch.tensor([i for seq in sequences for i in seq],dtype=torch.long)
        sizes=torch.tensor([len(seq) for seq in sequences],dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits=head(adapter(f));ctc=torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0,1),targets,lengths,ts,blank=0,zero_infinity=False)
        kl=baseline_kl(torch,logits,teachers,lengths,temperature=2.)
        rlogits=head(adapter(rf));rctc=torch.nn.functional.ctc_loss(rlogits.log_softmax(-1).transpose(0,1),ids,rl,sizes,blank=0,zero_infinity=False)
        rkl,_=replay_kl(torch,rlogits,rt,rl,{'frame_selection':'all','radius':0})
        memory=rctc+rkl;loss=ctc+.05*kl+memory
        if not torch.isfinite(loss):raise ValueError('nonfinite projected training loss')
        reference=[g.detach() for g in torch.autograd.grad(memory,parameters,retain_graph=True)]
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        before=[p.detach().clone() for p in parameters]
        optimizer.step();projection=project_optimizer_step(torch,parameters,before,reference)
        with torch.no_grad():
            check=head(adapter(rf));check_ctc=torch.nn.functional.ctc_loss(check.log_softmax(-1).transpose(0,1),ids,rl,sizes,blank=0,zero_infinity=False)
            check_kl,_=replay_kl(torch,check,rt,rl,{'frame_selection':'all','radius':0})
        events.append({**projection,'original_ctc':float(ctc.detach()),'memory_loss_before':float(memory.detach()),'memory_loss_after':float(check_ctc+check_kl)})
    return {'complete':True,'steps':events}


def gate(original,rendered,current,old,baseline,source):
    trial={'original_dev':original['metrics'],'rendered_dev':rendered['metrics']}
    current_result=promotion_gate(current,trial,source)
    if original['metrics']['per']>baseline['metrics']['per']:
        current_result['failures'].append('worse_than_pretrained')
    prior=old['metrics'];m=rendered['metrics']
    if m['per']>prior['per']+.03:current_result['failures'].append('rendered_old_teacher_overall_regression')
    for name,g in prior['groups'].items():
        if g['reference_count']>=20 and m['groups'][name]['per']>g['per']+.03:
            current_result['failures'].append('rendered_old_teacher_'+name+'_regression')
    current_result['passed']=not current_result['failures']
    return current_result


def run(suite:Path,seed:int):
    p=read_json(suite/'PROJECTED_REPLAY_POLICY.json')
    config=read_json(ROOT/'src/any2xsampa/config/projected_replay_v1.json')
    if any(p.get(k)!=v for k,v in config.items()) or seed not in p['seeds']:raise ValueError('unfrozen projected recipe')
    verify_file(Path(p['source_policy']['path']),p['source_policy']['sha256']);source=read_json(Path(p['source_policy']['path']));validate_source(source)
    verify_file(suite/'DECODER_COMPARISON.json',p['comparison_sha256']);verify_file(suite/'ERROR_DIAGNOSIS.json',p['diagnosis_sha256'])
    current=read_json(suite/'DECODER_COMPARISON.json')['aggregate']['candidate']
    out=suite/'projected-replay'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda,check_gpu_budget,gpu_total_used_bytes
    enforce_offline(out/'runtime-cache')
    if gpu_total_used_bytes()>14*1024**3:raise RuntimeError('card budget exceeded')
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.training.retention_adaptation import load_inputs,ReplayCycle,_evaluate,_evaluate_logits,make_residual_adapter
    from safetensors.torch import save_file
    with ProcessResourceMonitor(out/'monitor',timeout_seconds=1500):
        torch=configure_cuda();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        data=load_inputs(Path(source['source_run']),source['bindings']['manifest']['sha256'],torch)
        teacher,head=data['teacher'],data['head'];adapter=make_residual_adapter(torch).to('cuda');adapter.load_state_dict(teacher.state_dict(),strict=True)
        frozen={name:{k:v.detach().clone() for k,v in module.state_dict().items()} for name,module in [('teacher',teacher),('head',head)]}
        with torch.no_grad():
            for row in data['train']+data['replay']:
                if not torch.equal(head(adapter(row['features'].to('cuda'))).cpu(),row['teacher_logits']):raise ValueError('initialization replay differs')
        baseline=_evaluate_logits(data['dev'],data['labels']);old=_evaluate(data['protection'],data['labels'],torch,teacher,head)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-4);rng=random.Random(seed);cycle=ReplayCycle(data['replay'],seed)
        before=time.monotonic();started=datetime.now(timezone.utc).isoformat();best=eligible=None;stale=0;history=[];reason='maximum_epochs'
        for number in range(1,21):
            order=list(data['train']);rng.shuffle(order)
            event=epoch(torch,adapter,data,cycle,order,optimizer,before+1500)
            if not event['complete']:reason='incomplete_epoch_timeout';break
            adapter.eval();original=_evaluate(data['dev'],data['labels'],torch,adapter,head);rendered=_evaluate(data['protection'],data['labels'],torch,adapter,head)
            outcome=gate(original,rendered,current,old,baseline,source)
            event.update(epoch=number,original_dev=original['metrics'],rendered_dev=rendered['metrics'],gate=outcome,resources=check_gpu_budget(torch),elapsed_seconds=time.monotonic()-before)
            history.append(event);write_json_new(out/f'epoch-{number:02d}.json',event)
            score=original['metrics']['per']
            def capture():return {'epoch':number,'state':{k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},'original':original,'rendered':rendered}
            if best is None or score<best['original']['metrics']['per']:best=capture();stale=0
            else:stale+=1
            if outcome['passed'] and (eligible is None or score<eligible['original']['metrics']['per']):eligible=capture()
            print(seed,number,round(score,6),outcome['passed'],sum(x['projected'] for x in event['steps']),flush=True)
            if stale>=4:reason='early_stopping';break
        if best is None:raise RuntimeError('no completed epoch')
        for name,module in [('teacher',teacher),('head',head)]:
            if any(p.requires_grad for p in module.parameters()) or any(not torch.equal(v,frozen[name][k]) for k,v in module.state_dict().items()):raise ValueError('frozen parameter changed')
        checkpoints={}
        for role,value in [('best_overall',best),('best_admissible',eligible)]:
            checkpoints[role]=None
            if value is None:continue
            path=out/(role+'.safetensors');save_file(value['state'],str(path));ep=out/(role+'-evaluation.json');write_json_new(ep,{'original':value['original'],'rendered':value['rendered']})
            checkpoints[role]={'path':str(path),'sha256':sha256_file(path),'epoch':value['epoch'],'evaluation_path':str(ep),'evaluation_sha256':sha256_file(ep),'original_metrics':value['original']['metrics'],'rendered_metrics':value['rendered']['metrics']}
        report={'schema_version':'projected_replay_result_v1','seed':seed,'policy_sha256':sha256_file(suite/'PROJECTED_REPLAY_POLICY.json'),'checkpoints':checkpoints,'history':history,
                'step_zero_equal_all_training_rows':True,'teacher_unchanged':True,'head_unchanged':True,'backbone_loaded':False,'decoder':'greedy_ctc_v1','test_loaded':False,
                'projection_scope':'first_order_current_replay_batch_loss_only','stop_reason':reason,'elapsed_seconds':time.monotonic()-before,'started_at_utc':started,'ended_at_utc':datetime.now(timezone.utc).isoformat(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'training_unique_replay_windows':len(cycle.seen),'default_changed':False}
        write_json_new(out/'RESULT.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True);p.add_argument('--seed',type=int,choices=[17,29,43],required=True);a=p.parse_args()
    if not a.suite.resolve().is_relative_to(ROOT/'.state/runs'):raise ValueError('run must be under .state/runs')
    run(a.suite.resolve(),a.seed)

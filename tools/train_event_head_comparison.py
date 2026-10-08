"""One fixed H/E leave-song-out job, immutable starts and no retries."""
from __future__ import annotations
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file,sha256_object
from any2xsampa.training.event_head_protocol import load_protocol,load_cache,check_job,epoch_schedule,fold_steps
from any2xsampa.training.event_head_training import make_head,loss_on_row,predict
from any2xsampa.training.phone_error_metrics import phone_metrics
from input_native_probe_common import parameter_digest,compare_logits
from verify_input_supervision import export_replay


def evaluate(torch,head,arm,rows,spec,freeze,output):
    from safetensors.numpy import save_file
    from dataclasses import asdict
    from any2xsampa.adapters.segmental_event import decode_events,SegmentalEventRecognizer
    vocab=read_json(Path(freeze['vocabulary']['path']));labels={i:p for p,i in vocab.items()};details=[]
    head.eval()
    for row in rows:
        ids,q,b=predict(torch,head,row['features'],arm,spec);destination=output/row['example_id']
        if arm=='H':
            a=export_replay(q.numpy(),vocab,row,{'experimental_arm':'H','no_reference_conditioning':True,
                'snapshot_lock_sha256':sha256_file(Path(freeze['snapshot'])/'snapshot-lock.json'),
                'vocabulary_sha256':freeze['vocabulary']['sha256'],
                'independent_head_sha256':parameter_digest(head)},destination,Path(freeze['vocabulary']['path']))
            if a['prediction']!=[labels[i] for i in ids]:raise ValueError('H codec replay changed sequence')
            artifact=a
        else:
            decoded=decode_events(q.numpy(),b.numpy())
            if decoded['event_ids']!=ids:raise ValueError('exact FP32 vs independent FP64 event decode differs')
            destination.mkdir(parents=True,exist_ok=False)
            raw=destination/'scores.safetensors';save_file({'frame_log_probs':q.numpy(),'boundary_logits':b.numpy()},str(raw))
            vp=destination/'vocabulary.json';vp.write_bytes(Path(freeze['vocabulary']['path']).read_bytes())
            pp=destination/'events.json';write_json_new(pp,{'schema_version':'segmental_event_evidence_v1','scoring_spec':'mean_logp_start_hold_v1',
                'reference_conditioning':False,'input_audio':row['audio_metadata'],'crop':row['crop'],'geometry':row['geometry'],
                'scores':{'file':raw.name,'sha256':sha256_file(raw)},'vocabulary':{'file':vp.name,'sha256':sha256_file(vp)},
                'decoded_event_ids':ids,'head_sha256':parameter_digest(head),'times_are_ground_truth':False})
            a=SegmentalEventRecognizer(pp).recognize(Path(row['audio_path']));f=SegmentalEventRecognizer(pp,fixture=True).recognize(Path(row['audio_path']))
            if [asdict(s) for s in a]!=[asdict(s) for s in f]:raise ValueError('event codec differs')
            artifact={'event_path':str(pp),'event_sha256':sha256_file(pp),'scores_path':str(raw),'scores_sha256':sha256_file(raw),'real_fixture_equal':True}
        pred=[labels[i] for i in ids]
        details.append({'example_id':row['example_id'],'song_id':row['song_id'],'audio_role':row['audio_role'],
            'reference':row['target_ipa'],'prediction':pred,'metrics':phone_metrics([(row['target_ipa'],pred)]),**artifact})
    return {'details':details,'original':phone_metrics([(r['reference'],r['prediction']) for r in details if r['audio_role']=='original']),
            'rendered':phone_metrics([(r['reference'],r['prediction']) for r in details if r['audio_role']=='rendered'])}


def train(run,arm,held):
    freeze,spec,index=load_protocol(run,'CODE_REVIEW_training.json');check_job(run,arm,held,spec,worker=True)
    destination=run/'screen'/arm/held
    write_json_new(destination/'START.json',{'arm':arm,'held_song':held,'seed':17,'freeze_sha256':sha256_file(run/'FREEZE.json'),
        'review_sha256':sha256_file(run/'CODE_REVIEW_training.json'),'planned_epochs':20,'quality_training_start':True})
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda,check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from safetensors.torch import load_file,save_file
    enforce_offline(destination/'runtime-cache');started=time.monotonic()
    with serial_candidate_job(),ProcessResourceMonitor(destination/'monitor',timeout_seconds=spec['training']['maximum_job_seconds']):
        torch=configure_cuda();torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        rows=load_cache(torch,index['rows']);trainrows=[r for r in rows if r['song_id']!=held];heldrows=[r for r in rows if r['song_id']==held]
        if any(r['song_id']==held for r in trainrows) or not heldrows:raise ValueError('fold split leakage')
        state={k:v.cuda() for k,v in load_file(freeze['pretrained_head']['path'],device='cpu').items()}
        head=make_head(torch,state,arm).train();vocab=read_json(Path(freeze['vocabulary']['path']))
        initial=parameter_digest(head);checks=[]
        with torch.no_grad():
            for row in trainrows:
                checks.append(compare_logits(head.content(row['features'].cuda()).cpu().numpy(),row['prior'].numpy(),vocab))
        versions=[r['features']._version for r in rows]+[r['prior']._version for r in rows]
        schedule=epoch_schedule(index['rows'],index['unused_rng_rows'],held,17)
        frozen_schedule=read_json(run/'SCHEDULES.json')[held]
        if schedule!=frozen_schedule['epochs']:raise ValueError('fixed sampling/RNG order differs')
        write_json_new(destination/'INITIALIZATION.json',{'head_sha256':initial,'all_train_initial_logits_equal':True,
            'maximum_logit_error':max(c['maximum_absolute_error'] for c in checks),'train_example_ids':[r['example_id'] for r in trainrows],
            'held_example_ids':[r['example_id'] for r in heldrows],'parameter_count':sum(p.numel() for p in head.parameters()),
            'architecture_sha256':index['architecture']['sha256'],'shared_head_unchanged':True,'encoder_loaded':False})
        optimizer=torch.optim.AdamW(head.parameters(),lr=spec['training']['learning_rate'],weight_decay=spec['training']['weight_decay'],foreach=False)
        lookup={r['example_id']:r for r in trainrows};history=[];steps=0
        for epoch,order in enumerate(schedule,1):
            epoch_start=time.monotonic();totals={'sequence_loss_per_reference':0.,'kl':0.};head.train()
            for offset in range(0,len(order),8):
                if time.monotonic()-started>spec['training']['maximum_job_seconds']:raise TimeoutError('fixed job deadline; no partial selection')
                batch=order[offset:offset+8];optimizer.zero_grad(set_to_none=True)
                for key in batch:
                    loss,diag=loss_on_row(torch,head,lookup[key],arm,spec);(loss/len(batch)).backward()
                    for name in totals:totals[name]+=diag[name]
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
                if torch.cuda.memory_reserved()>7*1024**3:raise RuntimeError('process GPU cap exceeded')
            if any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('nonfinite trained parameters')
            event={'epoch':epoch,'optimizer_steps':(len(order)+7)//8,'cumulative_optimizer_steps':steps,
                'microsteps':len(order),'order_sha256':sha256_object(order),
                'means':{k:v/len(order) for k,v in totals.items()},'epoch_seconds':time.monotonic()-epoch_start,
                'elapsed_seconds':time.monotonic()-started,'memory':check_gpu_budget(torch),'held_out_metrics_used':False}
            write_json_new(destination/f'epoch-{epoch:02d}.json',event);history.append(event)
            print(arm,held,'epoch',epoch,'loss',round(event['means']['sequence_loss_per_reference'],6),'seconds',round(event['epoch_seconds'],2),flush=True)
        if steps!=frozen_schedule['optimizer_steps'] or versions!=[r['features']._version for r in rows]+[r['prior']._version for r in rows]:
            raise ValueError('update budget/cache mutation failure')
        checkpoint=destination/'final.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(checkpoint))
        result=evaluate(torch,head,arm,heldrows,spec,freeze,destination/'validation')
        payload={'complete':True,'arm':arm,'held_song':held,'seed':17,'selected_epoch':20,'history':history,
            'optimizer_steps':steps,'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),
            'head_state_sha256':parameter_digest(head),'validation':result,'schedule_sha256':frozen_schedule['sha256'],
            'freeze_sha256':sha256_file(run/'FREEZE.json'),'frozen_cache_unchanged':True,'encoder_loaded_during_training':False,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-started,
            'development_opened':False,'final_test_opened':False,'default_replaced':False}
    write_json_new(destination/'RESULT.json',payload);print('COMPLETE',arm,held,result['original']['edit_distance'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=['H','E'],required=True);p.add_argument('--held',required=True)
    a=p.parse_args();train(a.run.resolve(),a.arm,a.held)

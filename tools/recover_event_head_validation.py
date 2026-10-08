"""Read-only completion of a checkpoint saved after all 20 E epochs."""
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file
from any2xsampa.training.event_head_protocol import load_protocol,load_cache
from any2xsampa.training.event_head_training import make_head
from input_native_probe_common import parameter_digest,compare_logits
from event_head_validation_fast import evaluate_fast


def recover(run):
    freeze,spec,index=load_protocol(run,'CODE_REVIEW_recovery.json')
    incident=read_json(run/'RECOVERY_INCIDENT.json')
    for p,h in incident['source_bindings'].items():verify_file(Path(p),h)
    job=run/'screen/E/freedom-flower';output=run/'evaluation-recovery'
    if (job/'RESULT.json').exists() or output.exists():raise ValueError('recovery already performed or attempted')
    if incident['optimizer_steps_complete']!=440 or incident['optimizer_epochs_complete']!=20:raise ValueError('not an optimization-complete checkpoint')
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda,check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import query_total_gpu_bytes,ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from safetensors.torch import load_file
    from safetensors.numpy import load_file as load_np
    enforce_offline(run/'recovery-runtime-cache');probes=[query_total_gpu_bytes() for _ in range(3)]
    if max(probes)>14*1024**3:raise RuntimeError('card over limit before recovery')
    write_json_new(output/'START.json',{'cached_inference_only':True,'optimizer_steps':0,'quality_training_start':False,'gpu_queries':probes,
        'checkpoint_sha256':incident['checkpoint_sha256'],'review_sha256':sha256_file(run/'CODE_REVIEW_recovery.json')})
    t=time.monotonic()
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=1500):
        torch=configure_cuda();rows=load_cache(torch,index['rows']);held=[r for r in rows if r['song_id']=='freedom-flower']
        state={k:v.cuda() for k,v in load_file(freeze['pretrained_head']['path'],device='cpu').items()};head=make_head(torch,state,'E').eval()
        verify_file(job/'final.safetensors',incident['checkpoint_sha256']);head.load_state_dict(load_file(str(job/'final.safetensors'),device='cuda'),strict=True)
        before=parameter_digest(head);result=evaluate_fast(torch,head,'E',held,spec,freeze,output/'validation')
        partial=[];vocab=read_json(Path(freeze['vocabulary']['path']))
        for detail in result['details']:
            prior=job/'validation'/detail['example_id']/'events.json'
            if not prior.exists():continue
            old=read_json(prior);new=read_json(Path(detail['event_path']))
            if old['head_sha256']!=before or old['decoded_event_ids']!=new['decoded_event_ids']:raise ValueError('saved checkpoint/partial path differs')
            a=load_np(str(prior.parent/old['scores']['file']));b=load_np(detail['scores_path'])
            check=compare_logits(a['frame_log_probs'],b['frame_log_probs'],vocab)
            error=float(abs(a['boundary_logits']-b['boundary_logits']).max())
            if error>2e-5:raise ValueError('partial boundary evidence differs')
            partial.append({'example_id':detail['example_id'],'content':check,'boundary_error':error})
        if parameter_digest(head)!=before:raise ValueError('recovery modified checkpoint')
        history=[read_json(job/f'epoch-{i:02d}.json') for i in range(1,21)]
        evidence={'complete':True,'checkpoint_unchanged':True,'partial_evidence_checks':partial,
            'checkpoint_sha256':incident['checkpoint_sha256'],'head_state_sha256':before,'validation':result,
            'optimizer_steps':0,'quality_training_start':False,'cached_head_calls':len(held),
            'elapsed_seconds':time.monotonic()-t,'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    write_json_new(output/'RESULT.json',evidence)
    # This RESULT did not exist before; original START, epochs, checkpoint,
    # partial outputs, failed command/exit and failure record remain untouched.
    schedule=read_json(run/'SCHEDULES.json')['freedom-flower']
    write_json_new(job/'RESULT.json',{'complete':True,'arm':'E','held_song':'freedom-flower','seed':17,'selected_epoch':20,
        'history':history,'optimizer_steps':440,'checkpoint_path':str(job/'final.safetensors'),'checkpoint_sha256':incident['checkpoint_sha256'],
        'head_state_sha256':before,'validation':result,'schedule_sha256':schedule['sha256'],'freeze_sha256':sha256_file(run/'FREEZE.json'),
        'frozen_cache_unchanged':True,'cache_mutation_guard_passed_before_saved_checkpoint':True,'encoder_loaded_during_training':False,
        'peak_reserved_bytes':max(e['memory']['reserved_bytes'] for e in history),
        'elapsed_seconds':1500.,'optimizer_completion_seconds':history[-1]['elapsed_seconds'],
        'quality_process_exit_code':73,'quality_job_outcome':'20_epochs_saved_then_readonly_validation_recovered',
        'evaluation_recovery_path':str(output/'RESULT.json'),'evaluation_recovery_sha256':sha256_file(output/'RESULT.json'),
        'quality_training_restarted':False,'development_opened':False,'final_test_opened':False,'default_replaced':False})
    print('E CHECKPOINT EVALUATION RECOVERED',result['original']['edit_distance'],'errors; no optimizer step',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);recover(p.parse_args().run.resolve())

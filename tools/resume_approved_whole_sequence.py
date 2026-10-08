"""Resume the remaining whole-source experiment, preserving every failed attempt."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from record_whole_sequence_command import execute


def freeze_code(run:Path)->None:
    paths=[ROOT/'src/any2xsampa/training'/n for n in (
        'whole_sequence_supervision.py','whole_sequence_cache.py','whole_sequence_training.py',
        'whole_sequence_review.py','singing_domain_training.py','singing_domain_losses.py','singing_domain.py',
        'protected_adaptation.py','limited_data_artifacts.py','phone_error_metrics.py')]
    paths += [ROOT/'src/any2xsampa/adapters'/n for n in (
        'whole_sequence_inference.py','singing_domain_runtime.py','sequence_route_inference.py',
        'phoneticxeus_runtime.py','phoneticxeus_monitor.py','phoneticxeus.py','locked_audio_cache.py')]
    paths += [ROOT/'tools'/n for n in ('prepare_approved_whole_sequence.py','cache_approved_whole_sequence.py',
        'probe_approved_whole_sequence.py','train_approved_whole_sequence.py','run_approved_whole_sequence.py',
        'verify_approved_whole_sequence.py','evaluate_approved_whole_sequence.py',
        'record_whole_sequence_command.py','resume_approved_whole_sequence.py',
        'run_sequence_route_stages.py','evaluate_sequence_routes.py')]
    target=run/'CODE_FREEZE.json'
    if target.exists():
        for path,digest in read_json(target)['files'].items():verify_file(Path(path),digest)
    else:write_json_new(target,{'files':{str(p.resolve()):sha256_file(p) for p in paths},
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'before_quality_training':True})


def resource_precheck(run:Path)->bool:
    from any2xsampa.adapters.phoneticxeus_monitor import query_total_gpu_bytes
    used=query_total_gpu_bytes();limit=14*2**30;room=7*2**30
    status={'at_utc':datetime.now(timezone.utc).isoformat(),'total_used_bytes':used,
        'card_limit_bytes':limit,'required_headroom_for_maximum_own_reserved_bytes':room,
        'passed':used<=limit-room,'gpu_model_started':False,'other_processes_modified':False}
    folder=run/'resource-prechecks';folder.mkdir(exist_ok=True)
    path=folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    write_json_new(path,status)
    print('GPU PRECHECK',round(used/2**30,3),'GiB used; need <=7 GiB for the reserved 7 GiB job budget',flush=True)
    return status['passed']


def resume(run:Path)->int:
    freeze_code(run)
    if not resource_precheck(run):
        print('WAITING FOR GPU HEADROOM. No new model process started.',flush=True);return 2
    def call(name:str,tool:str,args:list[str],*,cpu:bool=False,timeout:int=1560)->None:
        if execute(run,name,ROOT/'tools'/tool,['--run',str(run),*args],cpu=cpu,timeout=timeout):
            raise RuntimeError('stage stopped; inspect its recorded evidence: '+name)
    if not (run/'SMOKE_ACCEPTED.json').exists():
        attempt=1
        while (run/('whole-smoke'+('' if attempt==1 else '-attempt'+str(attempt)))).exists():
            old=run/('whole-smoke'+('' if attempt==1 else '-attempt'+str(attempt)))
            failure=old/'monitor/resource-failure.json'
            if not failure.exists() or read_json(failure)['reason']!='total_gpu_memory_limit':
                raise RuntimeError('incomplete non-resource smoke needs inspection: '+str(old))
            attempt+=1
        call('smoke-attempt'+str(attempt),'cache_approved_whole_sequence.py',['--stage','smoke','--attempt',str(attempt)])
    accepted=read_json(run/'SMOKE_ACCEPTED.json');verify_file(Path(accepted['result_path']),accepted['result_sha256'])
    if not (run/'WHOLE_CACHE.json').exists():call('cache','cache_approved_whole_sequence.py',['--stage','cache'])
    if not (run/'whole-cuda-probe/RESULT.json').exists():call('whole-probe','probe_approved_whole_sequence.py',[])
    if not (run/'ROUTE_SELECTION.json').exists():
        call('screen-stage','run_approved_whole_sequence.py',['--stage','screen'],cpu=True,timeout=5*1560+60)
    if not (run/'checkpoint-verification-screen/RESULT.json').exists():
        call('verify-screen','verify_approved_whole_sequence.py',['--stage','screen'])
    if not (run/'FINAL_TRAINING.json').exists() and not (run/'FINAL_TRAINING_NOT_STARTED.json').exists():
        call('final-stage','run_approved_whole_sequence.py',['--stage','final'],cpu=True,timeout=3*1560+60)
    if (run/'FINAL_TRAINING.json').exists():
        if not (run/'checkpoint-verification-final/RESULT.json').exists():
            call('verify-final','verify_approved_whole_sequence.py',['--stage','final'])
        if not (run/'DEVELOPMENT_SELECTION.json').exists():call('development','evaluate_approved_whole_sequence.py',[])
    print('Completed all authorized whole-sequence stages; weights remain opt-in.',flush=True)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--freeze-only',action='store_true')
    a=p.parse_args()
    if a.freeze_only:freeze_code(a.run.resolve())
    else:sys.exit(resume(a.run.resolve()))

"""Continue only four unstarted E folds after read-only validation recovery."""
from pathlib import Path
import argparse,sys,time,subprocess,os
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file
from any2xsampa.training.event_head_protocol import load_protocol,check_job


def continue_screen(run):
    freeze,spec,_=load_protocol(run,'CODE_REVIEW_training_v2.json');incident=read_json(run/'RECOVERY_INCIDENT.json')
    recovery=read_json(run/'evaluation-recovery/RESULT.json')
    if not recovery['complete'] or recovery['optimizer_steps']!=0 or read_json(run/'recovery-exit.json')['exit_code']!=0:
        raise ValueError('checkpoint recovery must pass first')
    first=read_json(run/'screen/E/freedom-flower/RESULT.json')
    if first['quality_training_restarted'] is not False or first['quality_process_exit_code']!=73:raise ValueError('incident history obscured')
    from any2xsampa.adapters.phoneticxeus_monitor import query_total_gpu_bytes
    for key in incident['remaining_quality_jobs']:
        arm,song=key.split('--',1)
        if arm!='E' or song=='freedom-flower':raise ValueError('unplanned continuation job')
        destination=run/'screen'/arm/song
        if (destination/'RESULT.json').exists():
            d=read_json(destination/'RESULT.json')
            if not d['complete'] or d['freeze_sha256']!=sha256_file(run/'FREEZE.json'):raise ValueError('completed continuation job differs')
            verify_file(Path(d['checkpoint_path']),d['checkpoint_sha256']);continue
        check_job(run,arm,song,spec)
        probes=[query_total_gpu_bytes() for _ in range(3)]
        if max(probes)>14*1024**3:raise RuntimeError('card over limit; no quality process started')
        command=[freeze['python'],'-X','utf8',str(ROOT/'tools/train_event_head_comparison_v2.py'),'--run',str(run),'--held',song]
        write_json_new(run/'commands'/(key+'.command.json'),{'command':command,'gpu_preflight_bytes':probes,
            'quality_launch_counts_even_on_failure':True,'review_sha256':sha256_file(run/'CODE_REVIEW_training_v2.json'),
            'optimizer_loop_unchanged':True,'independent_codec_verification_deferred':True})
        t=time.monotonic();code=None;error=None
        try:
            with (run/'commands'/(key+'.log')).open('x',encoding='utf-8') as f:
                p=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1550,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                    env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
            code=p.returncode
        except Exception as exc:error=type(exc).__name__+': '+str(exc);raise
        finally:write_json_new(run/'commands'/(key+'.exit.json'),{'exit_code':code,'error':error,'elapsed_seconds':time.monotonic()-t})
        if code:raise RuntimeError(f'{key} failed ({code}); no retry')
        print('UNSTARTED FOLD COMPLETED',key,'seconds',round(time.monotonic()-t,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);continue_screen(p.parse_args().run.resolve())

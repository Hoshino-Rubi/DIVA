"""Serial ten-job screen; finished jobs are verified, never silently retried."""
from pathlib import Path
import argparse,sys,time,subprocess,os
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,verify_file
from any2xsampa.training.event_head_protocol import load_protocol,check_job


def launch(run):
    freeze,spec,_=load_protocol(run,'CODE_REVIEW_training.json')
    from any2xsampa.adapters.phoneticxeus_monitor import query_total_gpu_bytes
    for arm in spec['arms']:
        for held in spec['songs']:
            key=arm+'--'+held;done=run/'screen'/arm/held/'RESULT.json'
            if done.exists():
                old=read_json(done)
                if not old['complete'] or old['freeze_sha256']!=sha256_file(run/'FREEZE.json'):raise ValueError('completed job identity differs')
                verify_file(Path(old['checkpoint_path']),old['checkpoint_sha256']);continue
            check_job(run,arm,held,spec)
            probes=[query_total_gpu_bytes() for _ in range(3)]
            if max(probes)>14*1024**3:raise RuntimeError('card over limit; no worker launched')
            command=[freeze['python'],'-X','utf8',str(ROOT/'tools/train_event_head_comparison.py'),'--run',str(run),'--arm',arm,'--held',held]
            write_json_new(run/'commands'/(key+'.command.json'),{'command':command,'gpu_preflight_bytes':probes,'quality_launch_counts_even_on_failure':True,
                'review_sha256':sha256_file(run/'CODE_REVIEW_training.json')})
            t=time.monotonic();code=None;error=None
            try:
                with (run/'commands'/(key+'.log')).open('x',encoding='utf-8') as log:
                    p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1550,
                        env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                code=p.returncode
            except Exception as exc:error=type(exc).__name__+': '+str(exc);raise
            finally:write_json_new(run/'commands'/(key+'.exit.json'),{'exit_code':code,'error':error,'elapsed_seconds':time.monotonic()-t})
            if code:raise RuntimeError(f'{key} failed ({code}), group stopped; no retry')
            print('JOB COMPLETE',arm,held,'seconds',round(time.monotonic()-t,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);launch(p.parse_args().run.resolve())

"""Single-shot sequence: no parallel CUDA, no retries, explicit exit accounting."""
from __future__ import annotations
import argparse
from pathlib import Path
import os,subprocess,time
from acoustic_memory_common import ROOT,protocol,write_json_new,read_json


def execute(run: Path) -> None:
    freeze=protocol(run,review=True)
    write_json_new(run/'PIPELINE_START.json',{'automatic_retry':False,'phases':['execute','replay']})
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def call(key: str,args: list[str],timeout: float=1800) -> None:
        cmd=[freeze['python'],'-X','utf8',*args]
        write_json_new(run/'commands'/(key+'.command.json'),{'command':cmd,'timeout_seconds':timeout})
        before=time.monotonic()
        with (run/'commands'/(key+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill();process.wait();code=124
        write_json_new(run/'commands'/(key+'.exit.json'),{'exit_code':code,'elapsed_seconds':time.monotonic()-before})
        print(key,'exit',code,flush=True)
        if code:raise RuntimeError(key+' failed; no automatic retry')
    for phase in ('execute','replay'):
        call(phase+'-launcher',[str(ROOT/'tools/launch_acoustic_memory.py'),'--run',str(run.resolve()),'--phase',phase],1600)
        for stage in ('baseline','bank','query','score'):
            call(phase+'-'+stage,[str(ROOT/'tools/acoustic_memory_cpu.py'),'--run',str(run.resolve()),'--phase',phase,'--stage',stage],1530)
    call('verify',[str(ROOT/'tools/verify_acoustic_memory.py'),'--run',str(run.resolve())],1500)
    write_json_new(run/'PIPELINE_COMPLETE.json',{'complete':True,'regression_and_final_report_pending':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    try:execute(a.run)
    except Exception as exc:
        write_json_new(a.run/'PIPELINE_FAILURE.json',{'error':str(exc),'automatic_retry':False});raise

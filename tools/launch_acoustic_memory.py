"""Exactly two non-retryable cache-worker launch slots, no budget transfer."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import subprocess
import time
from acoustic_memory_common import ROOT, protocol, read_json, write_json_new, sha256_file


def launch(run: Path, phase: str) -> int:
    freeze=protocol(run,review=True)
    if phase not in ('execute','replay'):raise ValueError('unknown slot')
    command_path=run/'commands'/(phase+'.json')
    if command_path.exists() or (run/phase/'head/START.json').exists():raise ValueError('slot already used; no automatic retry')
    if phase=='replay' and (not (run/'commands/execute.exit.json').exists() or read_json(run/'commands/execute.exit.json')['exit_code']!=0):
        raise ValueError('execution head not complete')
    worker=ROOT/'tools/run_acoustic_memory_head.py'
    command=[freeze['python'],'-X','utf8',str(worker),'--run',str(run.resolve()),'--phase',phase]
    write_json_new(command_path,{'command':command,'phase':phase,'worker':str(worker.resolve()),
        'worker_sha256':sha256_file(worker),'review_sha256':sha256_file(run/'CODE_REVIEW.json'),
        'head_request_cap':229,'retry':False})
    env=dict(os.environ, PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    before=time.monotonic();code=-1;reason=None
    with (run/'commands'/(phase+'.log')).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write_json_new(run/'commands'/(phase+'.pid.json'),{'pid':process.pid})
        try:code=process.wait(timeout=1530)
        except subprocess.TimeoutExpired:
            process.kill();code=process.wait();reason='launcher_timeout_only_own_child_killed'
    write_json_new(run/'commands'/(phase+'.exit.json'),{'exit_code':code,'elapsed_seconds':time.monotonic()-before,
        'reason':reason,'automatic_retry':False,'command_sha256':sha256_file(command_path)})
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=('execute','replay'),required=True)
    a=p.parse_args();raise SystemExit(launch(a.run,a.phase))

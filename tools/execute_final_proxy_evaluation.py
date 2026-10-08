"""Resume the fixed final comparison serially; preserve blocked attempts."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.final_proxy_evaluation import verify_release
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,write_json_new
RUNTIME=ROOT/'.state/runs/limited-data-20260906T050251/runtime/Scripts/python.exe'


def execute(run:Path)->int:
    gate=run/'RELEASE.json';digest=sha256_file(gate);verify_release(gate,digest)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt=run/'attempts'/stamp;attempt.mkdir(parents=True)
    jobs=[]
    def job(name:str,args:list[str])->int:
        command=[str(RUNTIME),'-X','utf8',*args]
        start=datetime.now(timezone.utc).isoformat();before=time.monotonic()
        print('running',name,flush=True)
        with (attempt/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False,
                env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        entry=dict(name=name,command=command,started_utc=start,ended_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.monotonic()-before,exit_code=result.returncode,log=str(attempt/(name+'.log')))
        jobs.append(entry);write_json_new(attempt/(name+'.command.json'),entry)
        print('finished',name,'exit',result.returncode,flush=True)
        return result.returncode
    common=['--gate',str(gate),'--gate-sha256',digest]
    complete=run/'TEST_COMPLETE.json'
    if not complete.exists():
        if (run/'TEST_OPENING.json').exists():raise RuntimeError('prior opening is incomplete; retain evidence for recovery, do not silently repeat')
        output=attempt/'inference'
        rc=job('inference',['tools/run_final_proxy_evaluation.py',*common,'--output',str(output)])
        if rc:
            blocked=(output/'BLOCKED.json').exists()
            write_json_new(attempt/'ATTEMPT.json',dict(status='resource_blocked' if blocked else 'failed',jobs=jobs,
                test_opened=(run/'TEST_OPENING.json').exists(),resource_block=read_json(output/'BLOCKED.json') if blocked else None))
            print('attempt',attempt,flush=True);return 2 if blocked else rc
    evidence=Path(read_json(complete)['evidence']['path'])
    replay=attempt/'independent-replay'
    rc=job('independent-replay',['tools/run_final_proxy_evaluation.py',*common,'--output',str(replay),'--replay',str(evidence)])
    if rc:
        write_json_new(attempt/'ATTEMPT.json',dict(status='replay_failed_or_blocked',jobs=jobs,test_opened=True));return rc
    if (run/'COMPARISON.json').exists():raise RuntimeError('comparison already recorded; no repeat selection or scoring')
    rc=job('score',['tools/score_final_proxy_evaluation.py',*common,'--replay',str(replay/'REPLAY.json'),'--output',str(run)])
    if not rc:rc=job('metric-replay',['tools/score_final_proxy_evaluation.py',*common,'--replay',str(replay/'REPLAY.json'),'--output',str(attempt/'metric-replay'),'--verify',str(run/'COMPARISON.json')])
    write_json_new(attempt/'ATTEMPT.json',dict(status='complete' if not rc else 'score_failed',jobs=jobs,test_opened=True))
    print('attempt',attempt,flush=True)
    return rc


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve()
    if not run.is_relative_to(ROOT/'.state/runs'):raise ValueError('workspace run required')
    if (run/'COMPARISON.json').exists():raise ValueError('final comparison already completed; use report')
    raise SystemExit(execute(run))

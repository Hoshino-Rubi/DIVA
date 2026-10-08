"""Finish authorized stages serially; no minimum duration or idle loop."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new


def execute(run:Path,name:str,command:list[str],*,timeout:int)->None:
    folder=run/'completion-commands';folder.mkdir(exist_ok=True)
    exit_path=folder/(name+'.exit.json')
    if exit_path.exists():
        if read_json(exit_path)['exit_code']!=0:raise RuntimeError('previous completion stage failed: '+name)
        return
    write_json_new(folder/(name+'.command.json'),{'argv':command,'started_at_utc':datetime.now(timezone.utc).isoformat()})
    before=time.monotonic();print('START',name,flush=True)
    with (folder/(name+'.log')).open('x',encoding='utf-8') as stream:
        child=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,
            env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    write_json_new(exit_path,{'exit_code':child.returncode,'elapsed_seconds':time.monotonic()-before})
    if child.returncode:raise RuntimeError('completion stage failed: '+name)
    print('DONE',name,flush=True)


def finish(run:Path)->None:
    selection=read_json(run/'ROUTE_SELECTION.json');freeze=read_json(run/'FREEZE.json')
    runtime=freeze['training_python'];ordinary=str(ROOT/'.venv/Scripts/python.exe')
    command=lambda python,tool,*args:[python,'-X','utf8','-B',str(ROOT/'tools'/tool),'--run',str(run),*args]
    execute(run,'screen-checkpoints',command(runtime,'verify_sequence_route_checkpoints.py','--stage','screen'),timeout=1000)
    execute(run,'conditional-final',command(ordinary,'run_sequence_route_stages.py','--stage','final'),timeout=4900)
    if selection['passed']:
        execute(run,'final-checkpoints',command(runtime,'verify_sequence_route_checkpoints.py','--stage','final'),timeout=1000)
        execute(run,'development',command(runtime,'evaluate_sequence_routes.py'),timeout=1000)
    execute(run,'review-node-tests',['node','--test',str(ROOT/'tests/whole_sequence_review.test.cjs')],timeout=60)
    execute(run,'full-regression',[ordinary,'-X','utf8','-B','-m','pytest','-q','-p','no:cacheprovider',
        '--basetemp',str(run/'regression-temp'),'--junitxml',str(run/'regression.xml')],timeout=600)
    write_json_new(run/'COMPLETION_STAGES.json',{'screen_verified':True,'conditional_final_training':selection['passed'],
        'regression_command_completed':True,'no_active_model_job':True,'new_default_deployed':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    finish(parser.parse_args().run.resolve())

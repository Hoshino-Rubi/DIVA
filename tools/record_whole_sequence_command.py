"""Record an isolated subprocess's actual exit status and reproducible argv."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new


def execute(run:Path,name:str,script:Path,arguments:list[str],*,cpu:bool=False,timeout:int=1560)->int:
    directory=run/'commands';directory.mkdir(exist_ok=True)
    python=sys.executable if cpu else read_json(run/'FREEZE.json')['training_python']
    command=[python,'-X','utf8','-B',str(script.resolve()),*arguments]
    write_json_new(directory/(name+'.command.json'),{'argv':command,'cwd':str(ROOT),'started_at_utc':datetime.now(timezone.utc).isoformat()})
    before=time.monotonic();print('START',name,flush=True);timed_out=False
    with (directory/(name+'.log')).open('x',encoding='utf-8') as stream:
        try:
            child=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'})
            code=child.returncode
        except subprocess.TimeoutExpired:
            code=-1;timed_out=True
    write_json_new(directory/(name+'.exit.json'),{'exit_code':code,'timeout':timed_out,'elapsed_seconds':time.monotonic()-before})
    print('DONE',name,'exit',code,flush=True);return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--cpu',action='store_true');p.add_argument('--timeout',type=int,default=1560)
    p.add_argument('script',type=Path);p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();sys.exit(execute(a.run.resolve(),a.name,a.script,a.arguments,cpu=a.cpu,timeout=a.timeout))

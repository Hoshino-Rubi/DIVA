"""Serial post-ablation execution through verified local delivery."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,subprocess,os,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file

def finish(run:Path)->None:
    read_json(run/'RECIPE_SELECTION.json')
    py=str(ROOT/'.venv/Scripts/python.exe');gpu=read_json(run/'FREEZE.json')['runtime_python']
    def command(name:str,python:str,script:str,args:list[str],marker:Path,timeout:int=1200)->None:
        if marker.exists():
            if marker.suffix=='.json':read_json(marker)
            print('REUSE',name,flush=True);return
        argv=[python,'-X','utf8','-B',str(ROOT/script),*args]
        logdir=run/'commands';logdir.mkdir(exist_ok=True)
        write_json_new(logdir/(name+'.command.json'),{'argv':argv,'started_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha256_file(ROOT/script)})
        print('START',name,flush=True);before=time.monotonic()
        with (logdir/(name+'.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,shell=False,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONUTF8':'1'})
        write_json_new(logdir/(name+'.exit.json'),{'exit_code':result.returncode,'wall_seconds':time.monotonic()-before,'ended_at_utc':datetime.now(timezone.utc).isoformat()})
        if result.returncode or not marker.exists():raise RuntimeError(name+' did not complete; inspect its log')
        print('DONE',name,flush=True)
    common=['--run',str(run)]
    command('final-three-seeds',py,'tools/run_singing_domain_stages.py',common+['--stage','final'],run/'FINAL_TRAINING.json',5000)
    command('development-cache',gpu,'tools/cache_singing_development.py',common,run/'DEVELOPMENT_CACHE.json')
    for seed in (17,29,43):
        command('development-seed'+str(seed),gpu,'tools/evaluate_singing_domain.py',common+['--seed',str(seed)],run/'development-results'/('seed'+str(seed))/'RESULT.json')
    command('development-selection',py,'tools/evaluate_singing_domain.py',common+['--select'],run/'DEVELOPMENT_SELECTION.json')
    selected=read_json(run/'DEVELOPMENT_SELECTION.json')['research_seed']
    command('vocal-transfer-selected',gpu,'tools/evaluate_singing_domain.py',common+['--seed',str(selected),'--transfer'],run/'development-transfer'/('seed'+str(selected))/'RESULT.json')
    for seed in (17,29,43):
        job=run/'final-training'/('seed'+str(seed));out=run/'verification-final'/('seed'+str(seed))
        command('independent-seed'+str(seed),gpu,'tools/verify_singing_domain_job.py',common+['--job',str(job),'--output',str(out)],out/'VERIFICATION.json')
    command('standalone-entry',py,'tools/check_singing_inference_entry.py',common,run/'free-audio-entry-check.json')
    # Regression includes a small CUDA fixture; it runs only after all model jobs ended.
    marker=run/'REGRESSION_EXIT.json'
    if not marker.exists():
        argv=[py,'-X','utf8','-B','-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(run/'regression-temp'),'--junitxml',str(run/'regression.xml')]
        write_json_new(run/'commands/regression.command.json',{'argv':argv,'started_at_utc':datetime.now(timezone.utc).isoformat()})
        print('START full-regression',flush=True);before=time.monotonic()
        with (run/'regression.log').open('x',encoding='utf-8') as log:
            p=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900,shell=False,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
        write_json_new(marker,{'exit_code':p.returncode,'wall_seconds':time.monotonic()-before})
        if p.returncode:raise RuntimeError('regression failed; inspect regression.log')
        print('DONE full-regression',flush=True)
    elif read_json(marker)['exit_code']!=0:raise RuntimeError('previous regression failed')
    command('independent-audit',py,'tools/audit_singing_domain_results.py',common,run/'INDEPENDENT_AUDIT.json')
    command('final-report',py,'tools/report_singing_domain.py',common,run/'DELIVERY.json')
    print('LOCAL EXPERIMENTS COMPLETE',run,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);finish(p.parse_args().run.resolve())

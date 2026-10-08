"""Serial, resumable orchestration of predeclared song-grouped experiments."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,subprocess,time,json,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,verify_file
from any2xsampa.training.singing_domain import load_manifest,choice
from any2xsampa.training.phone_error_metrics import phone_metrics

def write_or_verify(path:Path,value:dict)->None:
    if path.exists():
        if read_json(path)!=value:raise ValueError('completed stage artifact differs during resume: '+str(path))
    else:write_json_new(path,value)

def execute(run:Path,arm:str,variant:str,held:str|None,seed:int,output:Path)->dict:
    done=output/'RESULT.json'
    if done.exists():
        result=read_json(done)
        if not result['complete'] or result['fixed_epoch']!=20 or result['policy_sha256']!=sha256_file(run/'POLICY.json') or result['cache_sha256']!=sha256_file(run/'CACHE.json'):
            raise ValueError('existing run is not a compatible complete job')
        if (result['arm'],result['variant'],result['held_song'],result['seed'])!=(arm,variant,held,seed):raise ValueError('job identity mismatch')
        verify_file(Path(result['checkpoint_path']),result['checkpoint_sha256'])
        return result
    if output.exists():raise RuntimeError(f'incomplete job requires inspection before continuing: {output}')
    python=read_json(run/'FREEZE.json')['runtime_python']
    command=[python,'-X','utf8','-B',str(ROOT/'tools/train_singing_domain.py'),'--run',str(run),'--arm',arm,'--variant',variant,'--seed',str(seed),'--output',str(output)]
    if held is not None:command+=['--held-song',held]
    logdir=run/'commands';logdir.mkdir(exist_ok=True)
    name=arm+'-'+variant+'-'+(held or 'all')+'-seed'+str(seed)
    write_json_new(logdir/(name+'.command.json'),{'argv':command,'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'policy_sha256':sha256_file(run/'POLICY.json'),'worker_sha256':sha256_file(ROOT/'tools/train_singing_domain.py')})
    before=time.monotonic();print('START',name,flush=True)
    with (logdir/(name+'.log')).open('x',encoding='utf-8') as log:
        process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,shell=False,timeout=1560,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'})
    write_json_new(logdir/(name+'.exit.json'),{'exit_code':process.returncode,'wall_seconds':time.monotonic()-before,'ended_at_utc':datetime.now(timezone.utc).isoformat()})
    if process.returncode:raise RuntimeError(f'{name} failed; see {logdir/(name+".log")}')
    result=read_json(done)
    print('DONE',name,'PER',result['final_validation']['metrics']['per'] if result['final_validation'] else '-',flush=True)
    return result

def summarize(results:list[dict],name:str)->dict:
    pairs=[];details=[]
    for result in results:
        assert result['complete'] and result['held_song'] is not None and result['fixed_epoch']==20
        for row in result['final_validation']['examples']:
            pairs.append((row['reference'],row['prediction']));details.append(row)
    metric=phone_metrics(pairs)
    return {'name':name,'complete':True,'fold_songs':[r['held_song'] for r in results],
        'reference_phones':metric['reference_phones'],'per':metric['per'],'trainable_parameters':results[0]['trainable_parameters'],
        'metrics':metric,'examples':details,'folds':[{'held_song':r['held_song'],'metrics':r['final_validation']['metrics'],
            'checkpoint_path':r['checkpoint_path'],'checkpoint_sha256':r['checkpoint_sha256'],'elapsed_seconds':r['elapsed_seconds']} for r in results]}

def run_stage(run:Path,stage:str)->None:
    spec,_=load_manifest(run)
    if stage=='architecture':
        summaries=[]
        for arm in spec['architecture_arms']:
            results=[execute(run,arm,'baseline',song,spec['training']['screen_seed'],run/'architecture'/arm/song) for song in spec['songs']]
            summary=summarize(results,arm);summaries.append(summary)
            write_or_verify(run/(arm+'-ARCHITECTURE.json'),summary)
        winner=choice(summaries)
        write_or_verify(run/'ARCHITECTURE_SELECTION.json',{'scope':'training_song_cross_validation_only','method':spec['selection'],
            'arms':summaries,'selected_arm':winner['name'],'dev_or_test_used':False})
        print('SELECTED ARCHITECTURE',winner['name'],winner['per'],flush=True)
    elif stage=='ablations':
        selected=read_json(run/'ARCHITECTURE_SELECTION.json');arm=selected['selected_arm']
        baseline=next(r for r in selected['arms'] if r['name']==arm);summaries=[{**baseline,'name':'baseline'}]
        for variant in spec['ablations']['variants']:
            results=[execute(run,arm,variant,song,spec['training']['screen_seed'],run/'ablations'/variant/song) for song in spec['songs']]
            summary=summarize(results,variant);summaries.append(summary);write_or_verify(run/(variant+'-ABLATION.json'),summary)
        winner=choice(summaries)
        write_or_verify(run/'RECIPE_SELECTION.json',{'selected_arm':arm,'selected_variant':winner['name'],
            'variants':summaries,'method':spec['selection'],'dev_or_test_used':False})
        print('SELECTED RECIPE',arm,winner['name'],winner['per'],flush=True)
    else:
        recipe=read_json(run/'RECIPE_SELECTION.json');arm=recipe['selected_arm'];variant=recipe['selected_variant'];results=[]
        for seed in spec['training']['final_seeds']:
            results.append(execute(run,arm,variant,None,seed,run/'final-training'/('seed'+str(seed))))
        write_or_verify(run/'FINAL_TRAINING.json',{'selected_arm':arm,'selected_variant':variant,
            'seeds':[{'seed':r['seed'],'path':r['checkpoint_path'],'sha256':r['checkpoint_sha256'],'result_path':str(Path(r['checkpoint_path']).parent/'RESULT.json')} for r in results],
            'dev_used_for_training_or_early_stopping':False,'default_replacement':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['architecture','ablations','final'],required=True);a=p.parse_args();run_stage(a.run.resolve(),a.stage)

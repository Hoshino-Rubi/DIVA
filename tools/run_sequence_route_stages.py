"""Serial/resumable execution with fixed-epoch, training-song-only selection."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from any2xsampa.training.phone_error_metrics import phone_metrics


def write_or_verify(path:Path,value:dict)->None:
    if path.exists():
        if read_json(path)!=value:raise ValueError('completed stage changed during resume: '+str(path))
    else:write_json_new(path,value)


def execute(run:Path,arm:str,held:str|None,seed:int)->dict:
    name=arm+'-'+(held or 'all')+'-seed'+str(seed)
    output=run/('screen' if held else 'final')/arm/(held or ('seed'+str(seed)))
    result_path=output/'RESULT.json'
    if result_path.exists():
        result=read_json(result_path)
        if not result['complete'] or (result['arm'],result['held_song'],result['seed'])!=(arm,held,seed):
            raise ValueError('incompatible existing result')
        verify_file(run/'POLICY.json',result['policy_sha256'])
        verify_file(Path(result['checkpoint_path']),result['checkpoint_sha256'])
        for p,digest in result['code_bindings'].items():verify_file(Path(p),digest)
        return result
    if output.exists():raise RuntimeError('incomplete job requires inspection: '+str(output))
    freeze=read_json(run/'FREEZE.json')
    command=[freeze['training_python'],'-X','utf8','-B',str(ROOT/'tools/train_sequence_route.py'),
        '--run',str(run),'--arm',arm,'--seed',str(seed),'--output',str(output)]
    if held:command+=['--held-song',held]
    commands=run/'commands';commands.mkdir(exist_ok=True)
    write_json_new(commands/(name+'.command.json'),{'argv':command,'started_at_utc':datetime.now(timezone.utc).isoformat()})
    before=time.monotonic();print('START',name,flush=True)
    with (commands/(name+'.log')).open('x',encoding='utf-8') as log:
        child=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1560,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
            env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'})
    write_json_new(commands/(name+'.exit.json'),{'exit_code':child.returncode,'elapsed_seconds':time.monotonic()-before})
    if child.returncode:raise RuntimeError('job failed; preserve partial artifacts and inspect '+name)
    result=read_json(result_path)
    print('DONE',name,'PER',result['final_validation']['metrics']['per'] if result['final_validation'] else 'not-evaluated',flush=True)
    return result


def summarize(results:list[dict])->dict:
    pairs=[];examples=[]
    for result in results:
        for row in result['final_validation']['examples']:
            pairs.append((row['reference'],row['prediction']));examples.append(row)
    return {'metrics':phone_metrics(pairs),'examples':examples,'folds':[
        {'held_song':r['held_song'],'metrics':r['final_validation']['metrics'],
         'optimizer_steps':r['optimizer_steps'],'checkpoint_path':r['checkpoint_path'],
         'checkpoint_sha256':r['checkpoint_sha256'],'elapsed_seconds':r['elapsed_seconds']} for r in results]}


def stages(run:Path,stage:str)->None:
    spec=read_json(run/'POLICY.json');freeze=read_json(run/'FREEZE.json')
    parent=Path(freeze['parents']['singing'])
    if stage=='screen':
        probe=read_json(run/'risk-cuda-probe/RESULT.json')
        if not probe['gradients_finite'] or not probe['frozen_parameters_unchanged']:raise ValueError('risk probe did not pass')
        whole=read_json(run/'WHOLE_SEQUENCE_AUDIT.json')
        if whole['eligible']:raise RuntimeError('whole sequence route needs its accepted cache before matched screening')
        write_or_verify(run/'WHOLE_SEQUENCE_TRAINING_BLOCKED.json',{'status':'data_gate_not_passed',
            'audit_sha256':sha256_file(run/'WHOLE_SEQUENCE_AUDIT.json'),'whole_quality_jobs_started':0,
            'loss_implementation':'sequence_risk.whole_sequence_ctc; fixture gradients/repeats/gaps tested',
            'no_substitute_rendered_training_started':True})
        results={}
        for arm in ('control','sequence_risk'):
            jobs=[execute(run,arm,song,17) for song in spec['songs']]
            results[arm]=summarize(jobs)
            write_or_verify(run/(arm+'-SCREEN.json'),results[arm])
        for left,right in zip(results['control']['folds'],results['sequence_risk']['folds']):
            if left['held_song']!=right['held_song'] or left['optimizer_steps']!=right['optimizer_steps']:
                raise ValueError('optimizer budgets differ between matched folds')
        prior=read_json(parent/'full-ARCHITECTURE.json')['metrics'];selection=spec['selection'];gates={}
        for arm,summary in results.items():
            metric=summary['metrics']
            if metric['reference_phones']!=1235 or metric['reference_phones']!=prior['reference_phones']:
                raise ValueError('cross-song denominator changed')
            gain=1-metric['per']/prior['per']
            regressions=[g for g,p in prior['groups'].items() if p['reference_count']>=selection['protected_group_min_references']
                and metric['groups'][g]['per']>p['per']+selection['maximum_group_PER_increase']]
            gates[arm]={'relative_gain_against_prior_folds':gain,'protected_regressions':regressions,
                'passed':gain>=selection['minimum_relative_PER_gain_against_prior_folds'] and not regressions}
        qualified=[a for a in results if gates[a]['passed']]
        winner=min(qualified,key=lambda a:(results[a]['metrics']['per'],results[a]['metrics']['insertions'],a)) if qualified else None
        write_or_verify(run/'ROUTE_SELECTION.json',{'passed':winner is not None,'selected_arm':winner,
            'arms':results,'gates':gates,'prior_full_folds':prior,'screen_quality_jobs_completed':10,
            'whole_sequence_quality_jobs_started':0,'development_used_for_selection':False,
            'final_test_predictions_opened':False,'default_replacement':False,
            'policy_sha256':sha256_file(run/'POLICY.json')})
        print('SELECTED',winner,gates,flush=True)
    else:
        selection=read_json(run/'ROUTE_SELECTION.json')
        if not selection['passed']:
            write_or_verify(run/'FINAL_TRAINING_NOT_STARTED.json',{'reason':'cross_song_gain_or_protection_gate_failed',
                'selection_sha256':sha256_file(run/'ROUTE_SELECTION.json'),'quality_training_jobs':0})
            print('FINAL NOT STARTED: cross-song gate failed',flush=True);return
        arm=selection['selected_arm']
        results=[execute(run,arm,None,seed) for seed in spec['training']['final_seeds']]
        write_or_verify(run/'FINAL_TRAINING.json',{'arm':arm,'seeds':[{'seed':r['seed'],
            'checkpoint_path':r['checkpoint_path'],'checkpoint_sha256':r['checkpoint_sha256']} for r in results],
            'default_replacement':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['screen','final'],required=True)
    args=parser.parse_args();stages(args.run.resolve(),args.stage)

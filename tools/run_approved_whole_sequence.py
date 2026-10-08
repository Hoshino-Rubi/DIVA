"""Five remaining whole-source folds, followed by at most three conditional final seeds."""
from pathlib import Path
import argparse,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,verify_file
from any2xsampa.adapters.whole_sequence_inference import verify_job
from record_whole_sequence_command import execute
from run_sequence_route_stages import summarize


def screening_gate(metric:dict,prior:dict,selection:dict)->dict:
    for item in (metric,prior):
        if (not math.isfinite(item['per']) or item['reference_phones'] <= 0
                or item['edit_distance'] != sum(item[k] for k in ('substitutions','deletions','insertions'))
                or abs(item['per']-item['edit_distance']/item['reference_phones']) > 1e-12):
            raise ValueError('invalid whole-screen metric arithmetic')
    if metric['groups'].keys()!=prior['groups'].keys() or any(
            metric['groups'][g]['reference_count']!=p['reference_count'] for g,p in prior['groups'].items()):
        raise ValueError('whole-screen protected-group coverage changed')
    if metric['reference_phones']!=1235 or metric['reference_phones']!=prior['reference_phones']:
        raise ValueError('whole route changed cross-song denominator')
    gain=1-metric['per']/prior['per']
    regressions=[g for g,p in prior['groups'].items() if p['reference_count']>=selection['protected_group_min_references']
        and metric['groups'][g]['per']>p['per']+selection['maximum_group_PER_increase']]
    return {'relative_gain_against_prior_folds':gain,'protected_regressions':regressions,
        'passed':gain>=selection['minimum_relative_PER_gain_against_prior_folds'] and not regressions}


def job(run:Path,held:str|None,seed:int)->dict:
    name='whole-'+(held or 'all')+'-seed'+str(seed)
    output=run/('screen' if held else 'final')/'whole_sequence'/(held or ('seed'+str(seed)))
    if (output/'RESULT.json').exists():return verify_job(run,output)
    if output.exists():raise RuntimeError('inspect incomplete whole job before resume: '+str(output))
    arguments=['--run',str(run),'--seed',str(seed),'--output',str(output)]
    if held:arguments+=['--held-song',held]
    if execute(run,name,ROOT/'tools/train_approved_whole_sequence.py',arguments):
        raise RuntimeError('whole quality job stopped: '+name)
    return verify_job(run,output)


def run_stage(run:Path,stage:str)->None:
    spec=read_json(run/'POLICY.json');freeze=read_json(run/'FREEZE.json')
    verify_file(run/'POLICY.json',freeze['policy_sha256'])
    prior_run=Path(freeze['parents']['routes']);singing=Path(freeze['parents']['singing'])
    if stage=='screen':
        results=[job(run,song,17) for song in spec['songs']]
        summary=summarize(results)
        control=read_json(prior_run/'control-SCREEN.json')
        if {r['example_id']:r['reference'] for r in summary['examples']}!={r['example_id']:r['reference'] for r in control['examples']}:
            raise ValueError('whole held windows/reference identities changed')
        prior=read_json(singing/'full-ARCHITECTURE.json')['metrics']
        gate=screening_gate(summary['metrics'],prior,spec['selection'])
        write_json_new(run/'whole_sequence-SCREEN.json',summary)
        write_json_new(run/'ROUTE_SELECTION.json',{**gate,'selected_arm':'whole_sequence' if gate['passed'] else None,
            'whole_sequence':summary,'matched_control':control,
            'prior_full_folds':prior,'new_screen_jobs':5,'previous_screen_jobs':10,
            'all_planned_screen_jobs':15,'development_used_for_selection':False,
            'final_test_predictions_opened':False,'default_replacement':False,
            'policy_sha256':sha256_file(run/'POLICY.json')})
        print('WHOLE SCREEN',summary['metrics']['edit_distance'],'/',1235,gate,flush=True)
    else:
        selection=read_json(run/'ROUTE_SELECTION.json')
        if not selection['passed']:
            write_json_new(run/'FINAL_TRAINING_NOT_STARTED.json',{'reason':'whole_cross_song_gain_or_protection_gate_failed',
                'selection_sha256':sha256_file(run/'ROUTE_SELECTION.json'),'new_final_seeds':0})
            print('FINAL NOT STARTED: whole-source gate failed',flush=True);return
        results=[job(run,None,seed) for seed in spec['training']['final_seeds']]
        write_json_new(run/'FINAL_TRAINING.json',{'arm':'whole_sequence','seeds':[{'seed':r['seed'],
            'checkpoint_path':r['checkpoint_path'],'checkpoint_sha256':r['checkpoint_sha256']} for r in results],
            'default_replacement':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['screen','final'],required=True)
    a=p.parse_args();run_stage(a.run.resolve(),a.stage)

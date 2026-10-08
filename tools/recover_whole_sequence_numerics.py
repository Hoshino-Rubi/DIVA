"""Recover one failed fold with a numerical implementation patch, preserving all attempts."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from any2xsampa.adapters.whole_sequence_inference import verify_job
from record_whole_sequence_command import execute
from run_sequence_route_stages import summarize
from run_approved_whole_sequence import screening_gate


def recover(run:Path)->None:
    spec=read_json(run/'POLICY.json');freeze=read_json(run/'FREEZE.json')
    addendum=read_json(run/'NUMERICAL_FIX_FREEZE.json')
    for p,d in addendum['files'].items():verify_file(Path(p),d)
    rows=read_json(run/'SCREEN_JOB_INDEX.json')['rows'];results=[]
    for row in rows:
        output=Path(row['job_path']);song=row['held_song']
        if not (output/'RESULT.json').exists():
            if output.exists():raise RuntimeError('inspect incomplete recovery: '+str(output))
            if row['implementation']!='two_grad_passes_v2':raise ValueError('previously completed original fold is missing')
            name='whole-v2-'+song+'-seed17'
            if execute(run,name,ROOT/'tools/train_approved_whole_sequence_v2.py',[
                '--run',str(run),'--held-song',song,'--seed','17','--output',str(output)]):
                raise RuntimeError('numerical recovery failed; preserve diagnostics: '+song)
        result=verify_job(run,output)
        if 'implementation_addendum_path' in result:
            verify_file(Path(result['implementation_addendum_path']),result['implementation_addendum_sha256'])
        results.append(result)
    summary=summarize(results)
    parent=Path(freeze['parents']['singing']);routes=Path(freeze['parents']['routes'])
    control=read_json(routes/'control-SCREEN.json');prior=read_json(parent/'full-ARCHITECTURE.json')['metrics']
    if {r['example_id']:r['reference'] for r in summary['examples']}!={r['example_id']:r['reference'] for r in control['examples']}:
        raise ValueError('recovery changed held-out reference identities')
    gate=screening_gate(summary['metrics'],prior,spec['selection'])
    write_json_new(run/'whole_sequence-SCREEN.json',summary)
    write_json_new(run/'ROUTE_SELECTION.json',{**gate,'selected_arm':'whole_sequence' if gate['passed'] else None,
        'whole_sequence':summary,'matched_control':control,'prior_full_folds':prior,
        'new_screen_completed_jobs':5,'new_screen_process_attempts':6,'failed_attempts':1,
        'previous_screen_completed_jobs':10,'all_planned_screen_completed_jobs':15,
        'all_screen_process_attempts_including_technical_failure':16,
        'technical_budget_exception':'one replay of the same failed fold; no additional recipe, seed or candidate selection',
        'screen_job_index_sha256':sha256_file(run/'SCREEN_JOB_INDEX.json'),
        'numerical_fix_sha256':sha256_file(run/'NUMERICAL_FIX_FREEZE.json'),
        'development_used_for_selection':False,'final_test_predictions_opened':False,'default_replacement':False,
        'policy_sha256':sha256_file(run/'POLICY.json')})
    print('RECOVERED SCREEN',summary['metrics']['edit_distance'],'/',1235,gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);recover(p.parse_args().run.resolve())

"""Independent CPU audit of fold boundaries, serialized metrics and data lineage."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,verify_file,write_json_new
from any2xsampa.training.singing_domain import policy

def distance(a:list,b:list)->int:
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def audit(run:Path)->None:
    spec=policy();refs={r['example_id']:r for r in read_json(run/'REFERENCES.json')['rows']}
    requests={r['example_id']:r for r in read_json(run/'AUDIO_REQUESTS.json')['rows']}
    for example,reference in refs.items():
        if example not in requests or any(reference[k]!=requests[example][k] for k in ('song_id','split','audio_role','audio_sha256','start_sec','end_sec')):
            raise ValueError('reference and audio request are not bound to the same source interval')
    completed=[];distances=0
    jobs=sorted(list(run.glob('architecture/*/*/RESULT.json'))+list(run.glob('ablations/*/*/RESULT.json'))+list(run.glob('final-training/*/RESULT.json')))
    if len(jobs)!=33:raise ValueError(f'expected 33 declared completed jobs, found {len(jobs)}')
    for path in jobs:
        result=read_json(path)
        if not result['complete'] or result['fixed_epoch']!=20:raise ValueError('incomplete fixed-epoch result')
        verify_file(Path(result['checkpoint_path']),result['checkpoint_sha256'])
        for source,digest in result['code_bindings'].items():verify_file(Path(source),digest)
        ids=result['training_example_ids'];held=result['held_song']
        if len(ids)!=len(set(ids)) or any(i not in requests for i in ids):raise ValueError('invalid training IDs')
        if any(requests[i]['song_id']==held or requests[i]['split']!='train' for i in ids):raise ValueError('held/development data entered optimization')
        expected={i for i,r in requests.items() if r['song_id']!=held}
        if set(ids)!=expected:raise ValueError('fold dropped or added source rows')
        if result['peak_reserved_bytes']>7*1024**3:raise ValueError('process resource limit exceeded')
        monitor=read_json(path.parent/'monitor/resource-monitor.json')
        if not monitor['operation_succeeded'] or monitor['maximum_sampled_total_used_bytes']>14*1024**3:raise ValueError('GPU monitor did not complete within budget')
        evaluation=result['final_validation']
        if held is not None:
            rows=evaluation['examples']
            expected_ids={i for i,r in refs.items() if r['song_id']==held and r['audio_role']=='original'}
            if {r['example_id'] for r in rows}!=expected_ids:raise ValueError('held original coverage changed')
            total=count=0
            for r in rows:
                if r['reference']!=refs[r['example_id']]['target_ipa']:raise ValueError('reference changed in evaluation')
                measured=distance(r['reference'],r['prediction']);distances+=1
                if measured!=r['metrics']['edit_distance']:raise ValueError('independent edit distance mismatch')
                total+=measured;count+=len(r['reference'])
            if total!=evaluation['metrics']['edit_distance'] or count!=evaluation['metrics']['reference_phones']:raise ValueError('fold aggregate mismatch')
        completed.append({'path':str(path.resolve()),'sha256':sha256_file(path),'checkpoint_sha256':result['checkpoint_sha256'],
            'arm':result['arm'],'variant':result['variant'],'held_song':held,'seed':result['seed']})
    architecture=read_json(run/'ARCHITECTURE_SELECTION.json')
    for group in architecture['arms']:
        if group['reference_phones']!=1235 or len(group['examples'])!=20:raise ValueError('architecture summary coverage changed')
        if sum(distance(r['reference'],r['prediction']) for r in group['examples'])!=group['metrics']['edit_distance']:raise ValueError('architecture summary distance mismatch')
    recipe=read_json(run/'RECIPE_SELECTION.json')
    for group in recipe['variants']:
        if group['reference_phones']!=1235 or len(group['examples'])!=20:raise ValueError('ablation summary coverage changed')
        if sum(distance(r['reference'],r['prediction']) for r in group['examples'])!=group['metrics']['edit_distance']:raise ValueError('ablation summary distance mismatch')
    devrefs={r['example_id']:r for r in read_json(run/'DEVELOPMENT_REFERENCES.json')['rows']}
    devchecks=0
    for path in sorted(run.glob('development-results/*/RESULT.json'))+sorted(run.glob('development-transfer/*/RESULT.json')):
        result=read_json(path)
        for r in result['examples']:
            if r['reference']!=devrefs[r['example_id']]['target_ipa'] or distance(r['reference'],r['prediction'])!=r['metrics']['edit_distance']:
                raise ValueError('development reference/distance differs')
            devchecks+=1
            if 'posterior_path' in r:verify_file(Path(r['posterior_path']),r['posterior_sha256'])
    # Preserve historical source/defaults, using the pre-existing run's independent baseline.
    preserved=read_json(ROOT/'.state/runs/dual-input-ctc-20260920T073445032166Z/PRESERVATION_BEFORE.json')['files']
    for path,digest in preserved.items():verify_file(Path(path),digest)
    for path,digest in read_json(run/'FREEZE.json')['sources'].items():verify_file(Path(path),digest)
    selection=read_json(run/'DEVELOPMENT_SELECTION.json')
    if not selection['vocal_transfer_not_used'] or selection['default_replacement']:raise ValueError('selection/deployment boundary violated')
    write_json_new(run/'INDEPENDENT_AUDIT.json',{'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'declared_jobs':len(completed),'jobs':completed,'fold_distance_checks':distances,'development_distance_checks':devchecks,
        'whole_song_holdouts_verified':True,'all_declared_reference_counts_preserved':True,'historical_files_unchanged':len(preserved),
        'source_files_unchanged':True,'new_final_test_predictions':False,'no_gpu_inference_in_this_audit':True})
    print('Audited',len(completed),'jobs;',distances,'held-window distances;',devchecks,'development comparisons.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run.resolve())

"""Audit all five fixed whole-source results against frozen prior and matched control."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from any2xsampa.training.whole_sequence_analysis import compare_systems
from any2xsampa.adapters.whole_sequence_inference import verify_job


def analyze(run:Path)->None:
    spec=read_json(run/'POLICY.json');freeze=read_json(run/'FREEZE.json')
    parent=Path(freeze['parents']['singing']);routes=Path(freeze['parents']['routes'])
    selection=read_json(run/'ROUTE_SELECTION.json')
    index=read_json(run/'SCREEN_JOB_INDEX.json') if (run/'SCREEN_JOB_INDEX.json').exists() else None
    job_paths={row['held_song']:Path(row['job_path']) for row in index['rows']} if index else {}
    for path,digest in read_json(run/'PRESERVATION_BEFORE.json')['files'].items():
        verify_file(Path(path),digest)
    parent_summary={r['held_song']:r for r in read_json(parent/'full-ARCHITECTURE.json')['folds']}
    control_summary={r['held_song']:r for r in read_json(routes/'control-SCREEN.json')['folds']}
    systems={'prior':[],'control':[],'sequence_risk':[],'whole_sequence':[]};folds=[]
    eligible={r['song_id'] for r in read_json(run/'WHOLE_REFERENCES.json')['songs']}
    for song in spec['songs']:
        prior_path=parent/'architecture/full'/song/'RESULT.json';prior=read_json(prior_path)
        control_path=routes/'screen/control'/song/'RESULT.json';control=read_json(control_path)
        risk=read_json(routes/'screen/sequence_risk'/song/'RESULT.json')
        whole=verify_job(run,job_paths.get(song,run/'screen/whole_sequence'/song))
        if 'implementation_addendum_path' in whole:
            verify_file(Path(whole['implementation_addendum_path']),whole['implementation_addendum_sha256'])
        if (prior['checkpoint_sha256']!=parent_summary[song]['checkpoint_sha256']
                or prior['final_validation']['metrics']!=parent_summary[song]['metrics']
                or control['checkpoint_sha256']!=control_summary[song]['checkpoint_sha256']
                or control['final_validation']['metrics']!=control_summary[song]['metrics']):
            raise ValueError('historical fold differs from its hash-bound aggregate')
        if whole['parent_checkpoint_sha256']!=prior['checkpoint_sha256'] or whole['parent_checkpoint_sha256']!=control['parent_checkpoint_sha256']:
            raise ValueError('whole route and control have different fold initialization')
        verify_file(Path(prior['checkpoint_path']),prior['checkpoint_sha256'])
        if whole['initial_validation']!=prior['final_validation']:
            raise ValueError('whole initialization prediction/metric replay differs')
        if (whole['optimizer_steps']!=control['optimizer_steps']
                or whole['training_example_ids']!=control['training_example_ids']
                or whole['training_songs']!=control['training_songs']):
            raise ValueError('matched continuation data/budget changed')
        expected_songs=eligible-{song}
        if set(whole['whole_training_songs'])!=expected_songs or song in whole['training_songs']:
            raise ValueError('whole-source leave-song-out contamination')
        if any([r['epoch'] for r in result['history']]!=list(range(1,21)) for result in (whole,control)):
            raise ValueError('fixed twenty-epoch budget incomplete')
        for epoch,base in zip(whole['history'],control['history']):
            if epoch['optimizer_steps']!=base['optimizer_steps'] or epoch['supervised_microsteps']!=base['supervised_microsteps']:
                raise ValueError('per-epoch original/rendered budgets changed')
            if {r['song_id'] for r in epoch['whole_sequences']}!=expected_songs or len(epoch['whole_sequences'])!=len(expected_songs):
                raise ValueError('whole source not processed exactly once per epoch')
            for row in epoch['whole_sequences']:
                if (row['coefficient']!=spec['whole_sequence']['coefficient'] or row['independent_chunk_ctc_used']
                        or row['maximum_recompute_logit_difference']>2e-5 or not row['reference_used_only_in_global_loss']):
                    raise ValueError('whole loss/recompute contract changed')
        losses={s:{'first':next(r['ctc'] for r in whole['history'][0]['whole_sequences'] if r['song_id']==s),
                   'last':next(r['ctc'] for r in whole['history'][-1]['whole_sequences'] if r['song_id']==s)} for s in sorted(expected_songs)}
        for name,result in [('prior',prior),('control',control),('sequence_risk',risk),('whole_sequence',whole)]:
            systems[name].extend(result['final_validation']['examples'])
        folds.append({'held_song':song,'whole_training_songs':sorted(expected_songs),'matched_optimizer_steps':whole['optimizer_steps'],
            'implementation_version':whole.get('implementation_version','whole_sequence_no_grad_first_pass_v1'),
            'short_training_example_ids_equal':True,'parent_checkpoint_equal':True,'step_zero_equal':True,
            'checkpoint_sha256':whole['checkpoint_sha256'],'elapsed_seconds':whole['elapsed_seconds'],
            'control_elapsed_seconds':control['elapsed_seconds'],'whole_extra_compute_seconds':whole['whole_extra_compute_seconds'],
            'peak_reserved_bytes':whole['peak_reserved_bytes'],'whole_ctc_losses':losses,
            'maximum_recompute_logit_difference':max(r['maximum_recompute_logit_difference'] for e in whole['history'] for r in e['whole_sequences'])})
    comparison=compare_systems(systems)
    metric=comparison['systems']['whole_sequence'];prior=comparison['systems']['prior'];control=comparison['systems']['control']
    if metric!=selection['whole_sequence']['metrics'] or control!=selection['matched_control']['metrics'] or prior!=selection['prior_full_folds']:
        raise ValueError('saved pooled screening metrics changed')
    if metric['reference_phones']!=1235 or metric['sequences']!=20:
        raise ValueError('screening denominator differs from approved short references')
    # Integer arithmetic independently verifies the fixed 5% and .03 protection thresholds.
    improvement=100*metric['edit_distance']<=95*prior['edit_distance']
    regressions=[]
    for group,base in prior['groups'].items():
        new=metric['groups'][group]
        if base['reference_count']>=20:
            base_errors=sum(base[k] for k in ('substitutions','deletions','insertions'))
            new_errors=sum(new[k] for k in ('substitutions','deletions','insertions'))
            if 100*(new_errors-base_errors)>3*base['reference_count']:regressions.append(group)
    gate=improvement and not regressions
    if gate!=selection['passed'] or regressions!=selection['protected_regressions']:
        raise ValueError('independent gate arithmetic disagrees')
    deltas=[]
    for row in comparison['paired_windows']:
        old=row['systems']['control'];new=row['systems']['whole_sequence']
        deltas.append({'example_id':row['example_id'],'song_id':row['song_id'],
            'errors_before':old['edit_distance'],'errors_after':new['edit_distance'],
            'delta_errors':new['edit_distance']-old['edit_distance'],
            'delta_insertions':new['insertions']-old['insertions'],'delta_deletions':new['deletions']-old['deletions']})
    analysis={**comparison,'fold_integrity_checks':folds,'independent_gate':{'passed':gate,'gain_passed':improvement,
        'protected_regressions':regressions,'integer_threshold_verification':True},
        'window_changes_against_control':sorted(deltas,key=lambda r:(r['delta_errors'],r['song_id'],r['example_id'])),
        'improved_windows':sum(r['delta_errors']<0 for r in deltas),'tied_windows':sum(r['delta_errors']==0 for r in deltas),
        'worse_windows':sum(r['delta_errors']>0 for r in deltas),
        'insufficient_groups':[g for g,v in metric['groups'].items() if v['reference_count']<20],
        'training_ctc_is_not_generalization_evidence':True,'new_human_review_requested':False,
        'all_screens_from_same_existing_five_songs_not_an_independent_new_test_set':True,
        'selection_sha256':sha256_file(run/'ROUTE_SELECTION.json'),
        'completed_screen_candidates':5,'screen_process_attempts':6 if index else 5,
        'failed_attempts':index['failed_attempts'] if index else []}
    output=run/'completion';output.mkdir(exist_ok=True)
    write_json_new(output/'ERROR_ANALYSIS.json',analysis)
    print('INDEPENDENT METRICS', {k:r['edit_distance'] for k,r in comparison['systems'].items()},'gate',gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);analyze(p.parse_args().run.resolve())

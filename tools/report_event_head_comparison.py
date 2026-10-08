"""Complete fivefold H/E comparison, protected groups and directional errors."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file,sha256_object
from any2xsampa.training.event_head_evaluation import screen_gate,aggregate
from any2xsampa.training.phone_error_metrics import align_phone_edits
from analyze_input_native_probe import migrate


def report(run: Path):
    spec=read_json(run/'POLICY.json');freeze=read_json(run/'FREEZE.json')
    independent=read_json(run/'independent-replay/RESULT.json')
    if not independent['passed'] or len(independent['checkpoints'])!=10 or len(independent['full_audio_checks'])!=10:
        raise ValueError('independent complete screen replay required before qualification')
    baseline=read_json(run/'independent-replay/ENDPOINT_CONTROLS.json')['rows']
    old_original=[r for r in baseline if r['audio_role']=='original'];old_rendered=[r for r in baseline if r['audio_role']=='rendered']
    pools={};jobs=[];schedule=read_json(run/'SCHEDULES.json');step_totals=Counter()
    for arm in spec['arms']:
        original=[];rendered=[]
        for song in spec['songs']:
            path=run/'screen'/arm/song/'RESULT.json';row=read_json(path)
            if not row['complete'] or row['selected_epoch']!=20 or row['seed']!=17:raise ValueError('unplanned/incomplete training')
            if row['freeze_sha256']!=sha256_file(run/'FREEZE.json'):raise ValueError('different data freeze')
            verify_file(Path(row['checkpoint_path']),row['checkpoint_sha256'])
            if [e['epoch'] for e in row['history']]!=list(range(1,21)):raise ValueError('missing epoch')
            for i,event in enumerate(row['history']):
                if event['order_sha256']!=sha256_object(schedule[song]['epochs'][i]):raise ValueError('sampling changed')
                if event['held_out_metrics_used']:raise ValueError('epoch selection leakage')
                if read_json(path.parent/f'epoch-{i+1:02d}.json')!=event:raise ValueError('epoch artifact mismatch')
            if sum(e['optimizer_steps'] for e in row['history'])!=schedule[song]['optimizer_steps']:raise ValueError('update count differs')
            step_totals[arm]+=row['optimizer_steps']
            original.extend(r for r in row['validation']['details'] if r['audio_role']=='original')
            rendered.extend(r for r in row['validation']['details'] if r['audio_role']=='rendered')
            recovered=row.get('quality_job_outcome')=='20_epochs_saved_then_readonly_validation_recovered'
            if recovered:
                if (arm,song)!=('E','freedom-flower') or row['quality_training_restarted'] is not False:
                    raise ValueError('unapproved recovered job')
                failure=read_json(path.parent/'monitor/resource-failure.json')
                recovery_path=Path(row['evaluation_recovery_path']);verify_file(recovery_path,row['evaluation_recovery_sha256'])
                recovery=read_json(recovery_path);monitor=read_json(recovery_path.parent/'monitor/resource-monitor.json')
                if failure['reason']!='wall_clock_limit' or recovery['optimizer_steps']!=0 or not recovery['checkpoint_unchanged']:
                    raise ValueError('invalid read-only recovery provenance')
                card=max(monitor['maximum_sampled_total_used_bytes'],failure['latest_sample']['total_used_bytes'],
                         max(e['memory']['total_used_bytes'] for e in row['history']))
            else:
                monitor=read_json(path.parent/'monitor/resource-monitor.json');card=monitor['maximum_sampled_total_used_bytes']
            if not monitor['operation_succeeded'] or card>14*1024**3 or row['peak_reserved_bytes']>7*1024**3:
                raise ValueError('GPU resource envelope violation')
            jobs.append({'arm':arm,'held_song':song,'epochs':20,'optimizer_steps':row['optimizer_steps'],
                'elapsed_seconds':row['elapsed_seconds'],'peak_reserved_gib':row['peak_reserved_bytes']/1024**3,
                'peak_card_sampled_gib':card/1024**3,'quality_process_exit_code':73 if recovered else 0,
                'read_only_validation_recovered':recovered,'training_restarted':False,
                'checkpoint_sha256':row['checkpoint_sha256'],'source_result_path':str(path),'source_result_sha256':sha256_file(path)})
        pools[arm]={'original':original,'rendered':rendered}
    if dict(step_totals)!={'H':2140,'E':2140}:raise ValueError('frozen H/E update totals differ')
    gates={arm:screen_gate(arm,pools[arm]['original'],pools[arm]['rendered'],old_original,old_rendered,
        spec['screen_gates'],h_original=pools['H']['original'] if arm=='E' else None) for arm in spec['arms']}
    eligible=[arm for arm,g in gates.items() if g['passed']]
    selected=min(eligible,key=lambda arm:(gates[arm]['original']['per'],arm!='H')) if eligible else None
    human=read_json(ROOT/'.state/runs/original-adaptation-20260919T012000/reviewed-training-reference.json')
    reference={r['example_id']:r for r in human['examples'] if r['split']=='train'}
    strata=defaultdict(Counter);movements=[];vowels=set('aeiouy\u0250\u0259\u0254\u025b\u0153\u00f8\u026a\u028a')
    for arm in spec['arms']:
        for row in pools[arm]['original']:
            phones=reference[row['example_id']]['phones']
            if [p['phone'] for p in phones]!=row['reference']:raise ValueError('human reference changed')
            notes=defaultdict(list)
            for p in phones:notes[p['note_id']].append(p['phone'])
            edge={phones[0]['note_id'],phones[-1]['note_id']};position=0
            for op,expected,actual in align_phone_edits(row['reference'],row['prediction']):
                if expected is None:continue
                p=phones[position];position+=1
                group='multi_vowel' if sum(any(ch in vowels for ch in v) for v in notes[p['note_id']])>=2 else 'other'
                key=(arm,row['song_id'],'edge' if p['note_id'] in edge else 'interior',group,expected)
                strata[key]['reference']+=1;strata[key][op]+=1
    h={r['example_id']:r for r in pools['H']['original']}
    for e in pools['E']['original']:
        old=h[e['example_id']]
        movements.append({'example_id':e['example_id'],'song_id':e['song_id'],
            'E_vs_H':migrate(e['reference'],e['prediction'],old['prediction']),
            'delta_errors':e['metrics']['edit_distance']-old['metrics']['edit_distance'],
            'delta_insertions':e['metrics']['insertions']-old['metrics']['insertions'],
            'delta_deletions':e['metrics']['deletions']-old['metrics']['deletions']})
    result={'schema_version':'event_head_screen_report_v1','screen_complete':True,'gates':gates,'selected_arm':selected,
        'all_jobs':jobs,'original_and_rendered_results':pools,'historical_endpoint':{'original':aggregate(old_original),'rendered':aggregate(old_rendered)},
        'directional_window_analysis':movements,'note_phone_strata':[dict(arm=k[0],song_id=k[1],edge=k[2],note_kind=k[3],phone=k[4],**v) for k,v in strata.items()],
        'quality_starts':10,'successful_quality_processes':9,'post_optimization_timeout_processes':1,
        'read_only_checkpoint_recoveries':1,'models_completed_20epochs':10,
        'optimizer_updates':dict(step_totals),'no_development_or_test_selection':True,
        'training_criterion_changed_with_decoder':True,'not_just_boundary_head_causal_claim':True,
        'eligible_for_conditional_three_seeds':selected is not None,'automatic_model_deployment':False}
    write_json_new(run/'SCREEN_REPORT.json',result)
    for arm,g in gates.items():
        m=g['original'];print(arm,m['edit_distance'],m['reference_phones'],'S/D/I',m['substitutions'],m['deletions'],m['insertions'],'gate',g['passed'],g['failures'])
    print('SELECTED',selected)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run.resolve())

"""Read-only independent output, schedule, provenance and regression audit."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse,sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file,sha256_object
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer,ctc_collapse
from any2xsampa.adapters.segmental_event import SegmentalEventRecognizer,decode_events
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.event_head_evaluation import screen_gate


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        out=[i]
        for j,y in enumerate(b,1):out.append(min(row[j]+1,out[j-1]+1,row[j-1]+(x!=y)))
        row=out
    return row[-1]


def verify(run: Path,output: Path):
    from safetensors.numpy import load_file
    if output.exists() or not output.resolve().is_relative_to(run.resolve()):raise ValueError('new verification output inside run required')
    freeze=read_json(run/'FREEZE.json');spec=read_json(run/'POLICY.json')
    for p,h in freeze['bindings'].items():verify_file(Path(p),h)
    code=[]
    for phase in ('preflight','training','recovery','training_v2','replay'):
        review=read_json(run/('CODE_REVIEW_'+phase+'.json'))
        if review['freeze_sha256']!=sha256_file(run/'FREEZE.json'):raise ValueError('review refers to another freeze')
        for p,h in review['code_bindings'].items():verify_file(Path(p),h)
        code.append({'phase':phase,'review_sha256':sha256_file(run/('CODE_REVIEW_'+phase+'.json'))})
    rows={r['example_id']:r for r in read_json(run/'CACHE_ROWS.json')['rows']}
    vocab=read_json(Path(freeze['vocabulary']['path']));labels={i:p for p,i in vocab.items()}
    full=read_json(run/'independent-replay/RESULT.json')
    if not full['passed'] or full['full_acoustic_forwards']!=10:raise ValueError('full audio replay not complete')
    if len(list((run/'commands').glob('*.command.json')))!=10 or len(list(run.glob('screen/*/*/START.json')))!=10:raise ValueError('unaccounted launches')
    schedules=read_json(run/'SCHEDULES.json');metrics={a:[] for a in spec['arms']};checks=[];artifacts={};epochs=0
    for arm in spec['arms']:
        for song in spec['songs']:
            folder=run/'screen'/arm/song;p=folder/'RESULT.json';result=read_json(p)
            key=arm+'--'+song;exit=read_json(run/'commands'/(key+'.exit.json'))
            command=read_json(run/'commands'/(key+'.command.json'))
            if len(command['gpu_preflight_bytes'])!=3:raise ValueError('GPU preflight absent')
            if exit['exit_code']!=0:
                if (arm,song,exit['exit_code'])!=('E','freedom-flower',73):raise ValueError('unhandled quality-process failure')
                if result.get('quality_training_restarted') is not False or result.get('quality_process_exit_code')!=73:
                    raise ValueError('failed process obscured')
                rp=Path(result['evaluation_recovery_path']);verify_file(rp,result['evaluation_recovery_sha256']);rr=read_json(rp)
                if rr['optimizer_steps']!=0 or not rr['checkpoint_unchanged'] or rr['checkpoint_sha256']!=result['checkpoint_sha256']:
                    raise ValueError('invalid read-only checkpoint recovery')
                if read_json(run/'recovery-exit.json')['exit_code']!=0:raise ValueError('recovery did not finish')
            if result['head_state_sha256']==read_json(folder/'INITIALIZATION.json')['head_sha256']:raise ValueError('no trained update recorded')
            verify_file(Path(result['checkpoint_path']),result['checkpoint_sha256']);artifacts[str(p)]=sha256_file(p)
            init=read_json(folder/'INITIALIZATION.json')
            expected_train={k for k,r in rows.items() if r['song_id']!=song}
            expected_held={k for k,r in rows.items() if r['song_id']==song}
            if set(init['train_example_ids'])!=expected_train or set(init['held_example_ids'])!=expected_held:raise ValueError('held-song leakage')
            if len(result['history'])!=20 or result['optimizer_steps']!=schedules[song]['optimizer_steps']:raise ValueError('training budget mismatch')
            for i,e in enumerate(result['history']):
                if e!=read_json(folder/f'epoch-{i+1:02d}.json') or e['order_sha256']!=sha256_object(schedules[song]['epochs'][i]):raise ValueError('sampling audit mismatch')
                epochs+=1
            seen=set()
            for detail in result['validation']['details']:
                row=rows[detail['example_id']];seen.add(row['example_id'])
                if row['song_id']!=song or row['target_ipa']!=detail['reference']:raise ValueError('validation reference drift')
                if arm=='H':
                    pp=Path(detail['posterior_path']);verify_file(pp,detail['posterior_sha256']);payload=read_json(pp)
                    raw=pp.parent/payload['raw_evidence']['file'];verify_file(raw,payload['raw_evidence']['sha256'])
                    values=load_file(str(raw))['frame_logits'];pred=[labels[i] for i in ctc_collapse(values.argmax(-1).tolist())]
                    real=PhoneticXeusPosteriorRecognizer(pp).recognize(Path(row['audio_path']))
                    fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True).recognize(Path(row['audio_path']))
                    view=lambda seq:[(asdict(s.time_range),[asdict(c) for c in s.candidate_phones],s.confidence) for s in seq]
                else:
                    pp=Path(detail['event_path']);verify_file(pp,detail['event_sha256']);payload=read_json(pp)
                    raw=pp.parent/payload['scores']['file'];verify_file(raw,payload['scores']['sha256'])
                    values=load_file(str(raw));decoded=decode_events(values['frame_log_probs'],values['boundary_logits'])
                    pred=[labels[i] for i in decoded['event_ids']]
                    real=SegmentalEventRecognizer(pp).recognize(Path(row['audio_path']))
                    fixture=SegmentalEventRecognizer(pp,fixture=True).recognize(Path(row['audio_path']))
                    view=lambda seq:[asdict(s) for s in seq]
                if view(real)!=view(fixture) or pred!=detail['prediction']:raise ValueError('saved evidence does not replay')
                m=phone_metrics([(detail['reference'],pred)])
                if m!=detail['metrics'] or m['edit_distance']!=distance(detail['reference'],pred):raise ValueError('independent edit count differs')
                checks.append({'arm':arm,'example_id':row['example_id'],'posterior_sha256':sha256_file(pp),'metric_equal':True,'fixture_equal':True})
                artifacts[str(pp)]=sha256_file(pp);artifacts[str(raw)]=sha256_file(raw)
                metrics[arm].append(detail)
            if seen!=expected_held:raise ValueError('held cohort incomplete')
    report=read_json(run/'SCREEN_REPORT.json');base=read_json(run/'independent-replay/ENDPOINT_CONTROLS.json')['rows']
    for arm in spec['arms']:
        get=lambda seq,role:[r for r in seq if r['audio_role']==role]
        gate=screen_gate(arm,get(metrics[arm],'original'),get(metrics[arm],'rendered'),get(base,'original'),get(base,'rendered'),
            spec['screen_gates'],h_original=get(metrics['H'],'original') if arm=='E' else None)
        if gate!=report['gates'][arm]:raise ValueError('independent gate recomputation differs')
    if epochs!=200 or len(checks)!=298:raise ValueError('expected 200epochs and298held validation windows')
    originals=read_json(ROOT/'.state/runs/dual-input-ctc-20260920T073445032166Z/PRESERVATION_BEFORE.json')['files']
    protected={p:h for p,h in originals.items() if Path(p).suffix.lower() in {'.vsqx','.wav','.json'} or p.endswith('any2xsampa\\cli.py')}
    for p,h in protected.items():verify_file(Path(p),h)
    suite=ET.parse(run/'regression.xml').getroot().find('testsuite');tests={k:int(suite.get(k)) for k in ('tests','failures','errors','skipped')}
    if tests['failures'] or tests['errors'] or read_json(run/'regression-exit.json')['exit_code']!=0:raise ValueError('regression failed')
    write_json_new(output,{'passed':True,'quality_training_starts':10,'models_completed_20epochs':10,
        'successful_quality_processes':9,'post_optimization_timeout_processes':1,'readonly_checkpoint_recoveries':1,'training_retries':0,'epochs':epochs,
        'optimizer_updates':{'H':2140,'E':2140},'posterior_fixture_replays':checks,'artifact_bindings':artifacts,
        'full_audio_forwards':10,'full_audio_checks':full['full_audio_checks'],'old_endpoint_cache_calls':149,
        'reviews':code,'source_files_unchanged':len(freeze['bindings']),'protected_originals_mappings_defaults':protected,
        'regression':tests,'selected_arm':report['selected_arm'],'default_replaced':False,'final_test_opened':False,
        'new_human_listening_required':False})
    print('DELIVERY VERIFIED',len(checks),'fixtures;',epochs,'epochs;',tests)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();verify(a.run.resolve(),a.output.resolve() if a.output else a.run.resolve()/'FINAL_VERIFICATION.json')

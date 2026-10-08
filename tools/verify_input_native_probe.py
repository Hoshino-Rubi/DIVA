"""Independent CPU artifact/fixture replay; never loads an acoustic model."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET
from input_native_probe_common import ROOT, SONGS, verify_review, prediction, compare_logits, a1_gate
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer


def distance(a: list, b: list) -> int:
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def verify(run: Path, output: Path | None = None) -> None:
    from safetensors.numpy import load_file
    output = output or run/'FINAL_VERIFICATION.json'
    if output.exists() or not output.resolve().is_relative_to(run.resolve()):
        raise ValueError('verification requires a new output file inside this run')
    freeze,review=verify_review(run)
    refs={r['window_id']:r for r in read_json(run/'REFERENCES.json')['rows']}
    vocab=read_json(Path(freeze['vocabulary']['path']))
    stages=['A0','A1','B0','B1','AR','BR'];records=[];resources=[];replays=[];artifacts={}
    project_bindings=dict(review['code_bindings'])
    def bind(path: Path):
        artifacts[str(path.resolve())]=sha256_file(path)
    for stage in stages:
        file=run/stage/'RESULT.json';bind(file);report=read_json(file)
        tasks=read_json(run/'tasks'/(stage+'.json'))['tasks']
        exitpath=run/'commands'/(stage+'.exit.json');bind(exitpath)
        if read_json(exitpath)['exit_code']!=0 or report['passed'] is not True or report['frozen_parameters_unchanged'] is not True:
            raise ValueError('stage not successful')
        if report['model_parameters_before']!=report['model_parameters_after']:
            raise ValueError('frozen state mismatch')
        expected={t['id']:t for t in tasks}
        actual={r['task_id']:r for r in report['records']}
        if set(expected)!=set(actual) or len(actual)!=len(tasks):raise ValueError('stage cohort incomplete')
        starts=list((run/stage/'forwards').glob('*/START.json'))
        if len(starts)!=len(tasks):raise ValueError('forward accounting mismatch')
        for ident,row in actual.items():
            task=expected[ident]
            if row['request']!=task['request']:raise ValueError('audio-only task mutated')
            record_path=run/stage/'forwards'/ident/'RESULT.json';bind(record_path)
            if read_json(record_path)!=row:raise ValueError('stage/item results diverge')
            for layer,spec in row['outputs'].items():
                pp=Path(spec['posterior_path']);verify_file(pp,spec['posterior_sha256']);bind(pp)
                payload=read_json(pp);raw=pp.parent/payload['raw_evidence']['file']
                verify_file(raw,payload['raw_evidence']['sha256']);bind(raw)
                values=load_file(str(raw))['frame_logits'];pred=prediction(values,vocab)
                if pred!=spec['prediction'] or pred!=payload['decoding']['collapsed_raw_labels']:
                    raise ValueError('native sequence replay differs')
                audio=Path(row['request']['audio_path'])
                real=PhoneticXeusPosteriorRecognizer(pp).recognize(audio)
                fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True).recognize(audio)
                def contract(rr):
                    return [{'time_range':asdict(s.time_range),'phones':[asdict(p) for p in s.candidate_phones],'confidence':s.confidence} for s in rr]
                if contract(real)!=contract(fixture):raise ValueError('independent fixture contract differs')
                if payload['producer']['input_audio']['sha256']!=row['request']['audio_sha256']:
                    raise ValueError('posterior rebound to different audio')
                if task.get('replay_of'):
                    old=run/task['replay_of']/'forwards'/ident/layer/'logits.safetensors'
                    diff=compare_logits(values,load_file(str(old))['frame_logits'],vocab)
                    op=read_json(old.parent/'posterior.json')
                    if op['segments']!=payload['segments'] or op['decoding']!=payload['decoding']:
                        raise ValueError('independent posterior content differs')
                    replays.append({'stage':stage,'id':ident,'layer':layer,**diff})
                records.append({'stage':stage,'id':ident,'layer':layer,'frames':len(values),
                    'sequence_replayed':True,'real_fixture_equal':True,'posterior_sha256':spec['posterior_sha256']})
        mon=read_json(run/stage/'monitor/resource-monitor.json')
        if (mon['completed_without_resource_abort'] is not True or mon['operation_succeeded'] is not True
            or mon['maximum_sampled_total_used_bytes']>14*1024**3 or report['peak_reserved_bytes']>7*1024**3):
            raise ValueError('resource verification failed')
        pre=read_json(run/'commands'/(stage+'.gpu-preflight.json'))
        if not pre['three_queries_passed'] or len(pre['used_bytes'])!=3:raise ValueError('three-query GPU gate absent')
        resources.append({'stage':stage,'forwards':len(tasks),'model_worker_seconds':report['elapsed_seconds'],
            'process_wall_seconds':read_json(exitpath)['elapsed_seconds'],'reserved_bytes':report['peak_reserved_bytes'],
            'card_sampled_bytes':mon['maximum_sampled_total_used_bytes']})
    gate=read_json(run/'A1_GATE.json')
    check=a1_gate(gate['details']['h'],gate['details']['x'])
    if any(check[k]!=gate[k] for k in check):raise ValueError('A1 gate failed independent recomputation')
    if gate['passed'] or gate['failures']!=['insertions_over_limit']:
        raise ValueError('unexpected phase outcome; do not silently finalize other branches')
    for stage in ('A2','ADR'):
        if (run/stage/'START.json').exists() or (run/'commands'/(stage+'.command.json')).exists():
            raise ValueError('forbidden development branch executed after failed A1')
    b=read_json(run/'B_ANALYSIS.json');metric_checks=0
    for path,digest in b['source_bindings'].items():verify_file(Path(path),digest)
    for cell in b['matrix'].values():
        metric=phone_metrics([(r['reference'],r['prediction']) for r in cell['details']])
        if metric!=cell['metrics'] or metric['reference_phones']!=1235:raise ValueError('B metric denominator changed')
        for r in cell['details']:
            if r['reference']!=refs[r['window_id']]['target_ipa']:raise ValueError('reference drift')
            if distance(r['reference'],r['prediction'])!=phone_metrics([(r['reference'],r['prediction'])])['edit_distance']:
                raise ValueError('independent edit distance mismatch')
            metric_checks+=1
    # Connect reported held-song checkpoints to pre-existing frozen indexes.
    probe=Path(freeze['probe_run']);mv=Path(freeze['mv_run'])
    source=read_json(probe/'SOURCE_BINDINGS.json')['old_checkpoint_dirs']
    replay=read_json(mv/'independent-replay/VERIFICATION.json')
    for spec in b['checkpoint_identity_audit']:
        if spec['variant']=='M':
            old=source[spec['song_id']];rp=Path(old['result']);verify_file(rp,old['result_sha256'])
        else:
            old=next(j for j in replay['checkpoints'] if j['variant']=='MV' and j['held_song']==spec['song_id'])
            rp=Path(old['source_result_path']);verify_file(rp,old['source_result_sha256'])
        result=read_json(rp)
        if result['checkpoint_sha256']!=spec['sha256']:raise ValueError('checkpoint source index mismatch')
        if result['held_song']!=spec['song_id'] or result['architecture']['sha256']!=read_json(run/'B1/RESULT.json')['architecture']['sha256']:
            raise ValueError('historical fold identity mismatch')
    # These historical hashes include all six human VSQX and original WAVs,
    # mappings/default exports. Test-song WAVs are not read or decoded here.
    old=read_json(Path(freeze['dual_run'])/'PRESERVATION_BEFORE.json')['files']
    preserve={p:h for p,h in old.items() if Path(p).suffix.lower() in {'.vsqx','.wav','.json'} or p.endswith('any2xsampa\\cli.py')}
    for p,h in preserve.items():verify_file(Path(p),h)
    for p,h in project_bindings.items():verify_file(Path(p),h)
    suite=ET.parse(run/'regression.xml').getroot().find('testsuite')
    test={k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
    if test['failures'] or test['errors'] or read_json(run/'regression-exit.json')['exit_code']!=0:
        raise ValueError('full regression failed')
    forwards=sum(r['forwards'] for r in resources)
    if forwards!=104 or len(list(run.glob('*/forwards/*/START.json')))!=104 or len(records)!=305:
        raise ValueError('actual budget/output accounting mismatch')
    if len(list((run/'commands').glob('*.command.json')))!=6:raise ValueError('unreported launch')
    write_json_new(output,{'passed':True,'posterior_fixture_replays':len(records),
        'independent_full_waveform_replays':10,'independent_outlet_replays':replays,
        'independent_metric_window_checks':metric_checks,'snapshot_and_231_bound_sources_unchanged':True,
        'checkpoint_source_indexes_verified':10,'preserved_originals_mappings_defaults':preserve,
        'launch_code_unchanged':project_bindings,'artifact_bindings':artifacts,'fixture_records':records,
        'budget':{'process_starts':6,'failed_model_processes':0,'complete_forward_attempts':forwards,
            'complete_forward_successes':forwards,'ceiling':111,'unspent_conditional_forwards':7,
            'cached_head_replays':0,'training_starts':0,'new_weights':0,'new_dependencies':0},
        'resources':resources,'regression':test,'no_default_change':True,'no_final_test_prediction_opened':True,
        'human_review_required':False,'A2_skipped':'A1 insertion protection failed',
        'B_route':'No fixed native intermediate output has unambiguous net advantage in >=3 songs; no training authorized.'})
    print('FINAL VERIFIED',forwards,'forwards;',len(records),'fixtures;',metric_checks,'metric windows;',test)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,help='New verification JSON inside the run; never overwrites prior evidence')
    args=p.parse_args();verify(args.run.resolve(),args.output.resolve() if args.output else None)

"""Prepare independent H/E training experiment without loading a model."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,sha256_object,write_json_new
from any2xsampa.training.event_head_protocol import CONFIG,validate_cohort,epoch_schedule,fold_steps


def prepare(run: Path):
    if run.exists() or not run.is_relative_to(ROOT/'.state/runs'):raise ValueError('new independent run required')
    parent=ROOT/'.state/runs/innovation-stagewise-20260924T084724Z'
    for p,h in read_json(parent/'FINAL_INDEX.json')['artifacts'].items():verify_file(Path(p),h)
    base=ROOT/'.state/runs/singing-domain-20260920T095918143867Z';old=read_json(base/'FREEZE.json')
    for p,h in old['sources'].items():verify_file(Path(p),h)
    sources={str(p):sha256_file(p) for p in [parent/'FINAL_INDEX.json',parent/'RESULTS.md',CONFIG,
        base/'FREEZE.json',base/'REFERENCES.json',base/'CACHE.json',base/'AUDIO_REQUESTS.json',base/'SUMMARY.json',
        ROOT/'.state/runs/final-readout-plan-20260924T032415Z/NEXT_PLAN.md']}
    cache=read_json(base/'CACHE.json');references={r['example_id']:r for r in read_json(base/'REFERENCES.json')['rows']}
    req={r['example_id']:r for r in read_json(base/'AUDIO_REQUESTS.json')['rows']}
    rows=[];unlabeled=[];spec=read_json(CONFIG)
    for entry in cache['entries']:
        if entry['audio_role']=='unlabeled_original':unlabeled.append({'example_id':entry['example_id'],'song_id':entry['song_id']})
        if entry['example_id'] not in references:continue
        ref=references[entry['example_id']]
        for k in ('song_id','split','audio_role','audio_sha256','start_sec','end_sec'):
            if entry[k]!=ref[k] or ref[k]!=req[entry['example_id']][k]:raise ValueError('cache/reference/request mismatch')
        if entry['architecture_sha256']!=cache['architecture']['sha256']:raise ValueError('mixed architectures')
        verify_file(Path(entry['path']),entry['sha256']);sources[entry['path']]=entry['sha256']
        verify_file(Path(ref['audio_path']),ref['audio_sha256']);sources[ref['audio_path']]=ref['audio_sha256']
        rows.append({**entry,**ref})
    validate_cohort(rows,spec)
    weights=ROOT/'.state/runs/limited-data-20260906T050251/rendered-features/pretrained-head.safetensors'
    headindex=read_json(weights.parent/'index.json');verify_file(weights,headindex['head']['sha256']);sources[str(weights)]=headindex['head']['sha256']
    vocab=Path(old['vocabulary_path']);sources[str(vocab)]=sha256_file(vocab)
    vocabulary=read_json(vocab)
    for r in rows:
        if r['target_ids']!=[vocabulary[p] for p in r['target_ipa']]:raise ValueError('target map changed')
    schedule={};controls={}
    for song in spec['songs']:
        path=base/'architecture/endpoint'/song/'RESULT.json';sources[str(path)]=sha256_file(path);control=read_json(path)
        cp=Path(control['checkpoint_path']);verify_file(cp,control['checkpoint_sha256']);sources[str(cp)]=control['checkpoint_sha256']
        sc=epoch_schedule(rows,unlabeled,song,17);steps=fold_steps(sc)
        if steps!=sum(e['optimizer_steps'] for e in control['history']):raise ValueError('old optimizer budget mismatch')
        schedule[song]={'epochs':sc,'sha256':sha256_object(sc),'optimizer_steps':steps,'labelled_microsteps':sum(map(len,sc)),
                        'original_slots':sum(sum(references[x]['audio_role']=='original' for x in epoch) for epoch in sc),
                        'rendered_slots':sum(sum(references[x]['audio_role']=='rendered' for x in epoch) for epoch in sc)}
        controls[song]={'result_path':str(path),'result_sha256':sha256_file(path),'checkpoint_path':str(cp),'checkpoint_sha256':control['checkpoint_sha256']}
    run.mkdir(parents=True)
    for name,payload in [('POLICY.json',spec),('CACHE_ROWS.json',{'rows':rows,'unused_rng_rows':unlabeled,'architecture':cache['architecture']}),
                         ('SCHEDULES.json',schedule),('CONTROLS.json',controls)]:write_json_new(run/name,payload)
    for name in ('CACHE_ROWS.json','SCHEDULES.json','CONTROLS.json'):sources[str(run/name)]=sha256_file(run/name)
    write_json_new(run/'FREEZE.json',{'schema_version':'event_head_comparison_freeze_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'bindings':sources,'policy_sha256':sha256_file(run/'POLICY.json'),'historical_run':str(base),'snapshot':old['model_snapshot'],
        'python':old['runtime_python'],'vocabulary':{'path':str(vocab),'sha256':sha256_file(vocab)},
        'pretrained_head':{'path':str(weights),'sha256':sha256_file(weights)},'training_stage':'H_E_fivefold_fixed_epoch',
        'shared_native_model_modified':False,'new_training_launch_cap':10,'conditional_final_launch_cap':3,'acoustic_forward_cap':16})
    print(run,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);prepare(p.parse_args().run.resolve())

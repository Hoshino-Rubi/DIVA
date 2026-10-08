"""Freeze accepted original windows and audibility development references for training."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,sha256_object,verify_file,write_json_new
from any2xsampa.training.reviewed_training_reference import SCHEMA,validate_training_reference
from any2xsampa.training.reviewed_audibility_reference import build_reviewed_reference
from tools.accept_reviewed_training_features import verified_reference,finish
from tools.merge_reviewed_training_acceptance import merge_rows
from tools.import_original_edge_review import load_verified_packet


def binding(path: Path) -> dict:
    return {'path':str(path.resolve()),'sha256':sha256_file(path)}


def build(cumulative_path: Path,dev_run: Path,output: Path) -> dict:
    cumulative=read_json(cumulative_path)
    if cumulative['manifest_hash']!=sha256_object({k:v for k,v in cumulative.items() if k!='manifest_hash'}):
        raise ValueError('cumulative manifest hash mismatch')
    refs=[];indices=[];accepted=[];sources=[]
    for spec in cumulative['source_bindings']:
        ap,rp=Path(spec['acceptance_path']),Path(spec['reference_path'])
        verify_file(ap,spec['acceptance_sha256']);verify_file(rp,spec['reference_sha256'])
        actual=finish(rp.parent,ap.parent,write=False)
        if actual!=read_json(ap):raise ValueError('acceptance replay mismatch')
        accepted.append(actual);refs.extend(r for r in verified_reference(rp.parent)['examples'] if r['status']=='human_confirmed')
        sources.append(binding(rp));indices.append(read_json(ap.parent/'features/index.json'))
    merged=merge_rows(accepted)
    if any(cumulative[k]!=v for k,v in merged.items()):raise ValueError('cumulative merge changed')
    inputs=read_json(dev_run/'INPUTS.json')
    dp=dev_run/'reviewed-reference.json';verify_file(dp,inputs['reference_sha256'])
    for key in ('packet','answers'):verify_file(Path(inputs[key]['path']),inputs[key]['sha256'])
    packet,_=load_verified_packet(Path(inputs['packet']['path']))
    dev=build_reviewed_reference(packet,read_json(Path(inputs['answers']['path'])))
    if dev!=read_json(dp):raise ValueError('dev answers replay mismatch')
    refs.extend(r for r in dev['examples'] if r['status']=='human_confirmed');sources.append(binding(dp))
    base=ROOT/'.state/runs/limited-data-20260906T050251'
    historical=read_json(base/'original-baselines/index.json')
    # Read only the metadata; never open any test evidence.
    dev_ids={r['example_id'] for r in dev['examples'] if r['status']=='human_confirmed'}
    historical['entries']=[e for e in historical['entries'] if e['example_id'] in dev_ids]
    indices.append(historical)
    reference={'schema_version':SCHEMA,'examples':refs,'reference_sources':sources,
               'absence_is_not_geometric_truncation':True,'test_songs_opened':False}
    reference['manifest_hash']=sha256_object(reference);validate_training_reference(reference)
    if (output/'original-training-manifest.json').exists():raise FileExistsError(output)
    cache=output/'original-features';cache.mkdir(parents=True,exist_ok=False)
    entries=[];model=indices[0]['model_metadata'];head=indices[0]['head']
    for index in indices:
        if index['model_metadata']!=model or index['head']['sha256']!=head['sha256'] or not index['all_parameters_frozen'] or not index['no_labels_or_score_received']:
            raise ValueError('mixed frozen model caches')
        for entry in index['entries']:
            if entry['test_outputs_sealed'] or entry['split'] not in {'train','dev'} or entry['audio_role']!='original':raise ValueError('sealed or wrong-role cache')
            old=Path(entry['evidence_path']);verify_file(old,entry['evidence_sha256'])
            target=cache/(entry['example_id']+'.safetensors');shutil.copyfile(old,target);verify_file(target,entry['evidence_sha256'])
            entries.append({**entry,'evidence_path':str(target.resolve()),'source_evidence_path':str(old)})
    hp=Path(head['path']);verify_file(hp,head['sha256']);shutil.copyfile(hp,cache/'pretrained-head.safetensors')
    feature_index={'schema_version':'phoneticxeus_feature_index_v1','purpose':'original-reviewed-v2',
        'source_dataset_hash':reference['manifest_hash'],'model_metadata':model,'entries':entries,
        'head':binding(cache/'pretrained-head.safetensors'),'all_parameters_frozen':True,'no_labels_or_score_received':True}
    refpath=output/'reviewed-training-reference.json';ip=cache/'index.json'
    write_json_new(refpath,reference);write_json_new(ip,feature_index)
    train=[r for r in refs if r['split']=='train'];devrows=[r for r in refs if r['split']=='dev']
    counts={'confirmed_train_windows':len(train),'confirmed_train_songs':len({r['song_id'] for r in train}),
        'confirmed_dev_windows':len(devrows),'confirmed_dev_reference_phones':sum(len(r['phones']) for r in devrows),'overlaps':[]}
    passed=counts['confirmed_train_windows']>=20 and counts['confirmed_train_songs']>=3 and counts['confirmed_dev_windows']>=3 and counts['confirmed_dev_reference_phones']>=100
    manifest={'schema_version':'original_training_manifest_v1','reference_mode':'confirmed_original_segments',
        'segment_reference':binding(refpath),'original_feature_index':binding(ip),
        'rendered_dataset':binding(base/'reference/rendered-proxy-dataset.json'),
        'rendered_feature_index':binding(base/'rendered-features/index.json'),
        'seed43_checkpoint':binding(ROOT/'.state/runs/quality-first8h-20260907T010738/training-v2/seed43/best_admissible.safetensors'),
        'vocabulary':binding(base/'model/ipa_vocab.json'),'gate':{'passed':passed,'counts':counts},
        'test_songs_opened':False,'automatic_model_promotion':False}
    manifest['manifest_hash']=sha256_object(manifest);write_json_new(output/'original-training-manifest.json',manifest)
    write_json_new(output/'BUILD.json',{'cumulative':binding(cumulative_path),'dev_inputs':binding(dev_run/'INPUTS.json'),'reference_sources':sources,'counts':counts,'training_started':False})
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cumulative',type=Path,required=True);p.add_argument('--dev-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'.state/runs'):raise ValueError('output must remain in runs')
    print(build(a.cumulative,a.dev_run,a.output.resolve())['gate'])

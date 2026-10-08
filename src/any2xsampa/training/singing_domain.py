"""Versioned data and experiment checks for singing-domain adaptation."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import math,re
from any2xsampa.training.limited_data_artifacts import read_json,sha256_object,verify_file

CONFIG=Path(__file__).parents[1]/'config/singing_domain_v1.json'

def policy() -> dict:
    return read_json(CONFIG)

def architecture_identity(snapshot: Mapping[str,Any], spec: Mapping[str,Any]) -> dict:
    expected=policy()['architecture']
    if dict(spec)!=expected:raise ValueError('unsupported singing-domain architecture')
    identity={'snapshot':dict(snapshot),'architecture':dict(spec),'audio_preprocessing':'mono_float32_scipy_resample_poly_16000_v1'}
    return {**identity,'sha256':sha256_object(identity)}

def validate_audio_row(row: Mapping[str,Any], *, allow_dev: bool=False) -> None:
    required={'example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec'}
    if set(row)!=required:raise ValueError('audio requests accept no target/text/score fields')
    if not isinstance(row['example_id'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',row['example_id']):raise ValueError('invalid or unsafe example ID')
    split=row['split'];song=row['song_id']
    if split=='train' and song not in policy()['songs']:raise ValueError('training song leakage')
    if split!='train' and not (allow_dev and split=='dev' and song=='shalong'):raise ValueError('development/test request not allowed')
    if row['audio_role'] not in {'original','rendered','separated_vocal','unlabeled_original'}:raise ValueError('unrecognized audio role')
    if split=='dev' and row['audio_role']=='unlabeled_original':raise ValueError('unlabeled development is forbidden')
    start,end=row['start_sec'],row['end_sec']
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in [start,end]):raise ValueError('non-finite crop')
    if start<0 or not 0<end-start<=20.000001:raise ValueError('crop outside bounded range')
    if not isinstance(row['audio_sha256'],str) or len(row['audio_sha256'])!=64:raise ValueError('invalid source hash')

def validate_cache_entry(entry: Mapping[str,Any], request: Mapping[str,Any], architecture_sha256: str) -> None:
    for key in ('example_id','song_id','split','audio_role','audio_sha256','start_sec','end_sec'):
        if entry.get(key)!=request.get(key):raise ValueError('cache source binding mismatch: '+key)
    if entry.get('architecture_sha256')!=architecture_sha256 or entry.get('cache_layer')!=15:raise ValueError('stale architecture/prefix cache')
    if entry.get('no_reference_received') is not True:raise ValueError('feature extractor received labels')
    if not isinstance(entry.get('frames'),int) or entry['frames']<1 or entry.get('width')!=1024:raise ValueError('invalid cache geometry')

def partition(rows: Sequence[Mapping[str,Any]], held_song: str | None) -> tuple[list[dict],list[dict]]:
    songs=policy()['songs']
    if held_song is not None and held_song not in songs:raise ValueError('unknown holdout song')
    if any(r.get('split')!='train' or r.get('song_id') not in songs for r in rows):raise ValueError('nontraining data in fold pool')
    if len({r['example_id'] for r in rows})!=len(rows):raise ValueError('duplicate fold input')
    train=[dict(r) for r in rows if r['song_id']!=held_song]
    validation=[dict(r) for r in rows if r['song_id']==held_song and r['audio_role']=='original']
    if held_song is not None and (not train or not validation):raise ValueError('empty training or held-original fold')
    return train,validation

def load_manifest(run: Path) -> tuple[dict,dict]:
    freeze=read_json(run/'FREEZE.json')
    for name in ('policy','audio_requests','references'):
        verify_file(run/freeze[name]['file'],freeze[name]['sha256'])
    spec=read_json(run/'POLICY.json')
    if spec!=policy():raise ValueError('frozen training policy differs from implementation version')
    refs=read_json(run/'REFERENCES.json')
    return spec,refs

def choice(results: Sequence[Mapping[str,Any]]) -> dict:
    complete=[r for r in results if r.get('complete') is True]
    if not complete:raise ValueError('no complete architecture/recipe result')
    for row in complete:
        if set(row['fold_songs'])!=set(policy()['songs']) or row['reference_phones']!=1235:
            raise ValueError('selection requires all five folds and unchanged denominator')
        if not math.isfinite(row['per']):raise ValueError('invalid selection score')
    return dict(min(complete,key=lambda r:(r['per'],r['trainable_parameters'],r['name'])))

"""Freeze source identity and non-training scope before sequential innovation probes."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.training.dual_input import validate_pair


def prepare(output: Path) -> None:
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'):
        raise ValueError('choose a new run directory')
    design=ROOT/'.state/runs/innovation-options-20260924T041026Z'
    for p,h in read_json(design/'INDEX.json')['artifacts'].items():verify_file(Path(p),h)
    dual=ROOT/'.state/runs/dual-input-ctc-20260920T073445032166Z'
    singing=ROOT/'.state/runs/singing-domain-20260920T095918143867Z'
    source_files=[design/'INDEX.json',design/'DESIGN_OPTIONS.md',
        ROOT/'src/any2xsampa/config/innovation_stagewise_v1.json']
    source_files += [dual/name for name in ('EVIDENCE.json','SCORING_REFERENCES.json','PAIRS.json','AUDIO_REQUESTS.json','EVALUATION.json','POLICY.json')]
    source_files += [singing/name for name in ('FREEZE.json','CACHE.json','REFERENCES.json')]
    bindings={str(p):sha256_file(p) for p in source_files}
    def bind(p: Path,h: str):
        verify_file(p,h);bindings[str(p.resolve())]=h
    candidate=ROOT/'.state/runs/guarded-retention-20260919T041500/CANDIDATE_FREEZE.json'
    bind(candidate,read_json(dual/'EVIDENCE.json')['candidate_sha256'])
    bundle=read_json(ROOT/'.state/runs/decoder-usability-20260920T011000/candidate-bundle-v1.json')
    vocabulary=bundle['vocabulary'];bind(Path(vocabulary['path']),vocabulary['sha256'])
    refs={r['example_id']:r for r in read_json(dual/'SCORING_REFERENCES.json')['rows'] if r['split']=='train'}
    requests={r['example_id']:r for r in read_json(dual/'AUDIO_REQUESTS.json')['requests']}
    evidence=[r for r in read_json(dual/'EVIDENCE.json')['entries'] if r['split']=='train']
    if len(refs)!=20 or len(evidence)!=40 or sum(len(r['reference']) for r in refs.values())!=1235:
        raise ValueError('fixed original training cohort changed')
    rows=[]
    for wid,ref in sorted(refs.items()):
        r={c:next(e for e in evidence if e['example_id']==wid and e['condition']==role)
           for c,role in [('x','mixed'),('v','vocal')]}
        pp=[]
        for c,e in r.items():
            pair=e['paired_record'];bind(Path(pair['path']),pair['sha256'])
            pair_data=read_json(Path(pair['path']));validate_pair(requests[wid],pair_data)
            for role,a in pair_data['audio'].items():bind(Path(a['path']),a['sha256'])
            for name in ('posterior','evidence'):
                bind(Path(e['artifact'][name+'_path']),e['artifact'][name+'_sha256'])
            p=read_json(Path(e['artifact']['posterior_path']))
            if p['producer']['model_metadata']['guarded_candidate_sha256']!=sha256_file(candidate):
                raise ValueError('mixed guarded model identity')
            if p['producer']['input_audio']['sha256']!=e['input_audio']['sha256']:
                raise ValueError('source audio mismatch')
            pp.append(p)
        if pp[0]['producer']['crop']!=pp[1]['producer']['crop'] or pp[0]['decoding']['geometry']!=pp[1]['decoding']['geometry']:
            raise ValueError('paired source clock or frame geometry mismatch')
        rows.append({'window_id':wid,'song_id':ref['song_id'],'reference':ref['reference'],
                     'reference_nature':'human_reviewed_edges_with_inherited_calibrated_source_ipa','sources':r})
    labeled=read_json(singing/'REFERENCES.json')['rows']
    cache=read_json(singing/'CACHE.json')
    original=[e for e in cache['entries'] if e['audio_role']=='original' and e['split']=='train']
    longest=min(original,key=lambda e:(-e['frames'],e['example_id']))
    bind(Path(longest['path']),longest['sha256'])
    if len(labeled)!=149 or len(original)!=20:raise ValueError('training cache count changed')
    output.mkdir(parents=True)
    write_json_new(output/'FACTOR_INPUTS.json',{'rows':rows,'same_G_weights':True,'reference_conditioning':False})
    write_json_new(output/'PROFILE_INPUT.json',{'entry':longest,'reference':next(r for r in labeled if r['example_id']==longest['example_id']),
        'purpose':'single_cached_prototype_forward_backward_no_optimizer_no_PER_selection',
        'architecture':cache['architecture']})
    for name in ('FACTOR_INPUTS.json','PROFILE_INPUT.json'):
        bindings[str(output/name)]=sha256_file(output/name)
    policy=read_json(ROOT/'src/any2xsampa/config/innovation_stagewise_v1.json')
    write_json_new(output/'POLICY.json',policy)
    write_json_new(output/'FREEZE.json',{'schema_version':'innovation_implementation_v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'bindings':bindings,
        'policy_sha256':sha256_file(output/'POLICY.json'),'vocabulary':vocabulary,
        'isolated_python':bundle['python'],'dual_run':str(dual),'singing_run':str(singing),
        'snapshot':bundle['snapshot'],'new_full_acoustic_forwards':0,'new_quality_training_starts':0,
        'cached_prototype_profile_job_cap':1,'optimizer_updates':0,
        'staged_scope':'factor experiment; event mathematical/gradient/API/resource feasibility; paired-source audit. No undefined fivefold training budget inferred.'})
    print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    prepare(p.parse_args().run.resolve())

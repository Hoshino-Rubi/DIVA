"""Fixed 2x2 saved-logit experiment, references used only after decoding."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
from pathlib import Path
import argparse
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.adapters.ctc_factor_exchange import exchange_factors,log_probabilities,reconstruction_check,diagnostics
from input_native_probe_common import a1_gate,aggregate,prediction,SONGS
from verify_input_supervision import export_replay


def verify_launch(run: Path, phase: str) -> dict:
    freeze=read_json(run/'FREEZE.json');review=read_json(run/('CODE_REVIEW_'+phase+'.json'))
    if review['approved_for_launch'] is not True or review['freeze_sha256']!=sha256_file(run/'FREEZE.json'):
        raise ValueError('review absent or stale')
    verify_file(run/'POLICY.json',freeze['policy_sha256'])
    for p,h in freeze['bindings'].items():verify_file(Path(p),h)
    for p,h in review['code_bindings'].items():verify_file(Path(p),h)
    p=Path(review['verification']['path']);verify_file(p,review['verification']['sha256'])
    if not read_json(p)['passed']:raise ValueError('prelaunch verification failed')
    return freeze


def run_probe(run: Path) -> None:
    import numpy as np
    from safetensors.numpy import load_file
    freeze=verify_launch(run,'factor');policy=read_json(run/'POLICY.json')['factor_exchange']
    if policy['primary']!='xv' or policy['cells']!=['xx','vv','xv','vx']:
        raise ValueError('unplanned factorization')
    out=run/'factor';write_json_new(out/'START.json',{'cells':policy['cells'],'primary':'xv',
        'planned_cached_decodes':80,'quality_training_starts':0,'full_acoustic_forwards':0})
    started=time.monotonic();inputs=read_json(run/'FACTOR_INPUTS.json')['rows']
    vocabulary=read_json(Path(freeze['vocabulary']['path']))
    pools={c:[] for c in policy['cells']};records=[];nearby=[]
    for row in inputs:
        wid=row['window_id'];sources=row['sources'];arrays={};payloads={}
        for c,e in sources.items():
            a=e['artifact'];verify_file(Path(a['evidence_path']),a['evidence_sha256'])
            verify_file(Path(a['posterior_path']),a['posterior_sha256'])
            arrays[c]=load_file(a['evidence_path'])['frame_logits']
            payloads[c]=read_json(Path(a['posterior_path']))
        frame_check=payloads['x']['producer']['crop']==payloads['v']['producer']['crop']
        if not frame_check or arrays['x'].shape!=arrays['v'].shape:raise ValueError('source frame mismatch')
        for cell in policy['cells']:
            dest=out/wid/cell
            write_json_new(out/'attempts'/(wid+'--'+cell+'.json'),{'window_id':wid,'cell':cell,'cached_decode_attempt':True})
            raw=exchange_factors(arrays[cell[0]],arrays[cell[1]]).astype(np.float32)
            normalized_error=float(np.max(np.abs(np.exp(raw.astype('float64')).sum(-1)-1)))
            if normalized_error>2e-6:raise ValueError('FP32 factor distribution not normalized')
            self_check=reconstruction_check(arrays[cell[0]],raw) if cell[0]==cell[1] else None
            p=payloads[cell[0]]['producer'];a=sources[cell[0]]['input_audio']
            entry={'audio_path':a['path'],'audio_metadata':p['input_audio'],'crop':p['crop'],
                   'geometry':payloads[cell[0]]['decoding']['geometry'],
                   'runtime':{'cpu_threads':4,'model_forward_executed':False,'operation':'factor_exchange_cached_logits_v1'}}
            metadata={**p['model_metadata'],'factor_exchange':{'cell':cell,'primary':cell=='xv',
                'source_posteriors':{c:e['artifact']['posterior_sha256'] for c,e in sources.items()},
                'source_audio_hashes':{c:e['input_audio']['sha256'] for c,e in sources.items()},
                'source_reference_used':False,'same_frame_no_warp':True}}
            artifact=export_replay(raw,vocabulary,entry,metadata,dest,Path(freeze['vocabulary']['path']))
            if artifact['prediction']!=prediction(raw,vocabulary):raise ValueError('codec changed sequence')
            if self_check and artifact['prediction']!=payloads[cell[0]]['decoding']['collapsed_raw_labels']:
                raise ValueError('historical control sequence did not replay')
            score_row={'window_id':wid,'song_id':row['song_id'],'reference':row['reference'],'prediction':artifact['prediction']}
            pools[cell].append(score_row)
            records.append({'window_id':wid,'song_id':row['song_id'],'cell':cell,'reconstruction':self_check,
                            'normalization_error':normalized_error,**artifact})
            if cell=='xv':nearby.append({'window_id':wid,**diagnostics(arrays['x'],arrays['v'],raw)})
        print('FACTOR',wid,'all four cells replayed',flush=True)
    metric={c:aggregate(rows) for c,rows in pools.items()}
    if metric['xx']['edit_distance']!=591 or metric['vv']['edit_distance']!=524:
        raise ValueError('frozen control metrics changed')
    gate=a1_gate(pools['xv'],pools['xx'])
    write_json_new(out/'RESULT.json',{'schema_version':'ctc_factor_exchange_probe_v1','complete':True,
        'metrics':metric,'per_song':{c:{s:aggregate([r for r in rows if r['song_id']==s]) for s in SONGS} for c,rows in pools.items()},
        'details':pools,'artifact_records':records,'diagnostics':nearby,'primary_gate':gate,
        'reverse_is_diagnostic_not_selectable':True,'full_acoustic_forwards':0,'quality_training_starts':0,
        'cached_decodes':len(records),'elapsed_seconds':time.monotonic()-started,
        'development_opened':False,'final_test_opened':False,'default_replaced':False})
    print('RESULT',{c:(m['edit_distance'],m['substitutions'],m['deletions'],m['insertions']) for c,m in metric.items()},gate['failures'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    run_probe(p.parse_args().run.resolve())

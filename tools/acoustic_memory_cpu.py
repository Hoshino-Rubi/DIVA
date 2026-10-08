"""Separate CPU bank, audio-only query, and post-freeze scoring stages."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
from acoustic_memory_common import protocol, read_json, write_json_new, bound, load_bound, load_arrays, save_arrays, deadline, sha256_file
import numpy as np
from any2xsampa.training.acoustic_memory import (SONGS, validate_query, validate_sources, normalized_keys,
    soft_ctc_values, memory_temperature, shuffled_values, exact_neighbors, softmax, memory_distribution, fuse)
from any2xsampa.adapters.phoneticxeus import ctc_collapse, emission_payload
from any2xsampa.training.phone_error_metrics import phone_metrics, align_phone_edits
from any2xsampa.training.event_head_evaluation import group_regressions, compare_songs


def baseline(run: Path, phase: str, check) -> dict:
    head = read_json(run / phase / 'head/RESULT.json')
    if head['complete'] is not True or head['head_requests'] != 229:
        raise ValueError('head predictions not frozen')
    frozen = bound(run / phase / 'head/RESULT.json')
    scoring = read_json(run / 'SCORING.json')['rows']  # only AFTER frozen head predictions
    lookup = {r['example_id']:r for r in head['rows'] if r['purpose'] == 'query'}
    freeze = protocol(run, review=True); vocab = load_bound(freeze['vocabulary']); labels = {i:p for p,i in vocab.items()}
    rows = []
    for ref in scoring:
        check(); r=lookup[ref['example_id']]; d=load_arrays(r['raw'])
        ids=ctc_collapse(d['logits'].argmax(1).tolist())
        pred=[labels[i] for i in ids]
        if pred != ref['baseline_prediction'] or ids != r['baseline_ids']:
            raise ValueError('baseline reproduction differs; stop before retrieval')
        rows.append({k:ref[k] for k in ('example_id','song_id','audio_role','reference')} | {'prediction':pred})
    metrics={role:phone_metrics([(r['reference'],r['prediction']) for r in rows if r['audio_role']==role]) for role in ('original','rendered')}
    if [metrics['original'][k] for k in ('reference_phones','edit_distance','substitutions','deletions','insertions')] != [1235,595,315,248,32] or [metrics['rendered'][k] for k in ('reference_phones','edit_distance')] != [4814,1551]:
        raise ValueError('reproduced baseline metric mismatch')
    return {'complete':True,'passed':True,'frozen_predictions':frozen,'rows':rows,'metrics':metrics,'matched_queries':len(rows)}


def bank(run: Path, phase: str, check) -> dict:
    if not read_json(run/phase/'baseline/RESULT.json')['passed']:raise ValueError('baseline gate not passed')
    sources=read_json(run/'SOURCES.json')['folds']
    head=read_json(run/phase/'head/RESULT.json')
    lookup={(r['fold'],r['example_id']):r for r in head['rows'] if r['purpose']=='source'}
    result={}
    for song in SONGS:
        rows=sources[song];validate_sources(rows,song)
        if rows != sorted(rows,key=lambda r:(r['song_id'],r['example_id'])):raise ValueError('bank tie order changed')
        keys=[];values=[];songs=[];spans=[];provenance=[];cursor=0;support=Counter()
        for row in rows:
            check();q=row['query']; h=lookup[song,row['example_id']]; d=load_arrays(h['raw'])
            if h['cache_sha256']!=q['sha256']:raise ValueError('bank cache hash changed')
            k=normalized_keys(d['z']);g,diag=soft_ctc_values(d['logits'],row['target_ids'])
            if k.shape!=(q['frames'],1024) or g.shape!=(q['frames'],428):raise ValueError('bank array shape')
            keys.append(k);values.append(g);songs.extend([row['song_id']]*len(k));spans.append((cursor,cursor+len(k)))
            provenance.append({'example_id':row['example_id'],'song_id':row['song_id'],'start_frame':0,
                'end_frame_exclusive':len(k),'bank_start':cursor,'bank_end_exclusive':cursor+len(k),
                'audio_sha256':q['audio_sha256'],'feature_sha256':q['sha256'],'reference_sha256':row['reference_sha256'],
                'checkpoint_sha256':h['checkpoint_sha256'],'head_artifact':h['raw'],'soft_alignment':diag})
            cursor+=len(k);support.update(row['target_ids'])
        key=np.concatenate(keys);gamma=np.concatenate(values)
        tau,nearest=memory_temperature(key,np.asarray(songs),check=check)
        shuffled,permutation=shuffled_values(gamma,spans)
        for lo,hi in spans:
            if not np.allclose(gamma[lo:hi].sum(0),shuffled[lo:hi].sum(0),atol=1e-9,rtol=1e-10):
                raise ValueError('shuffle did not preserve per-window marginals')
        path=run/phase/'bank'/song/'memory.safetensors'
        save_arrays(path,{'keys':key,'gamma':gamma,'shuffle_permutation':permutation,
            'cross_song_nearest_distances':nearest,'prior':gamma.mean(0)})
        result[song]={'raw':bound(path),'temperature':tau,'frame_count':len(key),'sources':provenance,
            'target_label_support':dict(support),'unrepresented_class_ids':[i for i in range(1,428) if i not in support],
            'all_frames_retained':True,'gamma_is_human_timing':False}
        print('bank',song,'frames',len(key),'temperature',tau,flush=True)
    return {'complete':True,'folds':result,'source_requests':len(lookup),'held_data_received':False}


def brute_neighbors(query: np.ndarray, keys: np.ndarray, k: int) -> tuple[np.ndarray,np.ndarray]:
    """Independent compiled exhaustive pairwise subtraction, no approximate index."""
    from scipy.spatial.distance import cdist
    d=cdist(query,keys,metric='sqeuclidean')
    ids=np.argsort(d,axis=1,kind='stable')[:,:min(k,len(keys))]
    return ids,np.take_along_axis(d,ids,axis=1)


def query(run: Path, phase: str, check) -> dict:
    # This function does not read SCORING/SOURCES/reference files.
    freeze=protocol(run,review=True);vocab=load_bound(freeze['vocabulary'])
    bank_result=read_json(run/phase/'bank/RESULT.json')
    head=read_json(run/phase/'head/RESULT.json')
    lookup={(r['fold'],r['example_id']):r for r in head['rows'] if r['purpose']=='query'}
    queries=read_json(run/'QUERIES.json')['rows']; output=[];brute=[]
    for song in SONGS:
        b=bank_result['folds'][song]; data=load_arrays(b['raw']);keys=data['keys'];g=data['gamma'];tau=b['temperature']
        shuffled=g[data['shuffle_permutation']];first_roles=set()
        for q in [q for q in queries if q['song_id']==song]:
            check();validate_query(q);h=lookup[song,q['example_id']];raw=load_arrays(h['raw'])
            if h['cache_sha256']!=q['sha256']:raise ValueError('query cache identity mismatch')
            key=normalized_keys(raw['z']);p=softmax(raw['logits'])
            indices,distances=exact_neighbors(key,keys,check=check)
            if phase=='replay' and q['audio_role'] not in first_roles:
                bi,bd=brute_neighbors(key,keys,16)
                if not np.array_equal(bi,indices) or not np.allclose(bd,distances,atol=1e-12,rtol=1e-12):
                    raise ValueError('independent brute-force neighbors differ')
                brute.append({'fold':song,'example_id':q['example_id'],'role':q['audio_role'],
                    'frames':len(key),'maximum_distance_error':float(abs(bd-distances).max())})
                first_roles.add(q['audio_role'])
            primary=fuse(p,memory_distribution(g,indices,distances,tau))
            arms={'endpoint':p,'memory':primary}
            if q['audio_role']=='original':
                arms['prior']=fuse(p,np.broadcast_to(data['prior'],p.shape))
                arms['shuffled']=fuse(p,memory_distribution(shuffled,indices,distances,tau))
            neighbor_path=run/phase/'query'/q['example_id']/'neighbors.safetensors'
            save_arrays(neighbor_path,{'indices':indices,'squared_distances':distances})
            for arm,distribution in arms.items():
                path=run/phase/'query'/q['example_id']/arm/'posterior.safetensors'
                # Preserve full FP64 distribution; stable public codec requires FP32 log evidence.
                if (distribution<=0).any():raise ValueError('zero posterior cannot be serialized without modifying evidence')
                logp=np.log(distribution).astype(np.float32)
                ids=ctc_collapse(distribution.argmax(1).tolist())
                if ids!=ctc_collapse(logp.argmax(1).tolist()):raise ValueError('FP32 posterior serialization changes sequence')
                save_arrays(path,{'probabilities':distribution,'frame_logits':logp})
                payload=emission_payload(logp,vocab,input_audio=q['audio_metadata'],crop=q['crop'],
                    geometry=q['geometry'],runtime={'offline':True,'cpu_threads':4,'head_device':'cuda:0','memory_device':'cpu'},
                    model_metadata={'vocabulary_sha256':freeze['vocabulary']['sha256'],'checkpoint_sha256':h['checkpoint_sha256'],
                        'method':arm,'memory_sha256':b['raw']['sha256'],'query_reference_received':False,
                        'posterior_representation':'FP32 log of saved full FP64 normalized distribution',
                        'reference_conditioned_training_bank_only':arm!='endpoint'})
                payload['raw_evidence']={'file':path.name,'sha256':sha256_file(path)}
                posterior=path.with_suffix('.json');write_json_new(posterior,payload)
                base_blank=raw['logits'].argmax(1)==0;new_ids=distribution.argmax(1)
                output.append({'example_id':q['example_id'],'song_id':song,'audio_role':q['audio_role'],'arm':arm,
                    'prediction_ids':ids,'prediction':payload['decoding']['collapsed_raw_labels'],
                    'posterior':bound(posterior),'raw':bound(path),'neighbors':bound(neighbor_path),
                    'blank_frames':{'baseline':int(base_blank.sum()),'candidate':int((new_ids==0).sum()),
                        'blank_to_phone':int((base_blank & (new_ids!=0)).sum()),
                        'phone_to_blank':int((~base_blank & (new_ids==0)).sum()),'queried':len(key)},
                    'bank_sha256':b['raw']['sha256'],'checkpoint_sha256':h['checkpoint_sha256']})
        print('query',song,'score units so far',len(output),flush=True)
    if len(output)!=338:raise ValueError('fixed scoring cohort incomplete')
    if phase=='replay' and len(brute)!=10:raise ValueError('independent nearest replay sample incomplete')
    return {'complete':True,'rows':output,'score_units':len(output),'brute_checks':brute,
        'query_reference_received':False,'dev_test_opened':False}


def score(run: Path, phase: str, check) -> dict:
    predictions=read_json(run/phase/'query/RESULT.json')
    if not predictions['complete'] or predictions['score_units']!=338:raise ValueError('predictions not frozen')
    prediction_binding=bound(run/phase/'query/RESULT.json')
    refs={r['example_id']:r for r in read_json(run/'SCORING.json')['rows']}
    pools=defaultdict(list);interior={}; metrics={}
    for pred in predictions['rows']:
        check();ref=refs[pred['example_id']]
        row={k:ref[k] for k in ('example_id','song_id','audio_role','reference')}
        row['prediction']=pred['prediction'];row['metrics']=phone_metrics([(row['reference'],row['prediction'])])
        pools[pred['arm']].append(row)
    for arm,rows in pools.items():
        metrics[arm]={role:phone_metrics([(r['reference'],r['prediction']) for r in rows if r['audio_role']==role])
            for role in ('original','rendered') if any(r['audio_role']==role for r in rows)}
        content=Counter()
        for row in [r for r in rows if r['audio_role']=='original']:
            phones=refs[row['example_id']]['phones'];notes=defaultdict(list)
            for p in phones:notes[p['note_id']].append(p['phone'])
            edges={phones[0]['note_id'],phones[-1]['note_id']};index=0
            for op,expected,actual in align_phone_edits(row['reference'],row['prediction']):
                if expected is None:continue
                note=phones[index]['note_id'];index+=1
                multi=sum(any(c in 'aeiouy\u0250\u0259\u0254\u025b\u0153\u00f8\u026a\u028a' for c in p) for p in notes[note])>=2
                if note not in edges and multi:content[op]+=1;content['reference_count']+=1
        interior[arm]=dict(content)
    primary=metrics['memory']['original'];base=metrics['endpoint']['original']
    render=metrics['memory']['rendered'];base_render=metrics['endpoint']['rendered']
    song=compare_songs([r for r in pools['memory'] if r['audio_role']=='original'],[r for r in pools['endpoint'] if r['audio_role']=='original'])
    failures=[]
    for key,limit in [('edit_distance',564),('insertions',32),('deletions',248)]:
        if primary[key]>limit:failures.append(key+'_over_limit')
    if len(song['improved'])<3:failures.append('fewer_than_three_songs_improve')
    if not song['macro_nonregressing']:failures.append('song_macro_regression')
    failures.extend('original_'+n+'_regression' for n in group_regressions(primary,base))
    if render['per']>base_render['per']+.03:failures.append('rendered_overall_regression')
    failures.extend('rendered_'+n+'_regression' for n in group_regressions(render,base_render))
    controls={arm:primary['edit_distance']<metrics[arm]['original']['edit_distance'] for arm in ('prior','shuffled')}
    return {'complete':True,'frozen_prediction_binding':prediction_binding,'metrics':metrics,'pools':dict(pools),
        'per_song':song,'gate':{'passed':not failures,'failures':failures,'automatic_promotion':False},
        'primary_strictly_beats_diagnostic_controls':controls,
        'acoustic_mechanism_not_established_by_single_shuffle':True,
        'internal_multivowel_content_errors':interior,'timings_are_phone_durations':False,
        'dev_test_opened':False,'reference_count_original':1235,'reference_count_rendered':4814}


if __name__=='__main__':
    import threading,os
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=('execute','replay'),required=True)
    p.add_argument('--stage',choices=('baseline','bank','query','score'),required=True);a=p.parse_args()
    out=a.run/a.phase/a.stage;before=time.monotonic()
    write_json_new(out/'START.json',{'stage':a.stage,'phase':a.phase,'cpu_threads':4,'pid':os.getpid(),'timeout_seconds':1500})
    def abort():
        write_json_new(out/'TIMEOUT.json',{'only_own_process_stopped':True,'timeout_seconds':1500});os._exit(73)
    timer=threading.Timer(1500,abort);timer.daemon=True;timer.start()
    try:
        from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
        enforce_offline(out/'offline');protocol(a.run,review=True)
        result=globals()[a.stage](a.run,a.phase,deadline(1500))
        result['elapsed_seconds']=time.monotonic()-before
        write_json_new(out/'RESULT.json',result)
    except Exception as exc:
        write_json_new(out/'FAILURE.json',{'error':str(exc),'type':type(exc).__name__,'automatic_retry':False});raise
    finally:timer.cancel()

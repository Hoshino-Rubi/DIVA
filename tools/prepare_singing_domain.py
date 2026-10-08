"""Freeze train-only domain experiments and audio-only cache requests."""
from __future__ import annotations
import argparse,sys,json,math
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,verify_file,write_json_new
from any2xsampa.training.singing_domain import policy,validate_audio_row

def prepare(run:Path)->None:
    spec=policy();requests=[];refs=[];sources={};exclusions=[]
    old=ROOT/'.state/runs/original-adaptation-20260919T012000'
    refp=old/'reviewed-training-reference.json'
    datasetp=ROOT/'.state/runs/limited-data-20260906T050251/reference/rendered-proxy-dataset.json'
    pairs_root=ROOT/'.state/runs/dual-input-ctc-20260920T073445032166Z'
    vp=ROOT/'.state/runs/limited-data-20260906T050251/model/src/model/xeusphoneme/resources/ipa_vocab.json'
    vocabulary=read_json(vp)
    def audio(row:dict,role:str,example_id:str|None=None)->dict:
        return {'example_id':example_id or row['example_id'],'song_id':row['song_id'],'split':'train','audio_role':role,
                'audio_path':row['audio_path'],'audio_sha256':row['audio_sha256'],'start_sec':row['start_sec'],'end_sec':row['end_sec']}
    original=[r for r in read_json(refp)['examples'] if r['split']=='train']
    assert len(original)==20 and sum(len(r['phones']) for r in original)==1235
    pair_index={r['example_id']:r for r in read_json(pairs_root/'PAIRS.json')['entries']}
    for row in original:
        assert row['status']=='human_confirmed' and row['audio_role']=='original'
        request=audio(row,'original');requests.append(request)
        target=[p['phone'] for p in row['phones']]
        refs.append({**request,'target_ipa':target,'target_ids':[vocabulary[p] for p in target],
                     'reference_nature':row['reference_nature'],'paired_example_id':row['example_id']+'--vocal'})
        item=pair_index[row['example_id']];verify_file(Path(item['path']),item['sha256']);pair=read_json(Path(item['path']))
        # Validate same original sampling interval through the prior frozen request.
        old_requests=read_json(pairs_root/'AUDIO_REQUESTS.json')
        old_rows=old_requests.get('rows',old_requests.get('requests',[]))
        original_request=next(x for x in old_rows if x['example_id']==row['example_id'])
        from any2xsampa.training.dual_input import validate_pair
        validate_pair(original_request,pair)
        assert original_request['audio_sha256']==row['audio_sha256']
        assert abs(original_request['start_sec']-row['start_sec'])<1e-8 and abs(original_request['end_sec']-row['end_sec'])<1e-8
        vocal=pair['audio']['vocal']
        vr={'example_id':row['example_id']+'--vocal','song_id':row['song_id'],'split':'train','audio_role':'separated_vocal',
            'audio_path':vocal['path'],'audio_sha256':vocal['sha256'],'start_sec':0.,'end_sec':vocal['samples']/44100}
        requests.append(vr)
        sources[item['path']]=item['sha256']
    rendered=[r for r in read_json(datasetp)['examples'] if r['split']=='train']
    for row in rendered:
        if not(row['eligible'] and row['training_eligible']):
            exclusions.append({'example_id':row['example_id'],'reason':row['exclusion_reasons']});continue
        ar=audio(row,'rendered');requests.append(ar)
        refs.append({**ar,'target_ipa':row['model_target_ipa'],'target_ids':row['model_target_ids'],
                     'reference_nature':'rendered_proxy_derived_not_original_acoustic_truth','paired_example_id':None})
    assert len([r for r in refs if r['audio_role']=='rendered'])==129
    # Uniform full-song candidates, selected without recognizer predictions.
    import numpy as np
    import soundfile as sf
    globalp=ROOT/'.state/runs/limited-data-20260906T050251/reference/reference-manifest.json'
    unlabeled=[];candidate_log=[]
    for song in read_json(globalp)['songs']:
        if song['song_id'] not in spec['songs']:continue
        source=song['original_audio'];verify_file(Path(source['path']),source['sha256'])
        known=[r for r in original if r['song_id']==song['song_id']]
        pool=[];duration=source['duration_sec'];rules=spec['ablations']
        with sf.SoundFile(source['path']) as stream:
            rate=stream.samplerate
            for start in range(0,int(duration-rules['unlabeled_window_seconds'])+1,int(rules['unlabeled_stride_seconds'])):
                end=start+rules['unlabeled_window_seconds'];why=[]
                margin=rules['supervised_exclusion_margin_seconds']
                if any(start<r['end_sec']+margin and end>r['start_sec']-margin for r in known):why.append('overlaps_confirmed_supervision_margin')
                stream.seek(round(start*rate));samples=stream.read(round((end-start)*rate),dtype='float32',always_2d=True)
                rms=float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
                db=20*math.log10(max(rms,1e-12))
                if db<rules['minimum_unlabeled_rms_dbfs']:why.append('below_rms_floor')
                record={'song_id':song['song_id'],'start_sec':start,'end_sec':end,'rms_dbfs':db,'exclusion_reasons':why}
                candidate_log.append(record)
                if not why:pool.append(record)
        maximum=rules['unlabeled_windows_per_song']
        indices=sorted(set(round(x) for x in np.linspace(0,len(pool)-1,min(maximum,len(pool))))) if pool else []
        for j in indices:
            chosen=pool[j];ar={'example_id':f"unlabeled-{song['song_id']}-{chosen['start_sec']:04d}",
                'song_id':song['song_id'],'split':'train','audio_role':'unlabeled_original',
                'audio_path':source['path'],'audio_sha256':source['sha256'],'start_sec':chosen['start_sec'],'end_sec':chosen['end_sec']}
            requests.append(ar);unlabeled.append(ar)
    for row in requests:validate_audio_row(row)
    assert len({r['example_id'] for r in requests})==len(requests)
    for r in requests:sources[r['audio_path']]=r['audio_sha256']
    for p in (refp,datasetp,vp,globalp):sources[str(p)]=sha256_file(p)
    run.mkdir(parents=True,exist_ok=True)
    paths={'policy':('POLICY.json',spec),'audio_requests':('AUDIO_REQUESTS.json',{'schema_version':'domain_audio_requests_v1','rows':requests,'labels_received':False}),
           'references':('REFERENCES.json',{'schema_version':'domain_training_references_v1','rows':refs,'unlabeled':unlabeled,'split_policy':'whole_song'})}
    freeze={'created_at_utc':datetime.now(timezone.utc).isoformat(),'sources':sources,'training_quality_experiment':True,
        'vocabulary_path':str(vp),'model_snapshot':str(vp.parents[4]),'runtime_python':str(ROOT/'.state/runs/limited-data-20260906T050251/runtime/Scripts/python.exe')}
    for key,(name,value) in paths.items():freeze[key]={'file':name,'sha256':write_json_new(run/name,value)}
    write_json_new(run/'FREEZE.json',freeze)
    write_json_new(run/'COVERAGE.json',{'original_windows':20,'original_reference_phones':1235,'rendered_windows':129,
        'rendered_excluded':exclusions,'unlabeled_windows':len(unlabeled),'unlabeled_seconds':sum(r['end_sec']-r['start_sec'] for r in unlabeled),
        'unlabeled_rms_does_not_establish_vocal_presence':True,'unlabeled_candidates':candidate_log,
        'unlabeled_by_song':{s:sum(r['song_id']==s for r in unlabeled) for s in spec['songs']}})
    preserved={str(p.resolve()):sha256_file(p) for p in (ROOT/'src').rglob('*') if p.is_file() and p.suffix in {'.py','.json'}}
    preserved.update(sources);write_json_new(run/'PRESERVATION_BEFORE.json',{'files':preserved})
    print(json.dumps({'run':str(run),'requests':len(requests),'references':len(refs),'unlabeled':len(unlabeled)},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();prepare(a.run.resolve())

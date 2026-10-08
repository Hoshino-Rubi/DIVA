"""Freeze and extract development inputs only after recipe selection."""
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file
from any2xsampa.training.singing_domain import validate_audio_row

def cache_dev(run:Path)->None:
    recipe=read_json(run/'RECIPE_SELECTION.json')
    if recipe['dev_or_test_used']:raise ValueError('recipe selection consumed development')
    source=ROOT/'.state/runs/original-adaptation-20260919T012000/reviewed-training-reference.json'
    originals=[r for r in read_json(source)['examples'] if r['split']=='dev']
    dataset=ROOT/'.state/runs/limited-data-20260906T050251/reference/rendered-proxy-dataset.json'
    rendered=[r for r in read_json(dataset)['examples'] if r['split']=='dev' and r['eligible']]
    assert len(originals)==3 and sum(len(r['phones']) for r in originals)==214 and len(rendered)==25
    pairs_root=ROOT/'.state/runs/dual-input-ctc-20260920T073445032166Z'
    pairs={r['example_id']:r for r in read_json(pairs_root/'PAIRS.json')['entries']}
    old_requests={r['example_id']:r for r in read_json(pairs_root/'AUDIO_REQUESTS.json')['requests']}
    requests=[];references=[];pair_bindings={}
    for row in originals:
        base={k:row[k] for k in ('example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec')}
        requests.append(base);phones=[p['phone'] for p in row['phones']]
        references.append({'example_id':row['example_id'],'source_example_id':row['example_id'],'cohort':'original_dev','target_ipa':phones})
        item=pairs[row['example_id']];verify_file(Path(item['path']),item['sha256']);pair=read_json(Path(item['path']))
        from any2xsampa.training.dual_input import validate_pair
        old=old_requests[row['example_id']];validate_pair(old,pair)
        if any(old[k]!=row[k] for k in ('audio_sha256','start_sec','end_sec')):raise ValueError('development pair source differs from confirmed reference')
        audio=pair['audio']['vocal'];verify_file(Path(audio['path']),audio['sha256'])
        vocal={**base,'example_id':row['example_id']+'--vocal','audio_role':'separated_vocal','audio_path':audio['path'],
               'audio_sha256':audio['sha256'],'start_sec':0.,'end_sec':audio['samples']/44100}
        requests.append(vocal);references.append({'example_id':vocal['example_id'],'source_example_id':row['example_id'],'cohort':'vocal_dev','target_ipa':phones})
        pair_bindings[vocal['example_id']]=item
    for row in rendered:
        ar={k:row[k] for k in ('example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec')}
        requests.append(ar);references.append({'example_id':row['example_id'],'source_example_id':row['example_id'],'cohort':'rendered_dev','target_ipa':row['model_target_ipa']})
    for row in requests:validate_audio_row(row,allow_dev=True)
    write_json_new(run/'DEVELOPMENT_REQUESTS.json',{'rows':requests,'recipe_selection_sha256':sha256_file(run/'RECIPE_SELECTION.json'),'pair_bindings':pair_bindings})
    write_json_new(run/'DEVELOPMENT_REFERENCES.json',{'rows':references,'source_reference_sha256':sha256_file(source),'rendered_dataset_sha256':sha256_file(dataset)})
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
    entries=[];started=time.monotonic();freeze=read_json(run/'FREEZE.json')
    with serial_candidate_job(),ProcessResourceMonitor(run/'development-cache-monitor',timeout_seconds=900),LockedAudioCache(max_bytes=16*1024**2,max_sources=8) as audio:
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),run/'development-runtime-cache',audio_cache=audio)
        for row in requests:entries.append(rt.save_prefix(row,run/'development-cache'/row['example_id'],allow_dev=True))
        write_json_new(run/'DEVELOPMENT_CACHE.json',{'architecture':rt.identity,'entries':entries,
            'requests_sha256':sha256_file(run/'DEVELOPMENT_REQUESTS.json'),'labels_supplied_to_encoder':False,
            'peak_reserved_bytes':rt.torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-started})
        print('Development cache:',len(entries),'windows',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);cache_dev(p.parse_args().run.resolve())

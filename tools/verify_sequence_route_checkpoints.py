"""Fresh-process checkpoint, metric, raw-audio/prefix, and posterior verification."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file


def verify(run:Path,stage:str)->None:
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.sequence_route_inference import verify_job,install_checkpoint,save_posterior,SequenceRoutePosteriorRecognizer
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.training.singing_domain_training import load_rows,predict,evaluate
    from any2xsampa.training.singing_domain import partition
    spec=read_json(run/'POLICY.json');parent=Path(read_json(run/'FREEZE.json')['parents']['singing'])
    if stage=='screen':
        if not (run/'ROUTE_SELECTION.json').exists():raise ValueError('screen not complete')
        jobs=[run/'screen'/arm/song for arm in ('control','sequence_risk') for song in spec['songs']]
    else:
        final=read_json(run/'FINAL_TRAINING.json')
        jobs=[run/'final'/final['arm']/('seed'+str(seed)) for seed in spec['training']['final_seeds']]
    output=run/('checkpoint-verification-'+stage);output.mkdir()
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=900):
        rt=SingingDomainRuntime(Path(read_json(parent/'FREEZE.json')['model_snapshot']),output/'runtime-cache')
        torch=rt.torch;rows,_=load_rows(parent,torch);labels=[p for p,i in sorted(rt.vocabulary.items(),key=lambda x:x[1])]
        checks=[]
        for job in jobs:
            result=verify_job(run,job);tail=install_checkpoint(rt,result)
            train,validation=partition(rows,result['held_song'])
            scored=evaluate(torch,tail,validation,labels) if validation else None
            if scored is not None and scored!=result['final_validation']:
                raise ValueError('fresh checkpoint prediction/metric mismatch')
            reference_row=next(r for r in train if r['audio_role']=='original')
            row={k:reference_row[k] for k in ('example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec')}
            row['end_sec']=row['start_sec']+8.
            # Both calls are audio-only and use the installed checkpoint in the full model.
            _,full,first=rt.base.encode(row)
            tensors,_=rt.encode_prefix(row)
            with torch.no_grad():
                cached,_=tail(tensors['prefix_features'].unsqueeze(0).cuda(),
                    torch.tensor([len(tensors['prefix_features'])],device='cuda'))
            difference=float((full-cached[0].cpu()).abs().max())
            if difference>2e-5 or not torch.equal(full.argmax(-1),cached[0].cpu().argmax(-1)):
                raise ValueError('fresh audio and prefix replay disagree')
            source={'input_audio':first['producer']['input_audio'],'crop':first['producer']['crop'],
                'geometry':first['decoding']['geometry'],'runtime':first['producer']['runtime']}
            name=result['arm']+'-'+(result['held_song'] or ('seed'+str(result['seed'])))
            posterior=save_posterior(rt,result,row,full,source,output/name)
            real=SequenceRoutePosteriorRecognizer(posterior).recognize(Path(row['audio_path']))
            fixture=SequenceRoutePosteriorRecognizer(posterior,fixture=True).recognize(Path(row['audio_path']))
            if [(p.time_range,p.candidate_phones) for p in real]!=[(p.time_range,p.candidate_phones) for p in fixture]:
                raise ValueError('real and fixture projections differ')
            checks.append({'job':str(job),'checkpoint_sha256':result['checkpoint_sha256'],
                'metrics_equal':scored is not None,'whole_held_song_exclusion_verified':result['held_song'] not in result['training_songs'] if result['held_song'] else None,
                'raw_audio_prefix_max_difference':difference,'real_fixture_equal':True,
                'posterior_path':str(posterior),'posterior_sha256':sha256_file(posterior),
                'audio_path':row['audio_path'],'evaluated_windows':len(validation)})
            print('VERIFIED',name,'maxdiff',difference,flush=True)
        write_json_new(output/'RESULT.json',{'checks':checks,'independent_process':True,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'no_optimizer_steps':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['screen','final'],required=True)
    args=parser.parse_args();verify(args.run.resolve(),args.stage)

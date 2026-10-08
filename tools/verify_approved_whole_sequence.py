"""Independent whole-route checkpoint, metrics and real/fixture replay."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file


def verify(run:Path,stage:str)->None:
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.whole_sequence_inference import verify_job,install_checkpoint,save_posterior,WholeSequencePosteriorRecognizer
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.training.singing_domain_training import load_rows,evaluate
    from any2xsampa.training.singing_domain import partition
    freeze=read_json(run/'FREEZE.json');spec=read_json(run/'POLICY.json');parent=Path(freeze['parents']['singing'])
    if stage=='screen':
        if not (run/'ROUTE_SELECTION.json').exists():raise ValueError('whole screening is incomplete')
        jobs=[run/'screen/whole_sequence'/song for song in spec['songs']]
    else:
        if not (run/'FINAL_TRAINING.json').exists():raise ValueError('whole final seeds are incomplete')
        jobs=[run/'final/whole_sequence'/('seed'+str(seed)) for seed in spec['training']['final_seeds']]
    output=run/('checkpoint-verification-'+stage);output.mkdir(exist_ok=False)
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=900):
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),output/'runtime-cache');torch=rt.torch
        rows,_=load_rows(parent,torch);labels=[p for p,i in sorted(rt.vocabulary.items(),key=lambda x:x[1])];checks=[]
        for job in jobs:
            result=verify_job(run,job);tail=install_checkpoint(rt,result)
            train,validation=partition(rows,result['held_song'])
            scored=evaluate(torch,tail,validation,labels) if validation else None
            if scored!=result['final_validation']:raise ValueError('fresh whole checkpoint predictions or metrics differ')
            source_row=next(r for r in train if r['audio_role']=='original')
            row={k:source_row[k] for k in ('example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec')}
            row['end_sec']=row['start_sec']+8.
            _,full,payload=rt.base.encode(row);features,_=rt.encode_prefix(row)
            with torch.no_grad():
                cached,_=tail(features['prefix_features'].unsqueeze(0).cuda(),torch.tensor([len(features['prefix_features'])],device='cuda'))
            difference=float((full-cached[0].cpu()).abs().max())
            if difference>2e-5 or not torch.equal(full.argmax(-1),cached[0].cpu().argmax(-1)):
                raise ValueError('whole-route raw audio/cache replay differs')
            origin={'input_audio':payload['producer']['input_audio'],'crop':payload['producer']['crop'],
                'geometry':payload['decoding']['geometry'],'runtime':payload['producer']['runtime']}
            name=result['held_song'] or 'seed'+str(result['seed'])
            posterior=save_posterior(rt,result,row,full,origin,output/name)
            real=WholeSequencePosteriorRecognizer(posterior).recognize(Path(row['audio_path']))
            fixture=WholeSequencePosteriorRecognizer(posterior,fixture=True).recognize(Path(row['audio_path']))
            if [(p.time_range,p.candidate_phones) for p in real]!=[(p.time_range,p.candidate_phones) for p in fixture]:
                raise ValueError('whole-route real/fixture projections differ')
            checks.append({'job':str(job),'checkpoint_sha256':result['checkpoint_sha256'],
                'metrics_equal':scored is not None,'raw_audio_prefix_max_difference':difference,'real_fixture_equal':True,
                'posterior_path':str(posterior),'posterior_sha256':sha256_file(posterior)})
            print('VERIFIED WHOLE',name,'maxdiff',difference,flush=True)
        write_json_new(output/'RESULT.json',{'checks':checks,'independent_process':True,'no_optimizer_steps':True,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['screen','final'],required=True)
    a=p.parse_args();verify(a.run.resolve(),a.stage)

"""Fresh-process full-audio/checkpoint/cache/fixture equivalence."""
from pathlib import Path
import argparse,sys,json
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.adapters.singing_domain_inference import load_domain_model,save_domain_posterior,SingingDomainPosteriorRecognizer
from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
from any2xsampa.adapters.candidate_inference import serial_candidate_job

def verify(run:Path,job:Path,output:Path)->None:
    output.mkdir(parents=True,exist_ok=False)
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=600):
        rt,tail,result=load_domain_model(run,job,output/'runtime-cache');torch=rt.torch
        index=read_json(run/'CACHE.json');requests=read_json(run/'AUDIO_REQUESTS.json')['rows']
        ids=set(result['validation_ids'] or [r['example_id'] for r in requests if r['audio_role']=='original'])
        selected=[r for r in requests if r['example_id'] in ids and r['audio_role']=='original'][:2]
        from safetensors.torch import load_file
        checks=[]
        smoke={**selected[0],'example_id':selected[0]['example_id']+'--smoke8','end_sec':selected[0]['start_sec']+8.}
        smoke_artifact=save_domain_posterior(rt,tail,result,smoke,output/'smoke8')
        smoke_pp=Path(smoke_artifact['posterior_path']);smoke_audio=Path(smoke['audio_path'])
        smoke_real=SingingDomainPosteriorRecognizer(smoke_pp).recognize(smoke_audio)
        smoke_fixture=SingingDomainPosteriorRecognizer(smoke_pp,fixture=True).recognize(smoke_audio)
        summary=lambda seq:[{'time_range':asdict(s.time_range),'phones':[asdict(p) for p in s.candidate_phones],'confidence':s.confidence} for s in seq]
        if summary(smoke_real)!=summary(smoke_fixture):raise ValueError('eight-second real/fixture mismatch')
        for row in selected:
            cached=next(r for r in index['entries'] if r['example_id']==row['example_id']);verify_file(Path(cached['path']),cached['sha256'])
            tensors=load_file(cached['path'],device='cpu')
            with torch.no_grad():
                features=tensors['prefix_features'].unsqueeze(0).cuda();lengths=torch.tensor([features.shape[1]],device='cuda')
                logits,_=tail(features,lengths,final_features=tensors['encoder_features'].unsqueeze(0).cuda() if tail.arm=='endpoint' else None)
            artifact=save_domain_posterior(rt,tail,result,row,output/row['example_id'])
            live=load_file(artifact['logits_path'])['frame_logits'];difference=float((live-logits[0].cpu()).abs().max())
            if difference>2e-5:raise ValueError('full audio inference differs from cached-prefix replay')
            posterior=Path(artifact['posterior_path']);audio=Path(row['audio_path'])
            real=SingingDomainPosteriorRecognizer(posterior).recognize(audio)
            fixture=SingingDomainPosteriorRecognizer(posterior,fixture=True).recognize(audio)
            view=lambda seq:[{'time_range':asdict(s.time_range),'candidate_phones':[asdict(p) for p in s.candidate_phones],'confidence':s.confidence} for s in seq]
            if view(real)!=view(fixture):raise ValueError('real/fixture output mismatch')
            checks.append({'example_id':row['example_id'],'maximum_logit_error':difference,'real_fixture_equal':True,'artifact':artifact})
        write_json_new(output/'VERIFICATION.json',{'job_result_sha256':sha256_file(job/'RESULT.json'),'checkpoint_sha256':result['checkpoint_sha256'],
            'fresh_process':True,'full_audio_inference':True,'checks':checks,'eight_second_smoke':smoke_artifact,
            'eight_second_real_fixture_equal':True,'smoke_has_no_reference_score':True,'dev_or_test_used':False})
        print(json.dumps({'checks':len(checks),'maximum_logit_error':max(c['maximum_logit_error'] for c in checks)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.run.resolve(),a.job.resolve(),a.output.resolve())

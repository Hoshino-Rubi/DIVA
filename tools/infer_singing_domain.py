"""Explicit experimental checkpoint inference; does not alter default recognizers."""
from pathlib import Path
from dataclasses import asdict
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.adapters.singing_domain_inference import load_domain_model,save_domain_posterior,SingingDomainPosteriorRecognizer
from any2xsampa.adapters.candidate_inference import audio_request,serial_candidate_job
from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
from any2xsampa.training.limited_data_artifacts import write_json_new,read_json,sha256_file

def infer(a:argparse.Namespace)->None:
    output=a.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'):raise ValueError('choose a new run output directory')
    request=audio_request(a.audio,a.start,a.duration);output.mkdir(parents=True)
    if a.fixture is not None:
        pp=a.fixture;payload=read_json(pp);crop=payload['producer']['crop']
        if abs(crop['requested_start_sec']-a.start)>1e-8 or abs(crop['requested_end_sec']-a.start-a.duration)>1e-8:raise ValueError('fixture crop mismatch')
        segments=SingingDomainPosteriorRecognizer(pp,fixture=True).recognize(a.audio)
    else:
        if a.run is None or a.job is None:raise ValueError('real experimental inference requires --run and --job')
        with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=900):
            rt,tail,checkpoint=load_domain_model(a.run.resolve(),a.job.resolve(),output/'runtime-cache')
            row={'example_id':'user-audio','song_id':'unassigned','split':'inference','audio_role':'original',
                **{k:request[k] for k in ('audio_path','audio_sha256','start_sec','end_sec')}}
            artifact=save_domain_posterior(rt,tail,checkpoint,row,output/'evidence',free_audio=True)
            pp=Path(artifact['posterior_path']);segments=SingingDomainPosteriorRecognizer(pp).recognize(a.audio)
    ipa=read_json(pp)['decoding']['collapsed_raw_labels']
    write_json_new(output/'RESULT.json',{'schema_version':'singing_domain_inference_result_v1','mode':'fixture' if a.fixture else 'real','posterior_path':str(pp.resolve()),'posterior_sha256':sha256_file(pp),
        'ipa_sequence':ipa,
        'segments':[asdict(s) for s in segments],'qualified_recognizer':False,'default_replacement':False,
        'phone_timing_is_emission_only':True,'audio_request':request})
    (output/'phones.ipa.txt').write_text(' '.join(ipa)+'\n',encoding='utf-8')
    print(output/'RESULT.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audio',type=Path,required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--job',type=Path);p.add_argument('--fixture',type=Path);p.add_argument('--output',type=Path,required=True);infer(p.parse_args())

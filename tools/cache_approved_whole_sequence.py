"""Audio-only serial full-source cache and real/fixture smoke in the existing runtime."""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file


def work(run:Path,stage:str,attempt:int=1)->None:
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
    from safetensors.torch import save_file
    freeze=read_json(run/'FREEZE.json')
    verify_file(run/'WHOLE_AUDIO_REQUESTS.json',freeze['whole_requests_sha256'])
    verify_file(run/'WHOLE_GEOMETRY.json',freeze['whole_geometry_sha256'])
    requests=read_json(run/'WHOLE_AUDIO_REQUESTS.json')['rows']
    geometry={r['example_id']:r for s in read_json(run/'WHOLE_GEOMETRY.json')['songs'] for r in s['chunks']}
    if attempt < 1:raise ValueError('attempt number must be positive')
    output=run/('whole-'+stage+('' if attempt==1 else '-attempt'+str(attempt)));output.mkdir(exist_ok=False)
    before=time.monotonic()
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=900),LockedAudioCache(max_bytes=16*1024**2) as audio:
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),output/'runtime-cache',audio_cache=audio)
        if stage=='smoke':
            checks=[]
            for seconds in (8.,20.):
                row={**requests[0],'example_id':'whole-smoke-'+str(int(seconds)),'start_sec':0.,'end_sec':seconds}
                tensors,payload=rt.encode_prefix(row)
                target=output/str(int(seconds));target.mkdir()
                raw=target/'features.safetensors';save_file(tensors,str(raw))
                vocab=target/'vocabulary.json';vocab.write_bytes((rt.base.snapshot/'src/model/xeusphoneme/resources/ipa_vocab.json').read_bytes())
                payload['raw_evidence']={'file':raw.name,'sha256':sha256_file(raw),'vocabulary_file':vocab.name,
                    'vocabulary_sha256':sha256_file(vocab),'logit_shape':list(tensors['frame_logits'].shape),'dtype':'float32'}
                posterior=target/'posterior.json';write_json_new(posterior,payload)
                real=PhoneticXeusPosteriorRecognizer(posterior).recognize(Path(row['audio_path']))
                fixture=PhoneticXeusPosteriorRecognizer(posterior,fixture=True).recognize(Path(row['audio_path']))
                if [(p.time_range,p.candidate_phones) for p in real]!=[(p.time_range,p.candidate_phones) for p in fixture]:
                    raise ValueError('whole smoke real/fixture mismatch')
                checks.append({'seconds':seconds,'real_fixture_equal':True,'frames':len(tensors['frame_logits']),
                    'posterior_path':str(posterior),'posterior_sha256':sha256_file(posterior)})
                print('SMOKE',seconds,'frames',len(tensors['frame_logits']),flush=True)
            write_json_new(output/'RESULT.json',{'checks':checks,'reference_received':False,'architecture':rt.identity,
                'elapsed_seconds':time.monotonic()-before,'peak_reserved_bytes':rt.torch.cuda.max_memory_reserved()})
            write_json_new(run/'SMOKE_ACCEPTED.json',{'result_path':str(output/'RESULT.json'),
                'result_sha256':sha256_file(output/'RESULT.json'),'attempt':attempt})
        else:
            smoke=read_json(run/'SMOKE_ACCEPTED.json')
            verify_file(Path(smoke['result_path']),smoke['result_sha256'])
            entries=[]
            for row in requests:
                entry=rt.save_prefix(row,output/row['example_id'])
                expected=geometry[row['example_id']]
                if entry['frames']!=expected['expected_frames'] or entry['crop']['source_start_sample']!=expected['source_start_sample'] or entry['crop']['source_end_sample_exclusive']!=expected['source_end_sample_exclusive']:
                    raise ValueError('actual cache differs from frozen sample/frame geometry')
                entries.append(entry);print('CACHE',len(entries),'/',len(requests),row['example_id'],entry['frames'],flush=True)
            write_json_new(run/'WHOLE_CACHE.json',{'entries':entries,'architecture':rt.identity,
                'audio_requests_sha256':freeze['whole_requests_sha256'],'geometry_sha256':freeze['whole_geometry_sha256'],
                'reference_received':False,'all_parameters_frozen':True,
                'elapsed_seconds':time.monotonic()-before,'peak_reserved_bytes':rt.torch.cuda.max_memory_reserved(),
                'audio_cache':audio.counters,'code_bindings':{str(Path(__file__).resolve()):sha256_file(Path(__file__))}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['smoke','cache'],required=True)
    p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();work(a.run.resolve(),a.stage,a.attempt)

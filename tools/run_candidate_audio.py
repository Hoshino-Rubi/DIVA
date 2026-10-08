"""Dedicated original-audio candidate runtime; labels are not accepted."""
from pathlib import Path
from contextlib import ExitStack
from dataclasses import asdict
import argparse,sys,time,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,sha256_object,write_json_new
from any2xsampa.adapters.candidate_inference import validate_bundle,validate_request


def run(a):
    verify_file(a.bundle,a.bundle_sha256);bundle=validate_bundle(a.bundle)
    verify_file(a.request,a.request_sha256);request=read_json(a.request);validate_request(request)
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'.state/runs'):raise ValueError('new run output required')
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,gpu_total_used_bytes,PhoneticXeusRuntime
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    enforce_offline(a.output/'runtime-cache')
    if gpu_total_used_bytes()>14*1024**3:raise RuntimeError('card budget exceeded before loading the model')
    from any2xsampa.adapters.guarded_phoneticxeus import GuardedPhoneHead
    from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
    from safetensors.numpy import load_file
    import numpy as np
    row={k:request[k] for k in ('audio_path','audio_sha256','start_sec','end_sec')}
    row.update(example_id='user-crop',song_id='user-selected-audio',split='inference',audio_role='user_input')
    started=time.monotonic();records=[]
    with ProcessResourceMonitor(a.output/'monitor',timeout_seconds=900),ExitStack() as stack:
        cache=None
        if a.audio_cache=='on':
            from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
            cache=stack.enter_context(LockedAudioCache(max_bytes=8*1024**2,max_sources=1))
        runtime=PhoneticXeusRuntime(Path(bundle['snapshot']),cache_dir=a.output/'runtime-cache',audio_cache=cache)
        spec=bundle['candidate'];head=GuardedPhoneHead(runtime.torch,runtime.model.model.ctc.ctc_lo,runtime.vocabulary,Path(spec['path']),spec['sha256'])
        loaded=time.monotonic()-started;first_logits=first_contract=None
        for i in range(1+a.warm_repeats):
            before=time.monotonic();f,_,payload=runtime.encode(row);encoded=time.monotonic()
            dest=a.output/f'iteration-{i:02d}';artifact=head.save(f,payload,dest);saved=time.monotonic()
            vocabulary=dest/'vocabulary.json';shutil.copyfile(bundle['vocabulary']['path'],vocabulary)
            pp=Path(artifact['posterior_path']);posterior=read_json(pp)
            # This is a newly created artifact; add adjacent vocabulary before publishing its final hash.
            posterior['raw_evidence']['vocabulary_file']='vocabulary.json'
            pp.write_text(__import__('json').dumps(posterior,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
            artifact['posterior_sha256']=sha256_file(pp)
            actual=PhoneticXeusPosteriorRecognizer(pp).recognize(Path(row['audio_path']))
            fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True).recognize(Path(row['audio_path']))
            projection=lambda rows:[{'time_range':asdict(s.time_range),'candidate_phones':[asdict(c) for c in s.candidate_phones],'confidence':s.confidence} for s in rows]
            contract=projection(actual)
            if contract!=projection(fixture):raise ValueError('real/fixture contract differs')
            logits=load_file(artifact['evidence_path'])['frame_logits']
            if first_logits is None:first_logits=logits;first_contract=contract
            elif not np.array_equal(logits,first_logits) or contract!=first_contract:raise ValueError('warm inference output differs')
            records.append({'iteration':i,'elapsed_seconds':time.monotonic()-before,'audio_encode_seconds':encoded-before,
                            'heads_and_save_seconds':saved-encoded,'replay_seconds':time.monotonic()-saved,
                            'artifact':artifact,'contract_sha256':sha256_object(contract),'real_fixture_equal':True,'logits_exact':True})
        report={'schema_version':'candidate_audio_worker_v1','model_load_seconds':loaded,'iterations':records,
                'first_result_worker_seconds':loaded+records[0]['elapsed_seconds'],'peak_reserved_bytes':runtime.torch.cuda.max_memory_reserved(),
                'audio_cache':cache.stats() if cache else None,'cpu_threads':4,'full_encoder_executed_each_iteration':True,'fp32':True,
                'bundle_sha256':a.bundle_sha256,'request_sha256':a.request_sha256,'labels_received':False}
    write_json_new(a.output/'RESULT.json',report)
    print('candidate audio complete',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','request','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('bundle-sha256','request-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--warm-repeats',type=int,choices=[0,5],default=0);p.add_argument('--audio-cache',choices=['on','off'],default='on')
    run(p.parse_args())

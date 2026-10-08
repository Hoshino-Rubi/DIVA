"""Opt-in real audio-only guarded candidate worker; one <=20s train/dev input."""
from pathlib import Path
import argparse
import sys,time
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.run_phoneticxeus_worker import validate_requests
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,sha256_object,write_json_new
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer,ctc_collapse
from any2xsampa.adapters.guarded_phoneticxeus import GuardedPhoneHead


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','requests','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('requests-sha256','candidate-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--warm-repeats',type=int,default=0,choices=[0,5]);a=p.parse_args()
    output=a.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'):raise ValueError('new run output required')
    verify_file(a.requests,a.requests_sha256);requests=read_json(a.requests);validate_requests(requests)
    if len(requests['rows'])!=1 or requests['rows'][0]['split'] not in {'train','dev'}:raise ValueError('one non-test request required')
    row=requests['rows'][0];before=time.monotonic()
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime,check_gpu_budget
    with ProcessResourceMonitor(output/'monitor',timeout_seconds=900):
        runtime=PhoneticXeusRuntime(a.snapshot,cache_dir=output/'runtime-cache')
        head=GuardedPhoneHead(runtime.torch,runtime.model.model.ctc.ctc_lo,runtime.vocabulary,a.candidate,a.candidate_sha256)
        loaded=time.monotonic()-before
        features,_,payload=runtime.encode(row);saved=head.save(features,payload,output/'evidence')
        vocab=a.snapshot/'src/model/xeusphoneme/resources/ipa_vocab.json';pp=Path(saved['posterior_path'])
        project=lambda values:[{'time_range':asdict(s.time_range),'candidate_phones':[asdict(c) for c in s.candidate_phones],'confidence':s.confidence} for s in values]
        actual=PhoneticXeusPosteriorRecognizer(pp,vocabulary_path=vocab).recognize(Path(row['audio_path']))
        fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True,vocabulary_path=vocab).recognize(Path(row['audio_path']))
        if project(actual)!=project(fixture):raise ValueError('real/fixture mismatch')
        cold=time.monotonic()-before
        from safetensors.numpy import load_file
        saved_tensors=load_file(saved['evidence_path']);warm=[];micro=[]
        for _ in range(a.warm_repeats):
            start=time.monotonic();f,_,q=runtime.encode(row);pred=head.predict(f);runtime.torch.cuda.synchronize()
            if not (pred['frame_logits']==saved_tensors['frame_logits']).all():raise ValueError('real repeat logits changed')
            warm.append(time.monotonic()-start)
        # Fixed features isolate adapter/head cost from encoder and file hashing.
        for _ in range(5):
            start=time.perf_counter()
            with runtime.torch.inference_mode():old=head.head(head.teacher(features.to('cuda'))).cpu().numpy()
            runtime.torch.cuda.synchronize();old_sec=time.perf_counter()-start
            start=time.perf_counter();head.predict(features);runtime.torch.cuda.synchronize();mix_sec=time.perf_counter()-start
            micro.append({'teacher_seconds':old_sec,'guarded_seconds':mix_sec})
        report={'requests_sha256':a.requests_sha256,'candidate_sha256':a.candidate_sha256,'row':row,'artifacts':saved,
            'real_fixture_equal':True,'contract_sha256':sha256_object(project(actual)),'model_loaded_seconds':loaded,'cold_end_to_end_seconds':cold,
            'warm_audio_encode_and_guard_seconds':warm,'micro_fixed_feature_adapter_seconds':micro,'cold_scope':'load+hash+audio+encode+save+real_fixture_replay',
            'warm_scope':'audio_read_hash+encode+two_adapters+guard; excludes save/replay','repeat_logits_exact':True,'resources':check_gpu_budget(runtime.torch),'peak_reserved_bytes':runtime.torch.cuda.max_memory_reserved(),
            'no_reference_or_lyrics_received':True,'test_opened':False,'default_changed':False}
        write_json_new(output/'RESULT.json',report);print('real guarded inference complete',cold,'seconds')

if __name__=='__main__':main()

"""Fresh CUDA process: replay all projected-replay checkpoints and posteriors."""
from pathlib import Path
from dataclasses import asdict
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    policy=read_json(run/'PROJECTED_REPLAY_POLICY.json');source=read_json(run/'POLICY.json')
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda
    enforce_offline(run/'checkpoint-replay/runtime-cache')
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.training.retention_adaptation import load_inputs,_evaluate,make_residual_adapter
    from any2xsampa.adapters.phoneticxeus import emission_payload,PhoneticXeusPosteriorRecognizer
    from safetensors.torch import load_file,save_file
    with ProcessResourceMonitor(run/'checkpoint-replay/monitor',timeout_seconds=600):
        torch=configure_cuda();data=load_inputs(Path(source['source_run']),source['bindings']['manifest']['sha256'],torch);reports=[]
        for seed in policy['seeds']:
            report_path=run/'projected-replay'/f'seed{seed}'/'RESULT.json';report=read_json(report_path)
            if report['policy_sha256']!=sha256_file(run/'PROJECTED_REPLAY_POLICY.json'):raise ValueError('training policy hash mismatch')
            local=[]
            for role,cp in report['checkpoints'].items():
                if cp is None:continue
                verify_file(Path(cp['path']),cp['sha256']);verify_file(Path(cp['evaluation_path']),cp['evaluation_sha256'])
                model=make_residual_adapter(torch).eval().requires_grad_(False).to('cuda');model.load_state_dict(load_file(cp['path']),strict=True)
                actual={'original':_evaluate(data['dev'],data['labels'],torch,model,data['head']),'rendered':_evaluate(data['protection'],data['labels'],torch,model,data['head'])}
                if actual!=read_json(Path(cp['evaluation_path'])):raise ValueError('independent checkpoint replay differs')
                artifacts=[]
                for row in data['dev']:
                    entry=next(e for e in data['original_index']['entries'] if e['example_id']==row['example_id'])
                    verify_file(Path(entry['posterior_path']),entry['posterior_sha256']);prior=read_json(Path(entry['posterior_path']))
                    with torch.inference_mode():logits=data['head'](model(row['features'].to('cuda'))).cpu()
                    output=run/'checkpoint-replay'/f'seed{seed}'/role/row['example_id'];output.mkdir(parents=True,exist_ok=False)
                    evidence=output/'evidence.safetensors';save_file({'frame_logits':logits},str(evidence))
                    producer=prior['producer'];payload=emission_payload(logits.numpy(),data['vocabulary'],input_audio=producer['input_audio'],crop=producer['crop'],model_metadata={**producer['model_metadata'],'projected_checkpoint_sha256':cp['sha256']},geometry=prior['decoding']['geometry'],runtime={'device':'cuda:0','offline':True,'fp32':True,'teacher_and_head_frozen':True})
                    payload.update(example_id=row['example_id'],song_id='shalong',split='dev',audio_role='original',raw_evidence={'file':evidence.name,'sha256':sha256_file(evidence)})
                    pp=output/'posterior.json';write_json_new(pp,payload);vocab=Path(data['manifest']['vocabulary']['path']);audio=Path(row['audio_path'])
                    real=PhoneticXeusPosteriorRecognizer(pp,vocabulary_path=vocab).recognize(audio);fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True,vocabulary_path=vocab).recognize(audio)
                    view=lambda rows:[{'time_range':asdict(s.time_range),'candidate_phones':[asdict(c) for c in s.candidate_phones],'confidence':s.confidence} for s in rows]
                    if view(real)!=view(fixture):raise ValueError('fixture differs')
                    artifacts.append({'posterior_path':str(pp),'posterior_sha256':sha256_file(pp),'real_fixture_equal':True})
                local.append({'role':role,'checkpoint_sha256':cp['sha256'],'predictions_and_metrics_equal':True,'posteriors':artifacts})
            reports.append({'seed':seed,'result_sha256':sha256_file(report_path),'checkpoints':local})
        write_json_new(run/'CHECKPOINT_REPLAY.json',{'schema_version':'projected_checkpoint_replay_v1','seeds':reports,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'test_loaded':False,'fresh_process':True})
    print('all 3 seeds independently replayed; 9 real/fixture posteriors equal')

if __name__=='__main__':main()

"""Authorized frozen test comparison using cached features; no labels in decoding."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.final_proxy_evaluation import verify_release,SYSTEMS
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,sha256_object,verify_file
from any2xsampa.adapters.phoneticxeus import emission_payload,PhoneticXeusPosteriorRecognizer
from any2xsampa.adapters.guarded_phoneticxeus import GuardedPhoneHead


def contract(rows: list) -> list[dict]:
    return [dict(time_range=asdict(s.time_range),candidate_phones=[asdict(c) for c in s.candidate_phones],confidence=s.confidence) for s in rows]


def validate_cached(request: dict,index: dict,vocabulary: dict,torch: Any) -> tuple[Any,Any,dict]:
    from safetensors.torch import load_file
    a,e=request['audio'],request['cache']
    for key in ('evidence','posterior'):verify_file(Path(e[key+'_path']),e[key+'_sha256'])
    verify_file(Path(a['audio_path']),a['audio_sha256'])
    tensors=load_file(e['evidence_path']);features,logits=tensors['encoder_features'],tensors['frame_logits']
    if tuple(features.shape)!=(e['frames'],1024) or tuple(logits.shape)!=(e['frames'],428):raise ValueError('cached tensor shape changed')
    if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in (features,logits)):raise ValueError('invalid frozen features')
    payload=read_json(Path(e['posterior_path']))
    if payload.get('test_outputs_sealed') is not True:raise ValueError('historical source seal changed')
    for k in ('example_id','song_id','split','audio_role'):
        if payload[k]!=a[k]:raise ValueError('cached posterior identity changed')
    p=payload['producer']
    if p['input_audio']['sha256']!=a['audio_sha256'] or p['model_metadata']!=index['model_metadata']:
        raise ValueError('cached posterior source/model mismatch')
    if p['crop']['requested_start_sec']!=a['start_sec'] or p['crop']['requested_end_sec']!=a['end_sec']:
        raise ValueError('cached posterior crop offset changed')
    if any(p.get(k) is not False for k in ('lyrics_used','score_used','rendered_reference_used')):
        raise ValueError('cached evidence used forbidden context')
    PhoneticXeusPosteriorRecognizer._verify_raw_projection(payload,Path(e['evidence_path']),vocabulary)
    return features,logits,payload


def save_posterior(system: str,logits: Any,source: dict,request: dict,gate: dict,gate_sha: str,
                   vocabulary: dict,output: Path) -> dict:
    from safetensors.numpy import save_file
    output.mkdir(parents=True,exist_ok=False)
    raw=output/'evidence.safetensors';save_file({'frame_logits':logits},str(raw))
    p=source['producer'];metadata={**p['model_metadata'],'final_evaluation_release_sha256':gate_sha,'evaluation_system':system}
    if system!='pretrained':metadata['teacher_checkpoint_sha256']=gate['bindings']['teacher_checkpoint']['sha256']
    if system=='frozen_guarded_candidate':
        metadata['student_checkpoint_sha256']=gate['bindings']['student_checkpoint']['sha256']
        metadata['guard_policy_sha256']=gate['bindings']['guard_policy']['sha256']
    payload=emission_payload(logits,vocabulary,input_audio=p['input_audio'],crop=p['crop'],model_metadata=metadata,
                             geometry=source['decoding']['geometry'],runtime={**p['runtime'],'encoder_evidence_reused':True,'head_replay_device':'cuda:0'})
    payload.update({k:request['audio'][k] for k in ('example_id','song_id','split','audio_role')})
    payload.update(test_outputs_sealed=False,final_evaluation_only=True,
                   evaluation_release=dict(sha256=gate_sha,scope=gate['scope'],inference_reference_used=False),
                   raw_evidence=dict(file=raw.name,sha256=sha256_file(raw),dtype='float32',full_label_count=428))
    payload['producer']['name']='phoneticxeus-final-'+system
    pp=output/'posterior.json';write_json_new(pp,payload)
    vp=Path(gate['bindings']['vocabulary']['path']);audio=Path(request['audio']['audio_path'])
    actual=PhoneticXeusPosteriorRecognizer(pp,vocabulary_path=vp).recognize(audio)
    fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True,vocabulary_path=vp).recognize(audio)
    if contract(actual)!=contract(fixture):raise ValueError('real/fixture posterior differs')
    return dict(system=system,posterior_path=str(pp.resolve()),posterior_sha256=sha256_file(pp),
                evidence_path=str(raw.resolve()),evidence_sha256=sha256_file(raw),
                prediction=payload['decoding']['collapsed_raw_labels'],
                real_fixture_equal=True,contract_sha256=sha256_object(contract(actual)),
                special_emissions=payload['decoding']['special_emissions'])


def run(gate_path: Path,gate_sha: str,output: Path,replay: Path|None=None) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda,check_gpu_budget,gpu_total_used_bytes,GIB
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    enforce_offline(output/'runtime-cache')
    gate,requests,index=verify_release(gate_path,gate_sha)
    used=gpu_total_used_bytes()
    if used>14*GIB:
        write_json_new(output/'BLOCKED.json',dict(reason='whole_gpu_above_14_gib',total_used_bytes=used,
            no_model_loaded=True,no_test_evidence_read=True,no_other_process_stopped=True,gate_sha256=gate_sha))
        raise RuntimeError('whole GPU exceeds approved 14 GiB budget; no CPU fallback')
    before=time.monotonic()
    with ProcessResourceMonitor(output/'monitor',timeout_seconds=600):
        torch=configure_cuda()
        from safetensors.torch import load_file
        from safetensors.numpy import load_file as load_numpy
        import numpy as np
        b=gate['bindings'];head=torch.nn.Linear(1024,428).float().eval().requires_grad_(False)
        head.load_state_dict(load_file(b['pretrained_head']['path']),strict=True);head=head.to('cuda')
        vocabulary=read_json(Path(b['vocabulary']['path']))
        engine=GuardedPhoneHead(torch,head,vocabulary,Path(b['candidate']['path']),b['candidate']['sha256'])
        modules={'head':head,'teacher':engine.teacher,'student':engine.student}
        before_weights={k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in modules.items()}
        original=None
        if replay:
            completed=read_json(gate_path.parent/'TEST_COMPLETE.json')
            if completed['evidence']['path']!=str(replay.resolve()):raise ValueError('replay path differs from completed run')
            verify_file(replay,completed['evidence']['sha256'])
            original=read_json(replay)
            if original.get('gate_sha256')!=gate_sha or original.get('status')!='complete':raise ValueError('replay does not match complete final comparison')
        else:
            write_json_new(gate_path.parent/'TEST_OPENING.json',dict(schema_version='final_proxy_test_opening_v1',
                gate_sha256=gate_sha,reference_sha256=b['reference']['sha256'],candidate_sha256=b['candidate']['sha256'],
                opened_utc=datetime.now(timezone.utc).isoformat(),output=str(output.resolve()),
                fixed_systems=SYSTEMS,selection_allowed=False))
        entries=[]
        for request in requests['rows']:
            start=time.monotonic();features,baseline,source=validate_cached(request,index,vocabulary,torch)
            check_gpu_budget(torch)
            with torch.inference_mode():fresh=head(features.to('cuda')).cpu()
            if not torch.allclose(fresh,baseline,atol=1e-4,rtol=1e-4) or not torch.equal(fresh.argmax(-1),baseline.argmax(-1)):
                raise ValueError('cached pretrained head replay changed')
            predicted=engine.predict(features)
            systems={'pretrained':baseline.numpy(),'old_seed43':predicted['teacher_logits'],'frozen_guarded_candidate':predicted['frame_logits']}
            eid=request['audio']['example_id'];local=[]
            for system,logits in systems.items():
                if replay:
                    prior=next(e for e in original['entries'] if e['example_id']==eid)
                    record=next(s for s in prior['systems'] if s['system']==system)
                    verify_file(Path(record['evidence_path']),record['evidence_sha256'])
                    verify_file(Path(record['posterior_path']),record['posterior_sha256'])
                    if not np.array_equal(logits,load_numpy(record['evidence_path'])['frame_logits']):raise ValueError('independent checkpoint replay logits differ')
                    pp=Path(record['posterior_path']);vp=Path(b['vocabulary']['path']);audio=Path(request['audio']['audio_path'])
                    actual=PhoneticXeusPosteriorRecognizer(pp,vocabulary_path=vp).recognize(audio)
                    fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True,vocabulary_path=vp).recognize(audio)
                    if contract(actual)!=contract(fixture) or sha256_object(contract(actual))!=record['contract_sha256']:
                        raise ValueError('independent replay contract changed')
                    payload=read_json(pp)
                    if payload['decoding']['collapsed_raw_labels']!=record['prediction']:raise ValueError('saved prediction drift')
                    local.append(dict(system=system,logits_exact=True,prediction_equal=True,real_fixture_equal=True,contract_sha256=record['contract_sha256']))
                else:local.append(save_posterior(system,logits,source,request,gate,gate_sha,vocabulary,output/eid/system))
            memory=check_gpu_budget(torch)
            entries.append(dict(example_id=eid,systems=local,guarded_frames=int(predicted['guard_mask'].sum()),
                                frames=features.shape[0],elapsed_seconds=time.monotonic()-start,memory=memory))
            print('verified' if replay else 'inferred',eid,'frames',features.shape[0],flush=True)
        if any(not torch.equal(value.detach().cpu(),before_weights[k][name]) for k,m in modules.items() for name,value in m.state_dict().items()):
            raise ValueError('frozen inference parameters changed')
        if any(p.requires_grad for m in modules.values() for p in m.parameters()):raise ValueError('inference parameters unexpectedly trainable')
        verify_release(gate_path,gate_sha)
        result=dict(schema_version='final_proxy_independent_replay_v1' if replay else 'final_proxy_model_evidence_v1',
            status='complete',gate_sha256=gate_sha,entries=entries,systems=SYSTEMS,
            device='cuda:0',dtype='float32',cpu_threads=4,offline=True,encoder_rerun=False,labels_received_by_predict=False,
            frozen_parameters_unchanged=True,elapsed_seconds=time.monotonic()-before,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            resources=check_gpu_budget(torch),test_outputs_opened=True,model_reselected=False,default_changed=False,
            independent_replay_source=({'path':str(replay),'sha256':sha256_file(replay)} if replay else None))
        result_path=output/('REPLAY.json' if replay else 'MODEL_EVIDENCE.json')
        write_json_new(result_path,result)
        if not replay:
            write_json_new(gate_path.parent/'TEST_COMPLETE.json',dict(gate_sha256=gate_sha,evidence=dict(path=str(result_path.resolve()),sha256=sha256_file(result_path)),completed_utc=datetime.now(timezone.utc).isoformat()))
        print('complete',round(result['elapsed_seconds'],3),'seconds',flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate',required=True,type=Path);parser.add_argument('--gate-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--replay',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'.state/runs'):raise ValueError('new run output required')
    run(args.gate.resolve(),args.gate_sha256,out,args.replay.resolve() if args.replay else None)


if __name__=='__main__':main()

"""Independent GPU head replay, old endpoint controls, then full audio checks."""
from __future__ import annotations
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file,sha256_object
from any2xsampa.training.event_head_protocol import load_protocol,load_cache
from any2xsampa.training.event_head_training import make_head,predict
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.training.phone_error_metrics import phone_metrics
from input_native_probe_common import compare_logits,parameter_digest


def replay(run: Path):
    freeze,spec,index=load_protocol(run,'CODE_REVIEW_replay.json')
    jobs=[]
    for arm in spec['arms']:
        for song in spec['songs']:
            p=run/'screen'/arm/song/'RESULT.json';r=read_json(p)
            if not r['complete'] or r['freeze_sha256']!=sha256_file(run/'FREEZE.json') or r['selected_epoch']!=20:
                raise ValueError('incomplete or stale quality job')
            verify_file(Path(r['checkpoint_path']),r['checkpoint_sha256']);jobs.append((p,r))
    output=run/'independent-replay'
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda,check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor,query_total_gpu_bytes
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from safetensors.torch import load_file
    from safetensors.numpy import load_file as load_np
    enforce_offline(run/'replay-runtime-cache');probes=[query_total_gpu_bytes() for _ in range(3)]
    if max(probes)>14*1024**3:raise RuntimeError('card budget before replay')
    write_json_new(output/'START.json',{'checkpoints':10,'old_endpoint_controls':5,'planned_full_acoustic_forwards':10,
        'quality_training':False,'gpu_preflight':probes,'review_sha256':sha256_file(run/'CODE_REVIEW_replay.json')})
    before=time.monotonic();reports=[];old_rows=[];full_checks=[]
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=1500):
        torch=configure_cuda();rows=load_cache(torch,index['rows']);lookup={r['example_id']:r for r in rows}
        state={k:v.cuda() for k,v in load_file(freeze['pretrained_head']['path'],device='cpu').items()}
        vocab=read_json(Path(freeze['vocabulary']['path']));labels={i:p for p,i in vocab.items()}
        controls=read_json(run/'CONTROLS.json')
        for song,source in controls.items():
            old=read_json(Path(source['result_path']));verify_file(Path(source['checkpoint_path']),source['checkpoint_sha256'])
            trained=load_file(source['checkpoint_path'],device='cuda')
            if set(trained)!={'residual.0.weight','residual.0.bias','residual.2.weight','residual.2.bias'}:raise ValueError('endpoint checkpoint changed architecture')
            expected={r['example_id']:r for r in old['final_validation']['examples']}
            for row in rows:
                if row['song_id']!=song:continue
                with torch.no_grad():
                    f=row['features'].cuda()
                    hidden=torch.nn.functional.linear(f,trained['residual.0.weight'],trained['residual.0.bias'])
                    residual=torch.nn.functional.linear(torch.nn.functional.gelu(hidden),trained['residual.2.weight'],trained['residual.2.bias'])
                    logits=torch.nn.functional.linear(f+residual,state['weight'],state['bias'])
                    ids=ctc_collapse(logits.argmax(-1).cpu().tolist());pred=[labels[i] for i in ids]
                if row['audio_role']=='original' and (expected[row['example_id']]['prediction']!=pred or expected[row['example_id']]['reference']!=row['target_ipa']):
                    raise ValueError('old endpoint original control did not replay')
                old_rows.append({'example_id':row['example_id'],'song_id':song,'audio_role':row['audio_role'],
                    'reference':row['target_ipa'],'prediction':pred,'metrics':phone_metrics([(row['target_ipa'],pred)]),
                    'checkpoint_sha256':source['checkpoint_sha256']})
        write_json_new(output/'ENDPOINT_CONTROLS.json',{'rows':old_rows,'completed_checkpoints':5,'cached_head_calls':149,'full_encoder_calls':0})
        for path,job in jobs:
            head=make_head(torch,state,job['arm']).eval();head.load_state_dict(load_file(job['checkpoint_path'],device='cuda'),strict=True)
            if parameter_digest(head)!=job['head_state_sha256']:raise ValueError('saved head state hash differs')
            checks=[]
            for original in job['validation']['details']:
                row=lookup[original['example_id']];ids,q,b=predict(torch,head,row['features'],job['arm'],spec)
                pred=[labels[i] for i in ids]
                if pred!=original['prediction'] or row['target_ipa']!=original['reference']:raise ValueError('checkpoint sequence/reference replay differs')
                if job['arm']=='H':
                    p=Path(original['posterior_path']);verify_file(p,original['posterior_sha256']);payload=read_json(p)
                    rawpath=p.parent/payload['raw_evidence']['file'];verify_file(rawpath,payload['raw_evidence']['sha256']);raw=load_np(str(rawpath))
                    error=compare_logits(q.numpy(),raw['frame_logits'],vocab);b_error=None
                else:
                    p=Path(original['event_path']);verify_file(p,original['event_sha256']);payload=read_json(p)
                    rawpath=p.parent/payload['scores']['file'];verify_file(rawpath,payload['scores']['sha256']);raw=load_np(str(rawpath))
                    error=compare_logits(q.numpy(),raw['frame_log_probs'],vocab);b_error=float(abs(b.numpy()-raw['boundary_logits']).max())
                    if b_error>2e-5:raise ValueError('event boundary logits differ')
                metric=phone_metrics([(row['target_ipa'],pred)])
                if metric!=original['metrics']:raise ValueError('independent metrics differ')
                checks.append({'example_id':row['example_id'],'content':error,'boundary_maximum_error':b_error,'metric_equal':True})
            reports.append({'arm':job['arm'],'held_song':job['held_song'],'source_result_sha256':sha256_file(path),
                'checkpoint_sha256':job['checkpoint_sha256'],'checks':checks})
            print('HEAD REPLAY',job['arm'],job['held_song'],len(checks),flush=True)
        # One predetermined original waveform for each checkpoint, never score-selected.
        from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
        runtime=SingingDomainRuntime(Path(freeze['snapshot']),output/'runtime-cache')
        for path,job in jobs:
            row=min((r for r in rows if r['song_id']==job['held_song'] and r['audio_role']=='original'),key=lambda r:(r['start_sec'],r['example_id']))
            key=job['arm']+'--'+job['held_song'];write_json_new(output/'forwards'/key/'START.json',{'example_id':row['example_id'],'arm':job['arm'],'complete_forward_attempt':True})
            request={k:row[k] for k in ('example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec')}
            live,_,payload=runtime.base.encode(request)
            if runtime.identity['sha256']!=row['architecture_sha256'] or payload['producer']['crop']!=row['crop']:raise ValueError('full waveform cache identity mismatch')
            head=make_head(torch,state,job['arm']).eval();head.load_state_dict(load_file(job['checkpoint_path'],device='cuda'),strict=True)
            ids,q,b=predict(torch,head,live,job['arm'],spec)
            expected=next(r for r in job['validation']['details'] if r['example_id']==row['example_id'])
            if [labels[i] for i in ids]!=expected['prediction']:raise ValueError('full audio head decode differs')
            cached_ids,cq,cb=predict(torch,head,row['features'],job['arm'],spec)
            error=compare_logits(q.numpy(),cq.numpy(),vocab)
            boundary=float((b-cb).abs().max()) if b is not None else None
            feature_error=float((live-row['features']).abs().max())
            if feature_error>2e-5 or (boundary is not None and boundary>2e-5):raise ValueError('full audio feature/boundary mismatch')
            record={'arm':job['arm'],'held_song':job['held_song'],'example_id':row['example_id'],
                'feature_maximum_error':feature_error,'content':error,'boundary_maximum_error':boundary,'event_sequence_equal':ids==cached_ids}
            write_json_new(output/'forwards'/key/'RESULT.json',record);full_checks.append(record)
            print('FULL AUDIO',key,'verified',flush=True)
        result={'passed':True,'checkpoints':reports,'endpoint_control_count':5,'head_cache_replays':sum(len(r['checks']) for r in reports),
            'control_cache_calls':149,'full_audio_checks':full_checks,'full_acoustic_forwards':len(full_checks),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-before,'quality_training_starts':0}
    write_json_new(output/'RESULT.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);replay(p.parse_args().run.resolve())

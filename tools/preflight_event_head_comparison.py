"""One accounted CUDA preflight: correctness, initialization and cost gate."""
from __future__ import annotations
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file
from any2xsampa.training.event_head_protocol import load_protocol,validate_cohort,load_cache
from any2xsampa.training.event_head_training import make_head,loss_on_row
from any2xsampa.training.segmental_event_exact import exact_sequence_nll,exact_viterbi
from any2xsampa.training.segmental_event import sequence_nll
from input_native_probe_common import compare_logits,parameter_digest


def preflight(run: Path):
    freeze,spec,index=load_protocol(run,'CODE_REVIEW_preflight.json');validate_cohort(index['rows'],spec)
    out=run/'preflight'
    if out.exists():raise FileExistsError('preflight already attempted; no retry')
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda,check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import query_total_gpu_bytes,ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from safetensors.torch import load_file
    from safetensors import safe_open
    enforce_offline(run/'preflight-runtime-cache');probes=[query_total_gpu_bytes() for _ in range(3)]
    if max(probes)>14*1024**3:raise RuntimeError('card over limit before preflight')
    write_json_new(out/'START.json',{'process_attempt':1,'max_gradient_cases':7,'quality_training':False,
        'optimizer_steps':0,'new_acoustic_forwards':0,'gpu_queries':probes,'review_sha256':sha256_file(run/'CODE_REVIEW_preflight.json')})
    with serial_candidate_job(),ProcessResourceMonitor(out/'monitor',timeout_seconds=900):
        torch=configure_cuda();torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        rows=load_cache(torch,index['rows']);headstate=load_file(freeze['pretrained_head']['path'],device='cpu')
        with safe_open(str(Path(freeze['snapshot'])/'model.safetensors'),framework='pt',device='cpu') as m:
            for k in ('weight','bias'):
                if not torch.equal(headstate[k],m.get_tensor('model.ctc.ctc_lo.'+k)):raise ValueError('pretrained head snapshot mismatch')
        headstate={k:v.cuda() for k,v in headstate.items()}
        heads={arm:make_head(torch,headstate,arm).eval() for arm in spec['arms']}
        initial={arm:parameter_digest(h) for arm,h in heads.items()}
        if not torch.equal(heads['H'].content.weight,heads['E'].content.weight):raise ValueError('H/E content initialization differs')
        vocab=read_json(Path(freeze['vocabulary']['path']));checks=[]
        with torch.no_grad():
            for row in rows:
                logits=heads['H'].content(row['features'].cuda()).cpu().numpy()
                check=compare_logits(logits,row['prior'].numpy(),vocab)
                checks.append({'example_id':row['example_id'],**check})
        # Two short correctness backward cases, not two trained models.
        x=torch.randn(12,8,device='cuda',requires_grad=True);b=torch.randn(12,device='cuda',requires_grad=True)
        q=x.log_softmax(-1);target=[1,1,2,3]
        slow=sequence_nll(torch,q,b,target);a=torch.autograd.grad(slow,(x,b),retain_graph=True)
        fast=exact_sequence_nll(torch,q,b,target);c=torch.autograd.grad(fast,(x,b))
        differences=[float((v-w).abs().max()) for v,w in zip(a,c)]
        for v,w in zip(a,c):torch.testing.assert_close(v,w,atol=spec['exact_engine']['gradient_absolute_tolerance'],rtol=spec['exact_engine']['gradient_relative_tolerance'])
        loss_error=abs(float(slow)-float(fast))
        if loss_error>spec['exact_engine']['loss_absolute_tolerance']:raise ValueError('CUDA exact loss differs')
        original=min((r for r in rows if r['audio_role']=='original'),key=lambda r:(-r['frames'],r['example_id']))
        rendered=min((r for r in rows if r['audio_role']=='rendered'),key=lambda r:(-r['frames'],r['example_id']))
        durations=[];decode_seconds=0.
        # Five real-cache gradient cases + two short comparisons = seven.
        for i,(arm,row) in enumerate([('E',original),('E',original),('E',rendered),('H',original),('H',rendered)]):
            write_json_new(out/f'case-{i:02d}-START.json',{'arm':arm,'example_id':row['example_id'],'optimizer_steps':0})
            head=heads[arm];head.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            loss,diag=loss_on_row(torch,head,row,arm,spec);loss.backward();torch.cuda.synchronize()
            seconds=time.monotonic()-t
            if any(not torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('nonfinite head gradient')
            durations.append({'arm':arm,'audio_role':row['audio_role'],'frames':row['frames'],'seconds':seconds,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'loss':float(loss.detach()),**diag})
            if arm=='E' and i==1:
                with torch.no_grad():
                    q,b=head(row['features'].cuda());t=time.monotonic();exact_viterbi(torch,q,b);torch.cuda.synchronize();decode_seconds=time.monotonic()-t
            head.zero_grad(set_to_none=True);check_gpu_budget(torch)
        if {a:parameter_digest(h) for a,h in heads.items()}!=initial:raise ValueError('preflight unexpectedly updated head')
        schedules=read_json(run/'SCHEDULES.json');cost={}
        for arm in spec['arms']:
            original_time=max(d['seconds'] for d in durations if d['arm']==arm and d['audio_role']=='original')
            rendered_time=max(d['seconds'] for d in durations if d['arm']==arm and d['audio_role']=='rendered')
            predictions={song:spec['cost_gate']['safety_factor']*(sc['original_slots']*original_time+sc['rendered_slots']*rendered_time)
                +spec['cost_gate']['fixed_process_and_evaluation_allowance_seconds'] for song,sc in schedules.items()}
            cost[arm]={'passed':max(predictions.values())<=spec['training']['maximum_job_seconds'],
                'fold_estimated_seconds':predictions,'longest_original_case_seconds':original_time,'longest_rendered_case_seconds':rendered_time}
        result={'passed':True,'correctness_passed':True,'initialization':checks,'all149_native_logit_checks_passed':True,
            'CUDA_gradient_maximum_errors':differences,'CUDA_loss_error':loss_error,'timing_cases':durations,'E_viterbi_seconds':decode_seconds,
            'cost_gate':cost,'parameters_unchanged':True,'cached_backward_cases':7,'optimizer_steps':0,'acoustic_forwards':0,
            'mixed_precision_note':'Neural/segment score FP32 CUDA; exact DP FP64 CPU, no duration or label pruning.'}
    write_json_new(out/'RESULT.json',result);print('PREFLIGHT_COST',cost,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);preflight(p.parse_args().run.resolve())

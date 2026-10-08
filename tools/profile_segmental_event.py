"""One bounded cached-evidence prototype profile, no optimizer or acoustic model."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from run_factor_exchange_probe import verify_launch
from input_native_probe_common import parameter_digest,compare_logits


def profile(run: Path) -> None:
    freeze=verify_launch(run,'event');review=read_json(run/'CODE_REVIEW_event.json')
    for p,h in review['data_bindings'].items():verify_file(Path(p),h)
    if not read_json(run/'factor/RESULT.json')['complete']:raise ValueError('execute scheme 2 first')
    output=run/'event-profile'
    if (output/'START.json').exists():raise ValueError('profile already started; no automatic retry')
    spec=read_json(run/'EVENT_SPEC.json')
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda,check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor,query_total_gpu_bytes
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    enforce_offline(output/'runtime-cache')
    probes=[query_total_gpu_bytes() for _ in range(3)]
    if any(v>14*1024**3 for v in probes):raise RuntimeError('GPU card budget exceeded before cached profile')
    write_json_new(output/'START.json',{'purpose':'prototype_cached_forward_backward_no_optimizer',
        'real_cache_cases':1,'synthetic_stress_cases':1,'optimizer_steps':0,'full_acoustic_forwards':0,
        'quality_training':False,'gpu_preflight_bytes':probes,'review_sha256':sha256_file(run/'CODE_REVIEW_event.json')})
    from any2xsampa.training.segmental_event import make_event_head,log_partitions
    from any2xsampa.adapters.segmental_event import decode_events,SegmentalEventRecognizer
    from safetensors.torch import load_file
    from safetensors.numpy import save_file
    from safetensors import safe_open
    import numpy as np
    started=time.monotonic();rows=[]
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=900):
        torch=configure_cuda();torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        headspec=spec['pretrained_head'];verify_file(Path(headspec['path']),headspec['sha256'])
        state=load_file(headspec['path'],device='cpu')
        with safe_open(str(Path(freeze['snapshot'])/'model.safetensors'),framework='pt',device='cpu') as full:
            if not torch.equal(state['weight'],full.get_tensor('model.ctc.ctc_lo.weight')) or not torch.equal(state['bias'],full.get_tensor('model.ctc.ctc_lo.bias')):
                raise ValueError('cached head differs from locked original phone projection')
        head=make_event_head(torch,state['weight'].cuda(),state['bias'].cuda())
        before=parameter_digest(head)
        source=read_json(run/'PROFILE_INPUT.json');entry=source['entry']
        verify_file(Path(entry['path']),entry['sha256']);raw=load_file(entry['path'],device='cpu')
        features=raw['encoder_features'].cuda()
        vocab=read_json(Path(freeze['vocabulary']['path']))
        with torch.no_grad():
            values=head.content(features).cpu().numpy()
        initial=compare_logits(values,raw['frame_logits'].numpy(),vocab)
        # Second case is a synthetic 20-second frame-geometry stress test,
        # not a new audio crop, reference, model-quality result or training job.
        generator=torch.Generator(device='cpu').manual_seed(29)
        stress=torch.randn((spec['synthetic_stress_frames'],1024),generator=generator,dtype=torch.float32).cuda()*.1
        cases=[('longest_original_cache',features,source['reference']['target_ids']),
               ('synthetic_long_window',stress,[1,2,2,3]*25)]
        for name,f,target in cases:
            write_json_new(output/(name+'-START.json'),{'frames':len(f),'labels':428,'target_events':len(target),
                'quality_reference':name=='longest_original_cache','optimizer_steps':0})
            t0=time.monotonic();head.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            q,b=head(f);den,num=log_partitions(torch,q,b,target);loss=den-num
            loss.backward();torch.cuda.synchronize()
            gradient={n:float(p.grad.detach().norm().cpu()) for n,p in head.named_parameters()}
            if any(not torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('nonfinite prototype gradient')
            memory=check_gpu_budget(torch)
            report={'case':name,'frames':len(f),'target_events':len(target),'log_partition_fp32':float(den.detach()),
                'target_log_partition_fp32':float(num.detach()),'nll':float(loss.detach()),'gradient_norms':gradient,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'memory_after':memory,
                'forward_backward_seconds':time.monotonic()-t0,'duration_cap_frames':None,
                'all_positive_durations_considered':True,'no_optimizer_update':True}
            if name=='longest_original_cache':
                score=q.detach().cpu().numpy();boundary=b.detach().cpu().numpy()
                decoded=decode_events(score,boundary)
                report['partition_fp32_vs_independent_fp64_error']=abs(report['log_partition_fp32']-decoded['log_partition'])
                if report['partition_fp32_vs_independent_fp64_error']>spec['partition_tolerance']:
                    raise ValueError('prototype partition precision check failed')
                dest=output/'evidence';dest.mkdir(parents=True,exist_ok=False)
                sp=dest/'scores.safetensors';save_file({'frame_log_probs':score,'boundary_logits':boundary},str(sp))
                vp=dest/'vocabulary.json';vp.write_bytes(Path(freeze['vocabulary']['path']).read_bytes())
                pp=dest/'events.json';write_json_new(pp,{'schema_version':'segmental_event_evidence_v1','scoring_spec':'mean_logp_start_hold_v1',
                    'reference_conditioning':False,'input_audio':entry['audio_metadata'],'crop':entry['crop'],'geometry':entry['geometry'],
                    'scores':{'file':sp.name,'sha256':sha256_file(sp)},'vocabulary':{'file':vp.name,'sha256':sha256_file(vp)},
                    'decoded_event_ids':decoded['event_ids'],'model_kind':'untrained_event_head_feasibility_only',
                    'encoder_features_sha256':entry['sha256'],'native_head_sha256':headspec['sha256'],
                    'not_a_quality_candidate':True,'times_are_ground_truth':False})
                real=SegmentalEventRecognizer(pp).recognize(Path(entry['audio_path']))
                fixture=SegmentalEventRecognizer(pp,fixture=True).recognize(Path(entry['audio_path']))
                if [asdict(s) for s in real]!=[asdict(s) for s in fixture]:raise ValueError('event real/fixture contract mismatch')
                report.update({'event_count':len(decoded['event_ids']),'real_fixture_equal':True,
                    'evidence_path':str(pp),'evidence_sha256':sha256_file(pp),'quality_metrics_intentionally_not_computed':True})
            write_json_new(output/(name+'-RESULT.json'),report);rows.append(report)
            del q,b,loss,den,num;head.zero_grad(set_to_none=True)
            print('EVENT_PROFILE',name,len(f),report['forward_backward_seconds'],report['peak_reserved_bytes'],flush=True)
        after=parameter_digest(head)
        if after!=before:raise ValueError('prototype parameters changed without an optimizer')
        result={'complete':True,'prototype_numerics_passed':True,'head_parameter_count':sum(p.numel() for p in head.parameters()),
            'head_parameters_before':before,'head_parameters_after':after,'initial_native_logits_check':initial,
            'records':rows,'elapsed_seconds':time.monotonic()-started,'quality_training_starts':0,'optimizer_steps':0,
            'new_acoustic_forwards':0,'cached_real_forward_backward_cases':1,'synthetic_stress_forward_backward_cases':1,
            'quality_improvement_not_tested':True,'default_replaced':False,'final_test_opened':False}
    write_json_new(output/'RESULT.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    profile(p.parse_args().run.resolve())

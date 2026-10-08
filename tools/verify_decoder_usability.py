"""Independent CPU decode/metric replay and same-input runtime comparison."""
from pathlib import Path
from statistics import median
from collections import Counter
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,verify_file,write_json_new
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.ctc_prefix import prefix_beam_decode
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.training.decoder_experiment import promotion_gate,validate_cohort,validate_policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    import numpy as np
    from safetensors.numpy import load_file
    policy=read_json(run/'POLICY.json');validate_policy(policy);cache=read_json(run/'HEAD_CACHE.json');validate_cohort(cache['entries'])
    for role in ('original','rendered'):
        if {r['song_id'] for r in cache['entries'] if r['split']=='train' and r['audio_role']==role}!={'promise','freedom-flower','queen-road-east','half-moon-night-song','like-you'}:raise ValueError('training song coverage differs')
    original=read_json(Path(policy['bindings']['segment_reference']['path']))
    rendered=read_json(Path(policy['bindings']['rendered_dataset']['path']))
    targets={('original',r['example_id'].lower()):[v['phone'] for v in r['source_phones'] if v['audibility'] in {'full','partial'}] for r in original['examples']}
    targets.update({('rendered',r['example_id'].lower()):r['model_target_ipa'] for r in rendered['examples'] if r['split'] in {'train','dev'} and r['eligible']})
    vocab=read_json(Path(policy['bindings']['vocabulary']['path']));labels={v:k for k,v in vocab.items()}
    summary=read_json(run/'DECODER_COMPARISON.json');rows=[];count=0
    for spec in summary['examples']:
        verify_file(Path(spec['path']),spec['sha256']);row=read_json(Path(spec['path']));target=targets[(row['audio_role'],row['example_id'].lower())]
        if row['reference']!=target:raise ValueError('reference source replay differs')
        verify_file(Path(row['evidence_path']),row['evidence_sha256']);raw=load_file(row['evidence_path'])
        for name,pred in row['predictions'].items():
            ids=prefix_beam_decode(raw['candidate'],beam_width=int(name[4:])).ids if name.startswith('beam') else ctc_collapse(raw[name].argmax(-1).tolist())
            actual=[labels[i] for i in ids]
            if actual!=pred or phone_metrics([(target,actual)])!=row['metrics'][name]:raise ValueError('independent decode/metric replay differs')
            count+=1
        rows.append(row)
    for system,cohorts in summary['aggregate'].items():
        for cohort,expected in cohorts.items():
            actual=phone_metrics([(r['reference'],r['predictions'][system]) for r in rows if r['cohort']==cohort and system in r['predictions']])
            if actual!=expected:raise ValueError('aggregate metric replay differs')
    for decoder in ('beam8','beam32'):
        if promotion_gate(summary['aggregate']['candidate'],summary['aggregate'][decoder],policy)!=summary['gates'][decoder]:raise ValueError('promotion replay differs')
    protocol=read_json(run/'BENCHMARK_PROTOCOL.json');performance={};common=None;checks=[]
    for job in protocol['jobs']:
        base=run/job['id'];exit=read_json(run/'jobs'/job['id']/'EXIT.json')
        if exit['exit_code']:raise ValueError('benchmark did not finish')
        result=read_json(base/'worker/RESULT.json');monitor=read_json(base/'worker/monitor/resource-monitor.json')
        if not monitor['operation_succeeded'] or result['peak_reserved_bytes']>7*1024**3 or monitor['maximum_sampled_total_used_bytes']>14*1024**3:raise ValueError('runtime resource gate failed')
        if len(result['iterations'])!=1+job['warm_repeats']:raise ValueError('missing warm iterations')
        for iteration in result['iterations']:
            art=iteration['artifact'];verify_file(Path(art['posterior_path']),art['posterior_sha256']);verify_file(Path(art['evidence_path']),art['evidence_sha256'])
            raw=load_file(art['evidence_path'])['frame_logits']
            if job['duration']==8:
                if common is None:common=raw
                elif not np.array_equal(common,raw):raise ValueError('cached/uncached/cold/hot logits differ')
            checks.append({'job':job['id'],'iteration':iteration['iteration'],'contract_sha256':iteration['contract_sha256']})
        if job['duration']!=8:continue
        entry=performance.setdefault(job['cache'],{'cold_end_to_end_seconds':[],'warm_complete_iteration_seconds':[],'peak_reserved_bytes':0,'peak_card_bytes':0})
        if job['warm_repeats']==0:entry['cold_end_to_end_seconds'].append(exit['elapsed_seconds'])
        else:entry['warm_complete_iteration_seconds']=[i['elapsed_seconds'] for i in result['iterations'][1:]]
        entry['peak_reserved_bytes']=max(entry['peak_reserved_bytes'],result['peak_reserved_bytes']);entry['peak_card_bytes']=max(entry['peak_card_bytes'],monitor['maximum_sampled_total_used_bytes'])
    for row in performance.values():
        if len(row['cold_end_to_end_seconds'])!=3 or len(row['warm_complete_iteration_seconds'])!=5:raise ValueError('incomplete benchmark')
        row['cold_median_seconds']=median(row['cold_end_to_end_seconds']);row['warm_median_seconds']=median(row['warm_complete_iteration_seconds'])
    old=ROOT/'.state/runs/guarded-retention-20260919T041500/real-smoke-cold1/evidence/evidence.safetensors'
    old_equal=np.array_equal(load_file(str(old))['frame_logits'],common)
    if not old_equal:raise ValueError('new entrypoint changed frozen legacy eight-second result')
    write_json_new(run/'INDEPENDENT_VERIFICATION.json',{'schema_version':'decoder_usability_independent_replay_v1','example_count':len(rows),'decoded_sequence_count':count,'source_references_replayed':True,
        'all_sequences_and_metrics_equal':True,'all_benchmark_logits_equal':True,'legacy_eight_second_logits_equal':old_equal,'runtime_checks':checks,'test_reopened':False})
    write_json_new(run/'PERFORMANCE_SUMMARY.json',{'schema_version':'candidate_complete_path_benchmark_v1','configurations':performance,
        'cold_relative_reduction':1-performance['on']['cold_median_seconds']/performance['off']['cold_median_seconds'],
        'warm_relative_reduction':1-performance['on']['warm_median_seconds']/performance['off']['warm_median_seconds'],
        'same_fp32_logits':True,'optimization_kind':'integration_of_preexisting_locked_audio_cache_not_new_algorithm','os_cache_flushed':False,
        'warm_scope':'audio read/full encoder/heads/save/real-fixture replay; excludes fresh process and model load','cold_scope':'complete fresh CLI process and result materialization',
        'initial_smoke_not_in_paired_benchmark':True})
    print('verified',len(rows),'examples,',count,'decoded sequences; all cache-mode logits and legacy result exact')
    print('benchmark',performance)

if __name__=='__main__':main()

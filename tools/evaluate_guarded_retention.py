"""Independent raw replay and frozen acoustic teacher-guard comparison."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,write_json_new,sha256_file
from any2xsampa.training.retention_adaptation import load_inputs
from any2xsampa.training.phoneticxeus_adaptation import make_residual_adapter
from any2xsampa.training.original_adaptation import _gate,_rendered_gate
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.adapters.teacher_guard import guard_logits


def evaluate(rows,data,adapter,policy,torch):
    raw_pairs=[];guard_pairs=[];raw_details=[];guard_details=[];counts=[]
    with torch.inference_mode():
        for row in rows:
            f=row['features'].to('cuda');t=data['head'](data['teacher'](f)).cpu().numpy();s=data['head'](adapter(f)).cpu().numpy()
            guarded,mask=guard_logits(s,t,data['vocabulary'],policy)
            target=row['model_target_ipa']
            for logits,pairs,details in ((s,raw_pairs,raw_details),(guarded,guard_pairs,guard_details)):
                prediction=[data['labels'][i] for i in ctc_collapse(logits.argmax(-1).tolist())]
                pairs.append((target,prediction));details.append({'example_id':row['example_id'],'song_id':row['song_id'],'reference':target,'prediction':prediction,'metrics':phone_metrics([(target,prediction)])})
            counts.append({'example_id':row['example_id'],'frames':len(mask),'guarded_frames':int(mask.sum())})
    return {'raw':{'metrics':phone_metrics(raw_pairs),'examples':raw_details},'guarded':{'metrics':phone_metrics(guard_pairs),'examples':guard_details},'counts':counts}


def run(destination):
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    enforce_offline(destination/'runtime-cache')
    from safetensors.torch import load_file
    policy=read_json(destination/'GUARD_POLICY.json')
    source_policy=read_json(ROOT/'.state/runs/retention-iteration-20260919T033200/POLICY.json')
    with ProcessResourceMonitor(destination/'resources',timeout_seconds=900):
        torch=configure_cuda();data=load_inputs(Path(source_policy['source_run']),source_policy['source_manifest_sha256'],torch)
        if policy['vocabulary_sha256']!=data['manifest']['vocabulary']['sha256']:raise ValueError('guard vocabulary binding mismatch')
        results=[]
        for spec in policy['report_bindings']:
            path=Path(spec['path']);verify_file(path,spec['sha256']);report=read_json(path);cp=report['checkpoints'][policy['checkpoint_role']]
            verify_file(Path(cp['path']),cp['sha256']);verify_file(Path(cp['evaluation_path']),cp['evaluation_sha256'])
            adapter=make_residual_adapter(torch).eval().requires_grad_(False).to('cuda');adapter.load_state_dict(load_file(cp['path']),strict=True)
            original=evaluate(data['dev'],data,adapter,policy,torch);rendered=evaluate(data['protection'],data,adapter,policy,torch)
            if read_json(Path(cp['evaluation_path']))!={'original':original['raw'],'rendered':rendered['raw']}:raise ValueError('independent checkpoint prediction/metric mismatch')
            og=_gate(original['guarded'],{'metrics':report['baseline_original']},{'metrics':report['teacher_original']})
            rg=_rendered_gate(rendered['guarded'],{'metrics':report['teacher_rendered']})
            result={'variant':report['variant']['id'],'seed':report['seed'],'checkpoint':cp,'original':original,'rendered':rendered,'original_gate':og,'rendered_gate':rg,'passed':og['passed'] and rg['passed'],'raw_independent_replay_equal':True}
            write_json_new(destination/report['variant']['id']/f'seed{report["seed"]}.json',result);results.append(result)
            print(result['variant'],result['seed'],original['guarded']['metrics']['per'],rendered['guarded']['metrics']['groups']['aspirated']['per'],result['passed'],flush=True)
        counts={v:sum(r['passed'] for r in results if r['variant']==v) for v in dict.fromkeys(r['variant'] for r in results)}
        eligible=[r for r in results if r['passed'] and counts[r['variant']]>=2]
        selected=min(eligible,key=lambda r:(r['original']['guarded']['metrics']['per'],r['seed'],r['checkpoint']['epoch'],list(counts).index(r['variant']))) if eligible else None
        summary={'schema_version':'guarded_retention_comparison_v1','guard_policy_sha256':sha256_file(destination/'GUARD_POLICY.json'),
         'passing_seeds_by_variant':counts,'selected':None if selected is None else {k:selected[k] for k in ('variant','seed','checkpoint','original_gate','rendered_gate')},
         'all_raw_replay_equal':True,'test_opened':False,'default_changed':False,'backbone_rerun':False,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
         'comparisons':[{'variant':r['variant'],'seed':r['seed'],'raw_original_per':r['original']['raw']['metrics']['per'],'guarded_original':r['original']['guarded']['metrics'],'guarded_rendered':r['rendered']['guarded']['metrics'],'passed':r['passed']} for r in results]}
        write_json_new(destination/'COMPARISON.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run(a.run.resolve())

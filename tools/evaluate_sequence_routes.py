"""Conditional final development/protection comparison, never training selection."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def release_gate(original:dict,rendered:dict,old_original:dict,old_rendered:dict,rules:dict)->dict:
    for value in (original,rendered,old_original,old_rendered):
        if any(isinstance(value[k],bool) or not isinstance(value[k],int) or value[k]<0 for k in ('reference_phones','edit_distance','substitutions','deletions','insertions')):
            raise ValueError('invalid development metric count')
        if (not value['reference_phones'] or not math.isfinite(value['per'])
                or abs(value['per']-value['edit_distance']/value['reference_phones'])>1e-12
                or value['edit_distance']!=sum(value[k] for k in ('substitutions','deletions','insertions'))):
            raise ValueError('invalid development metric arithmetic')
        for group in value['groups'].values():
            count=group['reference_count'];errors=sum(group[k] for k in ('substitutions','deletions','insertions'))
            if count and (not math.isfinite(group['per']) or abs(group['per']-errors/count)>1e-12):
                raise ValueError('invalid group metric arithmetic')
    for left,right in ((original,old_original),(rendered,old_rendered)):
        if (left['reference_phones']!=right['reference_phones'] or left['groups'].keys()!=right['groups'].keys()
                or any(left['groups'][k]['reference_count']!=right['groups'][k]['reference_count'] for k in left['groups'])):
            raise ValueError('development reference or group coverage changed')
    if original['reference_phones']!=214 or rendered['reference_phones']!=679:raise ValueError('development denominator changed')
    failures=[]
    for metric,limit in [('edit_distance','maximum_original_errors'),('insertions','maximum_original_insertions'),('deletions','maximum_original_deletions')]:
        if original[metric]>rules[limit]:failures.append('original_'+metric)
    for group,limit in [('aspirated','maximum_original_aspiration_errors'),('glottal_h','maximum_original_h_errors')]:
        if sum(original['groups'][group][k] for k in ('substitutions','deletions','insertions'))>rules[limit]:failures.append('original_'+group)
    if rendered['per']>old_rendered['per']+rules['maximum_protected_PER_increase']:failures.append('rendered_overall')
    for name,new,old in [('original',original,old_original),('rendered',rendered,old_rendered)]:
        for group,base in old['groups'].items():
            if base['reference_count']>=rules['minimum_group_references'] and new['groups'][group]['per']>base['per']+rules['maximum_protected_PER_increase']:
                failures.append(name+'_'+group+'_protected')
    return {'passed':not failures,'failures':failures,'default_replacement':False,
        'insufficient_original_groups':[g for g,v in original['groups'].items() if v['reference_count']<rules['minimum_group_references']]}


def evaluate(run:Path)->None:
    if not (run/'FINAL_TRAINING.json').exists():raise ValueError('no final training results; cannot evaluate or choose from screening on development')
    from safetensors.torch import load_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.sequence_route_inference import verify_job,install_checkpoint,save_posterior,SequenceRoutePosteriorRecognizer
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    spec=read_json(run/'POLICY.json');parent=Path(read_json(run/'FREEZE.json')['parents']['singing'])
    final=read_json(run/'FINAL_TRAINING.json')
    old=read_json(parent/'development-results/seed29/RESULT.json')['old_guarded']['metrics']
    refs={r['example_id']:r for r in read_json(parent/'DEVELOPMENT_REFERENCES.json')['rows']}
    index=read_json(parent/'DEVELOPMENT_CACHE.json');entries=index['entries']
    output=run/'development';output.mkdir()
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=900):
        rt=SingingDomainRuntime(Path(read_json(parent/'FREEZE.json')['model_snapshot']),output/'runtime-cache')
        if rt.identity['sha256']!=index['architecture']['sha256']:raise ValueError('development cache architecture differs')
        torch=rt.torch;labels=[p for p,i in sorted(rt.vocabulary.items(),key=lambda x:x[1])]
        summaries=[]
        for seed in spec['training']['final_seeds']:
            checkpoint=verify_job(run,run/'final'/final['arm']/('seed'+str(seed)))
            tail=install_checkpoint(rt,checkpoint);pairs={'original_dev':[],'rendered_dev':[]};examples=[]
            for entry in entries:
                ref=refs[entry['example_id']]
                if ref['cohort'] not in pairs:continue
                verify_file(Path(entry['path']),entry['sha256']);values=load_file(entry['path'])
                with torch.no_grad():logits,_=tail(values['prefix_features'].unsqueeze(0).cuda(),torch.tensor([entry['frames']],device='cuda'))
                ids=ctc_collapse(logits[0].argmax(-1).cpu().tolist());prediction=[labels[i] for i in ids]
                pair=ref['target_ipa'],prediction;pairs[ref['cohort']].append(pair)
                item={'example_id':entry['example_id'],'cohort':ref['cohort'],'reference':ref['target_ipa'],
                    'prediction':prediction,'metrics':phone_metrics([pair])}
                if ref['cohort']=='original_dev':
                    row={k:entry[k] for k in ('example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec')}
                    source={'input_audio':entry['audio_metadata'],'crop':entry['crop'],'geometry':entry['geometry'],
                        'runtime':{'device':'cuda:0','offline':True,'source':'verified_frozen_prefix','cache_sha256':entry['sha256']}}
                    pp=save_posterior(rt,checkpoint,row,logits[0],source,output/('seed'+str(seed))/entry['example_id'])
                    SequenceRoutePosteriorRecognizer(pp,fixture=True).recognize(Path(row['audio_path']))
                    item['posterior_path']=str(pp)
                examples.append(item)
            metrics={c:phone_metrics(p) for c,p in pairs.items()}
            gate=release_gate(metrics['original_dev'],metrics['rendered_dev'],old['original_dev'],old['rendered_dev'],spec['release'])
            result={'seed':seed,'metrics':metrics,'gate':gate,'examples':examples,'checkpoint_sha256':checkpoint['checkpoint_sha256']}
            write_json_new(output/('seed'+str(seed)+'-RESULT.json'),result);summaries.append(result)
            print('development seed',seed,'original errors',metrics['original_dev']['edit_distance'],'passed',gate['passed'],flush=True)
        passing=[r for r in summaries if r['gate']['passed']]
        chosen=min(passing or summaries,key=lambda r:(r['metrics']['original_dev']['per'],r['seed']))
        write_json_new(run/'DEVELOPMENT_SELECTION.json',{'research_seed':chosen['seed'],'passing_seeds':[r['seed'] for r in passing],
            'release_qualified':len(passing)>=spec['release']['minimum_passing_seeds'],'seeds':summaries,
            'vocal_transfer_used_for_selection':False,'default_replacement':False})
        # Exactly one transfer diagnostic after freezing the seed selection.
        checkpoint=verify_job(run,run/'final'/final['arm']/('seed'+str(chosen['seed'])))
        tail=install_checkpoint(rt,checkpoint);pairs=[];examples=[]
        for entry in entries:
            ref=refs[entry['example_id']]
            if ref['cohort']!='vocal_dev':continue
            verify_file(Path(entry['path']),entry['sha256']);values=load_file(entry['path'])
            with torch.no_grad():logits,_=tail(values['prefix_features'].unsqueeze(0).cuda(),torch.tensor([entry['frames']],device='cuda'))
            predicted=[labels[i] for i in ctc_collapse(logits[0].argmax(-1).cpu().tolist())]
            pair=ref['target_ipa'],predicted;pairs.append(pair)
            examples.append({'example_id':entry['example_id'],'reference':pair[0],'prediction':pair[1]})
        write_json_new(run/'VOCAL_TRANSFER.json',{'seed':chosen['seed'],'metrics':phone_metrics(pairs),'examples':examples,
            'used_for_selection':False,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    evaluate(parser.parse_args().run.resolve())

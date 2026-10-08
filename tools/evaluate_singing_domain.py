"""Evaluate frozen final seeds with identical original and rendered references."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import argparse,sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.training.singing_domain_evaluation import RELEASE_RULES,release_check
from any2xsampa.adapters.phoneticxeus import ctc_collapse,emission_payload
from any2xsampa.adapters.singing_domain_inference import load_domain_model,SingingDomainPosteriorRecognizer

def freeze_development(run:Path)->None:
    path=run/'DEVELOPMENT_POLICY.json'
    old=ROOT/'.state/runs/decoder-usability-20260920T011000'
    write_json_new(path,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'rules':RELEASE_RULES,
        'baseline_head_index':{'path':str(old/'HEAD_CACHE.json'),'sha256':sha256_file(old/'HEAD_CACHE.json')},
        'baseline_references':{'path':str(old/'SCORING_REFERENCES.json'),'sha256':sha256_file(old/'SCORING_REFERENCES.json')},
        'development_already_observed':True,'final_test_predictions_allowed':False,
        'labels_supplied_to_inference':False,'comparison_scope':'original214; rendered679; vocal transfer214 separate'})

def old_baselines(run:Path,refs:dict,labels:list[str])->dict:
    from safetensors.numpy import load_file
    spec=read_json(run/'DEVELOPMENT_POLICY.json')
    for name in ('baseline_head_index','baseline_references'):verify_file(Path(spec[name]['path']),spec[name]['sha256'])
    rows=read_json(Path(spec['baseline_head_index']['path']))['entries']
    oldrefs={r['cache_id']:r['phones'] for r in read_json(Path(spec['baseline_references']['path']))['rows']}
    pairs={'original_dev':[],'rendered_dev':[]};details=[]
    for row in rows:
        if row['cohort'] not in pairs:continue
        verify_file(Path(row['evidence_path']),row['evidence_sha256'])
        target=refs[row['example_id']]['target_ipa']
        if target!=oldrefs[row['cache_id']]:raise ValueError('old baseline reference differs from current development')
        logits=load_file(row['evidence_path'])['candidate']
        prediction=[labels[i] for i in ctc_collapse(logits.argmax(-1).tolist())]
        pairs[row['cohort']].append((target,prediction));details.append({'example_id':row['example_id'],'prediction':prediction})
    return {'metrics':{key:phone_metrics(value) for key,value in pairs.items()},'examples':details,'candidate':'prior_frozen_guarded_candidate'}

def evaluate_seed(run:Path,seed:int,*,transfer:bool=False)->None:
    policy=read_json(run/'DEVELOPMENT_POLICY.json')
    if policy['rules']!=RELEASE_RULES:raise ValueError('release policy changed')
    final=read_json(run/'FINAL_TRAINING.json');job=run/'final-training'/('seed'+str(seed))
    if seed not in [r['seed'] for r in final['seeds']]:raise ValueError('unplanned seed')
    if transfer and read_json(run/'DEVELOPMENT_SELECTION.json')['research_seed']!=seed:raise ValueError('transfer is allowed only after seed selection')
    output=run/('development-transfer' if transfer else 'development-results')/('seed'+str(seed));output.mkdir(parents=True,exist_ok=False)
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from safetensors.torch import load_file,save_file
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=900):
        rt,tail,checkpoint=load_domain_model(run,job,output/'runtime-cache');torch=rt.torch
        index=read_json(run/'DEVELOPMENT_CACHE.json')
        if index['architecture']['sha256']!=rt.identity['sha256']:raise ValueError('development architecture mismatch')
        refs={r['example_id']:r for r in read_json(run/'DEVELOPMENT_REFERENCES.json')['rows']}
        labels=[p for p,i in sorted(rt.vocabulary.items(),key=lambda x:x[1])]
        baseline=old_baselines(run,refs,labels);cohorts=('vocal_dev',) if transfer else ('original_dev','rendered_dev')
        pairs={key:[] for key in cohorts};pretrained={key:[] for key in pairs};details=[]
        for entry in index['entries']:
            if refs[entry['example_id']]['cohort'] not in cohorts:continue
            verify_file(Path(entry['path']),entry['sha256']);tensors=load_file(entry['path'],device='cpu')
            ref=refs[entry['example_id']];target=ref['target_ipa'];cohort=ref['cohort'];before=time.monotonic()
            with torch.no_grad():
                prefix=tensors['prefix_features'].unsqueeze(0).cuda()
                cached=tensors['encoder_features'].unsqueeze(0).cuda() if tail.arm=='endpoint' else None
                logits,_=tail(prefix,torch.tensor([prefix.shape[1]],device='cuda'),final_features=cached)
            values=logits[0].float().cpu().contiguous()
            prediction=[labels[i] for i in ctc_collapse(values.argmax(-1).tolist())]
            base=[labels[i] for i in ctc_collapse(tensors['frame_logits'].argmax(-1).tolist())]
            pairs[cohort].append((target,prediction));pretrained[cohort].append((target,base))
            details.append({'example_id':entry['example_id'],'cohort':cohort,'reference':target,'prediction':prediction,'metrics':phone_metrics([pairs[cohort][-1]])})
            if cohort in {'original_dev','vocal_dev'}:
                destination=output/entry['example_id'];destination.mkdir()
                metadata={**rt.base.identity,'singing_domain_architecture':rt.identity,'singing_domain_checkpoint':{
                    'arm':checkpoint['arm'],'variant':checkpoint['variant'],'seed':seed,'sha256':checkpoint['checkpoint_sha256'],'architecture_sha256':rt.identity['sha256']}}
                geometry=entry['geometry']
                payload=emission_payload(values.numpy(),rt.vocabulary,input_audio=entry['audio_metadata'],crop=entry['crop'],model_metadata=metadata,
                    geometry=geometry,runtime={'device':'cuda:0','precision':'float32','offline':True,'source':'frozen_prefix_cache_and_trained_tail',
                        'cache_sha256':entry['sha256'],'elapsed_seconds':time.monotonic()-before})
                raw=destination/'logits.safetensors';save_file({'frame_logits':values},str(raw))
                vp=destination/'vocabulary.json';vp.write_bytes((rt.base.snapshot/'src/model/xeusphoneme/resources/ipa_vocab.json').read_bytes())
                payload['raw_evidence']={'file':raw.name,'sha256':sha256_file(raw),'vocabulary_file':vp.name,'vocabulary_sha256':sha256_file(vp),'dtype':'float32','logit_shape':list(values.shape)}
                pp=destination/'posterior.json';write_json_new(pp,payload)
                real=SingingDomainPosteriorRecognizer(pp).recognize(Path(entry['audio_path']))
                fixture=SingingDomainPosteriorRecognizer(pp,fixture=True).recognize(Path(entry['audio_path']))
                view=lambda seq:[{'time_range':asdict(s.time_range),'phones':[asdict(p) for p in s.candidate_phones],'confidence':s.confidence} for s in seq]
                if view(real)!=view(fixture):raise ValueError('development real/fixture mismatch')
                details[-1].update(posterior_path=str(pp.resolve()),posterior_sha256=sha256_file(pp),real_fixture_equal=True)
        metrics={key:phone_metrics(value) for key,value in pairs.items()}
        gate=None if transfer else release_check(metrics['original_dev'],metrics['rendered_dev'],baseline['metrics']['original_dev'],baseline['metrics']['rendered_dev'])
        write_json_new(output/'RESULT.json',{'seed':seed,'checkpoint_sha256':checkpoint['checkpoint_sha256'],'metrics':metrics,
            'pretrained_metrics':{k:phone_metrics(v) for k,v in pretrained.items()},'old_guarded':baseline,'examples':details,'release_check':gate,
            'selection_input':'original_development_only; vocal diagnostics excluded','peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'reference_sha256':sha256_file(run/'DEVELOPMENT_REFERENCES.json'),'development_policy_sha256':sha256_file(run/'DEVELOPMENT_POLICY.json')})
        print('seed',seed,'transfer' if transfer else 'original','PER',metrics['vocal_dev' if transfer else 'original_dev']['per'],
              'release',gate['passed'] if gate else 'diagnostic only',gate['failures'] if gate else [],flush=True)

def select_seed(run:Path)->None:
    planned=read_json(run/'FINAL_TRAINING.json')['seeds']
    rows=[read_json(run/'development-results'/('seed'+str(r['seed']))/'RESULT.json') for r in planned]
    if {r['seed'] for r in rows}!={17,29,43}:raise ValueError('incomplete final seed comparison')
    passing=[r for r in rows if r['release_check']['passed']]
    candidate=min(passing or rows,key=lambda r:(r['metrics']['original_dev']['per'],r['seed']))
    write_json_new(run/'DEVELOPMENT_SELECTION.json',{'research_seed':candidate['seed'],
        'release_qualified':len(passing)>=RELEASE_RULES['minimum_passing_seeds'],
        'passing_seeds':[r['seed'] for r in passing],'method':'qualified first if any; original PER then seed; no vocal metrics',
        'seeds':[{'seed':r['seed'],'original':r['metrics']['original_dev'],'rendered':r['metrics']['rendered_dev'],'gate':r['release_check']} for r in rows],
        'all_seeds_reported':True,'vocal_transfer_not_used':True,'default_replacement':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');p.add_argument('--seed',type=int)
    p.add_argument('--transfer',action='store_true');p.add_argument('--select',action='store_true')
    a=p.parse_args()
    if a.freeze_only:freeze_development(a.run.resolve())
    elif a.select:select_seed(a.run.resolve())
    elif a.seed is not None:evaluate_seed(a.run.resolve(),a.seed,transfer=a.transfer)
    else:p.error('choose --freeze-only or --seed')

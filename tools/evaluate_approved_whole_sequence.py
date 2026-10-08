"""Fixed final-seed development checks for whole supervision; no final-test access."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from evaluate_sequence_routes import release_gate


def evaluate(run:Path)->None:
    if not (run/'FINAL_TRAINING.json').exists():raise ValueError('no final whole-source seeds to evaluate')
    if not (run/'checkpoint-verification-final/RESULT.json').exists():raise ValueError('independent checkpoint replay must precede development')
    from safetensors.torch import load_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.whole_sequence_inference import verify_job,install_checkpoint,save_posterior,WholeSequencePosteriorRecognizer
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    freeze=read_json(run/'FREEZE.json');spec=read_json(run/'POLICY.json');parent=Path(freeze['parents']['singing'])
    old=read_json(parent/'development-results/seed29/RESULT.json')['old_guarded']['metrics']
    refs={r['example_id']:r for r in read_json(parent/'DEVELOPMENT_REFERENCES.json')['rows']}
    index=read_json(parent/'DEVELOPMENT_CACHE.json');output=run/'development';output.mkdir(exist_ok=False)
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=900):
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),output/'runtime-cache');torch=rt.torch
        if rt.identity['sha256']!=index['architecture']['sha256']:raise ValueError('whole development architecture mismatch')
        labels=[p for p,i in sorted(rt.vocabulary.items(),key=lambda x:x[1])];summaries=[]
        def predict_entry(tail,entry):
            verify_file(Path(entry['path']),entry['sha256']);values=load_file(entry['path'])
            with torch.no_grad():
                logits,_=tail(values['prefix_features'].unsqueeze(0).cuda(),torch.tensor([entry['frames']],device='cuda'))
            prediction=[labels[i] for i in ctc_collapse(logits[0].argmax(-1).cpu().tolist())]
            return logits,prediction
        for seed in spec['training']['final_seeds']:
            result=verify_job(run,run/'final/whole_sequence'/('seed'+str(seed)));tail=install_checkpoint(rt,result)
            pairs={'original_dev':[],'rendered_dev':[]};examples=[]
            for entry in index['entries']:
                ref=refs[entry['example_id']]
                if ref['cohort'] not in pairs:continue
                logits,prediction=predict_entry(tail,entry);pair=(ref['target_ipa'],prediction)
                pairs[ref['cohort']].append(pair)
                item={'example_id':entry['example_id'],'cohort':ref['cohort'],'reference':pair[0],'prediction':pair[1],
                    'metrics':phone_metrics([pair])}
                if ref['cohort']=='original_dev':
                    row={k:entry[k] for k in ('example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec')}
                    source={'input_audio':entry['audio_metadata'],'crop':entry['crop'],'geometry':entry['geometry'],
                        'runtime':{'device':'cuda:0','offline':True,'source':'verified_frozen_prefix','cache_sha256':entry['sha256']}}
                    posterior=save_posterior(rt,result,row,logits[0],source,output/('seed'+str(seed))/entry['example_id'])
                    WholeSequencePosteriorRecognizer(posterior,fixture=True).recognize(Path(row['audio_path']))
                    item['posterior_path']=str(posterior)
                examples.append(item)
            metrics={cohort:phone_metrics(rows) for cohort,rows in pairs.items()}
            gate=release_gate(metrics['original_dev'],metrics['rendered_dev'],old['original_dev'],old['rendered_dev'],spec['release'])
            summary={'seed':seed,'metrics':metrics,'gate':gate,'examples':examples,'checkpoint_sha256':result['checkpoint_sha256']}
            write_json_new(output/('seed'+str(seed)+'-RESULT.json'),summary);summaries.append(summary)
            print('WHOLE DEVELOPMENT',seed,metrics['original_dev']['edit_distance'],gate['passed'],flush=True)
        passing=[r for r in summaries if r['gate']['passed']]
        chosen=min(passing or summaries,key=lambda r:(r['metrics']['original_dev']['per'],r['seed']))
        write_json_new(run/'DEVELOPMENT_SELECTION.json',{'research_seed':chosen['seed'],
            'passing_seeds':[r['seed'] for r in passing],'release_qualified':len(passing)>=spec['release']['minimum_passing_seeds'],
            'seeds':summaries,'vocal_transfer_used_for_selection':False,'default_replacement':False})
        result=verify_job(run,run/'final/whole_sequence'/('seed'+str(chosen['seed'])));tail=install_checkpoint(rt,result)
        pairs=[];examples=[]
        for entry in index['entries']:
            ref=refs[entry['example_id']]
            if ref['cohort']!='vocal_dev':continue
            _,prediction=predict_entry(tail,entry);pairs.append((ref['target_ipa'],prediction))
            examples.append({'example_id':entry['example_id'],'reference':ref['target_ipa'],'prediction':prediction})
        write_json_new(run/'VOCAL_TRANSFER.json',{'seed':chosen['seed'],'metrics':phone_metrics(pairs),'examples':examples,
            'used_for_selection':False,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);evaluate(p.parse_args().run.resolve())

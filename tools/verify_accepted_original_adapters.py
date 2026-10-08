"""Independent checkpoint replay, metrics and full-vocabulary posterior evidence."""
from pathlib import Path
import argparse
import sys
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,sha256_object,write_json_new
from any2xsampa.training.reviewed_training_reference import validate_training_reference
from any2xsampa.training.phoneticxeus_adaptation import make_residual_adapter,_load_cache
from any2xsampa.training.original_adaptation import _reference_ids,_load_evidence,_labels,_evaluate,_gate,_rendered_gate,validate_original_index
from any2xsampa.adapters.phoneticxeus import emission_payload,PhoneticXeusPosteriorRecognizer


def run(root: Path,output: Path) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    enforce_offline(output/'runtime-cache')
    from safetensors.torch import load_file,save_file
    manifest=read_json(root/'original-training-manifest.json')
    for key in ('segment_reference','original_feature_index','rendered_dataset','rendered_feature_index','seed43_checkpoint','vocabulary'):
        spec=manifest[key];verify_file(Path(spec['path']),spec['sha256'])
    reference=read_json(Path(manifest['segment_reference']['path']));validate_training_reference(reference)
    vocabulary=read_json(Path(manifest['vocabulary']['path']));labels=_labels(vocabulary)
    with ProcessResourceMonitor(output/'resources',timeout_seconds=600):
        torch=configure_cuda();index=read_json(Path(manifest['original_feature_index']['path']))
        refs=_reference_ids(reference['examples'],vocabulary);validate_original_index(index,refs,manifest['vocabulary']['sha256'])
        loaded=_load_evidence(index,refs,torch,root=root);dev=[r for r in loaded if r['split']=='dev']
        head=torch.nn.Linear(1024,428).eval().requires_grad_(False).to('cuda');verify_file(Path(index['head']['path']),index['head']['sha256']);head.load_state_dict(load_file(index['head']['path']))
        rendered,rhead,_=_load_cache(Path(manifest['rendered_feature_index']['path']),read_json(Path(manifest['rendered_dataset']['path'])),torch)
        rendered=[r for r in rendered if r['split']=='dev'];rhead.to('cuda')
        output.mkdir(parents=True,exist_ok=True);checks=[]
        for seed in (17,29,43):
            directory=root/f'original-training/seed{seed}';report=read_json(directory/'training-report.json')
            if report['manifest_sha256']!=sha256_file(root/'original-training-manifest.json'):raise ValueError('stale training report')
            checkpoint=report['checkpoints']['best_overall'];verify_file(Path(checkpoint['path']),checkpoint['sha256'])
            adapter=make_residual_adapter(torch).eval().requires_grad_(False).to('cuda');adapter.load_state_dict(load_file(checkpoint['path']))
            original=_evaluate(dev,labels,torch,adapter,head);protection=_evaluate(rendered,labels,torch,adapter,rhead)
            saved=read_json(directory/'best_overall-evaluation.json')
            if saved!={'original':original,'rendered':protection}:raise ValueError('independent predictions/metrics mismatch')
            gate=_gate(original,{'metrics':report['baseline_original']},{'metrics':report['seed43_original']})
            protect=_rendered_gate(protection,{'metrics':report['seed43_rendered']})
            contracts=[]
            for row in dev:
                entry=next(e for e in index['entries'] if e['example_id']==row['example_id'])
                pp=Path(entry['posterior_path']);verify_file(pp,entry['posterior_sha256']);old=read_json(pp);producer=old['producer']
                with torch.inference_mode():logits=head(adapter(row['features'].to('cuda'))).cpu()
                destination=output/f'seed{seed}'/row['example_id'];destination.mkdir(parents=True)
                raw=destination/'adapted-logits.safetensors';save_file({'frame_logits':logits},str(raw))
                metadata={**producer['model_metadata'],'adapter_checkpoint_sha256':checkpoint['sha256'],'adapter_role':'original_adaptation_unpromoted'}
                payload=emission_payload(logits.numpy(),vocabulary,input_audio=producer['input_audio'],crop=producer['crop'],model_metadata=metadata,geometry=old['decoding']['geometry'],runtime=producer['runtime'])
                payload['raw_evidence']={'file':raw.name,'sha256':sha256_file(raw)}
                path=destination/'posterior.json';write_json_new(path,payload)
                a=PhoneticXeusPosteriorRecognizer(path,vocabulary_path=Path(manifest['vocabulary']['path'])).recognize(Path(row['audio_path']))
                b=PhoneticXeusPosteriorRecognizer(path,fixture=True,vocabulary_path=Path(manifest['vocabulary']['path'])).recognize(Path(row['audio_path']))
                def contract(values):return [{'time_range':asdict(v.time_range),'candidate_phones':[asdict(c) for c in v.candidate_phones],'confidence':v.confidence} for v in values]
                if contract(a)!=contract(b):raise ValueError('real/fixture posterior mismatch')
                contracts.append({'example_id':row['example_id'],'posterior_sha256':sha256_file(path),'contract_sha256':sha256_object(contract(a)),'real_fixture_equal':True})
            checks.append({'seed':seed,'checkpoint_sha256':checkpoint['sha256'],'predictions_metrics_equal':True,'original_gate':gate,'rendered_gate':protect,'posteriors':contracts})
            print('verified seed',seed,flush=True)
        write_json_new(output/'VERIFICATION.json',{'checks':checks,'independent_process':True,'test_opened':False,'backbone_rerun':False,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'.state/runs'):raise ValueError('new output directory required')
    run(a.run_root.resolve(),a.output.resolve())

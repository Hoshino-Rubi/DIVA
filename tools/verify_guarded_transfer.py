"""Transfer replay only after candidate freeze, with unchanged Shalong reference."""
from pathlib import Path
import sys,argparse
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,sha256_object,write_json_new
from any2xsampa.training.retention_adaptation import load_inputs
from any2xsampa.adapters.guarded_phoneticxeus import GuardedPhoneHead
from any2xsampa.adapters.phoneticxeus import ctc_collapse,PhoneticXeusPosteriorRecognizer
from any2xsampa.training.phone_error_metrics import phone_metrics
from tools.replay_reviewed_audibility import validate_entry,QUALITY


def run(runroot: Path,output: Path) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline,configure_cuda
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    enforce_offline(output/'runtime-cache')
    from safetensors.torch import load_file
    policy=read_json(ROOT/'.state/runs/retention-iteration-20260919T033200/POLICY.json')
    frozen=runroot/'CANDIDATE_FREEZE.json';candidate=read_json(frozen)
    prior=read_json(QUALITY/'paired-diagnostics-v1/inputs.json')
    indexpath=QUALITY/'paired-phonetic-v1/index.json';verify_file(indexpath,prior['paired_index_sha256']);paired=read_json(indexpath)
    verify_file(Path(paired['paired_inputs_path']),paired['paired_inputs_sha256']);sources=read_json(Path(paired['paired_inputs_path']))
    pairs={r['context']['example_id']:r for r in sources['pairs'] if r['context']['song_id']=='shalong'}
    with ProcessResourceMonitor(output/'monitor',timeout_seconds=600):
        torch=configure_cuda();data=load_inputs(Path(policy['source_run']),policy['source_manifest_sha256'],torch)
        engine=GuardedPhoneHead(torch,data['head'],data['vocabulary'],frozen,sha256_file(frozen))
        evaluations={};posteriors=[]
        for condition in ('historical_raw','mixed','vocal'):
            teacher_pairs=[];student_pairs=[];guard_pairs=[];details=[]
            for row in data['dev']:
                if condition=='historical_raw':
                    entry=next(e for e in data['original_index']['entries'] if e['example_id']==row['example_id']);source_path=Path(row['audio_path'])
                else:
                    entry=next(e for e in paired['entries'] if e['original_example_id']==row['example_id'] and e['condition']==condition)
                    source=pairs[row['example_id']]['audio'][condition];source_path=Path(source['path']);verify_file(source_path,source['sha256'])
                validate_entry(row,entry,condition,pairs.get(row['example_id']))
                verify_file(Path(entry['evidence_path']),entry['evidence_sha256']);tensors=load_file(entry['evidence_path'])
                pp=Path(entry['posterior_path']);verify_file(pp,entry['posterior_sha256']);payload=read_json(pp)
                pred=engine.predict(tensors['encoder_features']);target=row['model_target_ipa'];local={}
                for key,pair_list in [('teacher_logits',teacher_pairs),('student_logits',student_pairs),('frame_logits',guard_pairs)]:
                    phones=[data['labels'][i] for i in ctc_collapse(pred[key].argmax(-1).tolist())];pair_list.append((target,phones));local[key]={'prediction':phones,'metrics':phone_metrics([(target,phones)])}
                saved=engine.save(tensors['encoder_features'],payload,output/condition/row['example_id'])
                path=Path(saved['posterior_path']);vp=Path(data['manifest']['vocabulary']['path'])
                a=PhoneticXeusPosteriorRecognizer(path,vocabulary_path=vp).recognize(source_path)
                b=PhoneticXeusPosteriorRecognizer(path,fixture=True,vocabulary_path=vp).recognize(source_path)
                contract=lambda rows:[{'time_range':asdict(s.time_range),'candidate_phones':[asdict(c) for c in s.candidate_phones],'confidence':s.confidence} for s in rows]
                if contract(a)!=contract(b):raise ValueError('transfer posterior replay mismatch')
                details.append({'example_id':row['example_id'],'systems':local,'reference':target,'guard_frames':int(pred['guard_mask'].sum()),'frames':len(pred['guard_mask'])})
                posteriors.append({**saved,'condition':condition,'example_id':row['example_id'],'real_fixture_equal':True,'contract_sha256':sha256_object(contract(a))})
            evaluations[condition]={'teacher':phone_metrics(teacher_pairs),'student':phone_metrics(student_pairs),'guarded':phone_metrics(guard_pairs),'examples':details}
            print(condition,evaluations[condition]['teacher']['per'],evaluations[condition]['guarded']['per'],flush=True)
        if evaluations['historical_raw']['guarded']!=candidate['development_metrics']['original']:raise ValueError('frozen candidate score changed')
        write_json_new(output/'TRANSFER.json',{'candidate_sha256':sha256_file(frozen),'selected_before_transfer':True,'selection_changed':False,'evaluations':evaluations,'posteriors':posteriors,'test_opened':False,'default_changed':False,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'.state/runs'):raise ValueError('new run output required')
    run(a.run.resolve(),a.output.resolve())

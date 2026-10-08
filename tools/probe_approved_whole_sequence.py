"""Bounded disposable real optimizer probe for approved whole-song CTC gradients."""
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file


def probe(run:Path)->None:
    from safetensors.torch import load_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime,make_tail,load_trainable_state
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
    from any2xsampa.training.whole_sequence_cache import load_whole_rows
    from any2xsampa.training.whole_sequence_supervision import backward_whole_sequence
    from any2xsampa.training.singing_domain_training import predict
    freeze=read_json(run/'FREEZE.json');spec=read_json(run/'POLICY.json')
    output=run/'whole-cuda-probe';output.mkdir(exist_ok=False);started=time.monotonic()
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=600):
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),output/'runtime-cache');torch=rt.torch
        torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        # The probe is disposable and has no validation or checkpoint selection.
        parent=read_json(Path(freeze['parents']['singing'])/'final-training/seed17/RESULT.json')
        verify_file(Path(parent['checkpoint_path']),parent['checkpoint_sha256'])
        tail=make_tail(torch,rt.model,'full');load_trainable_state(tail,load_file(parent['checkpoint_path']))
        rt.model.frontend.cpu();rt.model.preencoder.cpu()
        for block in list(rt.model.encoder.encoders)[:15]:block.cpu()
        rt.model.encoder.conditioning_layer.cpu()
        whole=load_whole_rows(run,torch,rt.identity['sha256'],None)
        frozen=[p for p in rt.model.parameters() if not p.requires_grad];versions=[p._version for p in frozen]
        parameters=[p for p in tail.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=spec['training']['learning_rate'],weight_decay=spec['training']['weight_decay'],foreach=False)
        optimizer.zero_grad(set_to_none=True);checks=[]
        for song in whole:
            result=backward_whole_sequence(torch,tail,song['chunks'],song['target_ids'],
                lambda r:predict(torch,tail,r)[0],coefficient=spec['whole_sequence']['coefficient'],
                deadline=started+580,device='cuda')
            checks.append({'song_id':song['song_id'],**result})
            print('PROBE',song['song_id'],'CTC',result['ctc'],'maxdiff',result['maximum_recompute_logit_difference'],flush=True)
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('non-finite or absent whole-source parameter gradients')
        torch.nn.utils.clip_grad_norm_(parameters,spec['training']['gradient_clip'],error_if_nonfinite=True)
        optimizer.step()
        if versions!=[p._version for p in frozen]:raise ValueError('whole probe mutated frozen parameters')
        write_json_new(output/'RESULT.json',{'checks':checks,'gradients_finite':True,'frozen_parameters_unchanged':True,
            'optimizer_steps':1,'checkpoint_saved':False,'quality_training_job':False,
            'policy_sha256':freeze['policy_sha256'],'cache_sha256':sha256_file(run/'WHOLE_CACHE.json'),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'memory':check_gpu_budget(torch),
            'elapsed_seconds':time.monotonic()-started,'held_or_development_predictions_computed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);probe(p.parse_args().run.resolve())

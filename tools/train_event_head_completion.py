"""Two explicitly authorized fresh E jobs, with recoverable epoch state."""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, sha256_file, sha256_object
from any2xsampa.training.event_head_protocol import load_cache, epoch_schedule
from any2xsampa.training.event_head_training import make_head, loss_on_row
from any2xsampa.training.event_head_resume import save_epoch_state
from input_native_probe_common import parameter_digest, compare_logits
from event_head_completion_common import load_completion, check_attempt
from event_head_validation_fast import evaluate_fast


def train(run: Path, held: str) -> None:
    auth, source, freeze, spec, index = load_completion(run, "CODE_REVIEW_training.json")
    ordinal = check_attempt(run, held, auth, worker=True)
    arm = "E"
    destination = run / "screen" / arm / held
    write_json_new(destination / "START.json", {
        "arm": arm, "held_song": held, "seed": 17, "quality_start_ordinal": ordinal,
        "freeze_sha256": sha256_file(source / "FREEZE.json"),
        "review_sha256": sha256_file(run / "CODE_REVIEW_training.json"),
        "planned_epochs": 20, "quality_training_start": True,
        "fresh_pretrained_initialization": True, "loaded_prior_failed_state": False,
        "replacement_of_failed_job": held == "promise",
    })
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline, configure_cuda, check_gpu_budget
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from safetensors.torch import load_file, save_file
    enforce_offline(destination / "runtime-cache")
    started = time.monotonic()
    with serial_candidate_job(), ProcessResourceMonitor(destination / "monitor", timeout_seconds=spec["training"]["maximum_job_seconds"]):
        torch=configure_cuda();torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        rows=load_cache(torch,index['rows']);trainrows=[r for r in rows if r['song_id']!=held];heldrows=[r for r in rows if r['song_id']==held]
        if any(r['song_id']==held for r in trainrows) or not heldrows:raise ValueError('fold split leakage')
        state={k:v.cuda() for k,v in load_file(freeze['pretrained_head']['path'],device='cpu').items()}
        head=make_head(torch,state,arm).train();vocab=read_json(Path(freeze['vocabulary']['path']))
        initial=parameter_digest(head);checks=[]
        with torch.no_grad():
            for row in trainrows:
                checks.append(compare_logits(head.content(row['features'].cuda()).cpu().numpy(),row['prior'].numpy(),vocab))
        versions=[r['features']._version for r in rows]+[r['prior']._version for r in rows]
        schedule=epoch_schedule(index['rows'],index['unused_rng_rows'],held,17)
        frozen_schedule=read_json(source/'SCHEDULES.json')[held]
        if schedule!=frozen_schedule['epochs']:raise ValueError('fixed sampling/RNG order differs')
        write_json_new(destination/'INITIALIZATION.json',{'head_sha256':initial,'all_train_initial_logits_equal':True,
            'maximum_logit_error':max(c['maximum_absolute_error'] for c in checks),'train_example_ids':[r['example_id'] for r in trainrows],
            'held_example_ids':[r['example_id'] for r in heldrows],'parameter_count':sum(p.numel() for p in head.parameters()),
            'architecture_sha256':index['architecture']['sha256'],'shared_head_unchanged':True,'encoder_loaded':False})
        optimizer=torch.optim.AdamW(head.parameters(),lr=spec['training']['learning_rate'],weight_decay=spec['training']['weight_decay'],foreach=False)
        lookup={r['example_id']:r for r in trainrows};history=[];steps=0
        binding = {
            "authorization_sha256": sha256_file(run / "AUTHORIZATION.json"),
            "freeze_sha256": sha256_file(source / "FREEZE.json"),
            "policy_sha256": sha256_file(source / "POLICY.json"),
            "review_sha256": sha256_file(run / "CODE_REVIEW_training.json"),
            "worker_sha256": sha256_file(Path(__file__)), "held_song": held,
            "schedule_sha256": frozen_schedule["sha256"], "initial_head_sha256": initial,
        }
        commits = []
        for epoch,order in enumerate(schedule,1):
            epoch_start=time.monotonic();totals={'sequence_loss_per_reference':0.,'kl':0.};head.train()
            for offset in range(0,len(order),8):
                if time.monotonic()-started>spec['training']['maximum_job_seconds']:raise TimeoutError('fixed job deadline; no partial selection')
                batch=order[offset:offset+8];optimizer.zero_grad(set_to_none=True)
                for key in batch:
                    loss,diag=loss_on_row(torch,head,lookup[key],arm,spec);(loss/len(batch)).backward()
                    for name in totals:totals[name]+=diag[name]
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
                if torch.cuda.memory_reserved()>7*1024**3:raise RuntimeError('process GPU cap exceeded')
            if any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('nonfinite trained parameters')
            event={'epoch':epoch,'optimizer_steps':(len(order)+7)//8,'cumulative_optimizer_steps':steps,
                'microsteps':len(order),'order_sha256':sha256_object(order),
                'means':{k:v/len(order) for k,v in totals.items()},'epoch_seconds':time.monotonic()-epoch_start,
                'elapsed_seconds':time.monotonic()-started,'memory':check_gpu_budget(torch),'held_out_metrics_used':False}
            write_json_new(destination/f'epoch-{epoch:02d}.json',event);history.append(event)
            save_started = time.monotonic()
            commit = save_epoch_state(torch, head, optimizer, destination / "resumable" / f"epoch-{epoch:02d}",
                                      epoch=epoch, steps=steps, binding=binding)
            commits.append({"epoch": epoch, "commit_path": str(commit), "commit_sha256": sha256_file(commit),
                            "save_seconds": time.monotonic() - save_started})
            print(arm,held,'epoch',epoch,'loss',round(event['means']['sequence_loss_per_reference'],6),
                  'seconds',round(event['epoch_seconds'],2),'state_saved',flush=True)
        if steps!=frozen_schedule['optimizer_steps'] or versions!=[r['features']._version for r in rows]+[r['prior']._version for r in rows]:
            raise ValueError('update budget/cache mutation failure')
        checkpoint=destination/'final.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(checkpoint))
        result = evaluate_fast(torch, head, arm, heldrows, spec, freeze, destination / "validation")
        payload={'complete':True,'arm':arm,'held_song':held,'seed':17,'selected_epoch':20,'history':history,
            'optimizer_steps':steps,'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),
            'head_state_sha256':parameter_digest(head),'validation':result,'schedule_sha256':frozen_schedule['sha256'],
            'freeze_sha256':sha256_file(source/'FREEZE.json'),'frozen_cache_unchanged':True,'encoder_loaded_during_training':False,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-started,
            'development_opened':False,'final_test_opened':False,'default_replaced':False,
            'resumable_epoch_commits':commits,'quality_start_ordinal':ordinal,
            'source_run':str(source),'authorization_sha256':sha256_file(run/'AUTHORIZATION.json')}
    write_json_new(destination/'RESULT.json',payload)
    print('COMPLETE',arm,held,result['original']['edit_distance'],flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser();p.add_argument("--run", type=Path, required=True);p.add_argument("--held", required=True)
    args = p.parse_args();train(args.run.resolve(), args.held)

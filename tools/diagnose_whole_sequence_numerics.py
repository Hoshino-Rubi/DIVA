"""Diagnose whole-CTC two-pass numerics on a frozen completed checkpoint; no updates."""
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file


def diagnose(run:Path)->None:
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.whole_sequence_inference import verify_job,install_checkpoint
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.training.whole_sequence_cache import load_whole_rows
    from any2xsampa.training.whole_sequence_training import predict
    from any2xsampa.training.whole_sequence_supervision import deterministic_checkpointed_tail
    freeze=read_json(run/'FREEZE.json');output=run/'numerical-diagnosis';output.mkdir(exist_ok=False)
    started=time.monotonic()
    with serial_candidate_job(),ProcessResourceMonitor(output/'monitor',timeout_seconds=600):
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),output/'runtime-cache');torch=rt.torch
        result=verify_job(run,run/'screen/whole_sequence/half-moon-night-song');tail=install_checkpoint(rt,result)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        rt.model.frontend.cpu();rt.model.preencoder.cpu()
        for block in list(rt.model.encoder.encoders)[:15]:block.cpu()
        rt.model.encoder.conditioning_layer.cpu()
        songs=load_whole_rows(run,torch,rt.identity['sha256'],'half-moon-night-song')
        states=[p._version for p in tail.parameters()];rows=[]
        with deterministic_checkpointed_tail(tail):
            for song in songs:
                for row in song['chunks']:
                    with torch.no_grad():a=predict(torch,tail,row)[0].detach().cpu().clone()
                    with torch.enable_grad():
                        value=predict(torch,tail,row)[0];b=value.detach().cpu().clone();del value
                        value=predict(torch,tail,row)[0];c=value.detach().cpu().clone();del value
                    event={'example_id':row['example_id'],'no_grad_vs_grad_max_difference':float((a-b).abs().max()),
                        'grad_vs_grad_max_difference':float((b-c).abs().max()),
                        'no_grad_vs_grad_argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1))),
                        'grad_vs_grad_argmax_equal':bool(torch.equal(b.argmax(-1),c.argmax(-1)))}
                    rows.append(event);print('NUMERIC',event,flush=True)
        if states!=[p._version for p in tail.parameters()]:raise ValueError('numerical diagnosis altered parameters')
        write_json_new(output/'RESULT.json',{'rows':rows,'checkpoint_sha256':result['checkpoint_sha256'],
            'attention_implementation':[{'type':type(b.attn).__name__,'use_sdpa':getattr(b.attn,'use_sdpa',None),
                'use_flash_attn':getattr(b.attn,'use_flash_attn',None)} for b in tail.blocks],
            'no_optimizer_steps':True,'parameters_unchanged':True,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'elapsed_seconds':time.monotonic()-started,'code_sha256':sha256_file(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run.resolve())

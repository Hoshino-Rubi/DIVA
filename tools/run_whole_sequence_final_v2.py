"""Conditional final seeds using the numerically consistent two-pass implementation."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file
from any2xsampa.adapters.whole_sequence_inference import verify_job
from record_whole_sequence_command import execute


def run_final(run:Path)->None:
    selection=read_json(run/'ROUTE_SELECTION.json')
    if not selection['passed']:
        write_json_new(run/'FINAL_TRAINING_NOT_STARTED.json',{'reason':'whole_cross_song_gain_or_protection_gate_failed',
            'selection_sha256':sha256_file(run/'ROUTE_SELECTION.json'),'new_final_seeds':0})
        print('FINAL NOT STARTED: whole screening failed',flush=True);return
    spec=read_json(run/'POLICY.json');results=[]
    for seed in spec['training']['final_seeds']:
        output=run/'final/whole_sequence'/('seed'+str(seed))
        if not (output/'RESULT.json').exists():
            if output.exists():raise RuntimeError('incomplete final seed requires inspection')
            if execute(run,'whole-final-v2-seed'+str(seed),ROOT/'tools/train_approved_whole_sequence_v2.py',[
                '--run',str(run),'--seed',str(seed),'--output',str(output)]):raise RuntimeError('final v2 seed failed')
        results.append(verify_job(run,output))
    write_json_new(run/'FINAL_TRAINING.json',{'arm':'whole_sequence','implementation_version':'two_grad_passes_v2',
        'seeds':[{'seed':r['seed'],'checkpoint_path':r['checkpoint_path'],'checkpoint_sha256':r['checkpoint_sha256']} for r in results],
        'default_replacement':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run_final(p.parse_args().run.resolve())

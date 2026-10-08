"""Thin isolated entry for one fixed approved-whole continuation job."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.whole_sequence_training import train_job
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--held-song')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();train_job(a.run.resolve(),a.held_song,a.seed,a.output.resolve())

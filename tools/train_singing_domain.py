"""Thin isolated worker for a single frozen singing-domain training job."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.singing_domain_training import train_job
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=['endpoint','lora','full'],required=True)
    p.add_argument('--variant',choices=['baseline','paired','unlabeled','attributes'],default='baseline');p.add_argument('--held-song')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not a.output.resolve().is_relative_to(a.run.resolve()):raise ValueError('training output outside current run')
    train_job(a.run.resolve(),a.arm,a.variant,a.held_song,a.seed,a.output.resolve())

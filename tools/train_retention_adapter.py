"""Run one predeclared offline retained original adapter job."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.retention_adaptation import run_retention

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--variant',required=True,choices=['replay_005','replay_100','symmetric_100','rehearsal_100'])
    p.add_argument('--seed',type=int,required=True,choices=[17,29,43])
    a=p.parse_args();suite=a.suite.resolve()
    if not suite.is_relative_to(ROOT/'.state/runs'):raise ValueError('suite must remain in runs')
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    with ProcessResourceMonitor(suite/a.variant/f'seed{a.seed}-monitor',timeout_seconds=1500):
        r=run_retention(suite,a.variant,a.seed)
    c=r['checkpoints'];print(f"variant={a.variant}; seed={a.seed}; per={c['best_overall']['original_metrics']['per']}; admissible={c['best_admissible'] is not None}")

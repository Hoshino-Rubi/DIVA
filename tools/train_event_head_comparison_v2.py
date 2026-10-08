"""Reuse the unchanged optimizer loop with read-only codec checks deferred."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.event_head_protocol import load_protocol
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file


def run(run_dir,held):
    load_protocol(run_dir,'CODE_REVIEW_training_v2.json')
    incident=read_json(run_dir/'RECOVERY_INCIDENT.json')
    if 'E--'+held not in incident['remaining_quality_jobs']:raise ValueError('v2 may only start previously unstarted E jobs')
    from event_head_validation_fast import evaluate_fast
    import train_event_head_comparison as old
    # Explicit, hash-reviewed callback substitution. The validated optimizer,
    # initializer, data loader and sampler are reused byte-for-byte.
    old.evaluate=evaluate_fast
    write_json_new(run_dir/'screen/E'/held/'SERIALIZATION_REVIEW.json',{
        'review_sha256':sha256_file(run_dir/'CODE_REVIEW_training_v2.json'),
        'original_optimizer_worker_sha256':sha256_file(ROOT/'tools/train_event_head_comparison.py'),
        'change':'evaluation callback saves raw evidence, independent codec checks deferred',
        'loss_optimizer_sampling_epochs_changed':False})
    old.train(run_dir,'E',held)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--held',required=True)
    a=p.parse_args();run(a.run.resolve(),a.held)

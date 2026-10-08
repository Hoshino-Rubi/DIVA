"""CPU-only verification before a human-reviewed quality-training launch."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from any2xsampa.training.input_supervision import load_manifest, policy, supervision_slots, shuffle_epoch
from any2xsampa.training.input_supervision_training import load_rows, require_screen_budget
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, write_json_new


def preflight(run: Path) -> None:
    import torch
    torch.set_num_threads(4)
    freeze, manifest, _ = load_manifest(run)
    require_screen_budget(run)
    rows, _ = load_rows(run, torch)
    old_root = Path(freeze['historical_run'])
    old_spec = read_json(old_root / 'POLICY.json')
    old_entries = read_json(old_root / 'CACHE.json')['entries']
    spec = policy()
    for key in ('epochs', 'weight_decay', 'gradient_accumulation', 'gradient_clip',
                'rendered_loss_weight', 'baseline_kl_weight', 'temperature'):
        if spec[key] != old_spec['training'][key]:
            raise ValueError('optimizer/loss policy differs from control: ' + key)
    if spec['learning_rate'] != old_spec['learning_rates']['full']:
        raise ValueError('learning rate differs from control')
    checks = []
    for variant in spec['variants']:
        for held in spec['songs']:
            slots = supervision_slots(rows, held, variant)
            old_train = [r for r in old_entries if r['song_id'] != held]
            old_slots = [r['example_id'] for r in old_train if r['audio_role'] in {'original', 'rendered'}
                         for _ in range(4 if r['audio_role'] == 'original' else 1)]
            ignored = [r['example_id'] for r in manifest['unused_unlabeled_rng_rows'] if r['song_id'] != held]
            old_ignored = [r['example_id'] for r in old_train if r['audio_role'] == 'unlabeled_original']
            if old_slots != [r['window_id'] for r in slots] or ignored != old_ignored:
                raise ValueError('pre-shuffle source order differs from actual control cache')
            old_job = read_json(old_root / 'architecture/full' / held / 'RESULT.json')
            a, b = random.Random(17), random.Random(17)
            for epoch in range(20):
                expected, unused = list(old_slots), list(old_ignored)
                a.shuffle(expected)
                a.shuffle(unused)
                actual = shuffle_epoch(slots, ignored, b)
                steps = (len(actual) + 7) // 8
                if (expected != [r['window_id'] for r in actual] or a.getstate() != b.getstate()
                        or steps != old_job['history'][epoch]['optimizer_steps']
                        or len(actual) != old_job['history'][epoch]['labelled_microsteps']):
                    raise ValueError('control schedule/RNG/update budget mismatch')
                checks.append({'variant': variant, 'held_song': held, 'epoch': epoch + 1,
                               'source_order_sha256': sha256_object(expected), 'optimizer_steps': steps})
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    write_json_new(run / 'PREFLIGHT.json', {
        'schema_version': 'input_supervision_preflight_v1', 'passed': True,
        'freeze_sha256': sha256_file(run / 'FREEZE.json'), 'validated_cache_rows': len(rows),
        'schedule_checks': checks, 'cpu_threads': 4, 'cuda_initialized': False,
        'quality_training_started': False, 'code_review_still_required': True,
        'budget': freeze['budget'],
    })
    print('PASS: 169 cached tensor rows; 200 exact source/RNG schedules; no CUDA; code review still required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    preflight(args.run.resolve())

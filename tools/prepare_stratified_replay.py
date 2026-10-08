"""Freeze one new recipe after the completed listening review, before training."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.stratified_replay import validate_policy


def prepare(source: Path, output: Path) -> None:
    analysis = read_json(output/'ANALYSIS.json')
    intake = read_json(output/'intake.json')
    verify_file(output/'intake.json', analysis['source_intake_sha256'])
    if any(intake[k] for k in ('formal_reference_created', 'training_started', 'source_reference_changed')):
        raise ValueError('diagnostic intake must not relabel or train')
    if analysis['teacher_support']['rendered_train']['windows'] != 129:
        raise ValueError('training support evidence missing')
    config_path = ROOT/'src/any2xsampa/config/stratified_projected_replay_v1.json'
    policy = read_json(config_path)
    paths = {
        'source_policy': source/'POLICY.json', 'comparison': source/'DECODER_COMPARISON.json',
        'prior_recipe': source/'PROJECTED_REPLAY_POLICY.json', 'diagnostic_intake': output/'intake.json',
        'diagnostic_analysis': output/'ANALYSIS.json', 'config': config_path,
        'worker': ROOT/'tools/train_stratified_replay.py',
        'loss_source': ROOT/'src/any2xsampa/training/stratified_replay.py',
        'projection_source': ROOT/'src/any2xsampa/training/projected_replay.py',
        'data_source': ROOT/'src/any2xsampa/training/retention_adaptation.py',
    }
    policy.update(bindings={key: {'path': str(path), 'sha256': sha256_file(path)} for key, path in paths.items()},
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  reason='Full-batch review supports retaining most proxies; rare protected frames are diluted by the old all-frame mean.',
                  uncorrected_local_diagnostic_items=['D04', 'D06'],
                  uncorrected_item_policy='Preserve all existing engineering-proxy targets and denominators; preferences are not full references.')
    policy['policy_sha256'] = sha256_object(policy)
    validate_policy(policy)
    write_json_new(output/'EXPERIMENT_POLICY.json', policy)
    # Record unchanged runtime/configuration and source-file hashes.
    preservation = {'schema_version': 'stratified_preservation_before_v1', 'files': {}}
    # Existing configuration and the existing recognizer/CLI remain unchanged.
    paths = list((ROOT/'src/any2xsampa/config').glob('*.json'))
    paths += list((ROOT/'src/any2xsampa/adapters').glob('*.py')) + [ROOT/'src/any2xsampa/cli.py']
    for path in paths:
        preservation['files'][str(path)] = sha256_file(path)
    manifest_path = ROOT/'.state/runs/limited-data-20260906T050251/reference/reference-manifest.json'
    verify_file(manifest_path, '120b45d2970e98d0bccba61d827004b72d50e531b739cfc52ac9f19bc10751c5')
    manifest = read_json(manifest_path)
    preservation['source_manifest_sha256'] = sha256_file(manifest_path)
    # Record only the train/dev songs; do not open observed final-test audio.
    for song in manifest['songs']:
        if song['split'] == 'test':
            continue
        for key in ('human_vsqx', 'original_audio', 'rendered_audio'):
            spec = song.get(key)
            if not spec:
                continue
            verify_file(Path(spec['path']), spec['sha256'])
            preservation['files'][spec['path']] = spec['sha256']
    preservation['historical_delivery_sha256'] = sha256_file(source/'DELIVERY.json')
    preservation['default_replacement'] = False
    write_json_new(output/'PRESERVATION_BEFORE.json', preservation)
    print('Frozen three seeds; no references, test audio, or default model changed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'.state/runs'):
        raise ValueError('output must remain under runs')
    prepare(args.source.resolve(), args.output.resolve())

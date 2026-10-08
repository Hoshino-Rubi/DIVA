"""Immutable delivery for the reviewed continuation, including the failed launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.input_supervision import load_manifest, supervision_slots, shuffle_epoch
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new


def finish(run: Path, replay_dir: Path) -> None:
    freeze, manifest, _ = load_manifest(run)
    analysis = read_json(run/'ANALYSIS.json')
    replay = read_json(replay_dir/'VERIFICATION.json')
    gate = read_json(run/'SCREEN_REPORT.json')
    incident = read_json(Path(freeze['bindings']['incident']['path']))
    regression = ET.parse(run/'final-regression.xml').getroot()
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in regression.iter('testsuite'))
              for key in ('tests', 'failures', 'errors', 'skipped')}
    counts['passed'] = counts['tests']-counts['failures']-counts['errors']-counts['skipped']
    if counts['failures'] or counts['errors'] or read_json(run/'final-regression-exit.json')['exit_code'] != 0:
        raise ValueError('regression did not pass')
    if not replay['all_checkpoints_verified'] or gate['selected_pipeline'] is not None:
        raise ValueError('this delivery is only for verified screen without promotion')
    schedules, jobs = [], []
    for path in sorted(run.glob('screen/*/*/RESULT.json')):
        row = read_json(path)
        slots = supervision_slots(manifest['rows'], row['held_song'], row['variant'])
        ignored = [r['example_id'] for r in manifest['unused_unlabeled_rng_rows'] if r['song_id'] != row['held_song']]
        rng = random.Random(row['seed'])
        for event in row['history']:
            ordered = shuffle_epoch(slots, ignored, rng)
            expected = {'source_order_sha256': sha256_object([r['window_id'] for r in ordered]),
                        'slot_order_sha256': sha256_object([r['slot_id'] for r in ordered]),
                        'view_order_sha256': sha256_object([r['condition'] for r in ordered]),
                        'rng_after_sha256': sha256_object(rng.getstate())}
            if any(event[k] != value for k, value in expected.items()):
                raise ValueError('actual sampling/RNG changed')
            schedules.append({'variant': row['variant'], 'held_song': row['held_song'], 'epoch': event['epoch'], **expected})
        jobs.append({'path': str(path), 'sha256': sha256_file(path),
                     'variant': row['variant'], 'held_song': row['held_song'], 'epochs': len(row['history'])})
    if len(jobs) != len(replay['checkpoints']) or len(jobs) != 9 or len(schedules) != 180:
        raise ValueError('available job coverage is incomplete')
    for binding in read_json(run/'PRESERVATION.json')['bindings']:
        verify_file(Path(binding['path']), binding['sha256'])
    write_json_new(run/'FINAL_SCHEDULE_AUDIT.json', {'actual_epoch_checks': schedules, 'all_passed': True})
    matrix = analysis['matrix']
    v_partial_errors = matrix['V(v)']['metrics']['edit_distance']
    import math
    required_maximum = math.floor(matrix['M(v)']['metrics']['edit_distance'] * .95 + 1e-12)
    missing_maximum = required_maximum - v_partial_errors
    raw_errors = [d['saved_logit_maximum_error'] for job in replay['checkpoints'] for d in job['details']
                  if d['saved_logit_maximum_error'] is not None]
    files = [run/name for name in ('FREEZE.json','CODE_REVIEW.json','PREFLIGHT.json','SCREEN_REPORT.json',
             'ANALYSIS.json','ANALYSIS.md','ERROR_ANALYSIS.json','PRESERVATION.json','FINAL_SCHEDULE_AUDIT.json',
             'REPLAY_REVIEW.json','REPLAY_PREFLIGHT.json','final-regression.xml','final-regression-exit.json')]
    files += [replay_dir/'VERIFICATION.json', replay_dir/'monitor/resource-monitor.json',
              run/'replay-exit.json', Path(freeze['bindings']['incident']['path'])]
    if read_json(run/'replay-exit.json')['exit_code'] != 0:
        raise ValueError('independent replay did not exit successfully')
    payload = {
        'schema_version': 'input_supervision_screen_delivery_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'stages': {'MV': 'completed_did_not_pass', 'V': 'four_folds_completed_one_measurement_timeout',
                   'independent_replay': 'completed', 'final_seeds_development': 'not_executed_gate_not_reached'},
        'valid_complete_jobs': jobs, 'valid_completed_epochs': len(schedules),
        'budget': analysis['budget'], 'failed_current_round_job': incident['attempts'][-1],
        'verification': {'checkpoints_replayed': len(replay['checkpoints']),
            'posterior_fixture_checks': sum(len(j['details']) for j in replay['checkpoints']),
            'maximum_saved_logit_error': max(raw_errors),
            'maximum_full_audio_logit_error': max(j['full_audio_maximum_logit_error'] for j in replay['checkpoints']),
            'schedule_epochs_checked': len(schedules), 'regression': counts},
        'V_missing_fold_arithmetic_only': {'four_fold_errors': v_partial_errors,
            'global_error_ceiling_for_5pct_vs_Mv': required_maximum,
            'missing_freedom_fold_maximum_errors_needed': missing_maximum,
            'does_not_predict_missing_result': True, 'other_gates_still_required': True},
        'artifacts': {str(p.resolve()): sha256_file(p) for p in files},
        'new_human_listening_required': False, 'default_replaced': False, 'new_final_test_predictions': False,
        'automatic_extra_budget_used': False,
    }
    write_json_new(run/'DELIVERY.json', payload)
    write_json_new(run/'CURRENT.json', {
        'status': 'delivered_no_eligible_pipeline_one_fold_unavailable',
        'report': str(run/'ANALYSIS.md'), 'delivery': str(run/'DELIVERY.json'),
        'running_model_jobs': 0, 'new_training_starts_consumed': 10, 'total_historical_starts': 19,
        'next_human_window': 'optional decision on one failed-fold retry or a different next direction; no listening form',
        'no_retry_authorized_or_started': True,
    })
    print('DELIVERED', run/'DELIVERY.json', counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    args = parser.parse_args()
    finish(args.run.resolve(), args.replay.resolve())

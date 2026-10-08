from copy import deepcopy

import pytest

from any2xsampa.training.input_supervision import policy
from tools.summarize_input_supervision_screen import matrix_report
from tools.report_input_supervision import verify_completed_job
from any2xsampa.training.limited_data_artifacts import sha256_file, write_json_new


def example_matrix(missing=False):
    rows = []
    for i in range(20):
        rows.append({'window_id': str(i), 'song_id': policy()['songs'][i//4],
                     'reference': ['i']*(61 if i < 19 else 76), 'prediction': ['i']*40})
    baseline = {'details': {c: deepcopy(rows) for c in ('x', 'v', 'h')}}
    jobs = []
    for variant in ('V', 'MV'):
        for song in policy()['songs']:
            if missing and variant == 'V' and song == policy()['songs'][0]:
                continue
            jobs.append({'variant': variant, 'held_song': song, 'details':
                [{**row, 'condition': c} for row in rows if row['song_id'] == song for c in ('x', 'v', 'h')]})
    return baseline, {'checkpoints': jobs}


def test_matrix_reports_all_nine_cells_without_input_selection():
    baseline, replay = example_matrix()
    matrix, windows = matrix_report(baseline, replay)
    assert len(matrix) == 9 and len(windows) == 9
    assert all(r['complete_five_folds'] and r['metrics']['reference_phones'] == 1235 for r in matrix.values())
    assert {k for k, r in matrix.items() if r['predeclared_primary']} == {'V(v)', 'MV(x)'}


def test_missing_fold_never_claims_full_reference_denominator():
    baseline, replay = example_matrix(missing=True)
    matrix, _ = matrix_report(baseline, replay)
    assert matrix['MV(x)']['complete_five_folds']
    assert not matrix['V(v)']['complete_five_folds']
    assert matrix['V(v)']['missing_windows'] == ['0', '1', '2', '3']
    assert matrix['V(v)']['metrics']['reference_phones'] == 1235 - 4*61
    assert matrix['V(v)']['matched_controls']['v']['metrics']['reference_phones'] == 1235 - 4*61
    assert matrix['V(v)']['matched_controls']['v']['scope'] == 'available_folds_diagnostic_not_selection'


@pytest.mark.parametrize('damage', ['reference', 'song', 'duplicate'])
def test_matrix_rejects_reference_or_coverage_corruption(damage):
    baseline, replay = example_matrix()
    row = replay['checkpoints'][0]['details'][0]
    if damage == 'reference': row['reference'] = ['different']
    if damage == 'song': row['song_id'] = 'shalong'
    if damage == 'duplicate': replay['checkpoints'][0]['details'].append(row)
    with pytest.raises(ValueError, match='reference identities'):
        matrix_report(baseline, replay)


@pytest.mark.parametrize('damage', ['none', 'schedule', 'freeze', 'checkpoint', 'exit', 'monitor'])
def test_job_report_requires_actual_execution_evidence(tmp_path, damage):
    freeze = {'code_bindings': {'trainer.py': '0'*64}}
    write_json_new(tmp_path/'FREEZE.json', freeze)
    write_json_new(tmp_path/'POLICY.json', {})
    song = policy()['songs'][0]
    job = tmp_path/'screen/MV'/song
    checkpoint = job/'checkpoint'
    job.mkdir(parents=True)
    checkpoint.write_bytes(b'test weights')
    write_json_new(job/'monitor/resource-monitor.json', {
        'operation_succeeded': damage != 'monitor', 'completed_without_resource_abort': True})
    write_json_new(tmp_path/'commands'/f'MV-{song}.exit.json', {'exit_code': 73 if damage == 'exit' else 0})
    history = [{'epoch': i, 'source_order_sha256': str(i), 'optimizer_steps': 22} for i in range(1, 21)]
    row = {'variant': 'MV', 'held_song': song, 'freeze_sha256': sha256_file(tmp_path/'FREEZE.json'),
           'policy_sha256': sha256_file(tmp_path/'POLICY.json'), 'code_bindings': freeze['code_bindings'],
           'frozen_parameters_unchanged': True, 'history': deepcopy(history),
           'initial_logit_errors': {'x': 0., 'v': 0., 'rendered': 0.},
           'checkpoint_path': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint)}
    schedule = {('MV', song, r['epoch']): r for r in history}
    if damage == 'schedule': row['history'][0]['source_order_sha256'] = 'wrong'
    if damage == 'freeze': row['freeze_sha256'] = 'wrong'
    if damage == 'checkpoint': checkpoint.write_bytes(b'changed')
    if damage == 'none':
        verify_completed_job(tmp_path, row, freeze, schedule)
    else:
        with pytest.raises(ValueError): verify_completed_job(tmp_path, row, freeze, schedule)


def test_error_diagnostics_keep_repeated_reference_positions_and_insertions():
    from tools.analyze_input_supervision_errors import indexed_edits, compare_rows
    states, insertions = indexed_edits(['i', 'i', 'kʰ'], ['i', 'i', 'k', 'h'])
    assert len(states) == 3 and len(insertions) == 1
    base = {'window_id': 'a', 'song_id': 'promise', 'reference': ['i', 'kʰ'], 'prediction': ['i', 'k']}
    candidate = {**base, 'prediction': ['i', 'kʰ', 'h']}
    result = compare_rows([base], [candidate])
    assert result['counts']['fixed_error'] == 1
    assert result['counts']['insertion_count_delta'] == 1
    assert result['old_metrics_same_windows']['edit_distance'] == result['new_metrics']['edit_distance'] == 1
    assert result['changes'][0]['reference_index'] == 1
    with pytest.raises(ValueError): compare_rows([base], [candidate, candidate])
    with pytest.raises(ValueError): compare_rows([base], [{**candidate, 'song_id': 'shalong'}])

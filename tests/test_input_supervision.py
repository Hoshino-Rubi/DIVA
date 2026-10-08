from copy import deepcopy
import random
import time

import pytest
from any2xsampa.training.input_supervision import policy, validate_rows, supervision_slots
from any2xsampa.training.input_supervision_evaluation import aggregate, by_song, compare_screen, choose_pipeline


def fixture_rows():
    rows = []
    for si, song in enumerate(policy()['songs']):
        for j in range(2):
            wid = f'{song}-{j}'
            base = dict(window_id=wid, song_id=song, split='train', audio_kind='original', source_order=si*10+j,
                        target_ipa=['i'], target_ids=[1], frames=5, source_audio_sha256=str(si)*64,
                        source_start_sec=j*10., source_end_sec=(j+1)*10.)
            for c in ('x', 'v'):
                rows.append({**base, 'row_id': wid+'--'+c, 'condition': c,
                             'reference_nature': 'confirmed_original' if c == 'x' else 'derived_from_confirmed_original_not_separately_audited'})
        rows.append({**base, 'row_id': song+'--rendered', 'window_id': song+'-rendered', 'condition': 'rendered',
                     'audio_kind': 'rendered', 'source_order': si*10+5, 'reference_nature': 'rendered_proxy'})
    return rows


@pytest.mark.parametrize('variant', ['V', 'MV'])
def test_fixed_slot_counts_and_all_held_derivatives_excluded(variant):
    rows = fixture_rows()
    validate_rows(rows, {'i': 1}, enforce_size=False)
    slots = supervision_slots(rows, 'freedom-flower', variant)
    assert len(slots) == 4*8+4
    assert all(r['song_id'] != 'freedom-flower' for r in slots)
    assert sum(r['condition'] == 'v' for r in slots) == (32 if variant == 'V' else 16)
    assert sum(r['condition'] == 'x' for r in slots) == (0 if variant == 'V' else 16)


@pytest.mark.parametrize('damage', ['song', 'role', 'reference', 'ids', 'nature', 'offset', 'frames', 'duplicate', 'overlap'])
def test_invalid_view_reference_or_sampling_rejected(damage):
    rows = fixture_rows()
    if damage == 'song': rows[0]['song_id'] = 'shalong'
    if damage == 'role': rows[1]['condition'] = 'h'
    if damage == 'reference': rows[1]['target_ipa'] = ['unknown']
    if damage == 'ids': rows[0]['target_ids'] = [0]
    if damage == 'nature': rows[1]['reference_nature'] = 'confirmed_original'
    if damage == 'offset': rows[1]['source_start_sec'] = .01
    if damage == 'frames': rows[0]['frames'] = 0
    if damage == 'duplicate': rows.append(rows[0])
    if damage == 'overlap':
        rows[2]['source_start_sec'] = rows[3]['source_start_sec'] = 9.
    with pytest.raises(ValueError): validate_rows(rows, {'i': 1}, enforce_size=False)


@pytest.mark.parametrize('variant', ['V', 'MV'])
def test_twenty_epoch_schedule_matches_actual_old_trainer(monkeypatch, variant):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(4)
    import any2xsampa.training.singing_domain_training as old
    import any2xsampa.training.input_supervision_training as new
    rows = fixture_rows()
    source = []
    for row in sorted((r for r in rows if r['condition'] in {'x', 'rendered'}), key=lambda r: r['source_order']):
        source.append({**row, 'example_id': row['window_id'], 'audio_role': row['audio_kind']})
        if row['condition'] == 'rendered':
            source.extend(dict(example_id=row['song_id']+str(i), song_id=row['song_id'], split='train', audio_role='unlabeled_original') for i in range(8))
    train = [r for r in source if r['song_id'] != 'freedom-flower']
    slots = supervision_slots(rows, 'freedom-flower', variant)
    ignored = [r['example_id'] for r in train if r['audio_role'] == 'unlabeled_original']
    seen = []
    def fake_loss(torch, tail, row, *unused):
        seen.append(row['window_id'])
        return tail.weight.sum()*0, {'ctc': 0., 'prior_kl': 0., 'paired_kl': 0., 'attribute_ctc': 0., 'aux_excluded': []}
    monkeypatch.setattr(old, 'supervised_loss', fake_loss)
    monkeypatch.setattr(new, 'supervised_loss', fake_loss)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 0)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    spec = {'training': {'original_repeats': 4, 'gradient_accumulation': 8, 'gradient_clip': 1.}}
    before, after = random.Random(17), random.Random(17)
    for _ in range(20):
        seen.clear()
        control = old.train_epoch(torch, model, train, 'baseline', spec, {}, optimizer, before, time.monotonic()+60)
        expected = list(seen)
        seen.clear()
        actual = new.train_epoch(torch, model, slots, ignored, optimizer, after, spec, time.monotonic()+60)
        assert seen == expected
        assert before.getstate() == after.getstate()
        assert actual['optimizer_steps'] == control['optimizer_steps']


def gate_fixture():
    details = []
    for index in range(20):
        n = 61 if index < 19 else 76
        details.append({'window_id': str(index), 'song_id': policy()['songs'][index//4],
                        'reference': ['i']*n, 'prediction': ['i']*(n-20)})
    base = {c: aggregate(details) for c in ('x', 'v')}
    songs = {c: by_song(details) for c in ('x', 'v')}
    folds = []
    for song in policy()['songs']:
        examples = [{**r, 'condition': 'v', 'prediction': r['reference'][:-10]} for r in details if r['song_id'] == song]
        evaluation = {'examples': examples, 'metrics': aggregate(examples)}
        folds.append({'schema_version': 'input_supervision_job_v2', 'held_song': song, 'complete': True,
                      'variant': 'V', 'seed': 17, 'fixed_epoch_count': 20,
                      'training_songs': [s for s in policy()['songs'] if s != song],
                      'validation_x': evaluation, 'validation_v': evaluation})
    return folds, base, songs, {c: deepcopy(details) for c in ('x', 'v')}


def test_gate_uses_exact_reference_and_requires_all_folds():
    folds, base, songs, details = gate_fixture()
    result = compare_screen('V', folds, base, songs, details)
    assert result['passed'] and len(result['strict_improved_songs']) == 5
    assert not compare_screen('V', folds[:-1], base, songs, details)['complete']
    folds[0]['validation_v']['examples'][0]['reference'][0] = 'h'
    with pytest.raises(ValueError, match='reference identity'): compare_screen('V', folds, base, songs, details)


@pytest.mark.parametrize('damage', ['duplicate_fold', 'legacy', 'training_leak', 'metric', 'seed'])
def test_gate_rejects_invalid_completed_artifacts(damage):
    folds, base, songs, details = gate_fixture()
    if damage == 'duplicate_fold': folds.append(folds[0])
    if damage == 'legacy': folds[0]['schema_version'] = 'input_supervision_job_start_v1'
    if damage == 'training_leak': folds[0]['training_songs'].append(folds[0]['held_song'])
    if damage == 'metric': folds[0]['validation_v']['metrics']['edit_distance'] = 0
    if damage == 'seed': folds[0]['seed'] = 43
    with pytest.raises(ValueError): compare_screen('V', folds, base, songs, details)


def test_tie_prefers_pipeline_without_separator_and_incomplete_never_selected():
    rows = [dict(variant=v, complete=True, passed=True, metric={'per': .4}) for v in ('V', 'MV')]
    assert choose_pipeline(rows) == 'MV'
    rows[0]['complete'] = False
    assert choose_pipeline(rows) is None


def test_pure_vocal_gain_must_beat_both_controls_and_macro():
    folds, base, songs, details = gate_fixture()
    # New V has ten deletions per window; M(x) has only one. M(v) remains worse.
    for r in details['x']: r['prediction'] = r['reference'][:-1]
    base['x'], songs['x'] = aggregate(details['x']), by_song(details['x'])
    result = compare_screen('V', folds, base, songs, details)
    assert 'x_relative_improvement_below_5pct' in result['failures']
    assert 'x_macro_regression' in result['failures']


def test_lower_overall_per_cannot_hide_aspiration_regression():
    folds, base, songs, details = gate_fixture()
    for c in ('x', 'v'):
        for row in details[c]:
            row['reference'] = ['kʰ']*2 + row['reference'][2:]
            row['prediction'] = row['reference'][:-20]
        base[c], songs[c] = aggregate(details[c]), by_song(details[c])
    for fold in folds:
        values = []
        for row in details['v']:
            if row['song_id'] == fold['held_song']:
                values.append({**row, 'condition': 'v', 'prediction': ['k']*2+row['reference'][2:-10]})
        fold['validation_v'] = {'examples': values, 'metrics': aggregate(values)}
    result = compare_screen('V', folds, base, songs, details)
    assert result['metric']['per'] < base['v']['per']
    assert 'aspirated' in result['protected_group_regressions']['v']
    assert not result['passed']


def test_incomplete_budget_blocks_before_launch(tmp_path, monkeypatch):
    import any2xsampa.training.input_supervision_training as training
    freeze = {'budget': {'remaining_screen_starts': 1}}
    monkeypatch.setattr(training, 'load_manifest', lambda run: (freeze, {}, {}))
    with pytest.raises(RuntimeError, match='insufficient authorized starts'):
        training.require_screen_budget(tmp_path)
    assert not list(tmp_path.iterdir())


def test_early_worker_failure_consumes_one_start_not_two(tmp_path, monkeypatch):
    import any2xsampa.training.input_supervision_training as training
    from any2xsampa.training.limited_data_artifacts import write_json_new
    monkeypatch.setattr(training, 'load_manifest', lambda run: ({'budget': {'remaining_screen_starts': 10}}, {}, {}))
    command = tmp_path / 'commands/V-freedom-flower.command.json'
    write_json_new(command, {'argv': ['python', 'worker']})
    with pytest.raises(RuntimeError, match='insufficient authorized starts'):
        training.require_screen_budget(tmp_path)
    # Only the dispatched worker can ignore its own still-pending command.
    training.require_screen_budget(tmp_path, current_launch='V-freedom-flower')
    job = tmp_path / 'screen/V/freedom-flower'
    write_json_new(job / 'START.json', {'budget_consumed_before_model_load': True})
    with pytest.raises(RuntimeError, match='insufficient authorized starts'):
        training.require_screen_budget(tmp_path, current_launch='V-freedom-flower')
    write_json_new(job / 'RESULT.json', {'complete': True})
    training.require_screen_budget(tmp_path)


def test_prior_failure_allows_only_remaining_predeclared_jobs(tmp_path, monkeypatch):
    import any2xsampa.training.input_supervision_training as training
    from any2xsampa.training.limited_data_artifacts import write_json_new
    freeze = {'budget': {'remaining_screen_starts': 9}, 'unavailable_screen_jobs': ['V-freedom-flower']}
    monkeypatch.setattr(training, 'load_manifest', lambda run: (freeze, {}, {}))
    training.require_screen_budget(tmp_path)
    with pytest.raises(ValueError, match='previously failed job'):
        training.train_job(tmp_path, 'V', 'freedom-flower', 17, tmp_path/'screen/V/freedom-flower')
    write_json_new(tmp_path/'commands/V-freedom-flower.command.json', {})
    with pytest.raises(ValueError, match='attempted retry'):
        training.require_screen_budget(tmp_path)


@pytest.mark.parametrize('damage', ['none', 'missing', 'unapproved', 'freeze', 'code', 'evidence'])
def test_training_review_is_hash_bound(tmp_path, monkeypatch, damage):
    import any2xsampa.training.input_supervision_training as training
    from any2xsampa.training.limited_data_artifacts import write_json_new, sha256_file
    freeze = {'code_bindings': {'reviewed.py': '0'*64}}
    write_json_new(tmp_path / 'FREEZE.json', freeze)
    evidence = tmp_path / 'verified.json'
    write_json_new(evidence, {'passed': True})
    monkeypatch.setattr(training, 'load_manifest', lambda run: (freeze, {}, {}))
    review = {'approved_for_training': True, 'freeze_sha256': sha256_file(tmp_path / 'FREEZE.json'),
              'code_bindings': deepcopy(freeze['code_bindings']),
              'verification_bindings': [{'path': str(evidence), 'sha256': sha256_file(evidence)}]}
    if damage == 'unapproved': review['approved_for_training'] = False
    if damage == 'freeze': review['freeze_sha256'] = '1'*64
    if damage == 'code': review['code_bindings']['reviewed.py'] = '1'*64
    if damage == 'evidence': evidence.write_text('{}', encoding='utf-8')
    if damage != 'missing': write_json_new(tmp_path / 'CODE_REVIEW.json', review)
    if damage == 'none':
        training.require_code_review(tmp_path)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            training.require_code_review(tmp_path)


def test_waveform_artifact_is_not_overwritten(tmp_path):
    import numpy as np
    from tools.input_condition_probe import _write_wav
    path = tmp_path/'clip.wav'
    _write_wav(path, np.zeros((441,2), dtype='float32'))
    before = path.read_bytes()
    with pytest.raises(FileExistsError): _write_wav(path, np.ones((441,2), dtype='float32')*.5)
    assert path.read_bytes() == before
    for values in (np.full((441,2), float('nan')), np.ones((441,2))):
        with pytest.raises(ValueError): _write_wav(tmp_path/'invalid.wav', values)


def test_bound_file_hash_rejects_cache_swap(tmp_path):
    from any2xsampa.training.limited_data_artifacts import verify_file, sha256_file
    path = tmp_path/'features.safetensors'
    path.write_bytes(b'original immutable prefix cache')
    digest = sha256_file(path)
    path.write_bytes(b'wrong window feature bytes')
    with pytest.raises(ValueError, match='SHA-256 mismatch'): verify_file(path, digest)


@pytest.mark.parametrize('damage', ['holdout', 'input'])
def test_gate_cannot_score_training_song_or_different_input(damage):
    folds, base, songs, details = gate_fixture()
    if damage == 'holdout':
        folds[0]['validation_v'], folds[1]['validation_v'] = folds[1]['validation_v'], folds[0]['validation_v']
    else:
        folds[0]['validation_v']['examples'][0]['condition'] = 'x'
    with pytest.raises(ValueError, match='wrong held song or input condition'):
        compare_screen('V', folds, base, songs, details)

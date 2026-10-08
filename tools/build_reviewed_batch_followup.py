"""Freeze five unused windows and two existing uncertain edges for one human pass."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.replacement_candidates import overlaps, select_candidates
from tools.batch_training_review import render_page, verify_sources_and_media
from tools.build_batch_training_review import OLD, PROGRESS, choice_for, proposals
from tools.accept_reviewed_training_features import verified_reference
from tools.prepare_original_training_candidates import _preview
from tools.reconcile_replacement_edges import load_verified_packet

BATCH = ROOT / '.state/runs/batch-training-review-20260918T001300'
INTAKE = ROOT / '.state/runs/batch-training-intake-20260918T020000'
FEATURES = ROOT / '.state/runs/batch-training-features-20260918T021000'
EDGE_INTAKE = ROOT / '.state/runs/replacement-edge-intake-20260917T224000'


def select_unused(packet: dict, verified: dict, cumulative: dict) -> tuple[list, dict]:
    sources = packet['source_bindings']
    history = read_json(OLD / 'HISTORY.json')
    location = read_json(Path(sources['location_review']['path']))
    # Every last-batch question is protected, including rejected/uncertain rows.
    protected = history['retained'] + location['items'] + cumulative['rows'] + packet['items']
    contexts = read_json(Path(sources['contexts']['path']))
    audit = read_json(PROGRESS / 'MULTIPATH_AUDIT.json')
    pool, excluded = [], []
    for context in contexts['contexts']:
        if context['split'] != 'train':
            continue
        alignment = read_json(Path(sources[context['context_id'] + '_saved_paths']['path']))
        if alignment['context'] != context:
            raise ValueError('saved context mismatch')
        values, failures = proposals(context, verified['songs'][context['song_id']], alignment,
                                     protected, history['rejected'], audit['anchors'], packet['policy'])
        pool.extend(values)
        excluded.extend(failures)
    selected = select_candidates(pool, cumulative['rows'], contexts['split']['train'], 5)
    if len(selected) != 5:
        raise ValueError('five unused windows unavailable; do not repeat prior questions')
    return selected, {'pool_count': len(pool), 'excluded': excluded, 'pool': pool,
                      'selection_uses_phone_predictions': False}


def pending_rows(packet: dict, reference: dict, songs: dict) -> list[dict]:
    r11 = deepcopy(next(r for r in packet['items'] if r['review_id'] == 'R11'))
    answer = deepcopy(next(r for r in reference['examples'] if r['review_id'] == 'R11')['raw_answer'])
    r11['alternatives'] = [c for c in r11['alternatives'] if c['choice_id'] == answer['choice']]
    r11.update(kind='retained', correspondence_preconfirmed=True, previous_answer=answer,
               previous_audit=answer,
               review_focus='已保留上轮首尾记录。仅需检查首字第二个音素的不确定状态；仍听不清可继续待定。重新选择结论并提交后才计为本轮回答。')
    edge_ref = verified_reference(EDGE_INTAKE)
    inputs = read_json(EDGE_INTAKE / 'INPUTS.json')
    edge_packet = load_verified_packet(Path(inputs['packet']['path']))
    clip = next(c for c in edge_packet['clips'] if c['location_review_id'] == 'L02')
    prior = next(r for r in edge_ref['examples'] if r['example_id'] == clip['example_id'])
    alternative = clip['correspondence_evidence']['selected_alternative']
    choice = choice_for(songs[clip['song_id']], alternative['start_sec'], alternative['end_sec'], 'human_reviewed_section', 0)
    choice.update(choice_id='A', catalog=deepcopy(clip['notes']))
    previous = {'review_id': 'L02', 'status': 'confirmed', 'choice': 'A', 'comment': '',
                'sequence_confirmed': True, 'edges': deepcopy(prior['edge_evidence'])}
    l02 = {k: clip[k] for k in ('song_id', 'split', 'start_sec', 'end_sec')}
    l02.update(review_id='L02', kind='retained', correspondence_preconfirmed=True,
               alternatives=[choice], previous_answer=previous, previous_audit=prior['edge_evidence'],
               review_focus='已保留此前首尾记录。仅需检查末字“空”的 /h/ 不确定状态；仍听不清可继续待定。重新选择结论并提交后才计为本轮回答。')
    return [l02, r11]


def build(output: Path) -> dict:
    if output.exists() or not output.is_relative_to(ROOT / '.state/runs'):
        raise ValueError('new run required')
    packet = read_json(BATCH / 'packet.json')
    verified = verify_sources_and_media(BATCH, packet)
    reference = verified_reference(INTAKE)
    cumulative = read_json(FEATURES / 'CUMULATIVE_DATA.json')
    if cumulative['manifest_hash'] != sha256_object({k: v for k, v in cumulative.items() if k != 'manifest_hash'}):
        raise ValueError('cumulative hash mismatch')
    from tools.merge_reviewed_training_acceptance import merge_rows
    from tools.accept_reviewed_training_features import finish
    pairs = [(PROGRESS, EDGE_INTAKE), (ROOT / '.state/runs/retained-edge-intake-20260917T233500',) * 2, (FEATURES, INTAKE)]
    rebuilt = merge_rows([finish(ref, acc, write=False) for acc, ref in pairs])
    if any(cumulative[k] != v for k, v in rebuilt.items()):
        raise ValueError('cumulative data cannot be reproduced')
    selected, audit = select_unused(packet, verified, cumulative)
    rows = pending_rows(packet, reference, verified['songs'])
    rows += [{**r, 'review_id': f'N{i:02}', 'kind': 'replacement',
              'correspondence_preconfirmed': False, 'previous_audit': None,
              'review_focus': '新十秒窗口；请在本页完成段落对应及两端音素。'} for i, r in enumerate(selected, 1)]
    for i, row in enumerate(rows):
        if any(overlaps(row, old) for old in cumulative['rows'] + rows[:i]):
            raise ValueError('review overlaps accepted data or another review')
    sources = deepcopy(packet['source_bindings'])
    for name, path in [('prior_batch', BATCH / 'packet.json'), ('prior_batch_answers', INTAKE / 'raw-answers.json'),
                       ('prior_batch_reference', INTAKE / 'reviewed-reference.json'), ('current_cumulative', FEATURES / 'CUMULATIVE_DATA.json'),
                       ('pending_edge_reference', EDGE_INTAKE / 'reviewed-reference.json'), ('pending_edge_inputs', EDGE_INTAKE / 'INPUTS.json'),
                       ('followup_builder', Path(__file__))]:
        sources[name] = {'path': str(path.resolve()), 'sha256': sha256_file(path)}
    policy = {'window_sec': 10, 'step_sec': .5, 'new_windows': 5, 'existing_pending_windows': 2,
              'prior_accepted_windows': 13, 'active_slots_after': 20,
              'all_previous_batch_original_ranges_protected': True,
              'prior_answers_are_drafts_only': True, 'mapping_aliases_added': False,
              'not_resubmitted': ['C16: unresolved mapping', 'R08: repeated section uncertain',
                                  'R03/R07/R10: rejected'], 'training_started': False}
    output.mkdir(parents=True)
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
    enforce_offline(output / 'runtime-cache')
    write_json_new(output / 'FROZEN_POLICY.json', policy)
    write_json_new(output / 'PROPOSAL_AUDIT.json', audit)
    media = output / 'media'
    media.mkdir()
    for row in rows:
        song = verified['songs'][row['song_id']]
        row.update(original_audio=deepcopy(song['original_audio']), source_vsqx=deepcopy(song['human_vsqx']), training_eligible=False)
        row['original'] = _preview(song['original_audio'], row, media, row['review_id'] + '-original', 'original')
        row['edge_media'] = {side: _preview(song['original_audio'], {'start_sec': row[side + '_sec'] - 1.25,
                             'end_sec': row[side + '_sec'] + 1.25}, media, row['review_id'] + '-' + side, 'original')
                             for side in ('start', 'end')}
        for choice in row['alternatives']:
            choice['preview'] = _preview(song['rendered_audio'], choice, media, row['review_id'] + '-' + choice['choice_id'], 'rendered')
    result = {'schema_version': 'batch_training_review_packet_v1', 'source_bindings': sources,
              'items': rows, 'prior_valid_windows': 13, 'active_slots_after': 20, 'new_slot_count': 5,
              'policy': policy, 'training_started': False, 'test_songs_opened': False}
    result['packet_sha256'] = sha256_object(result)
    write_json_new(output / 'packet.json', result)
    render_page(output, result)
    for spec in sources.values():
        verify_file(Path(spec['path']), spec['sha256'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output.resolve())
    print('items', len(result['items']), 'packet', result['packet_sha256'])

"""Reproduce review-history failures and test constrained localization offline."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new, sha256_object
from any2xsampa.training.reviewed_correspondence import (
    batch_history, history_failures, section_distance, descriptor_cost, constrained_path, dual_path_check,
)
from tools.accept_reviewed_training_features import verified_reference

RUNS = ROOT / '.state/runs'
ACCEPTED_RUNS = [
    'replacement-edge-intake-20260917T224000', 'retained-edge-intake-20260917T233500',
    'batch-training-intake-20260918T020000', 'reviewed-batch-followup-intake-20260918T140000',
]


def run(output: Path, *, vocal_contexts: bool = False) -> None:
    import numpy as np

    if output.exists() or not output.is_relative_to(RUNS):
        raise ValueError('new run required')
    output.mkdir(parents=True)
    started = time.monotonic()
    bindings = {}

    def load(name: str, path: Path, expected: str | None = None) -> dict:
        digest = verify_file(path, expected) if expected else sha256_file(path)
        bindings[name] = {'path': str(path.resolve()), 'sha256': digest}
        return read_json(path)

    latest = load('N07', RUNS / 'final-training-window-followup-20260918T223000/packet.json')
    sources = latest['source_bindings']
    ref = load('reference', Path(sources['reference']['path']), sources['reference']['sha256'])
    songs = {s['song_id']: s for s in ref['songs'] if s['split'] == 'train'}
    contexts = load('contexts', Path(sources['contexts']['path']), sources['contexts']['sha256'])['contexts']
    cumulative = load('accepted', RUNS / 'reviewed-batch-followup-features-20260918T141000/CUMULATIVE_DATA.json')
    if cumulative['manifest_hash'] != sha256_object({k: v for k, v in cumulative.items() if k != 'manifest_hash'}):
        raise ValueError('cumulative hash mismatch')
    accepted_ids = {r['example_id'].lower() for r in cumulative['rows']}
    sections, history = [], []
    for name in ACCEPTED_RUNS:
        base = RUNS / name
        verified = verified_reference(base)
        load(name, base / 'reviewed-reference.json')
        inputs = load(name + '_inputs', base / 'INPUTS.json')
        packet = load(name + '_packet', Path(inputs['packet']['path']), inputs['packet']['sha256'])
        answers = load(name + '_answers', Path(inputs['answers']['path']), inputs['answers']['sha256'])
        if packet['schema_version'] == 'batch_training_review_packet_v1':
            history += batch_history(packet, answers)
        for row in verified['examples']:
            if row.get('status') != 'human_confirmed' or row['example_id'].lower() not in accepted_ids:
                continue
            song = songs[row['song_id']]
            notes = {n['note_id']: n for n in song['notes']}
            if row['audio_sha256'] != song['original_audio']['sha256'] or row['source_vsqx_sha256'] != song['human_vsqx']['sha256']:
                raise ValueError('accepted section identity mismatch')
            sections.append({
                'example_id': row['example_id'].lower(), 'song_id': row['song_id'],
                'start_sec': row['start_sec'], 'end_sec': row['end_sec'],
                'rendered_start_sec': notes[row['note_ids'][0]]['start_sec'],
                'rendered_end_sec': notes[row['note_ids'][-1]]['end_sec'],
                'reference_source': name, 'scope': 'confirmed_section_not_exact_cut_time',
            })
    if len(sections) != len(accepted_ids) or {s['example_id'] for s in sections} != accepted_ids:
        raise ValueError('incomplete accepted section inventory')
    prior_history = load('old_history', RUNS / 'training-replacements-20260917T211100/HISTORY.json')
    # Add the latest N06/N07 answers, including the explicitly transcribed chat.
    p6 = load('N06', RUNS / 'final-training-window-review-20260918T150000/packet.json')
    before_n6 = history_failures(p6['items'][0], history)
    a6 = load('N06_answers', RUNS / 'final-training-window-rejected-20260918T213000/raw-answers.json')
    history += batch_history(p6, a6)
    before_n7 = history_failures(latest['items'][0], history)
    a7 = load('N07_answers', RUNS / 'n07-withdrawal-20260918T231226/conversation-answer.json')
    history += batch_history(latest, a7)
    policy = {
        'schema_version': 'reviewed_correspondence_diagnostic_v1',
        'descriptor_downsample': 4, 'frame_sec': 512 / 22050 * 4,
        'constraints': 'three_rows_at_reviewed_midpoint_within_entire_score_section',
        'held_out_rule': 'remove_entire_target_section_constraint_before_matching',
        'acceptance_metric': 'predicted_midpoint_inside_reviewed_section',
        'automatic_phone_timing_or_review_allowed': False,
        'history': 'all_supplied_batch_answers_including_uncertain_rejected_unanswered',
        'mapping_used_for_path_search': False, 'phone_predictions_used': False,
        'descriptor_condition': 'vocal_in_existing_contexts_mixed_elsewhere' if vocal_contexts else 'mixed',
        'original_span': 'frozen_window_start_to_end_no_unreviewed_intro_outro_constraint',
        'maximum_dual_path_disagreement_sec': .2,
        'training_started': False, 'test_songs_opened': False,
    }
    write_json_new(output / 'POLICY.json', policy)
    write_json_new(output / 'HISTORY.json', {'batch_reviews': history, 'older_reviews': prior_history,
        'duplicate_N06': before_n6, 'duplicate_N07': before_n7, 'accepted_sections': sections})
    rows, rejected_paths = [], []
    for sid, song in songs.items():
        spec = sources[sid + '_descriptors']
        verify_file(Path(spec['path']), spec['sha256'])
        bindings[sid + '_descriptors'] = spec
        descriptor_record = load(sid + '_descriptor_record', Path(sources[sid + '_descriptor_record']['path']),
                                 sources[sid + '_descriptor_record']['sha256'])
        for role in ('original', 'rendered'):
            if descriptor_record[role + '_audio_sha256'] != song[role + '_audio']['sha256']:
                raise ValueError('descriptor source mismatch')
        local = [s for s in sections if s['song_id'] == sid]
        # Do not force unreviewed intros/outros to match: versions can end at
        # different points. This crop uses only frozen original window times.
        first = int(np.floor(min(s['start_sec'] for s in local) / policy['frame_sec']))
        last = int(np.ceil(max(s['end_sec'] for s in local) / policy['frame_sec'])) + 1
        query_start = first * policy['frame_sec']
        with np.load(spec['path'], allow_pickle=False) as data:
            original = data['original'][:, ::4][:, first:last].copy()
            rendered = data['rendered'][:, ::4]
        if vocal_contexts:
            from any2xsampa.training.audio_section_correspondence import audio_descriptor
            times = query_start + np.arange(original.shape[1]) * policy['frame_sec']
            for context in (c for c in contexts if c['song_id'] == sid and c['split'] == 'train'):
                cid = context['context_id']
                spec = sources[cid + '_separation']
                separation = load(cid + '_separation', Path(spec['path']), spec['sha256'])
                if separation['context'] != context or not separation['clock']['no_time_stretch_or_compensation']:
                    raise ValueError('vocal context clock mismatch')
                spec = sources[cid + '_vocal']
                if any(separation['vocal'][k] != spec[k] for k in ('path', 'sha256')):
                    raise ValueError('vocal source identity mismatch')
                bindings[cid + '_vocal'] = spec
                vocal = audio_descriptor(spec)[:, ::4]
                clock_start = separation['clock']['effective_original_start_sec']
                vocal_times = clock_start + np.arange(vocal.shape[1]) * policy['frame_sec']
                inside = (times >= max(context['start_sec'], vocal_times[0])) & (times <= min(context['end_sec'], vocal_times[-1]))
                for channel in range(original.shape[0]):
                    original[channel, inside] = np.interp(times[inside], vocal_times, vocal[channel])
                print('loaded_existing_vocal', cid, flush=True)
        cost = descriptor_cost(original, rendered)
        full = constrained_path(cost, local, policy['frame_sec'], query_start_sec=query_start)
        write_json_new(output / 'paths' / (sid + '.json'), full)
        for target in local:
            context = next(c for c in contexts if c['song_id'] == sid and
                           c['start_sec'] <= target['start_sec'] and c['end_sec'] >= target['end_sec'])
            spec = sources[context['context_id'] + '_saved_paths']
            saved = load(context['context_id'], Path(spec['path']), spec['sha256'])
            old = {role: section_distance(data['candidates'][0], target) for role, data in saved['paths'].items()}
            held = constrained_path(cost, local, policy['frame_sec'], held_out_id=target['example_id'],
                                    query_start_sec=query_start)
            assert target['example_id'] not in held['constraint_example_ids']
            error = section_distance(held, target)
            record = {**target, 'old_top1_section_miss_sec': old,
                      'held_out_section_miss_sec': error,
                      'held_out_constraint_ids': held['constraint_example_ids'],
                      'held_out_passed': error == 0}
            rows.append(record)
            write_json_new(output / 'held-out' / (target['example_id'] + '.json'), held)
            print(sid, target['example_id'], 'old=',old, 'held_out_miss=',round(error, 3), flush=True)
        if sid == 'promise':
            for packet in (p6, latest):
                item = packet['items'][0]
                spec = sources[item['context_id'] + '_saved_paths']
                saved = load(item['context_id'], Path(spec['path']), spec['sha256'])
                for choice in item['alternatives']:
                    own = next(p for p in saved['paths'][choice['source_role']]['candidates'] if p['rank'] == choice['path_rank'])
                    other = 'mixed' if choice['source_role'] == 'vocal' else 'vocal'
                    checks = [dual_path_check(own, p, item['start_sec'], item['end_sec']) for p in saved['paths'][other]['candidates']]
                    rejected_paths.append({'review_id': item['review_id'], 'choice': choice['choice_id'],
                        'best_cross_input_disagreement_sec': min(c['maximum_disagreement_sec'] for c in checks),
                        'passes_dual_path_gate': any(c['passes'] for c in checks)})
    report = {
        'reviewed_windows': len(rows), 'held_out_passed': sum(r['held_out_passed'] for r in rows),
        'old_top1_passed': {role: sum(r['old_top1_section_miss_sec'][role] == 0 for r in rows) for role in ('mixed', 'vocal')},
        'rows': rows, 'rejected_candidate_cross_input_checks': rejected_paths,
        'new_review_page_generated': False, 'training_started': False, 'valid_training_windows': 19,
        'section_check_not_phone_accuracy': True, 'elapsed_seconds': time.monotonic() - started,
        'source_bindings': bindings,
    }
    write_json_new(output / 'RESULTS.json', report)
    print('completed', {k:report[k] for k in ('reviewed_windows','held_out_passed','old_top1_passed')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocal-contexts', action='store_true')
    args = parser.parse_args()
    run(args.output.resolve(), vocal_contexts=args.vocal_contexts)

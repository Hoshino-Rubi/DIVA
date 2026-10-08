"""Freeze reviewed training references; uncertain phones block the whole window."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.original_edge_review import _selection_range
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
from any2xsampa.training.reviewed_audibility_reference import _edge_states
from tools.build_original_edge_review import note_view, verify_media
from tools.import_replacement_location_answers import validate_edge_answers, validate_edge_packet


def assemble_row(clip: dict, edges: dict, reviewer: str) -> dict:
    row = {k: deepcopy(clip[k]) for k in ('example_id', 'song_id', 'split', 'audio_path', 'audio_sha256',
        'source_vsqx_sha256', 'reference_manifest_sha256', 'start_sec', 'end_sec')}
    row.update(status='pending', audio_role='original', reviewer=reviewer, edge_evidence=deepcopy(edges),
               source_phones=[], phones=[], omitted_phones=[], note_ids=[], blocking_reasons=[],
               training_eligible=False, ctc_frame_check='not_executed', missing_phone_cause='not_assessed')
    for key in ('candidate_origin', 'retained_review_id'):
        if key in clip:
            row[key] = clip[key]
    if any(e and e['status'] == 'wrong_correspondence' for e in edges.values()):
        row.update(status='rejected', blocking_reasons=['human_rejected_correspondence'])
        return row
    if any(e is None or e['status'] != 'confirmed' for e in edges.values()):
        row['blocking_reasons'] = ['incomplete_confirmed_edge_pair']
        return row
    if clip.get('source_sequence_approval_bound') is not True:
        row['blocking_reasons'] = ['source_sequence_approval_not_bound']
        return row
    try:
        notes, lo, hi = _selection_range(clip, edges['start'], edges['end'])
        lookup = {n['note_id']: n for n in notes}
        states, evidence = {}, defaultdict(list)
        for answer in edges.values():
            nid = answer['note_id']
            state = _edge_states(answer, lookup[nid])
            if nid in states and states[nid] != state:
                raise ValueError('same_note_edges_have_conflicting_audibility')
            if not any(v in {'full', 'partial'} for v in state.values()):
                raise ValueError('selected_edge_note_has_no_audible_phone')
            states[nid] = state
            evidence[nid].append(answer['review_id'])
    except ValueError as error:
        row['blocking_reasons'] = [str(error)]
        return row
    for note in notes:
        for index, phone in enumerate(note['phones']):
            state = states[note['note_id']][index] if note['note_id'] in states else 'full'
            value = {'note_id': note['note_id'], 'phone_index': index, 'phone': phone, 'audibility': state,
                     'evidence_scope': 'explicit_human_edge' if note['note_id'] in states else 'inherited_source_interior',
                     'review_ids': evidence.get(note['note_id'], [])}
            row['source_phones'].append(value)
            row['phones' if state in {'full', 'partial'} else 'omitted_phones'].append(deepcopy(value))
    row.update(status='human_confirmed', note_ids=[n['note_id'] for n in notes],
               source_phone_count=len(row['source_phones']), reference_phone_count=len(row['phones']),
               partial_phone_count=sum(p['audibility'] == 'partial' for p in row['phones']),
               catalog_outside_confirmed_note_range=[n['note_id'] for n in clip['notes'][:lo]+clip['notes'][hi+1:]],
               reference_nature='human_reviewed_edges_with_inherited_calibrated_source_ipa')
    return row


def build_reference(packet: dict, payload: dict) -> dict:
    answers = validate_edge_answers(packet, payload)
    groups = defaultdict(dict)
    for item in packet['items']:
        groups[item['example_id']][item['side']] = answers.get(item['review_id'])
    rows = [assemble_row(c, groups[c['example_id']], payload['reviewer']) for c in packet['clips']]
    confirmed = [r for r in rows if r['status'] == 'human_confirmed']
    unresolved = []
    for item in packet['items']:
        answer = answers.get(item['review_id'])
        if answer and answer['status'] == 'confirmed':
            note = next(n for n in item['notes'] if n['note_id'] == answer['note_id'])
            states = {p['phone_index']: p['edge_status'] for p in answer['phones']}
            for index, phone in enumerate(note['phones']):
                if states.get(index) in {None, 'uncertain'}:
                    unresolved.append({'review_id': item['review_id'], 'note_id': note['note_id'],
                        'lyric': note['lyric'], 'phone_index': index, 'phone': phone,
                        'state': states.get(index, 'unreviewed'), 'whole_window_excluded': True})
    result = {'schema_version': 'reviewed_training_audibility_reference_v1',
              'packet_sha256': packet['packet_sha256'], 'answers_sha256': sha256_object(payload),
              'examples': rows, 'unresolved_edge_phones': unresolved,
              'coverage': {'total_windows': len(rows), 'confirmed_windows': len(confirmed),
                  'confirmed_seconds': sum(r['end_sec']-r['start_sec'] for r in confirmed),
                  'confirmed_training_songs': len({r['song_id'] for r in confirmed}),
                  'reference_phones': sum(len(r['phones']) for r in confirmed),
                  'source_phones': sum(len(r['source_phones']) for r in confirmed),
                  'omitted_phones': sum(len(r['omitted_phones']) for r in confirmed),
                  'partial_phones': sum(r['partial_phone_count'] for r in confirmed)},
              'policy': {'training_started': False, 'ctc_frame_checks_completed': False,
                  'test_predictions_opened': False, 'internal_source_phones_unchanged': True,
                  'uncertain_windows_excluded_entirely': True, 'missing_cause_inferred': False,
                  'partial_phones_counted_once': True, 'full_song_timing_confirmed': False}}
    result['manifest_hash'] = sha256_object(result)
    return result


def load_verified_packet(path: Path) -> dict:
    packet = read_json(path)
    validate_edge_packet(packet)
    delivery_root = path.parent.parent
    delivery = read_json(delivery_root / 'DELIVERY.json')
    for name, digest in delivery['artifacts'].items():
        verify_file(delivery_root / name, digest)
    sources = packet['source_bindings']
    for spec in sources.values():
        verify_file(Path(spec['path']), spec['sha256'])
    verify_media(packet, path.parent)
    reference, _ = apply_original_mapping_overlay(read_json(Path(sources['reference']['path'])), Path(sources['mapping_evidence']['path']))
    triplets = read_json(Path(sources['reference_triplets']['path']))
    for clip in packet['clips']:
        song = next(s for s in reference['songs'] if s['song_id'] == clip['song_id'])
        if song['split'] != 'train' or clip['audio_sha256'] != song['original_audio']['sha256'] or clip['source_vsqx_sha256'] != song['human_vsqx']['sha256']:
            raise ValueError('source identity mismatch')
        verify_file(Path(clip['audio_path']), clip['audio_sha256'])
        verify_file(Path(clip['source_vsqx_path']), clip['source_vsqx_sha256'])
        catalog = {n['note_id']: note_view(n) for n in song['notes']}
        ids = [n['note_id'] for n in song['notes']]
        positions = [ids.index(n['note_id']) for n in clip['notes']]
        if positions != list(range(positions[0], positions[-1]+1)) or any(catalog[n['note_id']] != n for n in clip['notes']):
            raise ValueError('non-continuous or changed catalog')
        claim = next(c for c in triplets['items'] if c['song_id'] == song['song_id'])
        bound = claim['human_vsqx']['sha256'] == clip['source_vsqx_sha256'] and claim['human_claims'].get('phoneme_correctness') == 'user_asserted_correct'
        if clip['source_sequence_approval_bound'] != bound:
            raise ValueError('source sequence approval mismatch')
    return packet


def potential_window_count(history: dict, confirmed: list[dict], prior_accepted: list[dict]) -> int:
    # Retained candidates becoming references are the same windows, not extra data.
    return len({(r['song_id'], r['start_sec'], r['end_sec'])
                for r in history['retained'] + confirmed + prior_accepted})


def run(packet_path: Path, answers_path: Path, output: Path) -> dict:
    if output.exists() or not output.is_relative_to(ROOT / '.state/runs'):
        raise ValueError('output must be a new workspace run')
    packet = load_verified_packet(packet_path)
    payload = read_json(answers_path)
    result = build_reference(packet, payload)
    output.mkdir(parents=True)
    shutil.copyfile(answers_path, output / 'raw-answers.json')
    verify_file(output / 'raw-answers.json', sha256_file(answers_path))
    write_json_new(output / 'reviewed-reference.json', result)
    history = read_json(packet_path.parent.parent / 'history-preserved.json')
    prior = packet['source_bindings'].get('prior_data_acceptance')
    prior_accepted = read_json(Path(prior['path']))['rows'] if prior else []
    confirmed = [r for r in result['examples'] if r['status'] == 'human_confirmed']
    gate = {'schema_version': 'reviewed_training_gate_v1', 'training_started': False,
            'required_valid_windows': 20, 'required_training_songs': 3,
            'new_reference_confirmed_windows': result['coverage']['confirmed_windows'],
            'new_reference_confirmed_songs': result['coverage']['confirmed_training_songs'],
            'retained_old_candidates_total': len(history['retained']),
            'prior_data_accepted_windows': len(prior_accepted),
            'old_rejections_preserved': len(history['rejected']),
            'maximum_if_all_old_candidates_passed': potential_window_count(history, confirmed, prior_accepted),
            'ctc_frame_check': 'not_executed',
            'blocking_reasons': ['insufficient_confirmed_training_windows', 'ctc_frame_checks_not_executed'],
            'new_confirmed_references_are_not_training_eligibility': True}
    write_json_new(output / 'TRAINING_GATE.json', gate)
    write_json_new(output / 'INPUTS.json', {'packet': {'path': str(packet_path), 'sha256': sha256_file(packet_path)},
        'answers': {'path': str(answers_path), 'sha256': sha256_file(answers_path)},
        'source_bindings_verified': len(packet['source_bindings']), 'prediction_files_read': False,
        'command': ['.venv/Scripts/python.exe', 'tools/reconcile_replacement_edges.py', '--packet', str(packet_path),
                    '--answers', str(answers_path), '--output-dir', str(output)]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet.resolve(), args.answers.resolve(), args.output_dir.resolve())
    print(result['coverage'])
    print(result['unresolved_edge_phones'])


if __name__ == '__main__':
    main()

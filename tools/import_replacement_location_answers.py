"""Bind section-only answers and prepare edges without promoting phone truth."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from any2xsampa.training.limited_data_artifacts import (
    finite_number, read_json, sha256_file, sha256_object, verify_file, write_json_new,
)
from any2xsampa.training.original_edge_review import _validate_answer
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
from any2xsampa.training.replacement_candidates import overlaps
from tools.build_replacement_location_review import validate_answers
from tools.build_original_edge_review import note_view, verify_media

PACKET_SCHEMA = 'replacement_edge_review_packet_v1'
ANSWER_SCHEMA = 'replacement_edge_review_answers_v1'


def check_hash(packet: dict) -> None:
    if packet.get('packet_sha256') != sha256_object({k: v for k, v in packet.items() if k != 'packet_sha256'}):
        raise ValueError('packet hash mismatch')


def intake(packet: dict, answers: dict) -> dict:
    if packet.get('schema_version') != 'replacement_location_review_v1':
        raise ValueError('wrong location packet schema')
    validate_answers(packet, answers)
    if answers.get('scope') != 'correspondence_only' or answers.get('training_started') is not False:
        raise ValueError('answers exceed correspondence-only scope')
    if not isinstance(answers.get('reviewer'), str) or not answers['reviewer'].strip():
        raise ValueError('reviewer required')
    supplied = {a['review_id']: a for a in answers['answers']}
    rows = []
    for item in packet['items']:
        if item.get('split') != 'train':
            raise ValueError('only training correspondence may be imported')
        answer = supplied.get(item['review_id'])
        choice = answer['choice'] if answer else None
        selected = next((a for a in item['alternatives'] if a['choice_id'] == choice), None)
        rows.append({
            'review_id': item['review_id'], 'song_id': item['song_id'],
            'context_id': item['context_id'], 'start_sec': item['start_sec'], 'end_sec': item['end_sec'],
            'status': 'correspondence_confirmed' if selected else
                      'rejected' if choice == 'none' else 'uncertain' if choice == 'uncertain' else 'unanswered',
            'raw_answer': deepcopy(answer), 'selected_alternative': deepcopy(selected),
            'scope': 'this_original_window_to_selected_rendered_section_only',
            'phone_reference_confirmed': False, 'training_eligible': False,
            'global_timeline_confirmed': False,
        })
    return {'schema_version': 'replacement_location_intake_v1',
            'source_packet_sha256': packet['packet_sha256'], 'answers_sha256': sha256_object(answers),
            'reviewer': answers['reviewer'], 'rows': rows,
            'confirmed_correspondence_count': sum(r['status'] == 'correspondence_confirmed' for r in rows),
            'training_started': False, 'formal_phone_reference_created': False}


def validate_edge_packet(packet: dict) -> None:
    check_hash(packet)
    if packet.get('schema_version') != PACKET_SCHEMA:
        raise ValueError('wrong replacement edge schema')
    clips = {c['example_id']: c for c in packet['clips']}
    if not 1 <= len(clips) <= 3 or len(clips) != len(packet['clips']):
        raise ValueError('expected one to three distinct training clips')
    seen, sides = set(), {key: set() for key in clips}
    for clip in clips.values():
        start = finite_number(clip['start_sec'], 'start', minimum=0)
        end = finite_number(clip['end_sec'], 'end', minimum=0)
        if clip['split'] != 'train' or clip['song_id'] in {'shalong', 'ke-yi-ke-zai', 'king-of-karaoke'}:
            raise ValueError('non-training song forbidden')
        if abs(end-start-10) > 1e-6 or clip['training_eligible'] is not False:
            raise ValueError('window or eligibility changed')
    for item in packet['items']:
        rid, example, side = item['review_id'], item['example_id'], item['side']
        if rid in seen or example not in clips or side not in {'start', 'end'} or side in sides[example]:
            raise ValueError('invalid or duplicated edge')
        seen.add(rid)
        sides[example].add(side)
        if item['original_cut_sec'] != clips[example][side + '_sec']:
            raise ValueError('original cut mismatch')
        if item['duration_sec'] != 2.5 or item['cut_in_preview_sec'] != 1.25:
            raise ValueError('preview clock mismatch')
        catalog = {n['note_id']: n for n in clips[example]['notes']}
        if not item['notes'] or any(catalog.get(n['note_id']) != n for n in item['notes']):
            raise ValueError('edge notes differ from catalog')
    if any(s != {'start', 'end'} for s in sides.values()):
        raise ValueError('incomplete edge pair')


def validate_edge_answers(packet: dict, payload: dict) -> dict[str, dict]:
    validate_edge_packet(packet)
    if payload.get('schema_version') != ANSWER_SCHEMA or payload.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('wrong replacement edge answer binding')
    reviewer = payload.get('reviewer')
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(payload.get('answers'), list):
        raise ValueError('reviewer and answer list required')
    lookup = {i['review_id']: i for i in packet['items']}
    result = {}
    for answer in payload['answers']:
        rid = answer.get('review_id')
        if rid not in lookup or rid in result or answer.get('observation_schema') != 'phone_audibility_v2':
            raise ValueError('invalid replacement edge answer')
        _validate_answer(lookup[rid], answer, reviewer)
        result[rid] = deepcopy(answer)
    return result


def edge_notes(song: dict, cut: float) -> list[dict]:
    """Score neighbors are navigational hypotheses, not an original-audio clock."""
    return [note_view(n) for n in song['notes'] if n['start_sec'] < cut + 1.25 and n['end_sec'] > cut - 1.25]


def verify_sources(review_path: Path, packet: dict) -> tuple[dict, dict]:
    check_hash(packet)
    source_path = Path(packet['source_packet_path'])
    source = read_json(source_path)
    check_hash(source)
    if source['packet_sha256'] != packet['source_packet_sha256']:
        raise ValueError('candidate lineage mismatch')
    for spec in source['source_bindings'].values():
        verify_file(Path(spec['path']), spec['sha256'])
    delivery = read_json(source_path.parent / 'DELIVERY.json')
    artifact_hashes = {Path(name).as_posix(): digest for name, digest in delivery['artifacts'].items()}
    for name in ('HISTORY.json', 'location-review/review-packet.json'):
        verify_file(source_path.parent / name, artifact_hashes[name])
    history = read_json(source_path.parent / 'HISTORY.json')
    contexts = read_json(Path(source['source_bindings']['contexts']['path']))['contexts']
    contexts = {c['context_id']: c for c in contexts if c['split'] == 'train'}
    for item in packet['items']:
        context = contexts[item['context_id']]
        if item['song_id'] != context['song_id'] or item['split'] != 'train':
            raise ValueError('song or split mismatch')
        if not context['start_sec'] <= item['start_sec'] < item['end_sec'] <= context['end_sec']:
            raise ValueError('outside frozen context')
        if abs(item['end_sec'] - item['start_sec'] - 10) > 1e-6:
            raise ValueError('review window changed')
        if any(overlaps(item, old) for old in history['retained']):
            raise ValueError('overlap with retained window')
        verify_file(Path(item['alignment']['path']), item['alignment']['sha256'])
        for preview in [item['original']] + [a['preview'] for a in item['alternatives']]:
            verify_file(Path(preview['source_path']), preview['source_sha256'])
            for kind in ('audio', 'waveform'):
                path = (review_path.parent / 'media' / preview[kind + '_file']).resolve()
                if not path.is_relative_to((review_path.parent / 'media').resolve()):
                    raise ValueError('media path outside review directory')
                verify_file(path, preview[kind + '_sha256'])
    return source, history


def build_edges(result: dict, review: dict, reference: dict, sources: dict, media_dir: Path) -> dict:
    from tools.prepare_original_training_candidates import _preview

    lookup = {i['review_id']: i for i in review['items']}
    triplets = read_json(Path(sources['reference_triplets']['path']))
    clips, items = [], []
    for row in result['rows']:
        if row['status'] != 'correspondence_confirmed':
            continue
        song = next(s for s in reference['songs'] if s['song_id'] == row['song_id'])
        if song['split'] != 'train':
            raise ValueError('only training songs allowed')
        old = lookup[row['review_id']]
        choice = row['selected_alternative']
        for spec in (song['original_audio'], song['rendered_audio'], song['human_vsqx']):
            verify_file(Path(spec['path']), spec['sha256'])
        if old['original']['source_sha256'] != song['original_audio']['sha256'] or choice['preview']['source_sha256'] != song['rendered_audio']['sha256']:
            raise ValueError('selected media and reference sources differ')
        expected = [{k: n[k] for k in ('note_id', 'raw_lyric', 'model_target_ipa', 'mapping_status')}
                    for n in song['notes'] if n['start_sec'] < choice['end_sec'] and n['end_sec'] > choice['start_sec']]
        if choice['notes'] != expected:
            raise ValueError('selected reference notes differ from bound mapping')
        example = 'replacement-' + row['review_id']
        neighbors = {side: edge_notes(song, choice[side + '_sec']) for side in ('start', 'end')}
        if not all(neighbors.values()):
            raise ValueError('no edge note candidates')
        wanted = {n['note_id'] for group in neighbors.values() for n in group}
        indices = [i for i, n in enumerate(song['notes']) if n['note_id'] in wanted]
        catalog = [note_view(n) for n in song['notes'][min(indices):max(indices) + 1]]
        claim = next(c for c in triplets['items'] if c['song_id'] == song['song_id'])
        approval_bound = (claim['human_vsqx']['sha256'] == song['human_vsqx']['sha256'] and
                          claim['human_claims'].get('phoneme_correctness') == 'user_asserted_correct')
        clips.append({
            'example_id': example, 'location_review_id': row['review_id'], 'song_id': row['song_id'], 'split': 'train',
            'audio_path': song['original_audio']['path'], 'audio_sha256': song['original_audio']['sha256'],
            'source_vsqx_path': song['human_vsqx']['path'], 'source_vsqx_sha256': song['human_vsqx']['sha256'],
            'reference_manifest_sha256': sources['reference']['sha256'],
            'start_sec': row['start_sec'], 'end_sec': row['end_sec'], 'notes': catalog,
            'correspondence_evidence': deepcopy(row), 'training_eligible': False,
            'source_sequence_approval_bound': approval_bound,
            'source_sequence_approval': {'source_binding': 'reference_triplets',
                'scope': 'source_vsqx_phone_sequence_only_not_original_audio_timing',
                'human_claims': deepcopy(claim['human_claims']), 'pairing_status': claim['pairing_status'],
                'source_vsqx_sha256': claim['human_vsqx']['sha256']},
        })
        for side in ('start', 'end'):
            rid = row['review_id'] + ('-S' if side == 'start' else '-E')
            cut = row[side + '_sec']
            preview = _preview(song['original_audio'], {'start_sec': cut - 1.25, 'end_sec': cut + 1.25}, media_dir, rid, 'original')
            for kind in ('audio', 'waveform'):
                preview[kind + '_file'] = 'media/' + preview[kind + '_file']
            items.append({
                'review_id': rid, 'example_id': example, 'song_id': row['song_id'],
                'side': side, 'original_cut_sec': cut, 'cut_in_preview_sec': 1.25, 'duration_sec': 2.5,
                'notes': neighbors[side], 'media': [preview],
                'previous_audit': {'decision': 'corresponds: ' + row['raw_answer']['choice'],
                                   'final_audit_response': deepcopy(row['raw_answer'])},
                'question': row['song_id'] + ' / ' + row['review_id'] + ': 此前选定的对应保持不变，仅确认原唱切点内侧。',
                'score_neighbor_cut_sec': choice[side + '_sec'],
                'score_neighbor_scope': 'navigation_only_not_original_phone_time',
            })
    packet = {'schema_version': PACKET_SCHEMA, 'source_bindings': sources, 'clips': clips, 'items': items,
              'policy': {'training_started': False, 'formal_phone_reference_created': False,
                         'global_timeline_confirmed': False, 'missing_phone_cause': 'not_assessed',
                         'test_songs_opened': False, 'prior_correspondence_reasked': False}}
    packet['packet_sha256'] = sha256_object(packet)
    if items:
        validate_edge_packet(packet)
    return packet


def run(review_path: Path, answers_path: Path, output: Path) -> dict:
    if output.exists() or not output.resolve().is_relative_to((ROOT / '.state/runs').resolve()):
        raise ValueError('output must be a new workspace run')
    review, answers = read_json(review_path), read_json(answers_path)
    result = intake(review, answers)
    source, history = verify_sources(review_path, review)
    bindings = deepcopy(source['source_bindings'])
    for name, path in (('location_review', review_path), ('location_answers', answers_path)):
        bindings[name] = {'path': str(path.resolve()), 'sha256': sha256_file(path)}
    raw_reference = read_json(Path(bindings['reference']['path']))
    for name, spec in raw_reference['bindings'].items():
        verify_file(Path(spec['path']), spec['sha256'])
        bindings['reference_' + name] = deepcopy(spec)
    reference, _ = apply_original_mapping_overlay(raw_reference, Path(bindings['mapping_evidence']['path']))
    output.mkdir(parents=True)
    shutil.copyfile(answers_path, output / 'raw-answers.json')
    verify_file(output / 'raw-answers.json', bindings['location_answers']['sha256'])
    write_json_new(output / 'intake.json', result)
    write_json_new(output / 'history-preserved.json', history)
    page_dir = output / 'edge-review'
    (page_dir / 'media').mkdir(parents=True)
    packet = build_edges(result, review, reference, bindings, page_dir / 'media')
    write_json_new(page_dir / 'packet.json', packet)
    if packet['items']:
        from tools.render_original_edge_review_gui import render_packet
        verify_media(packet, page_dir)
        render_packet(page_dir)
    status = {'schema_version': 'replacement_location_delivery_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'answer_count': len(answers['answers']), 'confirmed_correspondence_count': result['confirmed_correspondence_count'],
              'unresolved_review_ids': [r['review_id'] for r in result['rows'] if r['status'] != 'correspondence_confirmed'],
              'next_edge_questions': len(packet['items']), 'new_formal_training_windows': 0,
              'training_started': False, 'required_training_windows': 20, 'required_training_songs': 3,
              'retained_old_candidates': len(history['retained']), 'retained_old_rejections': len(history['rejected']),
              'approved_local_duration_sec': sum(c['end_sec']-c['start_sec'] for c in packet['clips']),
              'source_bindings_verified': len(bindings), 'browser_listening_verified': False}
    for spec in bindings.values():
        verify_file(Path(spec['path']), spec['sha256'])
    write_json_new(output / 'STATUS.json', status)
    files = [p for p in output.rglob('*') if p.is_file()]
    write_json_new(output / 'DELIVERY.json', {'artifacts': {str(p.relative_to(output)): sha256_file(p) for p in files},
        'command': ['.venv/Scripts/python.exe', 'tools/import_replacement_location_answers.py', '--review-packet', str(review_path),
                    '--answers', str(answers_path), '--output-dir', str(output)]})
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-packet', required=True, type=Path)
    parser.add_argument('--answers', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(run(args.review_packet.resolve(), args.answers.resolve(), args.output_dir.resolve()))


if __name__ == '__main__':
    main()

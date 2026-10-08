"""Audit saved path alternatives and finish only necessary retained clip edges."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new, sha256_object
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
from any2xsampa.training.replacement_candidates import path_time, propose_windows
from tools.build_original_edge_review import note_view, verify_media
from tools.import_replacement_location_answers import edge_notes, validate_edge_packet, verify_sources
from tools.reconcile_replacement_edges import load_verified_packet

OLD = ROOT / '.state/runs/training-replacements-20260917T211100'
EDGES = ROOT / '.state/runs/replacement-location-intake-20260917T222000/edge-review/packet.json'
POLICY = ROOT / 'src/any2xsampa/config/local_correspondence_review_v2.json'


def anchor_failures(candidate: dict, anchors: list[dict]) -> list[str]:
    failed = []
    for anchor in anchors:
        try:
            value = path_time(candidate, (anchor['start_sec'] + anchor['end_sec']) / 2)
            if not anchor['rendered_start_sec'] <= value <= anchor['rendered_end_sec']:
                failed.append(anchor['review_id'])
        except ValueError:
            failed.append(anchor['review_id'])
    return failed


def inventory(history: dict, songs: dict) -> list[dict]:
    rows = []
    for old in history['retained']:
        final = old['history']['final_audit_response']
        notes = [n for n in songs[old['song_id']]['notes']
                 if n['start_sec'] < old['rendered_start_sec']+10 and n['end_sec'] > old['rendered_start_sec']]
        issues = [n['note_id'] for n in notes if n['mapping_status'] != 'resolved_proxy' or not n['model_target_ipa']]
        clean = old['decision'] == 'corresponds' and not final['comment'].strip() and not issues and bool(notes)
        rows.append({**deepcopy(old), 'clean_correspondence': clean, 'unresolved_mapping_notes': issues,
                     'needed': 'explicit_edge_phones' if clean else 'boundary_or_missing_word_repair',
                     'formal_reference_created': False, 'old_answer_reused': True})
    return rows


def select_retained(rows: list[dict], accepted: list[dict], song_order: list[str], limit: int) -> list[dict]:
    counts = Counter(r['song_id'] for r in accepted)
    order = {s: i for i, s in enumerate(song_order)}
    remaining = [r for r in rows if r['clean_correspondence']]
    result = []
    while remaining and len(result) < limit:
        row = min(remaining, key=lambda r: (counts[r['song_id']], order[r['song_id']], r['start_sec'], r['review_id']))
        result.append(row)
        # This batch favors distinct songs; it is not a new correspondence query.
        remaining = [r for r in remaining if r['song_id'] != row['song_id']]
    return result


def audit_paths(output: Path, policy: dict, sources: dict, songs: dict, history: dict,
                rows: list[dict], location_review: dict, location_intake: dict) -> dict:
    contexts = read_json(Path(sources['contexts']['path']))['contexts']
    contexts = [c for c in contexts if c['split'] == 'train']
    anchors = [{**r, 'context_id': r['history']['source_example_id']+'-context',
                'rendered_end_sec': r['rendered_start_sec']+10} for r in rows if r['clean_correspondence']]
    for r in location_intake['rows']:
        if r['status'] == 'correspondence_confirmed':
            anchors.append({**r, 'rendered_start_sec': r['selected_alternative']['start_sec'],
                            'rendered_end_sec': r['selected_alternative']['end_sec']})
    protected = history['retained'] + location_review['items']
    known_alignment_hashes = {i['context_id']: i['alignment']['sha256'] for i in location_review['items']}
    pool, records, total_windows = [], [], 0
    before = time.monotonic()
    for context in contexts:
        cid = context['context_id']
        path = OLD / 'alignments' / (cid+'.json')
        digest = verify_file(path, known_alignment_hashes[cid]) if cid in known_alignment_hashes else sha256_file(path)
        sources[cid+'_saved_paths'] = {'path': str(path), 'sha256': digest}
        alignment = read_json(path)
        if alignment['context'] != context:
            raise ValueError('stored paths belong to a different context')
        local = [a for a in anchors if a['context_id'] == cid]
        pairs = product(alignment['paths']['mixed']['candidates'][:3], alignment['paths']['vocal']['candidates'][:3])
        for mixed, vocal in pairs:
            failures = {role: anchor_failures(p, local) for role, p in [('mixed', mixed), ('vocal', vocal)]}
            record = {'context_id': cid, 'mixed_rank': mixed['rank'], 'vocal_rank': vocal['rank'],
                      'local_anchor_ids': [a['review_id'] for a in local], 'anchor_failures': failures}
            if any(failures.values()):
                record.update(status='rejected_by_local_section_evidence', windows_examined=0)
            else:
                candidates, excluded = propose_windows(context, songs[context['song_id']],
                    {'mixed': mixed, 'vocal': vocal}, protected, history['rejected'], policy)
                total_windows += len(candidates)+len(excluded)
                for c in candidates:
                    c.update(mixed_rank=mixed['rank'], vocal_rank=vocal['rank'],
                             machine_candidate_only=True, training_eligible=False)
                pool.extend(candidates)
                record.update(status='windows_checked', windows_examined=len(candidates)+len(excluded),
                    candidate_count=len(candidates),
                    excluded=[{k: r[k] for k in ('start_sec','end_sec','reasons')} for r in excluded])
            records.append(record)
        print(f'paths_checked={cid}; pairs_so_far={len(records)}', flush=True)
    unique = {}
    for row in pool:
        key = (row['song_id'], row['start_sec'], row['end_sec'], round(row['rendered_start_sec'],6), round(row['rendered_end_sec'],6))
        unique.setdefault(key, row)
    result = {'schema_version': 'local_multipath_audit_v2', 'policy_sha256': sha256_object(policy),
        'contexts': len(contexts), 'pair_count': len(records), 'window_pair_checks': total_windows,
        'anchors': anchors, 'pairs': records, 'machine_candidates': list(unique.values()),
        'new_human_candidates_admitted': 0, 'training_started': False, 'elapsed_seconds': time.monotonic()-before,
        'prediction_or_per_used': False, 'thresholds_relaxed': False, 'test_songs_opened': False}
    write_json_new(output / 'MULTIPATH_AUDIT.json', result)
    return result


def prepare_edges(output: Path, selected: list[dict], songs: dict, sources: dict) -> dict:
    from tools.prepare_original_training_candidates import _preview
    from tools.render_original_edge_review_gui import render_packet

    page = output / 'edge-review'
    media_dir = page / 'media'
    media_dir.mkdir(parents=True)
    triplets = read_json(Path(sources['reference_triplets']['path']))
    clips, items = [], []
    for row in selected:
        song = songs[row['song_id']]
        original = row['history']
        if original['source_audio_sha256'] != song['original_audio']['sha256'] or original['source_vsqx_sha256'] != song['human_vsqx']['sha256']:
            raise ValueError('retained source binding differs')
        verify_file(Path(song['human_vsqx']['path']), song['human_vsqx']['sha256'])
        nearby = {side: edge_notes(song, row['rendered_start_sec']+(10 if side=='end' else 0)) for side in ('start','end')}
        if not all(nearby.values()):
            raise ValueError('empty retained edge candidates')
        wanted = {n['note_id'] for values in nearby.values() for n in values}
        indices = [i for i,n in enumerate(song['notes']) if n['note_id'] in wanted]
        catalog = [note_view(n) for n in song['notes'][min(indices):max(indices)+1]]
        claim = next(c for c in triplets['items'] if c['song_id'] == row['song_id'])
        bound = claim['human_vsqx']['sha256']==song['human_vsqx']['sha256'] and claim['human_claims'].get('phoneme_correctness')=='user_asserted_correct'
        example = 'retained-'+row['review_id'].lower()
        clips.append({'example_id': example, 'song_id': row['song_id'], 'split': 'train',
            'candidate_origin': 'retained_human_candidate', 'retained_review_id': row['review_id'],
            'audio_path': song['original_audio']['path'], 'audio_sha256': song['original_audio']['sha256'],
            'source_vsqx_path': song['human_vsqx']['path'], 'source_vsqx_sha256': song['human_vsqx']['sha256'],
            'reference_manifest_sha256': sources['reference']['sha256'], 'start_sec': row['start_sec'], 'end_sec': row['end_sec'],
            'notes': catalog, 'source_sequence_approval_bound': bound, 'training_eligible': False,
            'source_sequence_approval': {'source_binding': 'reference_triplets', 'scope': 'source_vsqx_phone_sequence_only',
                 'human_claims': deepcopy(claim['human_claims']), 'source_vsqx_sha256': claim['human_vsqx']['sha256']},
            'correspondence_evidence': deepcopy(original)})
        for side in ('start','end'):
            rid = row['review_id']+('-S' if side=='start' else '-E')
            cut = row[side+'_sec']
            preview = _preview(song['original_audio'], {'start_sec': cut-1.25, 'end_sec': cut+1.25}, media_dir, rid, 'original')
            for kind in ('audio','waveform'):
                preview[kind+'_file'] = 'media/'+preview[kind+'_file']
            items.append({'review_id': rid, 'example_id': example, 'song_id': row['song_id'], 'side': side,
                'original_cut_sec': cut, 'duration_sec': 2.5, 'cut_in_preview_sec': 1.25,
                'notes': nearby[side], 'media': [preview], 'previous_audit': deepcopy(original),
                'question': original['song_name']+' / '+row['review_id']+'：整段对应已经确认，只补本侧边缘的听觉记录。',
                'score_neighbor_scope': 'navigation_only_not_original_phone_time'})
    packet = {'schema_version':'replacement_edge_review_packet_v1', 'clips': clips, 'items': items,
        'source_bindings': sources, 'policy': {'training_started': False, 'formal_phone_reference_created': False,
            'existing_correspondence_reasked': False, 'retained_candidates_not_counted_twice': True,
            'new_training_slots_used': 0, 'source_phone_sequence_approval_not_timing_approval': True}}
    packet['packet_sha256'] = sha256_object(packet)
    validate_edge_packet(packet)
    write_json_new(page/'packet.json',packet)
    verify_media(packet,page)
    render_packet(page)
    return packet


def run(output: Path) -> dict:
    source_packet = load_verified_packet(EDGES)
    sources = deepcopy(source_packet['source_bindings'])
    for name, path in [('progress_policy', POLICY), ('retained_history', OLD/'HISTORY.json'),
                       ('prior_data_acceptance', output/'DATA_ACCEPTANCE.json')]:
        sources[name] = {'path':str(path), 'sha256':sha256_file(path)}
    policy = read_json(POLICY)
    write_json_new(output/'FROZEN_REVIEW_POLICY.json',policy)
    location_review = read_json(Path(sources['location_review']['path']))
    _, history = verify_sources(Path(sources['location_review']['path']), location_review)
    location_intake_path = EDGES.parent.parent/'intake.json'
    sources['prior_location_intake'] = {'path':str(location_intake_path), 'sha256':sha256_file(location_intake_path)}
    location_intake = read_json(location_intake_path)
    reference,_ = apply_original_mapping_overlay(read_json(Path(sources['reference']['path'])),Path(sources['mapping_evidence']['path']))
    songs = {s['song_id']:s for s in reference['songs'] if s['split']=='train'}
    rows = inventory(history,songs)
    write_json_new(output/'RETAINED_INVENTORY.json', {'items':rows, 'phone_truth_created':False})
    shutil.copyfile(OLD/'HISTORY.json',output/'history-preserved.json')
    accepted = read_json(output/'DATA_ACCEPTANCE.json')['rows']
    selected = select_retained(rows, accepted, read_json(Path(sources['contexts']['path']))['split']['train'], policy['maximum_human_clips'])
    # Freeze selection before auditing path alternatives; never choose questions from recognizer scores.
    write_json_new(output/'NEXT_SELECTION.json', {'selected_ids':[r['review_id'] for r in selected],
        'prior_accepted_examples':[r['example_id'] for r in accepted], 'new_slots_used':0, 'maximum_questions':4})
    audit = audit_paths(output,policy,sources,songs,history,rows,location_review,location_intake)
    packet = prepare_edges(output,selected,songs,sources)
    for spec in sources.values():
        verify_file(Path(spec['path']),spec['sha256'])
    result = {'selected_ids':[r['review_id'] for r in selected], 'human_edges':len(packet['items']),
        'path_pairs':audit['pair_count'], 'machine_candidate_count':len(audit['machine_candidates']),
        'data_accepted_windows':len(accepted), 'training_started':False, 'packet_sha256':packet['packet_sha256']}
    write_json_new(output/'PROGRESS_STATUS.json',result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'.state/runs'):
        parser.error('output outside workspace')
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
    enforce_offline(output/'audit-runtime-cache')
    print(run(output))


if __name__=='__main__':
    main()

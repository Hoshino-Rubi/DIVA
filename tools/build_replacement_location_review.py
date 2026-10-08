"""Minimal correspondence-only review after dual audio paths disagree."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, sha256_object, write_json_new
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
from any2xsampa.training.replacement_candidates import path_time, overlaps


def choose_questions(failures: list[dict], song_order: list[str]) -> list[dict]:
    questions = []
    for song in song_order:
        candidates = [r for r in failures if r.get('song_id') == song and r['reasons'] == ['mixed_vocal_paths_disagree']]
        if candidates:
            questions.append(min(candidates, key=lambda r: (
                r['dual_path_disagreement_sec'], r['center_distance_sec'], r['context_id'], r['start_sec'])))
    return questions


def alternatives(row: dict, alignment: dict, song: dict, rejected: list[dict]) -> list[dict]:
    result = []
    # Two independent primary paths first, then existing alternatives. No score/PER.
    ordered = [(role, candidate) for rank in range(3) for role in ('vocal', 'mixed')
               for candidate in alignment['paths'][role]['candidates'] if candidate['rank'] == rank + 1]
    for role, candidate in ordered:
        try:
            start = path_time(candidate, row['start_sec'])
            end = path_time(candidate, row['end_sec'])
        except ValueError:
            continue
        if not 7 <= end-start <= 13 or not 0 <= start < end <= song['rendered_audio']['duration_sec']:
            continue
        if any(abs(start-a['start_sec']) < .5 and abs(end-a['end_sec']) < .5 for a in result):
            continue
        if any(overlaps(row, old) and abs(path_time(candidate, (max(row['start_sec'],old['start_sec']) +
               min(row['end_sec'],old['end_sec']))/2) - (old['rendered_start_sec'] +
               (max(row['start_sec'],old['start_sec']) + min(row['end_sec'],old['end_sec']))/2-old['start_sec'])) < .5
               for old in rejected):
            continue
        notes = [n for n in song['notes'] if n['start_sec'] < end and n['end_sec'] > start]
        if not notes:
            continue
        result.append({'choice_id': chr(65+len(result)), 'source_role': role,
            'path_rank': candidate['rank'], 'start_sec': start, 'end_sec': end,
            'notes': [{k:n[k] for k in ('note_id','raw_lyric','model_target_ipa','mapping_status')} for n in notes],
            'mapping_complete': all(n['mapping_status']=='resolved_proxy' for n in notes),
            'human_status': 'pending', 'training_eligible': False})
        if len(result) == 3:
            break
    return result


def validate_answers(packet: dict, payload: dict) -> None:
    if packet.get('packet_sha256') != sha256_object({k:v for k,v in packet.items() if k!='packet_sha256'}):
        raise ValueError('packet hash mismatch')
    if payload.get('schema_version') != 'replacement_location_answers_v1' or payload.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('wrong answer schema or packet')
    lookup={r['review_id']:r for r in packet['items']}
    seen=set()
    if not isinstance(payload.get('answers'),list):
        raise ValueError('answers must be a list')
    for answer in payload['answers']:
        rid=answer.get('review_id')
        if rid not in lookup or rid in seen:
            raise ValueError('unknown or duplicate review ID')
        seen.add(rid)
        allowed={a['choice_id'] for a in lookup[rid]['alternatives']}|{'none','uncertain'}
        if answer.get('choice') not in allowed or not isinstance(answer.get('comment'),str):
            raise ValueError('invalid review choice')
        if not isinstance(payload.get('reviewer'),str) or not payload['reviewer'].strip():
            raise ValueError('reviewer required')


def build(source: Path, output: Path) -> dict:
    from tools.prepare_original_training_candidates import _preview
    packet=read_json(source/'candidate-packet.json')
    if packet['packet_sha256'] != sha256_object({k:v for k,v in packet.items() if k!='packet_sha256'}):
        raise ValueError('source packet hash differs')
    if packet['replacement_count'] != 0:
        raise ValueError('location fallback only for zero admitted replacements')
    for spec in packet['source_bindings'].values():
        verify_file(Path(spec['path']),spec['sha256'])
    bindings=packet['source_bindings']
    reference,_=apply_original_mapping_overlay(read_json(Path(bindings['reference']['path'])),Path(bindings['mapping_evidence']['path']))
    history=read_json(source/'HISTORY.json')
    failures=read_json(source/'excluded-windows.json')['windows']
    order=read_json(Path(bindings['contexts']['path']))['split']['train']
    rows=choose_questions(failures,order)
    if output.exists():
        raise FileExistsError(output)
    media=output/'media';media.mkdir(parents=True)
    items=[]
    for row in rows:
        song=next(s for s in reference['songs'] if s['song_id']==row['song_id'])
        if song['split']!='train':
            raise ValueError('test/dev song forbidden')
        alignment_path=source/'alignments'/f"{row['context_id']}.json"
        alignment=read_json(alignment_path)
        choices=alternatives(row,alignment,song,history['rejected'])
        if not choices:
            continue
        rid=f"L{len(items)+1:02d}"
        item={**row,'review_id':rid,'alternatives':choices,
              'alignment':{'path':str(alignment_path.resolve()),'sha256':sha256_file(alignment_path)},
              'original':_preview(song['original_audio'],row,media,f'{rid}-original','original'),
              'scope':'correspondence_only_not_phone_truth','training_eligible':False}
        for choice in choices:
            choice['preview']=_preview(song['rendered_audio'],choice,media,f"{rid}-{choice['choice_id']}",'rendered')
        items.append(item)
    result={'schema_version':'replacement_location_review_v1','source_packet_sha256':packet['packet_sha256'],
            'source_packet_path':str((source/'candidate-packet.json').resolve()),
            'items':items,'retained_training_slots':packet['retained_slots'],
            'rejected_history_unchanged':True,'test_songs_opened':False,'training_started':False,
            'new_training_candidates_admitted':0,'old_training_active_count':7,
            'selection':'one_fully_mapped_nonoverlapping_disagreement_window_per_available_song',
            'no_automatic_phone_reference':True}
    result['packet_sha256']=sha256_object(result)
    write_json_new(output/'review-packet.json',result)
    template=(ROOT/'tools/replacement_location_gui.html').read_text(encoding='utf-8')
    import json
    encoded=json.dumps(result,ensure_ascii=False).replace('<',r'\u003c')
    (output/'index.html').write_text(template.replace('__PACKET_JSON__',encoded),encoding='utf-8')
    print(f'location_questions={len(items)}; alternatives={sum(len(i["alternatives"]) for i in items)}')
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if not output.is_relative_to(ROOT/'.state/runs'):
        raise ValueError('output outside workspace runs')
    build(args.source.resolve(),output)


if __name__=='__main__':
    main()

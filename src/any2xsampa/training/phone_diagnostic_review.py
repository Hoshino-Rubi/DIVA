"""Validate a diagnostic listening response without changing labels or training."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from any2xsampa.training.limited_data_artifacts import read_json,sha256_object,sha256_file,verify_file,write_json_new

DECISIONS={'pending','a','b','neither','uncertain','wrong_location'}


def validate_answers(packet:dict,answers:dict) -> list[dict]:
    digest=sha256_object({k:v for k,v in packet.items() if k!='packet_sha256'})
    if packet.get('schema_version')!='phone_diagnostic_review_packet_v1' or packet.get('packet_sha256')!=digest:
        raise ValueError('diagnostic packet hash mismatch')
    if answers.get('schema_version')!='phone_diagnostic_review_answers_v1' or answers.get('packet_sha256')!=digest:
        raise ValueError('answers belong to another diagnostic packet')
    if answers.get('scope')!='diagnostic_only_no_reference_or_training_update':
        raise ValueError('unsupported answer scope')
    expected={x['review_id']:x for x in packet['items']};rows=answers.get('responses');seen=set();result=[]
    if not isinstance(rows,list) or len(rows)!=len(expected):raise ValueError('all IDs, including pending answers, must be retained')
    for row in rows:
        if not isinstance(row,dict) or row.get('review_id') not in expected or row['review_id'] in seen:
            raise ValueError('unknown or duplicate review ID')
        seen.add(row['review_id']);choice=row.get('decision');statement=row.get('statement')
        if choice not in DECISIONS or not isinstance(statement,str):raise ValueError('invalid diagnostic response')
        item=expected[row['review_id']]
        if item.get('reference_update_allowed') is not False or item['song_id'] in {'shalong','ke-yi-ke-zai','king-of-karaoke'}:
            raise ValueError('unexpected review scope or song')
        result.append({**row,'note_id':item['note_id'],'song_id':item['song_id'],
                       'selected_ipa':item['choices'].get(choice),'selected_source':item['choice_sources'].get(choice),
                       'formal_reference_created':False,'training_allowed':False})
    return result


def import_answers(packet_path:Path,answers_path:Path,output:Path) -> dict:
    packet=read_json(packet_path);answers=read_json(answers_path);rows=validate_answers(packet,answers)
    for item in packet['items']:
        media=(packet_path.parent/'media'/item['media']['file']).resolve()
        if not media.is_relative_to((packet_path.parent/'media').resolve()):raise ValueError('media escapes packet')
        verify_file(media,item['media']['sha256']);verify_file(Path(item['source_audio_path']),item['source_audio_sha256'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'raw-answers.json').write_bytes(answers_path.read_bytes())
    report={'schema_version':'phone_diagnostic_intake_v1','packet_sha256':packet['packet_sha256'],'raw_answers_sha256':sha256_file(answers_path),
            'responses':rows,'formal_reference_created':False,'training_started':False,'source_reference_changed':False}
    write_json_new(output/'intake.json',report)
    return report

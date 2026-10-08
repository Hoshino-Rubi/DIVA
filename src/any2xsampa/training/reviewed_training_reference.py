"""Strict union of immutable reviewed audibility references, without recasting absence."""
from pathlib import Path
from typing import Any
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_object, finite_number
from any2xsampa.training.limited_data_reference import SONG_SPLIT

SCHEMA = 'reviewed_original_training_reference_v1'


def validate_training_reference(value: dict[str, Any]) -> None:
    if value.get('schema_version') != SCHEMA:
        from any2xsampa.training.segment_reference import validate_segment_reference
        validate_segment_reference(value)
        return
    if value.get('manifest_hash') != sha256_object({k:v for k,v in value.items() if k != 'manifest_hash'}):
        raise ValueError('reviewed training reference hash mismatch')
    expected = {}
    for source in value['reference_sources']:
        verify_file(Path(source['path']), source['sha256'])
        original = read_json(Path(source['path']))
        for row in original['examples']:
            if row['status'] == 'human_confirmed':
                if row['example_id'] in expected:
                    raise ValueError('duplicate source reference')
                expected[row['example_id']] = row
    rows = value['examples']
    if len(rows) != len(expected) or {r['example_id'] for r in rows} != set(expected):
        raise ValueError('reviewed source coverage changed')
    for row in rows:
        if row != expected[row['example_id']]:
            raise ValueError('reviewed phones or human evidence changed')
        if row['audio_role'] != 'original' or row['split'] not in {'train','dev'} or SONG_SPLIT.get(row['song_id']) != row['split']:
            raise ValueError('song leakage or original role mismatch')
        start=finite_number(row['start_sec'],'start',minimum=0)
        end=finite_number(row['end_sec'],'end',minimum=0)
        if not 0 < end-start <= 20.000001 or not row['phones'] or not row['edge_evidence']:
            raise ValueError('invalid confirmed segment')
        retained=[p for p in row['source_phones'] if p['audibility'] in {'full','partial'}]
        omitted=[p for p in row['source_phones'] if p['audibility'] in {'absent','legacy_outside_selection'}]
        if row['phones'] != retained or row['omitted_phones'] != omitted or len(retained)+len(omitted)!=len(row['source_phones']):
            raise ValueError('unaccounted phone omission')
        if any(p['evidence_scope']!='explicit_human_edge' for p in omitted):
            raise ValueError('interior omission')
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            if left['song_id']==right['song_id']:
                if left['audio_sha256']!=right['audio_sha256'] or (left['start_sec']<right['end_sec'] and right['start_sec']<left['end_sec']):
                    raise ValueError('overlapping or mixed-version original windows')

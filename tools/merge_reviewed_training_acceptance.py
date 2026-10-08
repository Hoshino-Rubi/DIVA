"""Revalidate and deduplicate original training data without starting training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from any2xsampa.training.limited_data_artifacts import finite_number, read_json, sha256_file, sha256_object, write_json_new
from any2xsampa.training.limited_data_reference import SONG_SPLIT
from tools.accept_reviewed_training_features import finish


def merge_rows(sources: list[dict]) -> dict:
    unique = {}
    duplicates = 0
    for source in sources:
        if source.get('schema_version') != 'reviewed_training_data_acceptance_v1':
            raise ValueError('wrong acceptance schema')
        for row in source['rows']:
            if row['status'] != 'data_accepted' or row['split'] != 'train' or row['audio_role'] != 'original' or SONG_SPLIT.get(row['song_id']) != 'train':
                raise ValueError('only accepted original training rows')
            start = finite_number(row['start_sec'], 'start', minimum=0)
            end = finite_number(row['end_sec'], 'end', minimum=0)
            if end <= start:
                raise ValueError('invalid sample interval')
            key = (row['song_id'], row['audio_sha256'], start, end)
            if key in unique:
                if unique[key] != row:
                    raise ValueError('conflicting repeated window; do not choose silently')
                duplicates += 1
                continue
            for old in unique.values():
                if old['song_id'] != row['song_id']:
                    continue
                if old['audio_sha256'] != row['audio_sha256']:
                    raise ValueError('multiple original versions for one song')
                if start < old['end_sec'] and old['start_sec'] < end:
                    raise ValueError('overlapping original training windows')
            unique[key] = deepcopy(row)
    rows = sorted(unique.values(), key=lambda r: (r['song_id'], r['start_sec']))
    songs = sorted({r['song_id'] for r in rows})
    return {'schema_version': 'cumulative_reviewed_training_data_v1', 'rows': rows,
            'valid_windows': len(rows), 'training_songs': songs, 'song_count': len(songs),
            'unique_audio_seconds': sum(r['end_sec']-r['start_sec'] for r in rows),
            'reference_phones': sum(r['reference_phones'] for r in rows), 'duplicate_rows_not_counted': duplicates,
            'required_windows': 20, 'required_training_songs': 3,
            'window_shortfall': max(0,20-len(rows)), 'song_shortfall': max(0,3-len(songs)),
            'quantity_gate_passed': len(rows)>=20 and len(songs)>=3,
            'training_started': False, 'dev_gate_checked_here': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs=2, action='append', required=True, metavar=('ACCEPTANCE_RUN','REFERENCE_RUN'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'):
        parser.error('use a new output file inside workspace runs')
    values, bindings = [], []
    for acceptance, reference in args.source:
        acceptance, reference = Path(acceptance).resolve(), Path(reference).resolve()
        file = acceptance/'DATA_ACCEPTANCE.json'
        saved = read_json(file)
        if finish(reference, acceptance, write=False) != saved:
            raise ValueError('acceptance cannot be reproduced from sources and feature cache')
        values.append(saved)
        bindings.append({'acceptance_path':str(file),'acceptance_sha256':sha256_file(file),
                         'reference_path':str(reference/'reviewed-reference.json'),
                         'reference_sha256':sha256_file(reference/'reviewed-reference.json')})
    result = merge_rows(values)
    result['source_bindings'] = bindings
    result['manifest_hash'] = sha256_object(result)
    write_json_new(output,result)
    print({k:v for k,v in result.items() if k not in {'rows','source_bindings'}})


if __name__=='__main__':
    main()

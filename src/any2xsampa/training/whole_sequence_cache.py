"""Read whole-song prefix caches only after approval, geometry and architecture checks."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file
from any2xsampa.training.singing_domain import validate_cache_entry
from any2xsampa.training.whole_sequence_supervision import validate_chunks


def load_whole_rows(run:Path, torch:Any, architecture_sha256:str, held_song:str|None)->list[dict]:
    from safetensors.torch import load_file
    freeze=read_json(run/'FREEZE.json')
    for filename,key in [('WHOLE_REFERENCES.json','whole_references_sha256'),
                         ('WHOLE_GEOMETRY.json','whole_geometry_sha256'),
                         ('WHOLE_AUDIO_REQUESTS.json','whole_requests_sha256'),
                         ('review-intake/REVIEW_INTAKE.json','review_intake_sha256')]:
        verify_file(run/filename,freeze[key])
    cache=read_json(run/'WHOLE_CACHE.json')
    if (cache['architecture']['sha256']!=architecture_sha256 or cache['reference_received'] is not False
            or cache['all_parameters_frozen'] is not True
            or cache['audio_requests_sha256']!=freeze['whole_requests_sha256']
            or cache['geometry_sha256']!=freeze['whole_geometry_sha256']):
        raise ValueError('stale or label-conditioned whole prefix cache')
    requests={r['example_id']:r for r in read_json(run/'WHOLE_AUDIO_REQUESTS.json')['rows']}
    entries={r['example_id']:r for r in cache['entries']}
    if len(entries)!=len(cache['entries']) or set(entries)!=set(requests):
        raise ValueError('whole cache coverage/duplicate mismatch')
    geometry={r['song_id']:r for r in read_json(run/'WHOLE_GEOMETRY.json')['songs']}
    spec=read_json(run/'POLICY.json');songs=[]
    if held_song is not None and held_song not in spec['songs']:
        raise ValueError('unknown held song for whole-sequence partition')
    for ref in read_json(run/'WHOLE_REFERENCES.json')['songs']:
        if ref['song_id'] not in spec['songs'] or ref['split']!='train' or ref['audio_role']!='original':
            raise ValueError('whole sequence training role/song leakage')
        if ref['song_id']==held_song:continue
        verify_file(Path(ref['audio_path']),ref['audio_sha256'])
        group=geometry[ref['song_id']];chunks=[]
        validate_chunks(group['chunks'],group['total_frames'])
        for selection in group['chunks']:
            entry=entries[selection['example_id']];request=requests[selection['example_id']]
            validate_cache_entry(entry,request,architecture_sha256)
            if entry['frames']!=selection['expected_frames'] or entry['song_id']!=ref['song_id']:
                raise ValueError('whole chunk source/frame mismatch')
            verify_file(Path(entry['path']),entry['sha256']);values=load_file(entry['path'],device='cpu')
            features=values['prefix_features']
            if features.shape!=(entry['frames'],1024) or features.dtype!=torch.float32 or not torch.isfinite(features).all():
                raise ValueError('invalid whole prefix tensor')
            chunks.append({**entry,**selection,'prefix_features':features})
        songs.append({**ref,'chunks':chunks,'total_frames':group['total_frames'],
            'whole_cache_sha256':sha256_file(run/'WHOLE_CACHE.json')})
    if not songs:raise ValueError('no eligible whole sequences after held-song exclusion')
    return songs

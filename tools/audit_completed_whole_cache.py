"""Independent completed-cache validation and source-duration accounting on CPU."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from any2xsampa.training.whole_sequence_analysis import source_coverage
from any2xsampa.training.singing_domain import validate_cache_entry
from any2xsampa.training.whole_sequence_supervision import validate_chunks


def audit(run:Path)->None:
    import numpy as np
    from safetensors.numpy import load_file
    freeze=read_json(run/'FREEZE.json');parent=Path(freeze['parents']['singing'])
    for filename,key in [('POLICY.json','policy_sha256'),('WHOLE_REFERENCES.json','whole_references_sha256'),
                         ('WHOLE_GEOMETRY.json','whole_geometry_sha256'),('WHOLE_AUDIO_REQUESTS.json','whole_requests_sha256')]:
        verify_file(run/filename,freeze[key])
    index=read_json(run/'WHOLE_CACHE.json');refs=read_json(run/'WHOLE_REFERENCES.json')['songs']
    if index['reference_received'] or not index['all_parameters_frozen']:
        raise ValueError('whole cache received targets or used trained parameters')
    if index['audio_requests_sha256']!=freeze['whole_requests_sha256'] or index['geometry_sha256']!=freeze['whole_geometry_sha256']:
        raise ValueError('cache source-request or geometry hash changed')
    requests={r['example_id']:r for r in read_json(run/'WHOLE_AUDIO_REQUESTS.json')['rows']}
    if len(index['entries'])!=32 or {e['example_id'] for e in index['entries']}!=set(requests):
        raise ValueError('whole cache coverage changed')
    geometry={s['song_id']:s for s in read_json(run/'WHOLE_GEOMETRY.json')['songs']}
    selection={r['example_id']:r for g in geometry.values() for r in g['chunks']};checks=[]
    for song in refs:
        verify_file(Path(song['audio_path']),song['audio_sha256'])
        validate_chunks(geometry[song['song_id']]['chunks'],geometry[song['song_id']]['total_frames'])
    for entry in index['entries']:
        validate_cache_entry(entry,requests[entry['example_id']],index['architecture']['sha256'])
        owner=selection[entry['example_id']];verify_file(Path(entry['path']),entry['sha256'])
        arrays=load_file(entry['path'])
        for name,width in [('prefix_features',1024),('encoder_features',1024),('frame_logits',428)]:
            if arrays[name].shape!=(owner['expected_frames'],width) or arrays[name].dtype!=np.float32 or not np.isfinite(arrays[name]).all():
                raise ValueError('invalid empirical cache tensor')
        crop=entry['crop']
        if crop['source_start_sample']!=owner['source_start_sample'] or crop['source_end_sample_exclusive']!=owner['source_end_sample_exclusive']:
            raise ValueError('actual audio samples differ from frozen global plan')
        checks.append({'example_id':entry['example_id'],'song_id':entry['song_id'],'frames':entry['frames'],
            'retained_global_frames':owner['global_end_frame']-owner['global_start_frame'],'sha256':entry['sha256']})
    short=[r for r in read_json(parent/'AUDIO_REQUESTS.json')['rows'] if r['audio_role']=='original']
    whole=[{'song_id':r['song_id'],'audio_sha256':r['audio_sha256'],'duration_sec':geometry[r['song_id']]['source_duration_sec']} for r in refs]
    coverage=source_coverage(short,whole)
    output=run/'completion';output.mkdir(exist_ok=True)
    write_json_new(output/'CACHE_AUDIT.json',{'entries':checks,'cache_sha256':sha256_file(run/'WHOLE_CACHE.json'),
        'empirical_tensor_shapes_and_sample_ranges_verified':True,'all_valid_tensors_finite_float32':True,
        'retained_global_frames':sum(r['retained_global_frames'] for r in checks),
        'coverage':coverage,'cpu_only':True,'model_predictions_used_to_change_references':False})
    print('CPU verified32caches,globalframes',sum(r['retained_global_frames'] for r in checks),
          'uniqueoriginalseconds',coverage['unique_original_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run.resolve())

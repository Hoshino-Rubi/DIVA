"""Independent CPU intake/provenance checks; no model or final-test audio access."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file
from any2xsampa.training.whole_sequence_review import validate_answers
from any2xsampa.adapters.audio_provenance import describe_audio_file


def audit(run:Path,answers:Path)->None:
    intake=read_json(run/'review-intake/REVIEW_INTAKE.json')
    verify_file(answers,intake['answers_sha256'])
    packet=Path(intake['packet_path']);verify_file(packet,intake['packet_sha256'])
    decisions=validate_answers(read_json(packet),read_json(answers),packet_sha256=intake['packet_sha256'])
    if decisions['accepted']!=intake['accepted']:raise ValueError('independent intake differs')
    preserved=read_json(run/'PRESERVATION_BEFORE.json')['files']
    for path,digest in preserved.items():verify_file(Path(path),digest)
    freeze=read_json(run/'FREEZE.json')
    for filename,key in [('POLICY.json','policy_sha256'),('WHOLE_REFERENCES.json','whole_references_sha256'),
                         ('WHOLE_GEOMETRY.json','whole_geometry_sha256'),('WHOLE_AUDIO_REQUESTS.json','whole_requests_sha256')]:
        verify_file(run/filename,freeze[key])
    refs=read_json(run/'WHOLE_REFERENCES.json')['songs']
    geom={s['song_id']:s for s in read_json(run/'WHOLE_GEOMETRY.json')['songs']}
    requests={r['example_id']:r for r in read_json(run/'WHOLE_AUDIO_REQUESTS.json')['rows']}
    checks=[]
    for ref in refs:
        info=describe_audio_file(Path(ref['audio_path']))
        if info['sha256']!=ref['audio_sha256']:raise ValueError('whole original WAV changed')
        group=geom[ref['song_id']];indices=[];end_sample=0;request_samples=[]
        for row in group['chunks']:
            r=requests[row['example_id']]
            if r['song_id']!=ref['song_id'] or r['audio_role']!='original' or r['split']!='train':
                raise ValueError('whole audio-role or split changed')
            if set(r)!={'example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec'}:
                raise ValueError('audio worker request contains extra reference conditions')
            first,last=round(r['start_sec']*info['sample_rate']),round(r['end_sec']*info['sample_rate'])
            if first!=row['source_start_sample'] or last!=row['source_end_sample_exclusive'] or first>end_sample:
                raise ValueError('audio crop samples changed or have gaps')
            request_samples.append((first,last));end_sample=max(end_sample,last)
            offset=round(r['start_sec']*50)
            indices.extend(range(offset+row['keep_start_frame'],offset+row['keep_end_frame']))
        expected=((info['frame_count']*16000+info['sample_rate']-1)//info['sample_rate']-400)//320+1
        if indices!=list(range(expected)) or request_samples[0][0]!=0 or end_sample!=info['frame_count']:
            raise ValueError('whole model frame/audio coverage is incomplete or duplicated')
        if group['total_frames']!=expected or ref['minimum_ctc_frames']>expected:
            raise ValueError('whole CTC feasibility differs')
        if ref['phone_timing_confirmed'] or ref['mapping_extended']:raise ValueError('review scope was expanded')
        checks.append({'song_id':ref['song_id'],'source_seconds':info['duration_sec'],'note_count':ref['note_count'],
            'target_phones':len(ref['target_ids']),'global_frames':expected,'chunks':len(group['chunks']),
            'unique_source_samples':info['frame_count'],'all_frames_owned_once':True,'all_audio_covered':True})
    result={'accepted':len(decisions['accepted']),'pending':len(decisions['pending']),'conflicts':len(decisions['conflicts']),
        'checks':checks,'original_answer_path':str(answers.resolve()),'original_answer_sha256':sha256_file(answers),
        'packet_sha256':intake['packet_sha256'],'preserved_files_checked':len(preserved),'all_preserved_files_unchanged':True,
        'models_or_dependencies_added':False,'test_predictions_opened':False,'inference_quality_not_measured':not (run/'ROUTE_SELECTION.json').exists()}
    write_json_new(run/'INTAKE_VERIFICATION.json',result)
    print('VERIFIED',len(checks),'whole sequences;',len(preserved),'preserved bindings unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--answers',type=Path,required=True)
    a=p.parse_args();audit(a.run.resolve(),a.answers.resolve())

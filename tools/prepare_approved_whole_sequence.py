"""Freeze the two reviewed whole-song sequences without using automatic clock paths."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file
from any2xsampa.training.whole_sequence_review import validate_answers
from any2xsampa.training.whole_sequence_supervision import accepted_reference, plan_chunks
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
from any2xsampa.training.limited_data_reference import load_vocabulary
from any2xsampa.adapters.audio_provenance import describe_audio_file
from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline


def prepare(run: Path, parent: Path) -> None:
    intake=read_json(run/'review-intake/REVIEW_INTAKE.json')
    packet_path=Path(intake['packet_path']); verify_file(packet_path,intake['packet_sha256'])
    raw_path=run/'review-intake/raw-answers.json'; verify_file(raw_path,intake['answers_sha256'])
    reviewed=validate_answers(read_json(packet_path),read_json(raw_path),packet_sha256=intake['packet_sha256'])
    if reviewed['accepted']!=intake['accepted'] or not reviewed['accepted']:
        raise ValueError('review intake changed or has no accepted whole sequences')
    previous=read_json(parent/'FREEZE.json'); verify_file(parent/'POLICY.json',previous['policy_sha256'])
    singing=Path(previous['parents']['singing'])
    policy=deepcopy(read_json(parent/'POLICY.json'))
    policy['schema_version']='whole_sequence_approved_policy_v1'
    policy['whole_sequence'].update(reference_nature='human_confirmed_whole_sequence_with_derived_engine_IPA',
        reference_gate='explicit whole order/repeat approval plus complete mapping; no inferred phone timing',
        forward_strategy='two-pass logit-gradient chain rule, one activation graph at a time',
        whole_branch_dropout=False,
        placement='one song at each evenly spaced optimizer batch; no added optimizer steps',
        remaining_screen_jobs=5, previous_screen_jobs=10,
        quality_initialization='original matching full-fold parent; never prior control or risk continuation')
    write_json_new(run/'POLICY.json',policy)
    preserved={}
    for source_run in (parent,singing):
        delivery=read_json(source_run/'DELIVERY.json')
        for row in delivery['source_files']+delivery['root_artifacts']:
            verify_file(Path(row['path']),row['sha256']);preserved[row['path']]=row['sha256']
    manifest_bind=reviewed['accepted'][0]['source_bindings']['reference_manifest']
    manifest=Path(manifest_bind['path']);verify_file(manifest,manifest_bind['sha256'])
    raw=read_json(manifest)
    for source in raw['bindings'].values():
        verify_file(Path(source['path']),source['sha256']);preserved[source['path']]=source['sha256']
    evidence=ROOT/'.state/runs/original-reference-eval-20260906T154652/manual-evidence/source.json'
    view,overlay=apply_original_mapping_overlay(raw,evidence)
    vocabulary=load_vocabulary(Path(raw['bindings']['vocabulary']['path']))
    songs={s['song_id']:s for s in view['songs']}; references=[];geometry=[];requests=[]
    for approval in reviewed['accepted']:
        for binding in approval['source_bindings'].values():
            verify_file(Path(binding['path']),binding['sha256']);preserved[binding['path']]=binding['sha256']
        song=songs[approval['song_id']]
        if song['song_id'] not in policy['songs']:
            raise ValueError('nontraining whole source approval')
        for path_key,hash_key in [('path','sha256'),('normalized_path','normalized_sha256')]:
            path=Path(song['human_vsqx'][path_key]);digest=song['human_vsqx'][hash_key]
            verify_file(path,digest);preserved[str(path)]=digest
        timeline=parse_vsqx_timeline(Path(song['human_vsqx']['normalized_path']))
        if [n['phonemes'] for n in timeline['notes']] != [n['raw_v4c_tokens'] for n in song['notes']]:
            raise ValueError('source VSQX sequence differs from the approved manifest')
        reference=accepted_reference(song,approval,vocabulary)
        audio=describe_audio_file(Path(reference['audio_path']))
        if audio['sha256']!=reference['audio_sha256']:raise ValueError('whole audio hash changed')
        plan=plan_chunks(song['song_id'],audio,policy['whole_sequence'])
        if reference['minimum_ctc_frames']>plan['total_frames']:
            raise ValueError('whole source has insufficient CTC frames including repeated labels')
        references.append(reference);geometry.append({'song_id':song['song_id'],**plan})
        for chunk in plan['chunks']:
            requests.append({'example_id':chunk['example_id'],'song_id':song['song_id'],'split':'train',
                'audio_role':'original','audio_path':reference['audio_path'],'audio_sha256':reference['audio_sha256'],
                'start_sec':chunk['start_sec'],'end_sec':chunk['end_sec']})
    if len({r['song_id'] for r in references})!=len(references):raise ValueError('duplicate whole source')
    write_json_new(run/'WHOLE_REFERENCES.json',{'songs':references,'overlay':overlay,
        'review_intake_sha256':sha256_file(run/'review-intake/REVIEW_INTAKE.json'),
        'source_manifest_sha256':manifest_bind['sha256'], 'old_reviewed_windows_modified':False,
        'labels_received_by_acoustic_worker':False,'phone_timing_confirmed':False})
    write_json_new(run/'WHOLE_GEOMETRY.json',{'songs':geometry,'prediction_free_sampling':True})
    write_json_new(run/'WHOLE_AUDIO_REQUESTS.json',{'rows':requests,'reference_received':False})
    audit=read_json(parent/'WHOLE_SEQUENCE_AUDIT.json')
    write_json_new(run/'REFERENCE_ACCEPTANCE.json',{'accepted_songs':[r['song_id'] for r in references],
        'target_phones':sum(len(r['target_ids']) for r in references),
        'source_seconds':sum(g['source_duration_sec'] for g in geometry),
        'excluded_songs':[{'song_id':s['song_id'],'unresolved_tokens':s['unresolved_tokens'],
                          'reasons':['no_new_whole_sequence_approval','unresolved_mapping']}
                         for s in audit['songs'] if s['song_id'] not in {r['song_id'] for r in references}],
        'automatic_clock_audit_remains_failed':True,
        'sequence_confirmation_does_not_certify_clock':True,
        'old_short_window_references_unchanged':True})
    for p in (manifest,evidence,Path(overlay['path']),packet_path,raw_path):preserved[str(p.resolve())]=sha256_file(p)
    write_json_new(run/'PRESERVATION_BEFORE.json',{'files':preserved})
    original_freeze=read_json(singing/'FREEZE.json')
    write_json_new(run/'FREEZE.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'parents':{'singing':str(singing),'routes':str(parent)},
        'training_python':previous['training_python'],'model_snapshot':original_freeze['model_snapshot'],
        'policy_sha256':sha256_file(run/'POLICY.json'),
        'whole_references_sha256':sha256_file(run/'WHOLE_REFERENCES.json'),
        'whole_geometry_sha256':sha256_file(run/'WHOLE_GEOMETRY.json'),
        'whole_requests_sha256':sha256_file(run/'WHOLE_AUDIO_REQUESTS.json'),
        'review_intake_sha256':sha256_file(run/'review-intake/REVIEW_INTAKE.json'),
        'default_replacement':False,'final_test_predictions_allowed':False})
    print('prepared',len(references),'songs',len(requests),'chunks',sum(len(r['target_ids']) for r in references),'phones')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();prepare(a.run.resolve(),a.parent.resolve())

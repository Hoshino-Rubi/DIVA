"""Freeze a reproducible, hash-bound final proxy evaluation before opening logits."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.final_test_mapping_review import verify_review_packet,assemble_proxy_reference
from tools.final_test_reference_review import bind
from any2xsampa.training.final_proxy_evaluation import (validate_reference,validate_requests,verify_release,
    AUDIO_KEYS,REQUEST_SCHEMA,GATE_SCHEMA,SYSTEMS,SCOPE)
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,sha256_object,verify_file
from any2xsampa.training.limited_data_reference import load_mapping,load_vocabulary,SPLITS
from any2xsampa.training.original_reference_mapping import documented_compound_mapping,OVERLAY_PATH
from any2xsampa.training.phoneticxeus_adaptation import required_ctc_frames

INDEX=ROOT/'.state/runs/limited-data-20260906T050251/original-baselines/index.json'
INDEX_SHA='1240f27cbf8aa2b62e6d3f8da21a95fbc1d11921a0ba25099f1e97fdf228a905'
LEGACY=ROOT/'.state/runs/guarded-retention-20260919T041500/FINAL_EVALUATION_POLICY.json'


def prepare(reference_path:Path,output:Path)->dict:
    inputs=read_json(reference_path.parent/'INPUTS.json')
    for spec in inputs.values():
        if isinstance(spec,dict) and {'path','sha256'}<=set(spec):verify_file(Path(spec['path']),spec['sha256'])
    review_path=Path(inputs['packet']['path']);review=verify_review_packet(review_path)
    answers_path=reference_path.parent/'raw-mapping-decisions.json'
    verify_file(answers_path,inputs['answers']['sha256'])
    decisions=read_json(answers_path)
    if decisions.get('automated_regression_fixture') or decisions.get('reviewer','').startswith('AUTOMATED_'):
        raise ValueError('automated fixture cannot release final test')
    b=review['source_bindings'];source=read_json(Path(b['original_packet']['path']));original=read_json(Path(b['original_answers']['path']))
    mapping=documented_compound_mapping(load_mapping(Path(b['base_mapping']['path'])),read_json(OVERLAY_PATH))
    vocabulary=load_vocabulary(Path(b['vocabulary']['path']))
    reference=read_json(reference_path)
    if reference!=assemble_proxy_reference(review,decisions,source,original,mapping,vocabulary):
        raise ValueError('proxy reference does not replay from actual human answers')
    if decisions.get('authorize_frozen_proxy_comparison') is not True:raise ValueError('proxy comparison not authorized')
    verify_file(INDEX,INDEX_SHA);index=read_json(INDEX)
    ready=validate_reference(reference,index,vocabulary)
    if not ready:raise ValueError('no complete approved proxy windows')
    candidate_path=Path(reference['candidate_binding']['path'])
    verify_file(candidate_path,reference['candidate_binding']['sha256']);candidate=read_json(candidate_path)
    legacy=read_json(LEGACY)
    if legacy['candidate']!=reference['candidate_binding'] or legacy['systems']!=SYSTEMS:
        raise ValueError('frozen final candidate or comparison changed')
    if candidate.get('test_opening') is not False or candidate.get('default_replacement') is not False:
        raise ValueError('candidate not pre-test frozen')
    bindings=dict(reference=bind(reference_path),mapping_decisions=bind(answers_path),mapping_review=bind(review_path),
        intake_inputs=bind(reference_path.parent/'INPUTS.json'),original_packet=b['original_packet'],original_answers=b['original_answers'],
        candidate=reference['candidate_binding'],legacy_policy=bind(LEGACY),source_index=bind(INDEX),vocabulary=b['vocabulary'],
        pretrained_head=index['head'],teacher_checkpoint={k:candidate['teacher_checkpoint'][k] for k in ('path','sha256')},
        student_checkpoint={k:candidate['student_checkpoint'][k] for k in ('path','sha256')},guard_policy=candidate['policy'],
        worker_source=bind(ROOT/'tools/run_final_proxy_evaluation.py'),gate_source=bind(ROOT/'src/any2xsampa/training/final_proxy_evaluation.py'),
        scorer_source=bind(ROOT/'tools/score_final_proxy_evaluation.py'),metrics_source=bind(ROOT/'src/any2xsampa/training/phone_error_metrics.py'),
        guard_source=bind(ROOT/'src/any2xsampa/adapters/guarded_phoneticxeus.py'),guard_rule_source=bind(ROOT/'src/any2xsampa/adapters/teacher_guard.py'))
    for k in ('base_mapping','mapping_overlay','proposals','manual_evidence'):
        bindings['reference_'+k]=b[k]
    source_manifest=read_json(Path(source['source_bindings']['reference']['path']))
    bindings['source_reference_manifest']=source['source_bindings']['reference']
    for song in source['songs']:
        bindings['source_audio_'+song['song_id']]=song['original_audio']
        bindings['source_vsqx_'+song['song_id']]=song['source_vsqx']
    for spec in bindings.values():verify_file(Path(spec['path']),spec['sha256'])
    # Check seals/hashes as bytes only; no posterior JSON or tensor loading here.
    entries={e['example_id']:e for e in index['entries']}
    requests=dict(schema_version=REQUEST_SCHEMA,split_sha256=sha256_object(SPLITS),contains_labels_or_score_context=False,
        rows=[dict(audio={k:r[k] for k in sorted(AUDIO_KEYS)},cache=deepcopy(entries[r['example_id']])) for r in ready])
    validate_requests(requests,index)
    for row in requests['rows']:
        for kind in ('evidence','posterior'):verify_file(Path(row['cache'][kind+'_path']),row['cache'][kind+'_sha256'])
    coverage=dict(total_windows=6,included_windows=len(ready),total_audio_seconds=108.,
        included_audio_seconds=sum(r['end_sec']-r['start_sec'] for r in ready),
        source_phone_count=sum(r['source_phone_count'] for r in ready),reference_phone_count=sum(len(r['phones']) for r in ready),
        absent_phones=sum(len(r['omitted_phones']) for r in ready),partial_phones=sum(r['partial_phone_count'] for r in ready),
        excluded=[dict(window_id=r['window_id'],reasons=r['blocking_reasons']) for r in reference['examples'] if not r['proxy_evaluation_eligible']],
        ctc_checks=[dict(window_id=r['window_id'],available_frames=entries[r['example_id']]['frames'],
            minimum_frames=required_ctc_frames([vocabulary[p['phone']] for p in r['phones']])) for r in ready])
    output.mkdir(parents=True,exist_ok=True)
    write_json_new(output/'AUDIO_FEATURE_REQUESTS.json',requests)
    write_json_new(output/'COVERAGE.json',coverage)
    policy=dict(schema_version='frozen_final_proxy_evaluation_policy_v1',systems=SYSTEMS,scope=SCOPE,
        reference=bindings['reference'],candidate=bindings['candidate'],decisions=bindings['mapping_decisions'],
        source_index=bindings['source_index'],scope_authorized_by_user=True,
        prior_policy=bindings['legacy_policy'],coverage=coverage,
        evaluation_only=True,original_ipa_accuracy_claim_allowed=False,training_allowed=False,
        model_reselection_allowed=False,mapping_reselection_after_scores_allowed=False,
        ctc_timestamps_as_phone_durations=False,default_replacement=False,
        candidate_not_promoted_by_final_scores=True,
        guard_groups_min_count=20,descriptive_regression_threshold=0.03,
        protocol_fixed_utc=datetime.now(timezone.utc).isoformat())
    write_json_new(output/'POLICY.json',policy)
    bindings['requests']=bind(output/'AUDIO_FEATURE_REQUESTS.json');bindings['policy']=bind(output/'POLICY.json')
    gate=dict(schema_version=GATE_SCHEMA,bindings=bindings,systems=SYSTEMS,scope=SCOPE,
        included_example_ids=[r['example_id'] for r in ready],coverage=coverage,
        execution_authorized=True,training_allowed=False,model_reselection_allowed=False,default_replacement=False,
        formal_original_ipa_accuracy_allowed=False,reference_frozen_before_predictions=True,
        test_predictions_opened_at_freeze=False,created_utc=datetime.now(timezone.utc).isoformat())
    gate['gate_hash']=sha256_object(gate);write_json_new(output/'RELEASE.json',gate)
    verify_release(output/'RELEASE.json',sha256_file(output/'RELEASE.json'))
    return gate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(ROOT/'.state/runs') or (out/'RELEASE.json').exists():raise ValueError('new evaluation release required')
    gate=prepare(a.reference.resolve(),out);print('frozen',len(gate['included_example_ids']),'windows',gate['coverage']['reference_phone_count'],'phones')
    print('release_sha256',sha256_file(out/'RELEASE.json'))

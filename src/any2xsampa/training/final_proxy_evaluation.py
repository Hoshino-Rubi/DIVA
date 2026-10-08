"""Read-only final evaluation release for an explicitly approved proxy reference."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256
from any2xsampa.training.limited_data_artifacts import (finite_number, read_json, require_digest,
    sha256_object, verify_file)
from any2xsampa.training.limited_data_reference import SONG_SPLIT, SPLITS
from any2xsampa.training.phoneticxeus_adaptation import required_ctc_frames
from any2xsampa.training.phone_error_metrics import phone_metrics

Json = dict[str, Any]
SYSTEMS = ['pretrained', 'old_seed43', 'frozen_guarded_candidate']
SCOPE = 'engine_proxy_diagnostic_not_original_ipa_accuracy'
REQUEST_SCHEMA = 'released_final_test_feature_requests_v1'
GATE_SCHEMA = 'approved_final_proxy_evaluation_release_v1'
AUDIO_KEYS = {'example_id','song_id','split','audio_role','audio_path','audio_sha256','start_sec','end_sec'}
CACHE_KEYS = {'example_id','song_id','split','audio_role','audio_sha256','start_sec','end_sec',
              'evidence_path','evidence_sha256','posterior_path','posterior_sha256','frames','feature_width',
              'vocabulary_sha256','model_snapshot_lock_sha256','test_outputs_sealed','created_at_utc'}


def validate_reference(reference: Json, index: Json, vocabulary: dict[str, int]) -> list[Json]:
    if (reference.get('schema_version') != 'approved_final_test_proxy_reference_v1' or
        reference.get('manifest_hash') != sha256_object({k:v for k,v in reference.items() if k!='manifest_hash'})):
        raise ValueError('proxy reference hash/schema mismatch')
    if (reference.get('comparison_scope') != SCOPE or reference.get('formal_original_ipa_accuracy_allowed') is not False or
        reference.get('training_allowed') is not False or reference.get('test_predictions_opened') is not False):
        raise ValueError('reference must be blind, proxy-only and excluded from training')
    rows=reference['examples']
    if len(rows)!=6 or {r['window_id'] for r in rows}!={f'T{i:02}' for i in range(1,7)}:
        raise ValueError('all six original review outcomes required')
    if len({r['example_id'] for r in rows})!=6:
        raise ValueError('duplicate final reference')
    if index.get('purpose')!='original-baselines' or index.get('split_sha256')!=sha256_object(SPLITS):
        raise ValueError('wrong source feature index')
    metadata=index['model_metadata']
    if (metadata.get('model_id'),metadata.get('revision'),metadata.get('weights_sha256'))!=(MODEL_ID,REVISION,WEIGHT_SHA256):
        raise ValueError('unapproved feature model')
    entries={e['example_id']:e for e in index['entries']}
    if len(entries)!=len(index['entries']):
        raise ValueError('duplicate cached examples')
    ready=[]
    for row in rows:
        if SONG_SPLIT.get(row['song_id'])!='test' or row['split']!='test' or row['audio_role']!='original' or row['training_eligible'] is not False:
            raise ValueError('final reference role/split mismatch')
        entry=entries.get(row['example_id'])
        if entry is None or entry.get('test_outputs_sealed') is not True:
            raise ValueError('source must be the frozen sealed cache')
        for k in AUDIO_KEYS-{'audio_path'}:
            if row[k]!=entry[k]:raise ValueError(f'cache/reference mismatch: {k}')
        start=finite_number(row['start_sec'],'start',minimum=0)
        end=finite_number(row['end_sec'],'end',minimum=0)
        if abs(end-start-18)>1e-6:raise ValueError('fixed eighteen-second window changed')
        if entry['feature_width']!=1024 or entry['frames']<1:
            raise ValueError('invalid cached geometry')
        if entry['vocabulary_sha256']!=metadata['vocabulary_sha256'] or entry['model_snapshot_lock_sha256']!=metadata['snapshot_lock_sha256']:
            raise ValueError('stale cache vocabulary/model')
        if row['status']=='proxy_reference_ready' and row.get('proxy_evaluation_eligible') is True:
            phones=row['phones']
            if not phones or any(p['phone'] not in vocabulary or p['phone'].startswith('<') or p['audibility'] not in {'full','partial'} for p in phones):
                raise ValueError('invalid reference phones')
            if row.get('reference_phone_count')!=len(phones) or row.get('original_ipa_ground_truth') is not False:
                raise ValueError('invalid proxy denominator/scope')
            if any(p['audibility']!='absent' for p in row['omitted_phones']):raise ValueError('unsupported reference omission')
            kept=[p for p in row['source_phones'] if p['audibility'] in {'full','partial'}]
            omitted=[p for p in row['source_phones'] if p['audibility']=='absent']
            if kept!=phones or omitted!=row['omitted_phones'] or len(kept)+len(omitted)!=len(row['source_phones']):
                raise ValueError('reference source phones were selectively removed')
            ids=[vocabulary[p['phone']] for p in phones]
            if required_ctc_frames(ids)>entry['frames']:raise ValueError('intact reference exceeds CTC frame capacity')
            ready.append(row)
        elif row['status']!='pending' or row.get('proxy_evaluation_eligible') is not False or row['phones']:
            raise ValueError('invalid pending reference')
    if reference['coverage']['proxy_ready_windows']!=len(ready) or reference['coverage']['reference_phones']!=sum(len(r['phones']) for r in ready):
        raise ValueError('proxy coverage mismatch')
    return ready


def validate_requests(requests: Json, index: Json) -> None:
    if set(requests)!={'schema_version','split_sha256','contains_labels_or_score_context','rows'} or requests['schema_version']!=REQUEST_SCHEMA:
        raise ValueError('invalid audio-only final request fields')
    if requests['contains_labels_or_score_context'] is not False or requests['split_sha256']!=sha256_object(SPLITS):
        raise ValueError('reference context or split is invalid')
    rows=requests['rows']
    if not rows or len(rows)>6 or len({r['audio']['example_id'] for r in rows})!=len(rows):
        raise ValueError('invalid final request count')
    entries={e['example_id']:e for e in index['entries']}
    for row in rows:
        if set(row)!={'audio','cache'} or set(row['audio'])!=AUDIO_KEYS or set(row['cache'])!=CACHE_KEYS:
            raise ValueError('worker cannot accept labels, lyrics, score or side context')
        a,e=row['audio'],row['cache']
        if entries.get(a['example_id'])!=e:raise ValueError('cache request differs from frozen index')
        if a['split']!='test' or SONG_SPLIT.get(a['song_id'])!='test' or a['audio_role']!='original' or e['test_outputs_sealed'] is not True:
            raise ValueError('release limited to sealed original test inputs')
        if any(a[k]!=e[k] for k in AUDIO_KEYS-{'audio_path'}):raise ValueError('audio/cache mismatch')
        require_digest(a['audio_sha256'],'audio hash')
        if abs(finite_number(a['end_sec'],'end')-finite_number(a['start_sec'],'start',minimum=0)-18)>1e-6:
            raise ValueError('fixed crop changed')


def verify_release(path: Path, expected_sha256: str) -> tuple[Json, Json, Json]:
    verify_file(path,expected_sha256)
    gate=read_json(path)
    if gate.get('schema_version')!=GATE_SCHEMA or gate.get('gate_hash')!=sha256_object({k:v for k,v in gate.items() if k!='gate_hash'}):
        raise ValueError('release hash/schema mismatch')
    if gate.get('systems')!=SYSTEMS or gate.get('scope')!=SCOPE or gate.get('execution_authorized') is not True:
        raise ValueError('wrong evaluation scope or authorization')
    for k in ('training_allowed','model_reselection_allowed','default_replacement','formal_original_ipa_accuracy_allowed'):
        if gate.get(k) is not False:raise ValueError('final gate permits an unauthorized action')
    if gate.get('reference_frozen_before_predictions') is not True or gate.get('test_predictions_opened_at_freeze') is not False:
        raise ValueError('release was not frozen while prediction blind')
    required={'reference','mapping_decisions','mapping_review','intake_inputs','original_packet','original_answers',
              'candidate','policy','legacy_policy','source_index','requests','vocabulary','pretrained_head',
              'teacher_checkpoint','student_checkpoint','guard_policy','worker_source'}
    if not required<=set(gate['bindings']):raise ValueError('incomplete release lineage')
    for binding in gate['bindings'].values():verify_file(Path(binding['path']),binding['sha256'])
    b=gate['bindings'];decisions=read_json(Path(b['mapping_decisions']['path']))
    if decisions.get('automated_regression_fixture') or decisions.get('reviewer','').startswith('AUTOMATED_'):
        raise ValueError('automated fixture is not a real human approval')
    if decisions.get('authorize_frozen_proxy_comparison') is not True or decisions.get('consent')!=SCOPE:
        raise ValueError('human proxy scope authorization missing')
    candidate=read_json(Path(b['candidate']['path']))
    for k in ('teacher_checkpoint','student_checkpoint'):
        if any(candidate[k][key]!=b[k][key] for key in ('path','sha256')):raise ValueError('candidate checkpoint changed')
    if candidate['policy']!=b['guard_policy']:raise ValueError('candidate guard changed')
    index=read_json(Path(b['source_index']['path']));requests=read_json(Path(b['requests']['path']))
    if index['head']!=b['pretrained_head']:raise ValueError('head binding changed')
    validate_requests(requests,index)
    if [r['audio']['example_id'] for r in requests['rows']]!=gate['included_example_ids']:
        raise ValueError('released set changed')
    return gate,requests,index


def score_predictions(reference: Json, predictions: Json) -> Json:
    rows=[r for r in reference['examples'] if r['proxy_evaluation_eligible']]
    expected={r['example_id'] for r in rows}
    if set(predictions)!=set(SYSTEMS) or any(set(predictions[s])!=expected for s in SYSTEMS):
        raise ValueError('all frozen systems must cover the identical released subset')
    systems={}
    for system in SYSTEMS:
        pairs=[];by_song:dict[str,list]= {};details=[]
        for row in rows:
            target=[p['phone'] for p in row['phones']];actual=predictions[system][row['example_id']]
            if not isinstance(actual,list) or any(not isinstance(p,str) or not p for p in actual):raise ValueError('invalid prediction')
            pairs.append((target,actual));by_song.setdefault(row['song_id'],[]).append((target,actual))
            details.append(dict(window_id=row['window_id'],example_id=row['example_id'],song_id=row['song_id'],
                                reference=target,prediction=actual,metrics=phone_metrics([(target,actual)])))
        systems[system]=dict(overall=phone_metrics(pairs),by_song={k:phone_metrics(v) for k,v in by_song.items()},examples=details)
    return dict(schema_version='frozen_final_proxy_comparison_v1',systems=systems,scope=SCOPE,
                coverage=reference['coverage'],reference_manifest_hash=reference['manifest_hash'],
                selected_after_test=False,default_replaced=False,training_performed=False,
                formal_original_ipa_accuracy_claim_allowed=False,ctc_times_are_phone_durations=False)

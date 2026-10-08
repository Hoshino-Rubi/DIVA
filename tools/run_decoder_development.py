"""Materialize fixed heads on CUDA, then evaluate two frozen acoustic decoders."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, write_json_new, sha256_file
from any2xsampa.training.decoder_experiment import validate_policy, validate_cohort, promotion_gate, reference_edits, sequence_digest
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def materialize(run: Path, policy: dict) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline, configure_cuda, gpu_total_used_bytes
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    enforce_offline(run/'head-runtime')
    if gpu_total_used_bytes() > 14 * 1024**3:
        raise RuntimeError('card exceeds budget before model initialization')
    from any2xsampa.training.retention_adaptation import load_inputs
    from any2xsampa.adapters.guarded_phoneticxeus import GuardedPhoneHead
    from safetensors.numpy import save_file
    start = time.monotonic()
    with ProcessResourceMonitor(run/'head-monitor', timeout_seconds=900):
        torch = configure_cuda()
        manifest = policy['bindings']['manifest']
        data = load_inputs(Path(policy['source_run']), manifest['sha256'], torch)
        spec = policy['bindings']['candidate']
        engine = GuardedPhoneHead(torch, data['head'], data['vocabulary'], Path(spec['path']), spec['sha256'])
        refs, records = [], []
        cohorts = [('original_train', data['train']), ('original_dev', data['dev']), ('rendered_train', data['replay']), ('rendered_dev', data['protection'])]
        for cohort, rows in cohorts:
            for row in rows:
                if row['split'] not in {'train', 'dev'}:
                    raise ValueError('test rows forbidden before prediction')
                result = engine.predict(row['features'])
                with torch.inference_mode():
                    baseline = data['head'](row['features'].to('cuda')).cpu().numpy()
                ident = cohort + '__' + row['example_id'].lower()
                dest = run/'head-cache'/f'{ident}.safetensors'
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise FileExistsError(dest)
                save_file({'pretrained': baseline, 'old_seed43': result['teacher_logits'], 'student': result['student_logits'],
                           'candidate': result['frame_logits'], 'guard_mask': result['guard_mask']}, str(dest))
                record = {k: row[k] for k in ('example_id', 'song_id', 'split', 'audio_role')}
                record.update(cohort=cohort, cache_id=ident, evidence_path=str(dest), evidence_sha256=sha256_file(dest), frames=len(baseline))
                records.append(record)
                targets = row.get('model_target_ipa')
                if targets is None:
                    targets = [data['labels'][i] for i in data['rendered_targets'][row['example_id']]['model_target_ids']]
                refs.append({'cache_id': ident, 'phones': targets})
        validate_cohort(records)
        # Structural freeze and exact parameter state checks against source tensors.
        from safetensors.torch import load_file
        for name, module in [('teacher_checkpoint', engine.teacher), ('student_checkpoint', engine.student)]:
            source = load_file(engine.candidate[name]['path'])
            if any(not torch.equal(value.cpu(), source[key]) for key, value in module.state_dict().items()):
                raise ValueError('frozen adapter changed')
        head_spec = data['original_index']['head']; source_head = load_file(head_spec['path'])
        if any(not torch.equal(value.cpu(), source_head[key]) for key, value in data['head'].state_dict().items()):
            raise ValueError('frozen head changed')
        write_json_new(run/'HEAD_CACHE.json', {'schema_version': 'decoder_frozen_heads_v1', 'policy_sha256': sha256_file(run/'POLICY.json'),
            'entries': records, 'sequence_sha256': sequence_digest(records), 'elapsed_seconds': time.monotonic()-start,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'frozen_parameters_equal': True, 'backbone_rerun': False})
        write_json_new(run/'SCORING_REFERENCES.json', {'schema_version': 'decoder_scoring_references_v1', 'rows': refs,
            'source_reference_sha256': policy['bindings']['segment_reference']['sha256'], 'not_passed_to_inference': True})
    print('HEAD_CACHE complete:', len(records), flush=True)


def evaluate(run: Path, policy: dict) -> None:
    import numpy as np
    from safetensors.numpy import load_file
    from any2xsampa.adapters.ctc_prefix import prefix_beam_decode
    cache = read_json(run/'HEAD_CACHE.json')
    if cache['policy_sha256'] != sha256_file(run/'POLICY.json') or sequence_digest(cache['entries']) != cache['sequence_sha256']:
        raise ValueError('cache or policy changed')
    validate_cohort(cache['entries'])
    references = {r['cache_id']:r['phones'] for r in read_json(run/'SCORING_REFERENCES.json')['rows']}
    vocabulary = read_json(Path(policy['bindings']['vocabulary']['path']))
    labels = {v:k for k,v in vocabulary.items()}
    if len(labels) != 428 or vocabulary['<blank>'] != 0:
        raise ValueError('vocabulary differs')
    original_refs = read_json(Path(policy['bindings']['segment_reference']['path']))
    source_refs = {r['example_id'].lower():r for r in original_refs['examples']}
    rows, diagnoses = [], []
    start = time.monotonic()
    for number, row in enumerate(cache['entries'], 1):
        verify_file(Path(row['evidence_path']), row['evidence_sha256'])
        raw = load_file(row['evidence_path']); predictions = {}; timings = {}
        for system in ('pretrained', 'old_seed43', 'student', 'candidate'):
            values = raw[system]
            if values.dtype != np.float32 or values.shape != (row['frames'], 428) or not np.isfinite(values).all():
                raise ValueError('invalid cached full-vocabulary logits')
            predictions[system] = [labels[i] for i in ctc_collapse(values.argmax(-1).tolist())]
        # No need to tune a decoder on rendered training examples.
        if row['cohort'] != 'rendered_train':
            for decoder in policy['decoders'][1:]:
                before = time.perf_counter()
                result = prefix_beam_decode(raw['candidate'], beam_width=decoder['beam_width'])
                predictions[decoder['id']] = [labels[i] for i in result.ids]
                timings[decoder['id']] = time.perf_counter() - before
        target = references[row['cache_id']]
        scores = {name: phone_metrics([(target, phones)]) for name, phones in predictions.items()}
        saved = {**row, 'reference': target, 'predictions': predictions, 'metrics': scores, 'decode_seconds': timings,
                 'argmax_blank_fraction': {name: float((raw[name].argmax(-1)==0).mean()) for name in ('pretrained','old_seed43','student','candidate')},
                 'guard_frames': int(raw['guard_mask'].sum()), 'special_emissions': {k:[p for p in v if p.startswith('<')] for k,v in predictions.items()}}
        if row['audio_role'] == 'original':
            ref = source_refs[row['example_id'].lower()]
            audible = [p for p in ref['source_phones'] if p['audibility'] in {'full', 'partial'}]
            if [p['phone'] for p in audible] != target:
                raise ValueError('phone-to-note attribution differs from fixed reference')
            edits = {name: reference_edits(target, pred) for name,pred in predictions.items()}
            saved.update(audio_path=ref['audio_path'], audio_sha256=ref['audio_sha256'], start_sec=ref['start_sec'], end_sec=ref['end_sec'],
                         source_phones=audible, edit_operations=edits)
            by_ref = {name:{e['reference_index']:e for e in ops if e['reference_index'] is not None} for name,ops in edits.items()}
            for i, phone in enumerate(audible):
                original = by_ref['old_seed43'][i]; current = by_ref['candidate'][i]
                if current['operation'] == 'match':
                    continue
                diagnoses.append({'cache_id': row['cache_id'], 'song_id': row['song_id'], 'split': row['split'], 'reference_index': i,
                    **phone, 'candidate_operation': current['operation'], 'candidate_prediction': current['prediction'],
                    'teacher_operation': original['operation'], 'teacher_prediction': original['prediction'],
                    'pretrained_operation': by_ref['pretrained'][i]['operation'],
                    'regression_from_teacher_match': original['operation']=='match',
                    'all_systems_disagree_with_reference': all(by_ref[n][i]['operation']!='match' for n in ('pretrained','old_seed43','student','candidate')),
                    'acoustic_timestamp_confirmed': False})
        dest = run/'examples'/f"{row['cache_id']}.json"
        write_json_new(dest, saved); rows.append(saved)
        print(number, '/', len(cache['entries']), row['cache_id'], flush=True)
    systems = ('pretrained','old_seed43','student','candidate','beam8','beam32')
    aggregates = {system:{cohort:phone_metrics([(r['reference'],r['predictions'][system]) for r in rows if r['cohort']==cohort and system in r['predictions']])
        for cohort in ('original_train','original_dev','rendered_train','rendered_dev') if not (system.startswith('beam') and cohort=='rendered_train')} for system in systems}
    frozen = read_json(Path(policy['bindings']['candidate']['path']))['development_metrics']
    if aggregates['candidate']['original_dev'] != frozen['original'] or aggregates['candidate']['rendered_dev'] != frozen['rendered']:
        raise ValueError('frozen candidate development replay differs')
    gates = {system:promotion_gate(aggregates['candidate'], aggregates[system], policy) for system in ('beam8','beam32')}
    passing = [name for name,gate in gates.items() if gate['passed']]
    selected = min(passing, key=lambda s:(aggregates[s]['original_dev']['per'], int(s[4:]))) if passing else 'candidate'
    write_json_new(run/'DECODER_COMPARISON.json', {'schema_version':'decoder_development_comparison_v1','policy_sha256':sha256_file(run/'POLICY.json'),
        'aggregate':aggregates,'gates':gates,'selected_decoder':selected,'candidate_development_exact':True,'test_reopened':False,
        'elapsed_seconds':time.monotonic()-start,'decode_seconds':{s:sum(r['decode_seconds'].get(s,0) for r in rows) for s in ('beam8','beam32')},
        'examples':[{'path':str(run/'examples'/f"{r['cache_id']}.json"),'sha256':sha256_file(run/'examples'/f"{r['cache_id']}.json")} for r in rows]})
    write_json_new(run/'ERROR_DIAGNOSIS.json', {'schema_version':'training_development_error_attribution_v1','errors':diagnoses,
        'interpretation':'Deterministic sequence-edit attribution, not verified acoustic phone boundaries; proxy reference unchanged.',
        'reference_scope':'human source note sequence and audibility; engineering IPA proxy expansion', 'test_reopened':False})
    print('DECODER complete:',selected,flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=('materialize','evaluate'),required=True);a=p.parse_args()
    run=a.run.resolve();policy=read_json(run/'POLICY.json');validate_policy(policy)
    if not run.is_relative_to(ROOT/'.state/runs'):
        raise ValueError('new artifacts must remain under runs')
    if a.stage=='materialize':materialize(run,policy)
    else:evaluate(run,policy)

if __name__=='__main__':main()

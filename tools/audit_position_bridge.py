"""Audit original/rendered phone-position provenance without creating new labels."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.training.ctc_position_bridge import position_occupancy,phone_key,exact_sequence_matches
from run_factor_exchange_probe import verify_launch


def audit(run: Path) -> None:
    from safetensors.numpy import load_file
    import numpy as np
    freeze=verify_launch(run,'bridge');review=read_json(run/'CODE_REVIEW_bridge.json')
    for p,h in review['data_bindings'].items():verify_file(Path(p),h)
    if not read_json(run/'event-profile/RESULT.json')['complete']:raise ValueError('event feasibility must finish before bridge audit')
    out=run/'bridge';write_json_new(out/'START.json',{'full_acoustic_forwards':0,'training_starts':0,
        'predeclared_scope':'exact sequence pairs; existing-note overlap; conditioned CTC occupancy audit only'})
    started=time.monotonic();base=Path(freeze['singing_run'])
    manifest_path=ROOT/'.state/runs/limited-data-20260906T050251/reference/reference-manifest.json'
    dataset_path=manifest_path.with_name('rendered-proxy-dataset.json')
    original_path=ROOT/'.state/runs/original-adaptation-20260919T012000/reviewed-training-reference.json'
    source=read_json(manifest_path);dataset=read_json(dataset_path)
    human={r['example_id']:r for r in read_json(original_path)['examples'] if r['split']=='train'}
    labels=read_json(base/'REFERENCES.json')['rows'];cache={r['example_id']:r for r in read_json(base/'CACHE.json')['entries']}
    reference={r['example_id']:r for r in labels};vocabulary=read_json(Path(freeze['vocabulary']['path']))
    rendered={r['example_id']:r for r in dataset['examples'] if r['example_id'] in reference and r['audio_role']=='rendered' and r['split']=='train'}
    if len(rendered)!=129 or len(human)!=20:raise ValueError('fixed train-only audit cohort changed')
    songs={r['song_id']:r for r in source['songs'] if r['split']=='train'}
    bindings={};notes={};rendered_keys={};occurrences=defaultdict(list)
    def bind(p,h=None):
        p=Path(p);actual=sha256_file(p)
        if h is not None and actual!=h:raise ValueError('bridge source hash mismatch: '+str(p))
        bindings[str(p)]=actual
    for song in songs.values():
        bind(song['human_vsqx']['path'],song['human_vsqx']['sha256'])
        for n in song['notes']:notes[n['note_id']]=n
    for wid,row in rendered.items():
        if not row['eligible'] or row['source_vsqx_sha256']!=songs[row['song_id']]['human_vsqx']['sha256']:
            raise ValueError('rendered reference eligibility/source mismatch')
        flat=[]
        for note in row['note_ids']:
            n=notes[note]
            if n['unresolved_tokens'] or n['unsupported_model_ipa']:raise ValueError('unresolved mapping in retained rendered window')
            flat.extend({'note_id':note,'phone_index':i,'phone':p} for i,p in enumerate(n['model_target_ipa']))
        if [p['phone'] for p in flat]!=reference[wid]['target_ipa'] or reference[wid]['target_ipa']!=row['model_target_ipa']:
            raise ValueError('rendered provenance flattening differs from frozen sequence')
        keys=[phone_key(row['source_vsqx_sha256'],p) for p in flat];rendered_keys[wid]=keys
        for i,key in enumerate(keys):occurrences[key].append((wid,i))
    originals={};summary=[];structural_pairs=[]
    for wid,h in sorted(human.items()):
        if h['status']!='human_confirmed' or h['source_vsqx_sha256']!=songs[h['song_id']]['human_vsqx']['sha256']:
            raise ValueError('original confirmation/version mismatch')
        if [p['phone'] for p in h['phones']]!=reference[wid]['target_ipa']:raise ValueError('original reference changed')
        keys=[phone_key(h['source_vsqx_sha256'],p) for p in h['phones']];originals[wid]=keys
        eligible_pairs=[dict(r,target_ipa=r['model_target_ipa']) for r in rendered.values()]
        original=dict(h,target_ipa=reference[wid]['target_ipa'])
        exact=exact_sequence_matches(original,eligible_pairs)
        exact_note_pairs=[r for r in exact if rendered_keys[r]==keys]
        local=[];reasons=Counter()
        for i,key in enumerate(keys):
            found=occurrences.get(key,[])
            if len(found)==1:
                rw,ri=found[0];item={'original_window':wid,'original_position':i,'rendered_window':rw,
                    'rendered_position':ri,'song_id':h['song_id'],'phone':key[3],'note_id':key[1],
                    'phone_index':key[2],'source_vsqx_sha256':key[0],'original_audibility':h['phones'][i]['audibility'],
                    'paired_full_sequence':rw in exact_note_pairs,'training_eligible':False,
                    'reason':'partial_sequence_overlap_only' if rw not in exact_note_pairs else 'requires_acoustic_reliability_check'}
                local.append(item);structural_pairs.append(item)
            elif not found:reasons['no_eligible_rendered_window_for_exact_note_phone']+=1
            else:reasons['ambiguous_duplicate_rendered_note_phone']+=1
        summary.append({'window_id':wid,'song_id':h['song_id'],'reference_phones':len(keys),
            'exact_sequence_matches':exact,'exact_sequence_and_note_matches':exact_note_pairs,
            'uniquely_overlapping_phone_positions':len(local),'unmatched_or_ambiguous':dict(reasons),
            'original_input_seconds':h['end_sec']-h['start_sec']})
    used=set(originals)|{p['rendered_window'] for p in structural_pairs}
    occupancies={};position_reports={};cached_count=0
    for wid in sorted(used):
        e=cache[wid];ref=reference[wid]
        for k in ('song_id','split','audio_role','audio_sha256','start_sec','end_sec'):
            if e[k]!=ref[k]:raise ValueError('cache/reference identity mismatch: '+k)
        if e['split']!='train' or e['song_id'] not in songs or e['architecture_sha256']!=read_json(base/'CACHE.json')['architecture']['sha256']:
            raise ValueError('cache split/architecture mismatch')
        bind(e['path'],e['sha256']);raw=load_file(e['path'])['frame_logits']
        if raw.shape!=(e['frames'],428) or not np.isfinite(raw).all():raise ValueError('invalid complete cache logits')
        ids=[vocabulary[p] for p in ref['target_ipa']]
        if ids!=ref['target_ids']:raise ValueError('target/vocabulary identity mismatch')
        d=position_occupancy(raw,ids);cached_count+=1
        # Stability means sensitivity to one-frame perturbation of model evidence,
        # NOT independently verified acoustic reliability. Reflect endpoints.
        shifted=[]
        for moved in (np.r_[raw[:1],raw[:-1]],np.r_[raw[1:],raw[-1:]]):
            shifted.append(position_occupancy(moved,ids)['position_weights']);cached_count+=1
        stats=[]
        for i,p in enumerate(d['positions']):
            center=d['position_weights'][:,i]
            variations=[float(.5*np.abs(center-w[:,i]).sum()) for w in shifted]
            stats.append({**p,'phone':ref['target_ipa'][i],
                'one_frame_shift_maximum_total_variation':max(variations),
                'central_90_width_seconds':p['central_90_width_frames']*.02,
                'perturbation_is_diagnostic_not_boundary_confirmation':True})
        occupancies[wid]=stats
        position_reports[wid]={'song_id':e['song_id'],'audio_role':e['audio_role'],'frames':e['frames'],
            'reference_phones':len(ids),'log_target_probability':d['log_target_probability'],
            'positions':stats,'source_cache_sha256':e['sha256'],'reference_conditioned_not_ground_truth':True}
    for pair in structural_pairs:
        a=occupancies[pair['original_window']][pair['original_position']]
        b=occupancies[pair['rendered_window']][pair['rendered_position']]
        pair['original_soft_position']=a;pair['rendered_soft_position']=b
        # No acoustic matching threshold is invented after seeing these values.
    by_song={s:{'windows':sum(r['song_id']==s for r in summary),
        'reference_phones':sum(r['reference_phones'] for r in summary if r['song_id']==s),
        'exact_full_sequence_pairs':sum(len(r['exact_sequence_and_note_matches']) for r in summary if r['song_id']==s),
        'overlapping_positions_audit_only':sum(p['song_id']==s for p in structural_pairs)} for s in songs}
    phone_counts=Counter(p['phone'] for p in structural_pairs)
    reason='no_exact_full_sequence_pairs' if not any(r['exact_sequence_and_note_matches'] for r in summary) else 'acoustic_reliability_not_independently_verified'
    write_json_new(out/'RESULT.json',{'schema_version':'position_bridge_feasibility_v1','complete':True,
        'training_ready':False,'blocking_reason':reason,'original_windows':20,'original_reference_phones':1235,
        'original_seconds':200,'rendered_candidate_windows':129,'by_song':by_song,'window_audit':summary,
        'structural_position_pairs':structural_pairs,'paired_position_phone_counts':dict(phone_counts),
        'soft_position_reports':position_reports,'cached_ctc_forward_backward_calls':cached_count,
        'marginal_evidence_sources':len(used),'source_bindings':bindings,'new_quality_training_starts':0,
        'new_audio_crops':0,'new_acoustic_forwards':0,'new_reference_labels':0,
        'reference_or_score_used_for_recognition':False,'elapsed_seconds':time.monotonic()-started,
        'interpretation':'Note/source identity gives structural correspondence only. Soft CTC positions are reference-conditioned; never acoustic truth. Partial-overlap pairs are diagnostic, not automatically authorized training pairs.',
        'request_relabeling':False,'next_option':'Specify partial-sequence, uncertainty-aware auxiliary supervision separately; do not manufacture exact pairs or recrop to force them.'})
    print('BRIDGE',by_song,'position_pairs',len(structural_pairs),'cached_FB',cached_count,'block',reason,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    audit(p.parse_args().run.resolve())

"""Prepare bounded combined correspondence/edge review from existing evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,sha256_object,verify_file,write_json_new
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
from any2xsampa.training.replacement_candidates import overlaps,path_time,select_candidates
from tools.build_original_edge_review import note_view
from tools.import_replacement_location_answers import edge_notes
from tools.prepare_training_progress_review import anchor_failures
from tools.reconcile_replacement_edges import load_verified_packet

PROGRESS=ROOT/'.state/runs/reviewed-training-progress-20260917T230000'
OLD=ROOT/'.state/runs/training-replacements-20260917T211100'
CUMULATIVE=ROOT/'.state/runs/retained-edge-intake-20260917T233500/CUMULATIVE_DATA.json'
POLICY=ROOT/'src/any2xsampa/config/batch_training_review_v1.json'


def choice_for(song: dict, start: float, end: float, role: str, rank: int) -> dict:
    notes=[n for n in song['notes'] if n['start_sec']<end and n['end_sec']>start]
    neighbors={side:edge_notes(song,t) for side,t in [('start',start),('end',end)]}
    gaps=[side for side in ('start','end') if not neighbors[side]]
    if not notes:
        raise ValueError('empty_reference_candidates')
    for side in gaps:
        neighbors[side]=[note_view(notes[0 if side=='start' else -1])]
    wanted={n['note_id'] for values in neighbors.values() for n in values}
    positions=[i for i,n in enumerate(song['notes']) if n['note_id'] in wanted]
    if not positions or not all(neighbors.values()):
        raise ValueError('empty_reference_candidates')
    return {'start_sec':start,'end_sec':end,'source_role':role,'path_rank':rank,
        'notes':[note_view(n) for n in notes], 'edge_notes':neighbors,
        'edge_score_gaps':gaps,
        'catalog':[note_view(n) for n in song['notes'][min(positions):max(positions)+1]],
        'mapping_complete':all(n['mapping_status']=='resolved_proxy' and n['model_target_ipa'] for n in notes),
        'training_eligible':False}


def proposals(context: dict,song: dict,alignment: dict,protected: list[dict],rejected: list[dict],
              anchors: list[dict],policy: dict) -> tuple[list[dict],list[dict]]:
    pool,excluded=[],[]
    paths=[(role,p) for rank in range(1,4) for role in ('vocal','mixed')
           for p in alignment['paths'][role]['candidates'] if p['rank']==rank]
    local=[a for a in anchors if a['context_id']==context['context_id']]
    for offset in range(round((context['end_sec']-context['start_sec']-10)/.5)+1):
        start=round(context['start_sec']+offset*.5,6)
        row={'song_id':song['song_id'],'split':'train','context_id':context['context_id'],
             'start_sec':start,'end_sec':start+10,
             'center_distance_sec':abs(start+5-(context['old_start_sec']+context['old_end_sec'])/2)}
        if any(overlaps(row,p) for p in protected):
            excluded.append({**row,'reason':'overlaps_previous_review_or_accepted_window'})
            continue
        choices,failures=[],[]
        for role,path in paths:
            reason=None
            if anchor_failures(path,local): reason='conflicts_with_local_confirmed_section'
            elif 1-path['mean_path_cost']<policy['minimum_similarity']: reason='low_audio_similarity'
            try:
                rs,re=path_time(path,start),path_time(path,start+10)
                if not 0<=rs<re<=song['rendered_audio']['duration_sec']: reason=reason or 'rendered_range_invalid'
                if not 7<=re-rs<=13: reason=reason or 'implausible_local_time_scale'
                for old in rejected:
                    if overlaps(row,old):
                        midpoint=(max(start,old['start_sec'])+min(start+10,old['end_sec']))/2
                        if abs(path_time(path,midpoint)-(old['rendered_start_sec']+midpoint-old['start_sec']))<.5:
                            reason=reason or 'repeats_rejected_correspondence'
                if not reason:
                    choice=choice_for(song,rs,re,role,path['rank'])
                    if not choice['mapping_complete']: reason='unresolved_whole_proposed_section'
                    elif any(abs(rs-c['start_sec'])<.5 and abs(re-c['end_sec'])<.5 for c in choices): reason='duplicate_alternative'
                    elif len(choices)<policy['maximum_alternatives']:
                        choice['choice_id']=chr(65+len(choices));choices.append(choice)
                    else: reason='alternative_limit'
            except ValueError as error: reason=str(error)
            if reason: failures.append({'source_role':role,'path_rank':path['rank'],'reason':reason})
        if choices:
            pool.append({**row,'alternatives':choices,'human_status':'pending','machine_accepted':False,
                         'training_eligible':False,'path_failures':failures})
        else: excluded.append({**row,'reason':'no_usable_alternative','path_failures':failures})
    return pool,excluded


def build(output: Path) -> dict:
    from tools.prepare_original_training_candidates import _preview
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'):
        raise ValueError('new run required')
    previous=load_verified_packet(PROGRESS/'edge-review/packet.json')
    sources=deepcopy(previous['source_bindings'])
    cumulative=read_json(CUMULATIVE)
    if cumulative['manifest_hash']!=sha256_object({k:v for k,v in cumulative.items() if k!='manifest_hash'}):
        raise ValueError('cumulative hash mismatch')
    delivery=read_json(CUMULATIVE.parent/'DELIVERY.json')
    verify_file(CUMULATIVE,delivery['artifacts']['CUMULATIVE_DATA.json'])
    policy=read_json(POLICY)
    output.mkdir(parents=True)
    write_json_new(output/'FROZEN_POLICY.json',policy)
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
    enforce_offline(output/'runtime-cache')
    for name,path in [('batch_policy',POLICY),('cumulative_acceptance',CUMULATIVE),('prior_history',OLD/'HISTORY.json')]:
        sources[name]={'path':str(path),'sha256':sha256_file(path)}
    reference,_=apply_original_mapping_overlay(read_json(Path(sources['reference']['path'])),Path(sources['mapping_evidence']['path']))
    songs={s['song_id']:s for s in reference['songs'] if s['split']=='train'}
    history=read_json(OLD/'HISTORY.json')
    shutil.copyfile(OLD/'HISTORY.json',output/'history-preserved.json')
    location=read_json(Path(sources['location_review']['path']))
    # Old retained windows and every recent location question remain protected.
    protected=history['retained']+location['items']+cumulative['rows']
    ctxdoc=read_json(Path(sources['contexts']['path']))
    contexts=[c for c in ctxdoc['contexts'] if c['split']=='train']
    audit=read_json(PROGRESS/'MULTIPATH_AUDIT.json')
    progress_delivery=read_json(PROGRESS/'DELIVERY.json')
    verify_file(PROGRESS/'MULTIPATH_AUDIT.json',progress_delivery['artifacts']['MULTIPATH_AUDIT.json'])
    sources['multipath_audit']={'path':str(PROGRESS/'MULTIPATH_AUDIT.json'),'sha256':sha256_file(PROGRESS/'MULTIPATH_AUDIT.json')}
    pool,excluded=[],[]
    for context in contexts:
        spec=sources[context['context_id']+'_saved_paths']
        verify_file(Path(spec['path']),spec['sha256'])
        alignment=read_json(Path(spec['path']))
        if alignment['context']!=context: raise ValueError('context mismatch')
        values,failures=proposals(context,songs[context['song_id']],alignment,protected,history['rejected'],audit['anchors'],policy)
        pool.extend(values);excluded.extend(failures)
    selected=select_candidates(pool,protected,ctxdoc['split']['train'],policy['maximum_new_slots'])
    write_json_new(output/'PROPOSAL_AUDIT.json',{'pool_count':len(pool),'excluded':excluded,
        'pool_windows':[{k:r[k] for k in ('song_id','context_id','start_sec','end_sec','center_distance_sec','path_failures')} for r in pool],
        'selected_new_windows':[{k:r[k] for k in ('song_id','context_id','start_sec','end_sec')} for r in selected],
        'policy_not_machine_acceptance':True})
    rows=[]
    for old in history['retained']:
        if old['review_id'] not in policy['old_pending_ids']: continue
        choice=choice_for(songs[old['song_id']],old['rendered_start_sec'],old['rendered_start_sec']+10,'human_reviewed_section',0)
        choice['choice_id']='A'
        rows.append({**deepcopy(old),'alternatives':[choice],'kind':'retained','correspondence_preconfirmed':True,
            'previous_audit':old['history'],'split':'train'})
    for i,row in enumerate(selected,1):
        rows.append({**row,'review_id':f'R{i:02}','kind':'replacement','correspondence_preconfirmed':False,
            'previous_audit':None})
    media=output/'media';media.mkdir()
    for row in rows:
        song=songs[row['song_id']]
        for role in ('original_audio','rendered_audio','human_vsqx'):
            verify_file(Path(song[role]['path']),song[role]['sha256'])
        row['original_audio']=deepcopy(song['original_audio']);row['source_vsqx']=deepcopy(song['human_vsqx'])
        row['original']=_preview(song['original_audio'],row,media,row['review_id']+'-original','original')
        row['edge_media']={}
        for side in ('start','end'):
            cut=row[side+'_sec']
            row['edge_media'][side]=_preview(song['original_audio'],{'start_sec':cut-1.25,'end_sec':cut+1.25},media,row['review_id']+'-'+side,'original')
        for choice in row['alternatives']:
            choice['preview']=_preview(song['rendered_audio'],choice,media,row['review_id']+'-'+choice['choice_id'],'rendered')
        row['training_eligible']=False
        print('prepared',row['review_id'],row['song_id'],flush=True)
    packet={'schema_version':'batch_training_review_packet_v1','items':rows,'source_bindings':sources,
        'prior_valid_windows':cumulative['valid_windows'],'active_slots_after':policy['existing_active_slots']+len(selected),
        'new_slot_count':len(selected),'policy':policy,'test_songs_opened':False,'training_started':False}
    packet['packet_sha256']=sha256_object(packet)
    write_json_new(output/'packet.json',packet)
    from tools.batch_training_review import validate_packet,render_page
    validate_packet(packet)
    render_page(output,packet)
    for spec in sources.values(): verify_file(Path(spec['path']),spec['sha256'])
    return packet


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=build(args.output.resolve())
    print('items',len(result['items']),'new',result['new_slot_count'],'active',result['active_slots_after'])

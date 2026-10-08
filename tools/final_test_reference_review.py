"""Blind final-test reference review after candidate freeze; no predictions opened."""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import argparse,base64,json,sys,shutil,wave
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,sha256_object,verify_file
from any2xsampa.training.limited_data_reference import load_mapping,load_vocabulary,project_tokens,SONG_SPLIT
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay,documented_compound_mapping,OVERLAY_PATH
from any2xsampa.training.original_edge_review import _validate_answer
from tools.build_original_edge_review import note_view
from tools.reconcile_replacement_edges import assemble_row
from tools.free_location_review import reference_preview
from tools.prepare_original_training_candidates import _preview
SCHEMA='frozen_final_test_review_packet_v1'
ANSWERS='frozen_final_test_review_answers_v1'


def bind(path:Path)->dict:
    return {'path':str(path.resolve()),'sha256':sha256_file(path)}


def test_view(sources:dict)->dict:
    for spec in sources.values():verify_file(Path(spec['path']),spec['sha256'])
    ref=read_json(Path(sources['reference']['path']))
    # Independently verify the existing documented ua+n rule; do not modify its
    # training-only application function or infer any other alias for test data.
    apply_original_mapping_overlay(ref,Path(sources['mapping_evidence']['path']))
    mapping=documented_compound_mapping(load_mapping(Path(ref['bindings']['mapping']['path'])),read_json(OVERLAY_PATH))
    vocabulary=load_vocabulary(Path(ref['bindings']['vocabulary']['path']))
    view=deepcopy(ref)
    for song in view['songs']:
        if song['split']=='test':
            for note in song['notes']:
                if 'ua_n' in note['raw_v4c_tokens']:note.update(project_tokens(note['raw_v4c_tokens'],mapping,vocabulary))
    return view


def validate_packet(packet:dict)->None:
    if packet.get('schema_version')!=SCHEMA or packet.get('packet_sha256')!=sha256_object({k:v for k,v in packet.items() if k!='packet_sha256'}):raise ValueError('final review packet hash mismatch')
    if packet.get('test_predictions_opened') is not False or packet.get('candidate_already_frozen') is not True or packet.get('training_allowed') is not False:raise ValueError('not a blind frozen reference packet')
    if len(packet['items'])!=6 or len({r['window_id'] for r in packet['items']})!=6:raise ValueError('six frozen test windows required')
    songs={s['song_id']:s for s in packet['songs']}
    if set(songs)!={'ke-yi-ke-zai','king-of-karaoke'}:raise ValueError('sealed song split changed')
    for item in packet['items']:
        if item['song_id'] not in songs or SONG_SPLIT[item['song_id']]!='test' or abs(item['end_sec']-item['start_sec']-18)>1e-6:raise ValueError('final window/split changed')
    if sorted(q['token'] for q in packet['mapping_questions'])!=['7','UN','iE_r','iUN','uo']:raise ValueError('mapping review scope changed')


def verify_packet(directory:Path,packet:dict)->None:
    validate_packet(packet);view=test_view(packet['source_bindings'])
    freeze=read_json(Path(packet['source_bindings']['candidate']['path']))
    if freeze.get('schema_version')!='guarded_phone_candidate_v1' or not freeze.get('selection_complete_before_separated_transfer'):raise ValueError('candidate not frozen before review')
    for name in ('teacher_checkpoint','student_checkpoint','policy'):
        s=freeze[name];verify_file(Path(s['path']),s['sha256'])
    frozen={r['example_id']:r for r in view['original_clips'] if r['split']=='test'}
    for song in packet['songs']:
        src=next(s for s in view['songs'] if s['song_id']==song['song_id'])
        expected=[{**note_view(n),'raw_v4c_tokens':n['raw_v4c_tokens'],'unresolved_tokens':n.get('unresolved_tokens',[]),'start_sec':n['start_sec'],'end_sec':n['end_sec']} for n in src['notes']]
        if song['catalog']!=expected:raise ValueError('test catalog differs from frozen source')
        for key,role in [('original_audio','original_audio'),('rendered_audio','rendered_audio'),('source_vsqx','human_vsqx')]:
            if song[key]!=src[role]:raise ValueError('test song identity mismatch')
            verify_file(Path(song[key]['path']),song[key]['sha256'])
        media=[(song['rendered_preview'],song['rendered_audio'],0.,song['rendered_audio']['duration_sec'])]
        for item in [i for i in packet['items'] if i['song_id']==song['song_id']]:
            old=frozen.get(item['source_example_id'])
            if old is None or (item['start_sec'],item['end_sec'])!=(old['start_sec'],old['end_sec']):raise ValueError('predeclared final window changed')
            media.append((item['original'],song['original_audio'],item['start_sec'],item['end_sec']))
            for side,m in item['edges'].items():media.append((m,song['original_audio'],item[side+'_sec']-1.25,item[side+'_sec']+1.25))
        for m,source,start,end in media:
            if m['source_sha256']!=source['sha256'] or m['source_crop']['requested_start_sec']!=start or m['source_crop']['requested_end_sec']!=end:raise ValueError('media source interval mismatch')
            for kind in ('audio','waveform'):
                path=(directory/'media'/m[kind+'_file']).resolve()
                if not path.is_relative_to((directory/'media').resolve()):raise ValueError('unsafe media path')
                verify_file(path,m[kind+'_sha256'])
            with wave.open(str(directory/'media'/m['audio_file']),'rb') as w:
                if abs(w.getnframes()/w.getframerate()-(end-start))>1/w.getframerate()+1e-8:raise ValueError('media timebase mismatch')
    # Full-song manual phone approval is distinct from the as-yet unconfirmed crops.
    claims=read_json(Path(packet['source_bindings']['reference_triplets']['path']))['items']
    for song in packet['songs']:
        c=next(v for v in claims if v['song_id']==song['song_id'])
        bound=c['human_vsqx']['sha256']==song['source_vsqx']['sha256'] and c['human_claims']['phoneme_correctness']=='user_asserted_correct'
        if not bound or song['source_sequence_approval_bound']!=bound:raise ValueError('source phone approval unbound')


def validate_answers(packet:dict,payload:dict)->list[dict]:
    validate_packet(packet)
    if payload.get('schema_version')!=ANSWERS or payload.get('packet_sha256')!=packet['packet_sha256']:raise ValueError('answers belong to another final packet')
    reviewer=payload.get('reviewer')
    if not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer)>120:raise ValueError('reviewer required')
    if payload.get('training_started') is not False or payload.get('test_predictions_opened') is not False:raise ValueError('review must remain prediction blind')
    rows=payload.get('answers',[])
    if len(rows)!=6 or {a['window_id'] for a in rows}!={i['window_id'] for i in packet['items']}:raise ValueError('six unique answers required')
    for a in rows:
        if a.get('status') not in {'confirmed','located','pending','unavailable'} or not isinstance(a.get('comment'),str) or len(a['comment'])>4000:raise ValueError('invalid answer status/comment')
        item=next(i for i in packet['items'] if i['window_id']==a['window_id']);song=next(s for s in packet['songs'] if s['song_id']==item['song_id'])
        if a['status'] in {'pending','unavailable'}:
            if a.get('sequence_confirmed') is not False or a.get('edges')!={}:raise ValueError('pending cannot assert reference')
            continue
        if a.get('sequence_confirmed') is not True or set(a.get('edges',{}))!={'start','end'}:raise ValueError('explicit range confirmation required')
        ids=[n['note_id'] for n in song['catalog']]
        try:lo,hi=[ids.index(a['edges'][side]['note_id']) for side in ('start','end')]
        except (ValueError,KeyError):raise ValueError('unknown edge note')
        if lo>hi:raise ValueError('noncontinuous/reversed range')
        if a['status']=='located':
            if any(a['edges'][side].get('phones')!=[] or a['edges'][side].get('whole_note_audibility','uncertain') not in {'full','partial','uncertain'} for side in ('start','end')):raise ValueError('located-only cannot assert phones')
            continue
        for side in ('start','end'):
            e=a['edges'][side]
            if e.get('review_id')!=a['window_id']+('-S' if side=='start' else '-E') or e.get('observation_schema')!='phone_audibility_v2' or e.get('status')!='confirmed':raise ValueError('invalid edge envelope')
            _validate_answer({'side':side,'notes':song['catalog']},e,reviewer)
            n=song['catalog'][ids.index(e['note_id'])]
            if [p['phone_index'] for p in e['phones']]!=list(range(len(n['phones']))):raise ValueError('all edge phones require observation')
    mapping=payload.get('mapping_responses',[])
    if len(mapping)!=5 or {r['token'] for r in mapping}!={q['token'] for q in packet['mapping_questions']}:raise ValueError('five mapping responses required')
    for r in mapping:
        if r.get('status') not in {'described','pending'} or not isinstance(r.get('description'),str) or len(r['description'])>4000:raise ValueError('invalid mapping note')
        if r['status']=='described' and not r['description'].strip():raise ValueError('empty mapping description')
    return rows


def assemble(packet:dict,payload:dict)->dict:
    rows=validate_answers(packet,payload);examples=[]
    for a in rows:
        item=next(i for i in packet['items'] if i['window_id']==a['window_id']);song=next(s for s in packet['songs'] if s['song_id']==item['song_id'])
        if a['status']=='confirmed':
            clip=dict(example_id=item['source_example_id'],song_id=song['song_id'],split='test',audio_path=song['original_audio']['path'],audio_sha256=song['original_audio']['sha256'],source_vsqx_sha256=song['source_vsqx']['sha256'],reference_manifest_sha256=packet['source_bindings']['reference']['sha256'],start_sec=item['start_sec'],end_sec=item['end_sec'],notes=song['catalog'],source_sequence_approval_bound=song['source_sequence_approval_bound'])
            row=assemble_row(clip,a['edges'],payload['reviewer'])
        else:
            row=dict(example_id=item['source_example_id'],song_id=song['song_id'],split='test',status='pending',phones=[],training_eligible=False,blocking_reasons=['reference_mapping_or_edges_pending' if a['status']=='located' else 'human_location_pending'])
        row.update(window_id=a['window_id'],raw_answer=deepcopy(a),unconfirmed_draft=deepcopy(payload.get('unconfirmed_drafts',{}).get(a['window_id'])));examples.append(row)
    result={'schema_version':'frozen_final_test_reference_intake_v1','packet_sha256':packet['packet_sha256'],'candidate_binding':packet['source_bindings']['candidate'],'examples':examples,'mapping_responses':deepcopy(payload['mapping_responses']),'mapping_descriptions_automatically_applied':False,'training_allowed':False,'test_predictions_opened':False}
    result['manifest_hash']=sha256_object(result);return result


def render(directory:Path,packet:dict)->None:
    verify_packet(directory,packet)
    media=[s['rendered_preview'] for s in packet['songs']]
    for item in packet['items']:media += [item['original'],*item['edges'].values()]
    audio={m['audio_file']:base64.b64encode((directory/'media'/m['audio_file']).read_bytes()).decode('ascii') for m in media}
    assets=ROOT/'tools/final_test_review_gui';text=(assets/'index.html').read_text(encoding='utf-8')
    enc=lambda x:json.dumps(x,ensure_ascii=True).replace('<',r'\u003c')
    for name,value in dict(PACKET=enc(packet),AUDIO=enc(audio),CSS=(assets/'style.css').read_text(encoding='utf-8'),STATE=(assets/'state.js').read_text(encoding='utf-8'),APP=(assets/'app.js').read_text(encoding='utf-8')).items():text=text.replace('/* INLINE_'+name+' */',value)
    with (directory/'index.html').open('x',encoding='utf-8') as f:f.write(text)


def build(candidate:Path,output:Path)->dict:
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'):raise ValueError('new run directory required')
    previous=read_json(ROOT/'.state/runs/free-location-review-20260918T235300/packet.json')
    sources={k:previous['source_bindings'][k] for k in ('reference','mapping_evidence','mapping_overlay','reference_triplets')};sources['candidate']=bind(candidate)
    view=test_view(sources);output.mkdir(parents=True);media=output/'media';media.mkdir()
    songs=[];items=[];questions={};coverage={};fixed=[r for r in view['original_clips'] if r['split']=='test']
    names={'ke-yi-ke-zai':'可一可再','king-of-karaoke':'K歌之王'}
    for song in [s for s in view['songs'] if s['split']=='test']:
        for name in ('original_audio','rendered_audio','human_vsqx'):verify_file(Path(song[name]['path']),song[name]['sha256'])
        sub=media/song['song_id'];sub.mkdir();preview=reference_preview(song['rendered_audio'],sub)
        for key in ('audio_file','waveform_file'):preview[key]=song['song_id']+'/'+preview[key]
        catalog=[{**note_view(n),'raw_v4c_tokens':n['raw_v4c_tokens'],'unresolved_tokens':n.get('unresolved_tokens',[]),'start_sec':n['start_sec'],'end_sec':n['end_sec']} for n in song['notes']]
        songs.append(dict(song_id=song['song_id'],song_name=names[song['song_id']],catalog=catalog,original_audio=song['original_audio'],rendered_audio=song['rendered_audio'],source_vsqx=song['human_vsqx'],rendered_preview=preview,source_sequence_approval_bound=True))
        coverage[song['song_id']]={'notes':len(catalog),'resolved_notes':sum(n['mapping_status']=='resolved_proxy' for n in catalog),'unresolved_notes':sum(n['mapping_status']!='resolved_proxy' for n in catalog)}
        for n in catalog:
            for token in n['unresolved_tokens']:
                q=questions.setdefault(token,{'token':token,'contexts':[]})
                q['contexts'].append({'song_id':song['song_id'],'song_name':names[song['song_id']],'note_id':n['note_id'],'lyric':n['lyric'],'raw_v4c_tokens':n['raw_v4c_tokens'],'start_sec':n['start_sec'],'end_sec':n['end_sec']})
        for old in [r for r in fixed if r['song_id']==song['song_id']]:
            wid=f'T{len(items)+1:02d}'
            original=_preview(song['original_audio'],old,media,wid+'-original','original')
            edges={side:_preview(song['original_audio'],dict(start_sec=old[side+'_sec']-1.25,end_sec=old[side+'_sec']+1.25),media,wid+'-'+side,'original') for side in ('start','end')}
            items.append(dict(window_id=wid,source_example_id=old['example_id'],song_id=song['song_id'],start_sec=old['start_sec'],end_sec=old['end_sec'],original=original,edges=edges))
    packet={'schema_version':SCHEMA,'source_bindings':sources,'songs':songs,'items':items,'mapping_questions':sorted(questions.values(),key=lambda q:q['token']),'mapping_coverage':coverage,
            'mapping_policy':'existing_v4_plus_documented_ua_n_only; other_aliases_unresolved','candidate_already_frozen':True,'training_allowed':False,'test_predictions_opened':False,'machine_reference_preselected':False,'original_test_audio_read_for_blind_review_only':True}
    packet['packet_sha256']=sha256_object(packet);write_json_new(output/'packet.json',packet);render(output,packet)
    write_json_new(output/'BUILD.json',{'candidate':sources['candidate'],'packet_sha256':packet['packet_sha256'],'windows':6,'edge_questions':12,'mapping_descriptions':len(questions),'predictive_models_run':False,'test_predictions_opened':False,'mapping_coverage':coverage})
    return packet

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path);p.add_argument('--packet',type=Path);p.add_argument('--answers',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'.state/runs') or out.exists():raise ValueError('new run output required')
    if a.candidate:
        d=build(a.candidate.resolve(),out);print('packet',d['packet_sha256'])
    else:
        d=read_json(a.packet);verify_packet(a.packet.parent,d);result=assemble(d,read_json(a.answers));out.mkdir();shutil.copyfile(a.answers,out/'raw-answers.json');write_json_new(out/'reviewed-reference.json',result);write_json_new(out/'INPUTS.json',{'packet':bind(a.packet),'answers':bind(a.answers)});print('confirmed',sum(r['status']=='human_confirmed' for r in result['examples']))

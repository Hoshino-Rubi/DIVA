"""Validate combined human answers and render an offline batch review."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sys
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json,sha256_file,sha256_object,verify_file,write_json_new,finite_number
from any2xsampa.training.limited_data_reference import SONG_SPLIT
from any2xsampa.training.original_edge_review import _validate_answer
from tools.reconcile_replacement_edges import assemble_row

PACKET_SCHEMA='batch_training_review_packet_v1'
ANSWER_SCHEMA='batch_training_review_answers_v1'


def validate_packet(packet: dict) -> None:
    if packet.get('schema_version')!=PACKET_SCHEMA or packet.get('packet_sha256')!=sha256_object({k:v for k,v in packet.items() if k!='packet_sha256'}):
        raise ValueError('batch packet hash or schema mismatch')
    if not 1<=len(packet['items'])<=16 or packet['active_slots_after']>20:
        raise ValueError('batch slot limit exceeded')
    seen=set()
    for row in packet['items']:
        if row['review_id'] in seen or row['split']!='train' or SONG_SPLIT.get(row['song_id'])!='train':
            raise ValueError('duplicate ID or non-training song')
        seen.add(row['review_id'])
        start=finite_number(row['start_sec'],'start',minimum=0)
        end=finite_number(row['end_sec'],'end',minimum=0)
        if abs(end-start-10)>1e-6 or row['training_eligible'] is not False:
            raise ValueError('invalid window or automatic eligibility')
        if not 1<=len(row['alternatives'])<=3 or len({c['choice_id'] for c in row['alternatives']})!=len(row['alternatives']):
            raise ValueError('invalid alternatives')
        for c in row['alternatives']:
            ids=[n['note_id'] for n in c['catalog']]
            if not ids or len(ids)!=len(set(ids)):
                raise ValueError('invalid candidate note catalog')
            for note in c['catalog']:
                if not isinstance(note['phones'],list) or any(not isinstance(p,str) or not p for p in note['phones']):
                    raise ValueError('invalid candidate phones')
        for old in packet['items']:
            if old is row: break
            if old['song_id']==row['song_id'] and start<old['end_sec']-1e-7 and old['start_sec']<end-1e-7:
                raise ValueError('batch original windows overlap')


def validate_answers(packet: dict,payload: dict) -> dict[str,dict]:
    validate_packet(packet)
    if payload.get('schema_version')!=ANSWER_SCHEMA or payload.get('packet_sha256')!=packet['packet_sha256']:
        raise ValueError('answers belong to a different batch')
    if not isinstance(payload.get('reviewer'),str) or not payload['reviewer'].strip() or len(payload['reviewer'])>120:
        raise ValueError('reviewer required')
    if not isinstance(payload.get('answers'),list) or payload.get('training_started') is not False:
        raise ValueError('invalid answer envelope')
    items={r['review_id']:r for r in packet['items']}
    result={}
    for answer in payload['answers']:
        rid=answer.get('review_id')
        if rid not in items or rid in result:
            raise ValueError('unknown or duplicate review ID')
        row=items[rid]
        if not isinstance(answer.get('comment'),str) or len(answer['comment'])>4000:
            raise ValueError('invalid raw comment')
        if answer.get('status') not in {'confirmed','uncertain','rejected'}:
            raise ValueError('invalid answer status')
        if answer['status']!='confirmed':
            if answer.get('edges')!={} or answer.get('sequence_confirmed') is not False or answer.get('choice') is not None:
                raise ValueError('unconfirmed answer cannot assert reference')
        else:
            choice=next((c for c in row['alternatives'] if c['choice_id']==answer.get('choice')),None)
            if choice is None or answer.get('sequence_confirmed') is not True or set(answer.get('edges',{}))!={'start','end'}:
                raise ValueError('explicit section and continuous sequence confirmation required')
            if row['correspondence_preconfirmed'] and answer['choice']!='A':
                raise ValueError('old correspondence cannot silently change')
            for side in ('start','end'):
                edge=answer['edges'][side]
                if edge.get('review_id')!=rid+('-S' if side=='start' else '-E') or edge.get('status')!='confirmed' or edge.get('observation_schema')!='phone_audibility_v2':
                    raise ValueError('invalid edge assertion')
                _validate_answer({'side':side,'notes':choice['catalog']},edge,payload['reviewer'])
                note=next(n for n in choice['catalog'] if n['note_id']==edge['note_id'])
                if {p['phone_index'] for p in edge['phones']}!=set(range(len(note['phones']))):
                    raise ValueError('each edge phone must be explicitly reviewed')
            positions={n['note_id']:i for i,n in enumerate(choice['catalog'])}
            if positions[answer['edges']['start']['note_id']]>positions[answer['edges']['end']['note_id']]:
                raise ValueError('start note follows end note')
        result[rid]=deepcopy(answer)
    return result


def verify_sources_and_media(directory: Path,packet: dict) -> dict:
    validate_packet(packet)
    for spec in packet['source_bindings'].values(): verify_file(Path(spec['path']),spec['sha256'])
    sources=packet['source_bindings']
    from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
    from tools.build_original_edge_review import note_view
    ref,_=apply_original_mapping_overlay(read_json(Path(sources['reference']['path'])),Path(sources['mapping_evidence']['path']))
    songs={s['song_id']:s for s in ref['songs'] if s['split']=='train'}
    triplets=read_json(Path(sources['reference_triplets']['path']))
    for row in packet['items']:
        song=songs[row['song_id']]
        if row['original_audio']!=song['original_audio'] or row['source_vsqx']!=song['human_vsqx']:
            raise ValueError('source identity mismatch')
        preview=row['original']
        if preview['source_sha256']!=song['original_audio']['sha256'] or preview['source_crop']['requested_start_sec']!=row['start_sec'] or preview['source_crop']['requested_end_sec']!=row['end_sec']:
            raise ValueError('formal original preview clock mismatch')
        verify_file(Path(row['source_vsqx']['path']),row['source_vsqx']['sha256'])
        catalog={n['note_id']:note_view(n) for n in song['notes']}
        ids=[n['note_id'] for n in song['notes']]
        for choice in row['alternatives']:
            positions=[ids.index(n['note_id']) for n in choice['catalog']]
            if positions!=list(range(positions[0],positions[-1]+1)) or any(catalog[n['note_id']]!=n for n in choice['catalog']):
                raise ValueError('reference catalog changed or non-continuous')
            preview=choice['preview']
            if preview['source_sha256']!=song['rendered_audio']['sha256'] or preview['source_crop']['requested_start_sec']!=choice['start_sec'] or preview['source_crop']['requested_end_sec']!=choice['end_sec']:
                raise ValueError('alternative preview clock mismatch')
        previews=[row['original'],*row['edge_media'].values(),*[c['preview'] for c in row['alternatives']]]
        for preview in previews:
            verify_file(Path(preview['source_path']),preview['source_sha256'])
            for kind in ('audio','waveform'):
                path=(directory/'media'/preview[kind+'_file']).resolve()
                if not path.is_relative_to((directory/'media').resolve()): raise ValueError('unsafe media path')
                verify_file(path,preview[kind+'_sha256'])
            with wave.open(str(directory/'media'/preview['audio_file']),'rb') as wav:
                if abs(wav.getnframes()/wav.getframerate()-preview['duration_sec'])>1/wav.getframerate():
                    raise ValueError('audio duration mismatch')
        for side,preview in row['edge_media'].items():
            if preview['source_sha256']!=song['original_audio']['sha256'] or abs(preview['source_crop']['requested_start_sec']-(row[side+'_sec']-1.25))>1e-8 or abs(preview['duration_sec']-2.5)>1e-8:
                raise ValueError('edge audio clock mismatch')
    return {'songs':songs,'triplets':triplets}


def assemble(packet: dict,payload: dict,verified: dict) -> dict:
    answers=validate_answers(packet,payload)
    rows=[]
    for item in packet['items']:
        answer=answers.get(item['review_id'])
        if not answer or answer['status']!='confirmed':
            rows.append({'review_id':item['review_id'],'status':'rejected' if answer and answer['status']=='rejected' else 'pending',
                         'raw_answer':deepcopy(answer),'phones':[],'training_eligible':False})
            continue
        choice=next(c for c in item['alternatives'] if c['choice_id']==answer['choice'])
        song=verified['songs'][item['song_id']]
        claim=next(c for c in verified['triplets']['items'] if c['song_id']==item['song_id'])
        clip={'example_id':'batch-'+item['review_id'].lower(),'song_id':item['song_id'],'split':'train',
            'audio_path':song['original_audio']['path'],'audio_sha256':song['original_audio']['sha256'],
            'source_vsqx_sha256':song['human_vsqx']['sha256'],
            'reference_manifest_sha256':packet['source_bindings']['reference']['sha256'],
            'start_sec':item['start_sec'],'end_sec':item['end_sec'],'notes':choice['catalog'],
            'source_sequence_approval_bound':claim['human_vsqx']['sha256']==song['human_vsqx']['sha256'] and claim['human_claims'].get('phoneme_correctness')=='user_asserted_correct'}
        if item['kind']=='retained': clip.update(candidate_origin='retained_human_candidate',retained_review_id=item['review_id'])
        row=assemble_row(clip,answer['edges'],payload['reviewer'])
        row.update(review_id=item['review_id'],raw_answer=deepcopy(answer),selected_choice=choice['choice_id'])
        rows.append(row)
    result={'schema_version':'batch_reviewed_training_reference_v1','packet_sha256':packet['packet_sha256'],
            'answers_sha256':sha256_object(payload),'examples':rows,'training_started':False,
            'confirmed_count':sum(r['status']=='human_confirmed' for r in rows)}
    result['manifest_hash']=sha256_object(result)
    return result


def render_page(directory: Path,packet: dict) -> None:
    verify_sources_and_media(directory,packet)
    audio={}
    for row in packet['items']:
        for preview in row['edge_media'].values():
            name=preview['audio_file'];audio[name]=base64.b64encode((directory/'media'/name).read_bytes()).decode('ascii')
    assets=ROOT/'tools/batch_training_review_gui'
    encoded=lambda value:json.dumps(value,ensure_ascii=False).replace('<',r'\u003c').replace('>',r'\u003e').replace('&',r'\u0026')
    values={'PACKET':encoded(packet),'AUDIO':encoded(audio),'STATE':(assets/'state.js').read_text(encoding='utf-8'),
            'APP':(assets/'app.js').read_text(encoding='utf-8'),'CSS':(assets/'style.css').read_text(encoding='utf-8')}
    html=re.sub(r'/\* INLINE_([A-Z]+) \*/',lambda m:values[m[1]],(assets/'index.html').read_text(encoding='utf-8'))
    with (directory/'index.html').open('x',encoding='utf-8') as file: file.write(html)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',required=True,type=Path)
    parser.add_argument('--answers',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'): parser.error('new run required')
    packet=read_json(args.packet);payload=read_json(args.answers)
    verified=verify_sources_and_media(args.packet.resolve().parent,packet)
    result=assemble(packet,payload,verified)
    output.mkdir(parents=True)
    shutil.copyfile(args.answers,output/'raw-answers.json')
    write_json_new(output/'reviewed-reference.json',result)
    write_json_new(output/'INPUTS.json',{'packet':{'path':str(args.packet.resolve()),'sha256':sha256_file(args.packet)},
        'answers':{'path':str(args.answers.resolve()),'sha256':sha256_file(args.answers)},'training_started':False})
    print('confirmed references:',result['confirmed_count'],'training: not started')

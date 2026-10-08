"""CPU-only independent replay of language probes and offline review packet integrity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file


def distance(a:list[str],b:list[str])->int:
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def audit(run:Path)->None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
    enforce_offline(run/'input-audit-cache')
    from any2xsampa.adapters.powsm_language import PowsmLanguagePosteriorRecognizer
    from any2xsampa.training.phone_error_metrics import phone_metrics
    from any2xsampa.training.whole_sequence_review import validate_answers
    freeze=read_json(run/'FREEZE.json');checks=[]
    refs={r['example_id']:r['target_ipa'] for r in read_json(run/'POWSM_SCORING_REFERENCES.json')['rows']}
    devrefs={r['example_id']:r['target_ipa'] for r in read_json(Path(freeze['parents']['powsm'])/'SCORING_REFERENCES.json')['rows']}
    selection=read_json(run/'POWSM_LANGUAGE_SELECTION.json')
    for folder,language in [('powsm-unknown','<unk>'),('powsm-cantonese','<yue>'),('powsm-cantonese-dev','<yue>')]:
        result=read_json(run/folder/'RESULT.json');pairs=[]
        for path,digest in result['code_bindings'].items():verify_file(Path(path),digest)
        for row in result['entries']:
            pp=Path(row['posterior_path']);verify_file(pp,row['posterior_sha256'])
            payload=read_json(pp)
            if payload['producer']['model_metadata']['language_condition']!=language:raise ValueError('language identity changed')
            timed=True
            try:PowsmLanguagePosteriorRecognizer(pp,fixture=True).recognize(Path(row['request']['audio_path']))
            except ValueError as exc:
                if 'phone emissions in non-audio positions cannot become timed PhoneSegments' not in str(exc):raise
                timed=False
            target=(devrefs if result['development'] else refs)[row['request']['example_id']]
            if row['prediction']!=payload['decoding']['prediction_phones']:raise ValueError('prediction changed')
            pair=(target,row['prediction']);pairs.append(pair)
            metric=phone_metrics([pair])
            if metric['edit_distance']!=distance(*pair):raise ValueError('independent edit distance disagrees')
            checks.append({'folder':folder,'example_id':row['request']['example_id'],'raw_projection_verified':True,
                'timed_contract_available':timed,'errors':metric['edit_distance'],'posterior_sha256':row['posterior_sha256']})
        aggregate=phone_metrics(pairs)
        expected=read_json(run/'POWSM_LANGUAGE_DEVELOPMENT.json')['metrics'] if result['development'] else selection['conditions'][result['condition']]['metrics']
        if aggregate!=expected:raise ValueError('aggregate language metrics changed')
    review=run/'whole-sequence-review';packet=read_json(review/'packet.json');build=read_json(review/'BUILD.json')
    verify_file(review/'packet.json',build['packet_sha256']);verify_file(review/'index.html',build['index_sha256'])
    text=(review/'index.html').read_text(encoding='utf-8')
    embedded=json.loads(re.search(r'<script id="packet" type="application/json">(.*?)</script>',text,re.S).group(1))
    if embedded['packet']!=packet or embedded['packet_sha256']!=build['packet_sha256']:raise ValueError('embedded packet differs')
    media=[]
    for item in packet['items']:
        for role,row in item['assets'].items():
            path=(review/row['file']).resolve()
            if not path.is_relative_to(review.resolve()):raise ValueError('review media path escapes packet')
            verify_file(path,row['sha256']);verify_file(Path(item['source_bindings'][role]['path']),row['sha256'])
            media.append({'review_id':item['review_id'],'role':role,'sha256':row['sha256'],'bit_exact_source_copy':True})
    pending={'schema_version':'whole_sequence_review_answers_v1','packet_sha256':build['packet_sha256'],'answers':[
        {'review_id':r['review_id'],'status':'pending','statement':'fixture pending export',
         'phone_timing_confirmed':False,'source_bindings':r['source_bindings']} for r in packet['items']]}
    validated=validate_answers(packet,pending,packet_sha256=build['packet_sha256'])
    if len(validated['pending'])!=2 or validated['accepted']:raise ValueError('pending export was incorrectly promoted')
    write_json_new(run/'INPUT_AUDIT.json',{'independent_language_replays':checks,'metric_distances_verified':len(checks),
        'review_media':media,'review_packet_sha256':build['packet_sha256'],'pending_export_verified':True,
        'browser_visual_validation':False,'new_human_answers_consumed':False})
    print('verified language outputs',len(checks),'review media',len(media),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    audit(parser.parse_args().run.resolve())

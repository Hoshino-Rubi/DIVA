"""CPU-only complete native-outlet report with alignment-ambiguity accounting."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from input_native_probe_common import ROOT, SONGS, aggregate, prediction
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from any2xsampa.training.phone_error_metrics import align_phone_edits, GROUPS


def optimal_statuses(reference: list, predicted: list) -> list[set[str]]:
    """Statuses possible on ANY minimum-edit path; no fabricated time alignment."""
    n, m = len(reference), len(predicted)
    f = [[0]*(m+1) for _ in range(n+1)]
    b = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): f[i][0]=i; b[i][m]=n-i
    for j in range(m+1): f[0][j]=j; b[n][j]=m-j
    for i in range(1,n+1):
        for j in range(1,m+1):
            f[i][j]=min(f[i-1][j]+1,f[i][j-1]+1,f[i-1][j-1]+(reference[i-1]!=predicted[j-1]))
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            b[i][j]=min(b[i+1][j]+1,b[i][j+1]+1,b[i+1][j+1]+(reference[i]!=predicted[j]))
    best=f[n][m]; result=[set() for _ in reference]
    for i in range(n):
        for j in range(m+1):
            if f[i][j]+1+b[i+1][j]==best: result[i].add('deletion')
            if j<m:
                same=reference[i]==predicted[j]
                if f[i][j]+(not same)+b[i+1][j+1]==best:
                    result[i].add('match' if same else 'substitution')
    if any(not s for s in result): raise ValueError('unreachable reference alignment')
    return result


def migrate(ref: list, earlier: list, later: list) -> dict:
    a,b=optimal_statuses(ref,earlier),optimal_statuses(ref,later)
    rows=[]
    for i,(x,y) in enumerate(zip(a,b)):
        ambiguous=('match' in x and len(x)>1) or ('match' in y and len(y)>1)
        restored=x=={'match'} and 'match' not in y
        lost=y=={'match'} and 'match' not in x
        rows.append({'reference_index':i,'phone':ref[i],'earlier_statuses':sorted(x),'later_statuses':sorted(y),
                     'correctness_ambiguous':ambiguous,'earlier_only_definitely_correct':restored,
                     'later_only_definitely_correct':lost})
    return {'positions':rows,'net_unambiguous_recovery':sum(r['earlier_only_definitely_correct']-r['later_only_definitely_correct'] for r in rows),
            'ambiguous_positions':sum(r['correctness_ambiguous'] for r in rows)}


def analyze(run: Path) -> None:
    from safetensors.numpy import load_file
    from safetensors import safe_open
    freeze=read_json(run/'FREEZE.json')
    for path,h in freeze['bindings'].items():verify_file(Path(path),h)
    refs={r['window_id']:r for r in read_json(run/'REFERENCES.json')['rows']}
    vocab=read_json(Path(freeze['vocabulary']['path']))
    source={r['example_id']:r for r in read_json(Path(freeze['dual_run'])/'SCORING_REFERENCES.json')['rows'] if r['split']=='train'}
    b1=read_json(run/'B1/RESULT.json')
    if b1['passed'] is not True or len(b1['records'])!=60:raise ValueError('B1 incomplete')
    pools=defaultdict(list); identities={}; bindings={}; cache=read_json(Path(freeze['probe_run'])/'CACHE.json')
    def bind(path,digest=None):
        path=Path(path);actual=sha256_file(path)
        if digest is not None and actual!=digest:raise ValueError('evidence hash mismatch: '+str(path))
        bindings[str(path)]=actual
    def decode(artifact):
        pp=Path(artifact['posterior_path']);bind(pp,artifact['posterior_sha256']);payload=read_json(pp)
        raw=pp.parent/payload['raw_evidence']['file'];bind(raw,payload['raw_evidence']['sha256'])
        pred=prediction(load_file(str(raw))['frame_logits'],vocab)
        if pred!=artifact['prediction']:raise ValueError('saved prediction does not replay')
        return pred,payload
    for record in b1['records']:
        wid,c=record['task_id'].rsplit('--',1);ref=refs[wid]
        for layer,artifact in record['outputs'].items():
            pred,payload=decode(artifact)
            identities[wid,c]=(payload['producer']['input_audio'],payload['producer']['crop'])
            pools['P'+layer,c].append({'window_id':wid,'song_id':ref['song_id'],'reference':ref['target_ipa'],'prediction':pred})
    # M/MV have only last-four-block checkpoint deltas; hash and tensor names prove
    # no native layer/head weights were substituted. Never load them into G.
    checkpoints=[]
    probe=Path(freeze['probe_run']); mv=Path(freeze['mv_run'])
    history=ROOT/'.state/runs/singing-domain-20260920T095918143867Z'
    for variant,base in [('M',history/'architecture/full'),('MV',mv/'screen/MV')]:
        for song in SONGS:
            rp=base/song/'RESULT.json';bind(rp);r=read_json(rp)
            if r['held_song']!=song or r['frozen_parameters_unchanged'] is not True or r['architecture']['sha256']!=b1['architecture']['sha256']:
                raise ValueError('historical frozen architecture/holdout identity mismatch')
            cp=Path(r['checkpoint_path']);bind(cp,r['checkpoint_sha256'])
            with safe_open(str(cp),framework='np') as stream: keys=list(stream.keys())
            if not keys or any(not any(k.startswith(f'blocks.{i}.') for i in range(4)) for k in keys):
                raise ValueError('checkpoint modifies something beyond last four blocks')
            checkpoints.append({'variant':variant,'song_id':song,'path':str(cp),'sha256':r['checkpoint_sha256'],
                'tensor_count':len(keys),'native_layers_and_head_excluded':True,'architecture_sha256':r['architecture']['sha256']})
    old=read_json(probe/'RESULT.json');pc={r['condition_id']:r for r in cache['entries']}
    oldmv=read_json(mv/'independent-replay/VERIFICATION.json')
    comparable=[]
    for c in ('x','v','h'):
        for row in old['details'][c]:
            bind(row['logits_path'],row['logits_sha256'])
            pred=prediction(load_file(row['logits_path'])['frame_logits'],vocab)
            e=pc[row['condition_id']];identity=(e['audio_metadata'],e['crop'])
            if pred!=row['prediction'] or row['reference']!=refs[row['window_id']]['target_ipa']:
                raise ValueError('M prediction/reference mismatch')
            same=identity==identities[row['window_id'],c]
            comparable.append({'variant':'M','window_id':row['window_id'],'condition':c,'directly_comparable':same})
            if same:pools['M',c].append(row)
    for job in oldmv['checkpoints']:
        if job['variant']!='MV':continue
        for row in job['details']:
            pred,payload=decode(row);wid=row['window_id'];c=row['condition']
            if row['reference']!=refs[wid]['target_ipa']:raise ValueError('MV reference mismatch')
            identity=(payload['producer']['input_audio'],payload['producer']['crop'])
            same=identity==identities[wid,c]
            comparable.append({'variant':'MV','window_id':wid,'condition':c,'directly_comparable':same,
                'reason':None if same else 'original_full_source_crop_is_not_reencoded_paired_x'})
            if same:pools['MV',c].append(row)
    matrix={};strata=defaultdict(Counter);vowels=set('aeiouy\u0250\u0259\u0254\u025b\u0153\u00f8\u026a\u028a')
    for (model,c),rows in pools.items():
        if len(rows)!=20:raise ValueError('do not silently aggregate partial cells')
        met=aggregate(rows)
        if met['reference_phones']!=1235:raise ValueError('scoring denominator changed')
        matrix[model+'('+c+')']={'metrics':met,'per_song':{s:aggregate([r for r in rows if r['song_id']==s]) for s in SONGS},'details':rows}
        for row in rows:
            phones=source[row['window_id']]['source_phones'];notes=defaultdict(list)
            if [p['phone'] for p in phones]!=row['reference']:raise ValueError('note reference identity changed')
            for p in phones:notes[p['note_id']].append(p['phone'])
            edges={phones[0]['note_id'],phones[-1]['note_id']};pos=0
            for op,expected,actual in align_phone_edits(row['reference'],row['prediction']):
                if expected is None:continue
                p=phones[pos];pos+=1
                kind='multi_vowel' if sum(any(ch in vowels for ch in v) for v in notes[p['note_id']])>=2 else 'other'
                key=(model,c,row['song_id'],'edge' if p['note_id'] in edges else 'interior',kind,expected)
                strata[key]['reference']+=1;strata[key][op]+=1
    migration=[];decisions=[]
    # h is primary; x/v are explanatory. No per-window best layer or oracle.
    for c in ('h','x','v'):
        look={(model,r['window_id']):r for (model,condition),rows in pools.items() if condition==c for r in rows}
        for layer in (4,8,12):
            for later in ('P19','M','MV'):
                if (later,c) not in pools:continue
                songcounts=defaultdict(Counter);phonecounts=defaultdict(Counter)
                for wid,ref in refs.items():
                    evidence=migrate(ref['target_ipa'],look['P'+str(layer),wid]['prediction'],look[later,wid]['prediction'])
                    migration.append({'condition':c,'earlier':'P'+str(layer),'later':later,'window_id':wid,'song_id':ref['song_id'],**evidence})
                    for p in evidence['positions']:
                        for counter in (songcounts[ref['song_id']],phonecounts[(ref['song_id'],p['phone'])]):
                            counter['earlier_only']+=p['earlier_only_definitely_correct'];counter['later_only']+=p['later_only_definitely_correct'];counter['ambiguous']+=p['correctness_ambiguous']
                nets={s:v['earlier_only']-v['later_only'] for s,v in songcounts.items()}
                positive=[s for s,n in nets.items() if n>0]
                robust=sum(nets.values())>0 and len(positive)>=3 and sum(nets.values())-max(nets.values())>0
                decisions.append({'condition':c,'earlier':'P'+str(layer),'later':later,'song_counts':dict(songcounts),
                    'unambiguous_net_by_song':nets,'strict_positive_songs':positive,'net_not_single_song_driven':robust,
                    'primary_input':c=='h','eligible_for_separate_proposal_not_training':c=='h' and robust,
                    'phone_strata':[{'song_id':k[0],'phone':k[1],**v} for k,v in sorted(phonecounts.items())]})
    write_json_new(run/'B_ANALYSIS.json',{'schema_version':'native_outlet_diagnostic_v1','matrix':matrix,
        'input_comparability':comparable,'checkpoint_identity_audit':checkpoints,'source_bindings':bindings,
        'native_outputs':240,'reference_phones_each_cell':1235,'source_reference_unchanged':True,
        'note_phone_strata':[dict(model=k[0],condition=k[1],song_id=k[2],edge=k[3],note_kind=k[4],phone=k[5],**v) for k,v in sorted(strata.items())],
        'migration':migration,'route_evidence':decisions,'new_training':False,'new_test_predictions':False,
        'limitations':['Output readability is not absence of hidden information.','Reference edit paths are not acoustic timestamps.',
            'All references remain scored; ambiguous match/error positions excluded only from directional evidence.',
            'MV(x) is not directly comparable: original source crop differs from paired recoded x. No extra inference added.']})
    for name,r in matrix.items():
        m=r['metrics'];print(name,m['edit_distance'],'S/D/I',m['substitutions'],m['deletions'],m['insertions'])
    print('PRIMARY_ROUTE',[(d['earlier'],d['later'],d['unambiguous_net_by_song'],d['eligible_for_separate_proposal_not_training']) for d in decisions if d['primary_input']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run.resolve())

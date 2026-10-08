"""Evaluate frozen GAME and pYIN outputs; never infer, shift, warp, or rewrite them."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.local.common import read,sha,write
from any2xsampa.music.note_metrics import NoteEvent,evaluate_notes
from any2xsampa.adapters.game_onnx import replay_result
from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline


def sequence_errors(reference: list[int],prediction: list[int]) -> dict:
    n,m=len(reference),len(prediction)
    cost=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):cost[i][0]=i
    for j in range(m+1):cost[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):cost[i][j]=min(cost[i-1][j-1]+(reference[i-1]!=prediction[j-1]),cost[i-1][j]+1,cost[i][j-1]+1)
    i,j=n,m;counts={'substitutions':0,'deletions':0,'insertions':0,'exact_pitch_pairs':0};ops=[]
    while i or j:
        if i and j and cost[i][j]==cost[i-1][j-1]+(reference[i-1]!=prediction[j-1]):
            op='equal' if reference[i-1]==prediction[j-1] else 'substitution'
            counts['exact_pitch_pairs' if op=='equal' else 'substitutions']+=1
            ops.append({'operation':op,'reference_index':i-1,'prediction_index':j-1,'reference_pitch':reference[i-1],'predicted_pitch':prediction[j-1]});i-=1;j-=1
        elif i and cost[i][j]==cost[i-1][j]+1:
            counts['deletions']+=1;ops.append({'operation':'deletion','reference_index':i-1,'reference_pitch':reference[i-1]});i-=1
        else:
            counts['insertions']+=1;ops.append({'operation':'insertion','prediction_index':j-1,'predicted_pitch':prediction[j-1]});j-=1
    return {'reference_notes':n,'predicted_notes':m,'edit_distance':cost[n][m],**counts,
            'sequence_error_rate':cost[n][m]/n if n else None,'timing_evaluated':False,'operations':list(reversed(ops))}


def evaluate(run: Path,reference_manifest: Path,output: Path) -> dict:
    selection=read(run/'SELECTED_INPUTS.json');review=read(run/'BASELINE_REVIEW.json')
    if sha(run/'SELECTED_INPUTS.json')!=review['input_sha256']:raise ValueError('Selection changed')
    reference_file=Path(selection['reference_file'])
    if sha(reference_file)!=selection['reference_sha256']:raise ValueError('Reference changed')
    examples={x['example_id']:x for x in read(reference_file)['examples']}
    source=read(reference_manifest);songs={x['song_id']:x for x in source['songs']}
    audit={x['song']:x for x in read(run/'REFERENCE_CLOCK_AUDIT.json')['rows']}
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for job in selection['jobs']:
        if not job['id'].startswith('original-'):continue
        ex=examples[job['source_example_id']];song=songs[job['song_id']]
        if ex['split']!='train' or song['split']!='train':raise ValueError('Only frozen train songs may enter qualification')
        if ex['reference_manifest_sha256']!=sha(reference_manifest):raise ValueError('Reference manifest differs')
        spec=song['human_vsqx'];file=Path(spec.get('normalized_path',spec['path']))
        if sha(file)!=spec.get('normalized_sha256',spec['sha256']) or ex['source_vsqx_sha256']!=spec['sha256']:
            raise ValueError('Source VSQX hash differs')
        timeline=parse_vsqx_timeline(file)['notes']
        if len(timeline)!=len(song['notes']):raise ValueError('Note indexing changed')
        by_id={}
        for note,source_note in zip(song['notes'],timeline,strict=True):
            if abs(note['start_sec']-source_note['start_sec'])>1e-5 or abs(note['end_sec']-source_note['end_sec'])>1e-5:
                raise ValueError('VSQX note clock/index changed')
            by_id[note['note_id']]=source_note
        refs=[{'note_id':key,**by_id[key]} for key in ex['note_ids']]
        prediction_path=Path(job['output'])/'prediction.json'
        if sha(prediction_path)!=review['predictions'][job['id']]:raise ValueError('Unfrozen model prediction')
        candidate=read(prediction_path)
        basefile=run/'baseline'/job['id']/'prediction.json';base=read(basefile)
        if base['game_prediction_sha256']!=sha(prediction_path) or base['input']!=candidate['input']:
            raise ValueError('Controls do not use the same input')
        replay_result(prediction_path,output/(job['id']+'-replay.json'))
        record={'song_id':job['song_id'],'example_id':ex['example_id'],
                'reference_vsqx_path':str(file),'reference_vsqx_sha256':sha(file),
                'reference_notes':refs,'reference_claims':song['source_claims'],'clock_audit':audit[job['song_id']],
                'source_approval_scope':'calibrated phone sequence and coarse music claims; not new millisecond boundary annotations',
                'prediction_sha256':sha(prediction_path),'baseline_sha256':sha(basefile),'systems':{}}
        for name,pred in [('game',candidate),('old_pyin',base)]:
            notes=pred['notes']
            sequence=sequence_errors([n['pitch'] for n in refs],[n['pitch_midi'] for n in notes])
            data={'pitch_sequence':sequence,'note_count':len(notes)}
            if audit[job['song_id']]['clock_plausible']:
                start,end=ex['start_sec'],ex['end_sec']
                # Every reference note in the frozen correspondence must overlap;
                # edge clipping is recorded and shared by both systems.
                inside=[n for n in refs if n['end_sec']>start and n['start_sec']<end]
                ref_events=[NoteEvent(max(start,n['start_sec']),min(end,n['end_sec']),n['pitch']) for n in inside]
                est=[NoteEvent(n['start_sec'],n['end_sec'],n['pitch_midi']) for n in notes]
                data['project_clock_conditional']=evaluate_notes(ref_events,est,reference_clock_confirmed=False)
                data['edge_clipped_reference_notes']=[n['note_id'] for n in inside if n['start_sec']<start or n['end_sec']>end]
                data['outside_reference_note_ids']=[n['note_id'] for n in refs if n not in inside]
            else:
                data['timing_not_scored_reason']='Known audio/VSQX clock offset; no confirmed exact transformation. No optimized time shift was fitted.'
            record['systems'][name]=data
        records.append(record)
    totals={}
    for name in ('game','old_pyin'):
        keys=('reference_notes','predicted_notes','edit_distance','substitutions','deletions','insertions','exact_pitch_pairs')
        sums={key:sum(r['systems'][name]['pitch_sequence'][key] for r in records) for key in keys}
        sums['sequence_error_rate']=sums['edit_distance']/sums['reference_notes']
        totals[name]=sums
    report={'schema_version':'game_note_qualification_comparison_v1','records':records,'pitch_sequence_totals':totals,
            'formal_original_timing_windows':0,'joint_quality_target_passed':False,
            'reason':'Exact original note timing clock not established; sequence errors are diagnostics only.',
            'default_model_changed':False,'new_model_forwards_this_evaluation':0}
    write(output/'comparison.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=evaluate(a.run,a.reference,a.output)
    print(json.dumps({'totals':report['pitch_sequence_totals'],'timing_windows':report['formal_original_timing_windows'],
        'songs':[{k:r[k] for k in ('song_id','example_id')}|{'game_errors':r['systems']['game']['pitch_sequence']['edit_distance'],'old_errors':r['systems']['old_pyin']['pitch_sequence']['edit_distance']} for r in report['records']]},ensure_ascii=False))
if __name__=='__main__':main()

"""Post-experiment comparisons; never consulted for training or checkpoint selection."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Sequence
import math
from any2xsampa.training.phone_error_metrics import phone_metrics


def scalar_distance(reference:Sequence[str],prediction:Sequence[str])->int:
    previous=list(range(len(prediction)+1))
    for i,expected in enumerate(reference,1):
        current=[i]
        for j,actual in enumerate(prediction,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(expected!=actual)))
        previous=current
    return previous[-1]


def compare_systems(systems:dict[str,list[dict]])->dict[str,Any]:
    if len(systems)<2:raise ValueError('paired analysis needs at least two systems')
    canonical=None;lookups={};totals={}
    for name,examples in systems.items():
        rows={r['example_id']:r for r in examples}
        if len(rows)!=len(examples):raise ValueError('duplicate comparison example')
        identities={k:(r['song_id'],tuple(r['reference'])) for k,r in rows.items()}
        if canonical is not None and identities!=canonical:
            raise ValueError('paired comparison changed reference identity or denominator')
        canonical=identities;lookups[name]=rows;pairs=[]
        for row in examples:
            pair=(row['reference'],row['prediction']);metric=phone_metrics([pair])
            if scalar_distance(*pair)!=metric['edit_distance']:
                raise ValueError('independent edit count mismatch')
            if row.get('metrics') is not None and row['metrics']!=metric:
                raise ValueError('saved per-window metric differs from replayed prediction')
            pairs.append(pair)
        totals[name]=phone_metrics(pairs)
    by_song={};paired=[]
    for example_id,(song,_) in canonical.items():
        record={'example_id':example_id,'song_id':song,'reference_phones':len(lookups[next(iter(systems))][example_id]['reference'])}
        record['systems']={name:phone_metrics([(rows[example_id]['reference'],rows[example_id]['prediction'])]) for name,rows in lookups.items()}
        paired.append(record)
    for song in sorted({song for song,_ in canonical.values()}):
        by_song[song]={name:phone_metrics([(row['reference'],row['prediction']) for row in rows.values() if row['song_id']==song])
                       for name,rows in lookups.items()}
    return {'systems':totals,'by_song':by_song,'paired_windows':sorted(paired,key=lambda r:(r['song_id'],r['example_id'])),
        'independent_scalar_distances_verified':sum(len(rows) for rows in lookups.values()),
        'same_references_and_windows':True,'timing_accuracy_claimed':False}


def source_coverage(short:list[dict],whole:list[dict])->dict:
    intervals=defaultdict(list);short_intervals=defaultdict(list)
    for row in short:
        if row['split']!='train' or row['audio_role']!='original':
            raise ValueError('coverage requires original training audio')
        key=(row['song_id'],row['audio_sha256'])
        start,end=row['start_sec'],row['end_sec']
        if not all(math.isfinite(v) for v in (start,end)) or not 0<=start<end:
            raise ValueError('invalid coverage interval')
        intervals[key].append((start,end));short_intervals[key].append((start,end))
    for row in whole:
        key=(row['song_id'],row['audio_sha256'])
        if any(song==row['song_id'] and digest!=row['audio_sha256'] for song,digest in intervals):
            raise ValueError('whole and short audio versions differ')
        if not math.isfinite(row['duration_sec']) or row['duration_sec']<=0:raise ValueError('invalid whole duration')
        intervals[key].append((0.,row['duration_sec']))
    def merged(values:list[tuple[float,float]])->float:
        start,end=sorted(values)[0];total=0.
        for left,right in sorted(values)[1:]:
            if left<=end:end=max(end,right)
            else:total+=end-start;start,end=left,right
        return total+end-start
    detail=[{'song_id':song,'audio_sha256':digest,'unique_original_seconds':merged(values),
             'previous_unique_original_seconds':merged(short_intervals[(song,digest)]) if short_intervals[(song,digest)] else 0.}
            for (song,digest),values in sorted(intervals.items())]
    before=sum(r['previous_unique_original_seconds'] for r in detail)
    after=sum(r['unique_original_seconds'] for r in detail)
    return {'by_song':detail,'previous_unique_original_seconds':before,'unique_original_seconds':after,
        'additional_unique_original_seconds':after-before,
        'includes_instrumental_and_pause_audio':True,'all_seconds_are_not_claimed_vocal_or_new_annotations':True}

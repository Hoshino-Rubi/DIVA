"""Fixed H/E screen gates; all comparisons require identical references."""
from __future__ import annotations
from collections import Counter
from typing import Sequence
from any2xsampa.training.phone_error_metrics import phone_metrics


def aggregate(rows: Sequence[dict]) -> dict:
    return phone_metrics([(r['reference'],r['prediction']) for r in rows])


def same_references(rows: Sequence[dict],baseline: Sequence[dict]) -> None:
    def identity(rr):
        if len({r['example_id'] for r in rr})!=len(rr):raise ValueError('duplicate evaluation window')
        return {r['example_id']:(r['song_id'],r['reference']) for r in rr}
    if identity(rows)!=identity(baseline):raise ValueError('different evaluation references')


def group_regressions(new: dict,old: dict,minimum: int=20,tolerance: float=.03) -> list[str]:
    if set(new['groups'])!=set(old['groups']):raise ValueError('group inventory mismatch')
    bad=[]
    for name,base in old['groups'].items():
        group=new['groups'][name]
        if group['reference_count']!=base['reference_count']:raise ValueError('group denominator changed')
        if base['reference_count']>=minimum and group['per']>base['per']+tolerance:bad.append(name)
    return bad


def compare_songs(rows: Sequence[dict],baseline: Sequence[dict]) -> dict:
    same_references(rows,baseline)
    songs=sorted({r['song_id'] for r in rows})
    a={s:aggregate([r for r in rows if r['song_id']==s]) for s in songs}
    b={s:aggregate([r for r in baseline if r['song_id']==s]) for s in songs}
    return {'current':a,'baseline':b,'improved':[s for s in songs if a[s]['edit_distance']<b[s]['edit_distance']],
        'macro_nonregressing':sum(m['per'] for m in a.values())<=sum(m['per'] for m in b.values())}


def screen_gate(arm: str,original: list[dict],rendered: list[dict],old_original: list[dict],old_rendered: list[dict],
                rules: dict,*,h_original: list[dict]|None=None) -> dict:
    same_references(original,old_original);same_references(rendered,old_rendered)
    new,old=aggregate(original),aggregate(old_original);synth,oldsynth=aggregate(rendered),aggregate(old_rendered)
    if len(original)!=20 or new['reference_phones']!=1235 or len(rendered)!=129 or old['edit_distance']!=595:
        raise ValueError('frozen original/rendered cohort or endpoint control changed')
    songs=compare_songs(original,old_original);fail=[]
    for metric,bound in [('edit_distance','maximum_errors'),('insertions','maximum_insertions'),('deletions','maximum_deletions')]:
        if new[metric]>rules[bound]:fail.append(metric+'_over_limit')
    if len(songs['improved'])<rules['minimum_improved_songs_vs_endpoint']:fail.append('fewer_than_three_songs_improve_vs_endpoint')
    if not songs['macro_nonregressing']:fail.append('macro_regression_vs_endpoint')
    fail.extend('original_'+name+'_regression' for name in group_regressions(new,old))
    if synth['per']>oldsynth['per']+.03:fail.append('rendered_overall_regression')
    fail.extend('rendered_'+name+'_regression' for name in group_regressions(synth,oldsynth))
    comparison=None
    if arm=='E':
        if h_original is None:raise ValueError('event arm requires complete H control')
        same_references(original,h_original);h=aggregate(h_original);hs=compare_songs(original,h_original)
        # Integer inequality prevents a rounded 4.98% from passing 5%.
        if new['edit_distance']*100>h['edit_distance']*95:fail.append('E_gain_vs_H_below_5pct')
        if len(hs['improved'])<3:fail.append('E_fewer_than_three_songs_improve_vs_H')
        if not hs['macro_nonregressing']:fail.append('E_macro_regression_vs_H')
        for metric in ('insertions','deletions'):
            if new[metric]>h[metric]:fail.append('E_'+metric+'_regression_vs_H')
        fail.extend('E_'+name+'_regression_vs_H' for name in group_regressions(new,h))
        comparison={'baseline_H':h,'by_song':hs,'relative_error_improvement':(h['edit_distance']-new['edit_distance'])/h['edit_distance'] if h['edit_distance'] else None}
    elif arm!='H':raise ValueError('unknown arm')
    return {'passed':not fail,'failures':fail,'original':new,'rendered':synth,'endpoint_original':old,
        'endpoint_rendered':oldsynth,'per_song':songs,'event_vs_H':comparison,'development_used':False,
        'new_independent_generalization_claim':False,'automatic_default_replacement':False}

"""Development comparisons remain separate from the train-song research selection."""
from __future__ import annotations
from typing import Mapping,Any
import math

RELEASE_RULES={'maximum_original_errors':87,'maximum_original_insertions':11,'maximum_original_deletions':25,
    'maximum_aspirated_errors':1,'maximum_h_errors':2,'minimum_group_references':20,
    'maximum_rendered_per_increase':.03,'maximum_group_per_increase':.03,'minimum_passing_seeds':2,
    'selection':'original_development_per_then_seed','automatic_default_replacement':False}

def release_check(original:Mapping[str,Any],rendered:Mapping[str,Any],old_original:Mapping[str,Any],old_rendered:Mapping[str,Any])->dict:
    for metric in (original,rendered,old_original,old_rendered):
        for key in ('reference_phones','edit_distance','insertions','deletions','substitutions'):
            if isinstance(metric[key],bool) or not isinstance(metric[key],int) or metric[key]<0:raise ValueError('invalid metric count')
        if not metric['reference_phones'] or not math.isfinite(metric['per']) or abs(metric['per']-metric['edit_distance']/metric['reference_phones'])>1e-12:raise ValueError('invalid PER')
        if metric['edit_distance']!=sum(metric[k] for k in ('insertions','deletions','substitutions')):raise ValueError('inconsistent error totals')
    if original['reference_phones']!=214 or old_original['reference_phones']!=214 or rendered['reference_phones']!=679 or old_rendered['reference_phones']!=679:
        raise ValueError('development denominator changed')
    failures=[];rules=RELEASE_RULES
    for metric,bound in [('edit_distance','maximum_original_errors'),('insertions','maximum_original_insertions'),('deletions','maximum_original_deletions')]:
        if original[metric]>rules[bound]:failures.append('original_'+metric)
    for group,key in [('aspirated','maximum_aspirated_errors'),('glottal_h','maximum_h_errors')]:
        g=original['groups'][group]
        if sum(g[k] for k in ('substitutions','deletions','insertions'))>rules[key]:failures.append('original_'+group)
    if rendered['per']>old_rendered['per']+rules['maximum_rendered_per_increase']:failures.append('rendered_overall')
    for cohort,new,old in [('original',original,old_original),('rendered',rendered,old_rendered)]:
        for group,base in old['groups'].items():
            if base['reference_count']>=rules['minimum_group_references'] and new['groups'][group]['per']>base['per']+rules['maximum_group_per_increase']:
                failures.append(cohort+'_'+group+'_protected')
    return {'passed':not failures,'failures':failures,'rules':rules,'default_replacement':False,
        'insufficient_original_groups':[name for name,g in original['groups'].items() if g['reference_count']<rules['minimum_group_references']]}

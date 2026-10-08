import pytest
from any2xsampa.training.event_head_evaluation import screen_gate,same_references,group_regressions


def originals(errors):
    rows=[]
    for i in range(20):
        n=62 if i<15 else 61;wrong=errors//20+(i<errors%20)
        rows.append({'example_id':str(i),'song_id':'song'+str(i%5),'reference':['i']*n,'prediction':['z']*wrong+['i']*(n-wrong)})
    return rows


def rendered():
    return [{'example_id':'r'+str(i),'song_id':'song'+str(i%5),'reference':['i','u'],'prediction':['i','u']} for i in range(129)]


RULES={'maximum_errors':564,'maximum_insertions':32,'maximum_deletions':248,'minimum_improved_songs_vs_endpoint':3}


def test_H_and_E_need_absolute_and_matched_comparator_gain():
    base=originals(595);synth=rendered();h=originals(564)
    assert screen_gate('H',h,synth,base,synth,RULES)['passed']
    assert screen_gate('E',originals(530),synth,base,synth,RULES,h_original=h)['passed']
    g=screen_gate('E',originals(536),synth,base,synth,RULES,h_original=h)
    assert not g['passed'] and 'E_gain_vs_H_below_5pct' in g['failures']
    assert not screen_gate('H',originals(565),synth,base,synth,RULES)['passed']


def test_no_partial_fold_or_reference_rebinding():
    with pytest.raises(ValueError):same_references(originals(590)[:19],originals(590))
    a=originals(590);a[0]['song_id']='test'
    with pytest.raises(ValueError):same_references(a,originals(590))
    with pytest.raises(ValueError):same_references(a+[a[0]],a+[a[0]])


def test_group_protection_not_replaced_with_overall_gain():
    old={'groups':{'aspirated':{'reference_count':30,'per':14/30}}}
    new={'groups':{'aspirated':{'reference_count':30,'per':15/30}}}
    assert group_regressions(new,old)==['aspirated']
    new['groups']['aspirated']['reference_count']=29
    with pytest.raises(ValueError):group_regressions(new,old)

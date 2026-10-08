from pathlib import Path
import copy
import pytest
from any2xsampa.training.decoder_experiment import promotion_gate, validate_cohort, reference_edits
from any2xsampa.training.limited_data_artifacts import read_json
from any2xsampa.training.phone_error_metrics import phone_metrics


def policy():
    return read_json(Path(__file__).parents[1]/'src/any2xsampa/config/ctc_prefix_experiment_v1.json')


def cohorts():
    result=[]
    for role,split,count,song in [('original','train',20,'promise'),('original','dev',3,'shalong'),('rendered','train',129,'promise'),('rendered','dev',25,'shalong')]:
        for i in range(count):
            result.append(dict(cache_id=f'{role}_{split}_{i}',cohort=f'{role}_{split}',audio_role=role,split=split,song_id=song))
    return result


@pytest.mark.parametrize('mutation', ['test','role','duplicate','count'])
def test_cohort_rejects_leak_role_and_stale_membership(mutation):
    rows=cohorts();validate_cohort(rows)
    if mutation=='test': rows[0]['song_id']='king-of-karaoke'
    elif mutation=='role': rows[0]['audio_role']='rendered'
    elif mutation=='duplicate': rows[1]['cache_id']=rows[0]['cache_id']
    else: rows.pop()
    with pytest.raises(ValueError): validate_cohort(rows)


def test_improved_per_cannot_hide_extra_deletions_or_class_regression():
    metrics=phone_metrics([(['h']*25+['s']*75,['h']*20+['s']*50)])
    base={'original_dev':metrics,'rendered_dev':metrics}
    trial=copy.deepcopy(base)
    trial['original_dev']['per']=metrics['per']*.8
    trial['original_dev']['deletions']+=1
    trial['rendered_dev']['groups']['glottal_h']['per']+=.04
    result=promotion_gate(base,trial,policy())
    assert not result['passed']
    assert 'original_deletions_increased' in result['failures']
    assert 'rendered_dev:glottal_h:regression' in result['failures']


def test_reference_change_is_not_scoring_improvement():
    metrics=phone_metrics([(['h','s'],['s'])]);base={'original_dev':metrics,'rendered_dev':metrics}
    trial=copy.deepcopy(base);trial['original_dev']['reference_phones']=1
    with pytest.raises(ValueError):promotion_gate(base,trial,policy())


def test_edit_attribution_retains_repeats_and_insertions():
    result=reference_edits(['a','a','t'],['a','a','s','t'])
    assert [r['reference_index'] for r in result if r['reference'] is not None]==[0,1,2]
    assert [r['prediction_index'] for r in result if r['prediction'] is not None]==[0,1,2,3]
    assert sum(r['operation']=='insertion' for r in result)==1

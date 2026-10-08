from pathlib import Path
import random
import pytest
from any2xsampa.training.event_head_protocol import epoch_schedule,fold_steps,check_job
from any2xsampa.training.limited_data_artifacts import write_json_new


def test_schedule_keeps_old_unused_rng_consumption_and_isolation():
    rows=[dict(example_id=str(i),song_id='a' if i<4 else 'b',split='train',audio_role='original' if i%2 else 'rendered') for i in range(9)]
    unl=[{'example_id':'u'+str(i),'song_id':'a'} for i in range(8)]
    actual=epoch_schedule(rows,unl,'b',17,3);rng=random.Random(17);expected=[]
    for _ in range(3):
        ids=[]
        for r in rows:
            if r['song_id']!='b':ids.extend([r['example_id']]*(4 if r['audio_role']=='original' else 1))
        rng.shuffle(ids);other=[r['example_id'] for r in unl];rng.shuffle(other);expected.append(ids)
    assert actual==expected and all(set(x)<=set('0123') for x in actual)
    assert fold_steps(actual)==6


def test_nontraining_row_rejected():
    with pytest.raises(ValueError):epoch_schedule([dict(example_id='test',split='test',song_id='a',audio_role='original')],[],None,17)


def test_retry_budget_and_preflight_cost_are_enforced(tmp_path):
    spec={'arms':['H','E'],'songs':['s']}
    write_json_new(tmp_path/'preflight/RESULT.json',{'passed':True,'cost_gate':{'H':{'passed':True},'E':{'passed':False}}})
    check_job(tmp_path,'H','s',spec)
    with pytest.raises(ValueError):check_job(tmp_path,'E','s',spec)
    with pytest.raises(ValueError):check_job(tmp_path,'H','s',spec,worker=True)
    write_json_new(tmp_path/'commands/H--s.command.json',{})
    check_job(tmp_path,'H','s',spec,worker=True)
    with pytest.raises(ValueError):check_job(tmp_path,'H','s',spec)
    write_json_new(tmp_path/'screen/H/s/START.json',{})
    with pytest.raises(ValueError):check_job(tmp_path,'H','s',spec,worker=True)

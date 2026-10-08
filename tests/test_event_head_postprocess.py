from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from finish_event_head_screen import readiness
from any2xsampa.training.limited_data_artifacts import write_json_new


def test_pending_training_is_not_restarted(tmp_path):
    s={'arms':['H','E'],'songs':['s']}
    r=readiness(tmp_path,s);assert not r['ready'] and not r['blocked'] and len(r['missing'])==2
    write_json_new(tmp_path/'commands/H--s.exit.json',{'exit_code':0});write_json_new(tmp_path/'screen/H/s/RESULT.json',{'complete':True})
    assert readiness(tmp_path,s)['missing']==['E--s']


def test_unrecovered_failure_blocks_postprocess(tmp_path):
    write_json_new(tmp_path/'commands/E--s.exit.json',{'exit_code':73})
    assert readiness(tmp_path,{'arms':['E'],'songs':['s']})['blocked']


def test_only_documented_readonly_recovery_allowed(tmp_path):
    write_json_new(tmp_path/'commands/E--freedom-flower.exit.json',{'exit_code':73})
    write_json_new(tmp_path/'screen/E/freedom-flower/RESULT.json',{'complete':True,'quality_training_restarted':False})
    assert readiness(tmp_path,{'arms':['E'],'songs':['freedom-flower']})['ready']

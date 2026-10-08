import copy
import pytest
from any2xsampa.training.phone_diagnostic_review import validate_answers
from any2xsampa.training.limited_data_artifacts import sha256_object


def sample():
 p={'schema_version':'phone_diagnostic_review_packet_v1','items':[{'review_id':'D01','note_id':'promise-n001','song_id':'promise','reference_update_allowed':False,'choices':{'a':['u','e'],'b':['e']},'choice_sources':{'a':'proxy_reference','b':'candidate'}}]}
 p['packet_sha256']=sha256_object(p)
 a={'schema_version':'phone_diagnostic_review_answers_v1','packet_sha256':p['packet_sha256'],'scope':'diagnostic_only_no_reference_or_training_update','responses':[{'review_id':'D01','decision':'pending','statement':'原话'}]}
 return p,a


@pytest.mark.parametrize('decision',['pending','a','b','neither','uncertain','wrong_location'])
def test_every_allowed_answer_preserves_quote_without_promoting_reference(decision):
 p,a=sample();a['responses'][0]['decision']=decision;r=validate_answers(p,a)[0]
 assert r['statement']=='原话' and not r['formal_reference_created'] and not r['training_allowed']
 if decision=='b':assert r['selected_ipa']==['e']


def test_forged_packet_missing_or_foreign_answers_rejected():
 p,a=sample();p['items'][0]['choices']['a']=['x']
 with pytest.raises(ValueError):validate_answers(p,a)
 p,a=sample();a['responses']=[]
 with pytest.raises(ValueError):validate_answers(p,a)
 p,a=sample();a['responses'][0]['review_id']='D02'
 with pytest.raises(ValueError):validate_answers(p,a)

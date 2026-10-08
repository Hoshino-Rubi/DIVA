from copy import deepcopy
import pytest
from tools.build_replacement_location_review import choose_questions, alternatives, validate_answers
from test_replacement_candidates import fixtures
from any2xsampa.training.limited_data_artifacts import sha256_object


def test_location_questions_only_use_single_disagreement_failures():
    valid=dict(song_id='a',reasons=['mixed_vocal_paths_disagree'],dual_path_disagreement_sec=.5,
               center_distance_sec=0,context_id='c',start_sec=1)
    bad={**valid,'dual_path_disagreement_sec':.1,'reasons':['mixed_vocal_paths_disagree','whole_window_unresolved_mapping']}
    assert choose_questions([bad,valid],['a','b'])==[valid]


def test_alternatives_deduplicate_and_never_reoffer_rejected_pair():
    context,song,paths,_=fixtures()
    song['rendered_audio']={'duration_sec':100}
    row={**context,'start_sec':5,'end_sec':15}
    first={**paths['vocal'],'rank':1}
    second=deepcopy(first);second['rank']=2
    for a in second['path_anchors']:
        a['rendered_sec']+=20
    alignment={'paths':{r:{'candidates':[first,second]} for r in ('mixed','vocal')}}
    old=dict(song_id='song',start_sec=5,end_sec=15,rendered_start_sec=6,review_id='C01')
    choices=alternatives(row,alignment,song,[old])
    assert len(choices)==1 and choices[0]['start_sec']==26
    assert choices[0]['training_eligible'] is False


def packet_and_answers():
    packet={'schema_version':'replacement_location_review_v1','items':[{'review_id':'L01','alternatives':[{'choice_id':'A'}]}]}
    packet['packet_sha256']=sha256_object(packet)
    answers={'schema_version':'replacement_location_answers_v1','packet_sha256':packet['packet_sha256'],
             'reviewer':'fixture','answers':[{'review_id':'L01','choice':'none','comment':'only part matches'}]}
    return packet,answers


def test_none_and_uncertain_are_valid_not_forced_to_a_match():
    packet,answers=packet_and_answers()
    for choice in ('A','none','uncertain'):
        answers['answers'][0]['choice']=choice
        validate_answers(packet,answers)


@pytest.mark.parametrize('kind',['hash','id','duplicate','choice','reviewer'])
def test_location_answer_import_rejects_invalid_binding(kind):
    packet,answers=packet_and_answers()
    if kind=='hash': answers['packet_sha256']='0'*64
    elif kind=='id': answers['answers'][0]['review_id']='B01'
    elif kind=='duplicate': answers['answers'].append(deepcopy(answers['answers'][0]))
    elif kind=='choice': answers['answers'][0]['choice']='D'
    else: answers['reviewer']=' '
    with pytest.raises(ValueError): validate_answers(packet,answers)

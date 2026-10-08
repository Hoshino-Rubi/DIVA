from __future__ import annotations
import copy
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from any2xsampa.music.game_notes import decode_regions
from any2xsampa.music.note_metrics import NoteEvent,match_notes,evaluate_notes,_maximum_matching
from any2xsampa.adapters.game_onnx import checked_request


def raw_example():
    boundaries=np.zeros((1,100),dtype=bool);boundaries[0,[20,27,50]]=True
    return {'durations':np.array([[.2,.07,.23,.5]],np.float32),'presence':np.array([[False,True,True,True]]),
            'scores':np.array([[0,60.1,60.1,62.2]],np.float32),'maskN':np.ones((1,4),bool),
            'maskT':np.ones((1,100),bool),'boundaries':boundaries}


def test_rest_preserves_original_time_same_pitch_notes_and_short_notes():
    r=decode_regions(raw_example(),start_sec=20,duration_sec=1)
    assert [n['pitch_midi'] for n in r['notes']]==[60,60,62]
    assert [n['start_sec'] for n in r['notes']]==pytest.approx([20.2,20.27,20.5])
    assert r['notes'][0]['duration_sec']==pytest.approx(.07)
    assert len(r['regions'])==4 and r['regions'][0]['presence'] is False
    assert not r['quality_validated']

@pytest.mark.parametrize('field',['maskN','maskT'])
def test_nonprefix_padding_is_rejected(field):
    r=raw_example();r[field][0,1]=False
    with pytest.raises(ValueError,match='Padding'):decode_regions(r,start_sec=0,duration_sec=1)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,128])
def test_invalid_voiced_pitch_cannot_be_repaired_by_clipping(value):
    r=raw_example();r['scores'][0,1]=value
    with pytest.raises(ValueError):decode_regions(r,start_sec=0,duration_sec=1)


def test_duration_must_agree_with_model_boundaries():
    r=raw_example();r['durations'][0,1]=.2
    with pytest.raises(ValueError,match='Durations'):decode_regions(r,start_sec=0,duration_sec=1)


def test_empty_output_is_valid_if_model_predicts_only_rest():
    r=raw_example();r['presence'][:]=False
    assert decode_regions(r,start_sec=0,duration_sec=1)['notes']==[]


def test_rounding_at_half_step_is_explicit_and_padding_is_ignored():
    r=raw_example();r['scores'][0,1]=60.5
    assert decode_regions(r,start_sec=0,duration_sec=1)['notes'][0]['pitch_midi']==61
    for name,extra in [('durations',0),('scores',0),('presence',False),('maskN',False)]:
        r[name]=np.pad(r[name],((0,0),(0,1)),constant_values=extra)
    assert len(decode_regions(r,start_sec=0,duration_sec=1)['notes'])==3

@pytest.mark.parametrize('field',['lyrics','notes','pitch','vsqx','known_durations','language','phones'])
def test_audio_request_rejects_reference_conditions(field):
    req={'audio_path':'x.wav','audio_sha256':'a'*64,'start_sec':0,'end_sec':10,field:[]}
    with pytest.raises(ValueError,match='audio'):checked_request(req)


def test_one_to_one_matching_counts_duplicates_and_missing_notes():
    ref=[NoteEvent(0,1,60),NoteEvent(1,2,60)]
    pred=[ref[0],ref[0]]
    r=evaluate_notes(ref,pred,reference_clock_confirmed=True)
    assert r['pitch_onset_offset']['matches']==1
    assert r['pitch_onset_offset']['false_positives']==1
    assert r['pitch_onset_offset']['false_negatives']==1
    assert not r['quality_target_passed']


def test_augmenting_path_is_not_greedy_and_has_maximum_cardinality():
    assert len(_maximum_matching([[0,1],[0]],2))==2
    ref=[NoteEvent(.04,1,60),NoteEvent(0,1,60)]
    pred=[NoteEvent(.02,1,60),NoteEvent(.08,1,60)]
    assert len(match_notes(ref,pred))==2


def test_long_note_end_error_never_gets_percentage_relaxation():
    ref=[NoteEvent(0,10,60)];pred=[NoteEvent(0,10.1,60)]
    r=evaluate_notes(ref,pred,reference_clock_confirmed=True)
    assert r['pitch_onset']['matches']==1
    assert r['pitch_onset_offset']['matches']==0
    assert r['timing_on_pitch_onset_matches']['offset']['maximum_abs_sec']==pytest.approx(.1)


def test_pitch_boundary_joint_rule_and_unconfirmed_clock():
    ref=[NoteEvent(0,1,60)]
    r=evaluate_notes(ref,[NoteEvent(.05,1.05,72)],reference_clock_confirmed=True)
    assert r['boundaries_without_pitch']['matches']==1 and r['octave_errors_on_boundary_matches']==1
    assert r['pitch_onset_offset']['matches']==0
    assert not evaluate_notes(ref,ref)['quality_target_passed']
    assert evaluate_notes(ref,ref,reference_clock_confirmed=True)['quality_target_passed']


def test_boundaries_and_pitch_are_separate_graph_outputs(monkeypatch,tmp_path):
    from any2xsampa.adapters import game_onnx as g
    monkeypatch.setattr(g,'load_snapshot',lambda _:({}, {'languages':{'yue':3},'samplerate':44100}))
    raw=raw_example();calls=[]
    class Fake:
        def __init__(self,name):self.name=name
        def get_providers(self):return ['CPUExecutionProvider']
        def get_inputs(self):return [SimpleNamespace(name=x) for x in g.EXPECTED_INPUTS[self.name]]
        def get_outputs(self):return [SimpleNamespace(name=x) for x in g.EXPECTED_OUTPUTS[self.name]]
        def run(self,names,data):
            assert set(data)==g.EXPECTED_INPUTS[self.name];calls.append((self.name,data))
            if self.name=='encoder':return [np.zeros((1,100,256),np.float32)]*2+[raw['maskT']]
            if self.name=='segmenter':
                assert not data['known_boundaries'].any() and data['language'].tolist()==[3]
                return [raw['boundaries']]
            if self.name=='bd2dur':return [raw['durations'],raw['maskN']]
            return [raw['presence'],raw['scores']]
    runtime=g.GameOnnxRuntime(tmp_path,device='cpu',session_factory=lambda file:Fake(file.stem))
    decoded=runtime.infer(np.zeros((1,44100),np.float32))
    assert len(calls)==11 and len(decoded['boundary_iterations'])==8
    assert len(decode_regions(decoded,start_sec=0,duration_sec=1)['notes'])==3


def test_cuda_is_not_silently_substituted_by_cpu(tmp_path):
    from any2xsampa.adapters.game_onnx import GameOnnxRuntime
    with pytest.raises(ValueError,match='CPU only'):GameOnnxRuntime(tmp_path,device='cuda')


def test_snapshot_hash_or_language_mismatch_fails_before_session_creation(tmp_path,monkeypatch):
    from any2xsampa.adapters import game_onnx as g
    called=[]
    def forbidden(*args):called.append(True);raise AssertionError('should not load')
    with pytest.raises(FileNotFoundError):g.GameOnnxRuntime(tmp_path,device='cpu',session_factory=forbidden)
    assert called==[]


def test_snapshot_rejects_changed_weight_and_missing_cantonese(tmp_path):
    import json,hashlib
    from any2xsampa.music.game_notes import load_snapshot,EXPECTED_ARCHIVE,REVISION,GRAPH_NAMES
    def freeze(language):
        config={'samplerate':44100,'timestep':.01,'loop':True,'embedding_dim':256,'languages':language}
        (tmp_path/'config.json').write_text(json.dumps(config),encoding='utf8')
        for name in GRAPH_NAMES:(tmp_path/(name+'.onnx')).write_bytes(b'fixture-not-an-executable-graph')
        files=[{'path':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
               for f in tmp_path.iterdir() if f.name!='snapshot-lock.json']
        (tmp_path/'snapshot-lock.json').write_text(json.dumps({'schema_version':'game_onnx_snapshot_v1','revision':REVISION,
            'archive_sha256':EXPECTED_ARCHIVE,'files':files}),encoding='utf8')
    freeze({'yue':3})
    assert load_snapshot(tmp_path)[1]['languages']['yue']==3
    (tmp_path/'encoder.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):load_snapshot(tmp_path)
    freeze({'zh':4})
    with pytest.raises(ValueError,match='Cantonese'):load_snapshot(tmp_path)


def test_raw_replay_preserves_notes_and_rejects_changed_evidence(tmp_path):
    from any2xsampa.local.common import write,read,sha
    from any2xsampa.adapters.game_onnx import replay_result
    raw=raw_example();source=tmp_path/'raw.npz';np.savez_compressed(source,**raw)
    pred=decode_regions(raw,start_sec=5,duration_sec=1)
    pred.update(input={'model_duration_sec':1},raw_evidence={'file':source.name,'sha256':sha(source)})
    file=tmp_path/'prediction.json';write(file,pred)
    proof=replay_result(file,tmp_path/'replay.json')
    assert proof['notes']==pred['notes'] and proof['model_forwards']==0
    source.write_bytes(source.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='evidence'):replay_result(file,tmp_path/'bad.json')


def test_maximum_matching_agrees_with_exhaustive_small_oracle():
    import itertools
    for bits in range(1<<9):
        edges=[[j for j in range(3) if bits&(1<<(i*3+j))] for i in range(3)]
        best=0
        for choices in itertools.product(range(-1,3),repeat=3):
            assigned=[j for j in choices if j>=0]
            if len(set(assigned))!=len(assigned):continue
            if all(j<0 or j in edges[i] for i,j in enumerate(choices)):best=max(best,len(assigned))
        assert len(_maximum_matching(edges,3))==best


def test_pitch_sequence_diagnostics_do_not_disguise_octaves_or_duplicates():
    from tools.evaluate_game_note_qualification import sequence_errors
    r=sequence_errors([60,60,62],[60,62])
    assert r['deletions']==1 and r['edit_distance']==1
    r=sequence_errors([60],[72,72])
    assert r['substitutions']==1 and r['insertions']==1 and r['edit_distance']==2


def test_review_rejects_code_changes_and_unregistered_output(tmp_path,monkeypatch):
    from tools import run_game_note_job as w
    from any2xsampa.local.common import write,sha
    monkeypatch.setattr(w,'ROOT',tmp_path)
    code=tmp_path/'worker.py';code.write_text('original')
    request=tmp_path/'request.json';write(request,{'fixture':'no inference'})
    (tmp_path/'snapshot-lock.json').write_text('{}')
    target=tmp_path/'job'
    review={'schema_version':'game_note_launch_review_v1','optimizer_updates':0,'device':'cpu',
        'code_sha256':{'worker.py':sha(code)},'validation':[], 'snapshot_lock_sha256':sha(tmp_path/'snapshot-lock.json'),
        'jobs':[{'output':str(target),'request_sha256':sha(request)}], 'maximum_jobs':1,'maximum_graph_calls_per_job':11}
    p=tmp_path/'review.json';write(p,review)
    assert w.verify_review(p,request,tmp_path,target)['device']=='cpu'
    with pytest.raises(ValueError,match='authorized'):w.verify_review(p,request,tmp_path,tmp_path/'wrong')
    code.write_text('changed')
    with pytest.raises(ValueError,match='invalidated'):w.verify_review(p,request,tmp_path,target)

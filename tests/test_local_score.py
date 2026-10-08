from __future__ import annotations
import copy
import json
from pathlib import Path
import threading
import wave
from xml.etree import ElementTree as ET
import pytest
from any2xsampa.local.common import ROOT, Cancelled, read, sha, write
from any2xsampa.local.score_export import make_score, export_score
from any2xsampa.local.score_tasks import import_score, export_saved_score, build_score
from any2xsampa.local.vsqx_export import export_vsqx_from_result

@pytest.fixture
def result_file(tmp_path):
    audio=tmp_path/'音频.wav'
    with wave.open(str(audio),'wb') as out:
        out.setparams((1,2,16000,0,'NONE','not compressed'));out.writeframes(bytes(16000*2*3))
    file=tmp_path/'result.json'
    write(file,{'schema_version':'local_task_result_v1','mode':'real','input':{
        'path':str(audio),'sha256':sha(audio),'start_sec':0,'end_sec':3},'segments':[
        {'start_sec':.25,'end_sec':.27,'normalized_ipa':'pʰ'},
        {'start_sec':.5,'end_sec':.52,'normalized_ipa':'ɐ'},
        {'start_sec':1.2,'end_sec':1.22,'normalized_ipa':'i'},
        {'start_sec':2.7,'end_sec':2.72,'normalized_ipa':'kˀ'}]})
    return file

@pytest.fixture
def pitch():
    return {'note_candidates':[{'start_sec':.2,'duration_sec':.5,'midi':48},
        {'start_sec':1,'duration_sec':.7,'midi':67},
        {'start_sec':2.5,'duration_sec':.4,'midi':72}]}

def test_audio_notes_keep_pitch_durations_and_all_phone_evidence(result_file,pitch,tmp_path):
    source=read(result_file);original=copy.deepcopy(source)
    score=make_score(source,pitch,120)
    assert [n['pitch'] for n in score['notes']]==[48,67,72]
    assert [n['start_sec'] for n in score['notes']]==[.2,1,2.5]
    assert score['notes'][0]['duration_sec']==pytest.approx(.5)
    assert score['notes'][0]['v4c_phonemes']==['p_h','a']
    assert score['notes'][2]['placeholder'] is True
    assert score['notes'][2]['ipa']==['kˀ']
    assert score['notes'][2]['unsupported_phones']==['kˀ']
    assert source==original
    assert sorted([i for n in score['notes'] for i in n['phone_indices']]+score['unassigned_phone_indices'])==list(range(4))
    file=tmp_path/'score.json';write(file,score)
    target=tmp_path/'自动草稿.vsqx';export_score(file,target)
    tree=ET.parse(target);ns={'v':tree.getroot().tag.split('}')[0][1:]}
    notes=tree.findall('.//v:note',ns)
    assert [int(n.findtext('v:n',namespaces=ns)) for n in notes]==[48,67,72]
    assert [int(n.findtext('v:t',namespaces=ns)) for n in notes]==[192,960,2400]
    assert [int(n.findtext('v:dur',namespaces=ns)) for n in notes]==[480,672,384]
    assert tree.findtext('.//v:tempo/v:v',namespaces=ns)=='12000'
    from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline
    timeline=parse_vsqx_timeline(target)
    assert timeline['notes'][0]['start_sec']==pytest.approx(.2,abs=1/240)
    assert timeline['notes'][1]['start_sec']==pytest.approx(1.0,abs=1/240)
    assert all(n.find('v:p',ns).get('lock')=='1' for n in notes)
    assert read(target.with_name(target.name+'.provenance.json'))['transpose_semitones']==0

def test_no_pitch_evidence_cannot_become_fixed_c4(result_file):
    with pytest.raises(ValueError,match='No voiced pitch'):
        make_score(read(result_file),{'note_candidates':[]},120)

@pytest.mark.parametrize('value',[float('nan'),True,128,-1,60.3])
def test_invalid_pitch_fails(result_file,pitch,value):
    pitch['note_candidates'][0]['midi']=value
    with pytest.raises(ValueError):make_score(read(result_file),pitch,120)

def test_import_preserves_original_and_exports_phone_draft_with_same_note_geometry(result_file,tmp_path):
    template=ROOT/'examples/accepted/congqian_onnx_miku_v4c.vsqx'
    before=template.read_bytes();result_before=result_file.read_bytes()
    info=import_score({'result_path':str(result_file),'template_path':str(template),'output_dir':str(tmp_path/'import')},threading.Event(),lambda _:None)
    out=tmp_path/'导出.vsqx'
    export_saved_score({'score_path':info['score_path'],'score_sha256':info['score_sha256'],'output':str(out)})
    assert Path(info['preserved_vsqx_path']).read_bytes()==before==template.read_bytes()
    assert out.exists()
    assert result_before==result_file.read_bytes()
    score=read(Path(info['score_path']))
    assert score['source']=='supplied_vsqx_phone_draft' and score['audio_reestimated'] is False
    assert Path(score['suggested_vsqx_path']).read_bytes()==out.read_bytes()
    assert score['phone_suggestion_changed_note_p_count'] >= 0
    assert 'original_v4c_phonemes' in score['notes'][0]
    assert 'proposed_v4c_phonemes' in score['notes'][0]
    with pytest.raises(FileExistsError):export_saved_score({'score_path':info['score_path'],'score_sha256':info['score_sha256'],'output':str(out)})

def test_tampered_audio_or_score_is_rejected(result_file,pitch,tmp_path):
    template=ROOT/'examples/accepted/congqian_onnx_miku_v4c.vsqx'
    info=import_score({'result_path':str(result_file),'template_path':str(template),'output_dir':str(tmp_path/'import')},threading.Event(),lambda _:None)
    Path(info['score_path']).write_text('{}')
    with pytest.raises(ValueError,match='hash'):export_saved_score({'score_path':info['score_path'],'score_sha256':info['score_sha256'],'output':str(tmp_path/'bad.vsqx')})
    Path(read(result_file)['input']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='audio'):build_score({'result_path':str(result_file)},threading.Event(),lambda _:None)

def test_ctc_phone_repetition_is_not_removed_when_suggesting(tmp_path):
    from any2xsampa.local.vsqx_export import _suggestion_for_note
    rules={'i':{'tokens':['i']}}
    suggestion=_suggestion_for_note([{'start_sec':.1,'end_sec':.12,'normalized_ipa':'i'},
                                    {'start_sec':.3,'end_sec':.32,'normalized_ipa':'i'}],0,1,rules)
    assert suggestion['phones']==['i','i'] and suggestion['tokens']==['i','i']

def test_note_proposal_does_not_export_partial_unknowns(result_file,tmp_path):
    source=ROOT/'examples/accepted/congqian_onnx_miku_v4c.vsqx'
    tree=ET.parse(source);root=tree.getroot();ns=root.tag.split('}')[0]+'}'
    part=root.find('.//'+ns+'vsPart');part.find(ns+'t').text='0'
    for n in list(part.findall(ns+'note'))[1:]:part.remove(n)
    note=part.find(ns+'note');note.find(ns+'t').text='0';note.find(ns+'dur').text='4000'
    original_p=note.findtext(ns+'p')
    template=tmp_path/'template.vsqx';tree.write(template,encoding='utf8')
    out=tmp_path/'proposal.vsqx'
    export_vsqx_from_result(result_path=result_file,template_path=template,output_path=out,apply_phone_suggestions=True)
    assert ET.parse(out).findtext('.//'+ns+'note/'+ns+'p')==original_p

def test_build_score_job_is_serial_cancellable_and_links_result(result_file,pitch,tmp_path,monkeypatch):
    from any2xsampa.local import score_tasks
    closed=[]
    class Worker:
        def __init__(self,python,log,**kw):assert kw['module']=='any2xsampa.local.pitch_worker'
        def call(self,action,params,cancel,progress,**kw):
            file=Path(params['output_dir'])/'pitch-analysis.json'
            write(file,{**pitch,'result_sha256':sha(result_file)})
            return {'pitch_path':str(file),'pitch_sha256':sha(file)}
        def close(self):closed.append(True)
    monkeypatch.setattr(score_tasks,'WorkerClient',Worker)
    answer=build_score({'result_path':str(result_file),'output_dir':str(tmp_path/'built')},threading.Event(),lambda _:None)
    assert closed==[True] and answer['model_forwards']==0
    assert answer['note_count']==3 and Path(answer['vsqx_path']).exists()
    assert read(result_file.parent/'score-link.json')['score_sha256']==answer['score_sha256']
    flag=threading.Event();flag.set()
    with pytest.raises(Cancelled):build_score({'result_path':str(result_file),'output_dir':str(tmp_path/'cancelled')},flag,lambda _:None)
    assert not (tmp_path/'cancelled').exists()

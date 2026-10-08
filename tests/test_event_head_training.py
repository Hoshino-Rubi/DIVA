from pathlib import Path
import sys
import wave
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from any2xsampa.training.event_head_training import make_head
from any2xsampa.training.limited_data_artifacts import write_json_new,sha256_file
from any2xsampa.adapters.audio_provenance import describe_audio_file


def test_independent_heads_content_equal_and_parameter_counts():
    torch=pytest.importorskip('torch');torch.manual_seed(17)
    state={'weight':torch.randn(428,1024),'bias':torch.randn(428)}
    h=make_head(torch,state,'H');e=make_head(torch,state,'E')
    assert sum(p.numel() for p in h.parameters())==438700
    assert sum(p.numel() for p in e.parameters())==439725
    assert h.content.weight.data_ptr()!=state['weight'].data_ptr()!=e.content.weight.data_ptr()
    x=torch.randn(4,1024);hq,_=h(x);eq,b=e(x)
    torch.testing.assert_close(hq,eq,atol=0,rtol=0);assert b.count_nonzero()==0


@pytest.mark.parametrize('arm',['H','E'])
def test_real_validation_artifact_serializes_and_replays(tmp_path,monkeypatch,arm):
    torch=pytest.importorskip('torch');pytest.importorskip('safetensors')
    import train_event_head_comparison as module
    from any2xsampa.adapters.segmental_event import decode_events
    from any2xsampa.adapters.phoneticxeus import ctc_collapse
    audio=tmp_path/'audio.wav'
    with wave.open(str(audio),'wb') as f:f.setparams((1,2,16000,3200,'NONE',''));f.writeframes(b'\0\0'*3200)
    v={'<blank>':0,**{f'p{k}':k for k in range(1,428)}};vp=tmp_path/'vocab.json';write_json_new(vp,v)
    snapshot=tmp_path/'snapshot';write_json_new(snapshot/'snapshot-lock.json',{'fixture':True})
    state={'weight':torch.randn(428,1024),'bias':torch.randn(428)};head=make_head(torch,state,arm)
    raw=np.full((8,428),-9.,dtype='float32');raw[:,1]=9.
    q=torch.from_numpy(raw).log_softmax(-1);b=torch.full((8,),-8.);b[0]=8.
    ids=ctc_collapse(q.argmax(-1).tolist()) if arm=='H' else decode_events(q.numpy(),b.numpy())['event_ids']
    monkeypatch.setattr(module,'predict',lambda *_:(ids,q,b if arm=='E' else None))
    row={'example_id':'example','song_id':'song','audio_role':'original','target_ipa':['p1'],'features':torch.zeros(8,1024),
        'audio_path':str(audio),'audio_metadata':describe_audio_file(audio),
        'crop':{'effective_start_sec':0.,'effective_end_sec':.2},
        'geometry':{'frame_count':8,'stride_samples':320,'receptive_field_samples':400},'runtime':{'fixture':True}}
    freeze={'snapshot':str(snapshot),'vocabulary':{'path':str(vp),'sha256':sha256_file(vp)}}
    r=module.evaluate(torch,head,arm,[row],{},freeze,tmp_path/'output')
    assert r['details'][0]['real_fixture_equal'] and r['original']['reference_phones']==1

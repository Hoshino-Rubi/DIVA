from pathlib import Path
import sys,wave
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from any2xsampa.training.limited_data_artifacts import write_json_new,read_json,sha256_file
from any2xsampa.adapters.audio_provenance import describe_audio_file
from any2xsampa.training.event_head_training import make_head


def test_fast_evaluation_preserves_predictions_metrics_and_event_codec(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');pytest.importorskip('safetensors')
    import event_head_validation_fast as fast
    import train_event_head_comparison as original
    from any2xsampa.adapters.segmental_event import decode_events,SegmentalEventRecognizer
    from dataclasses import asdict
    from safetensors.numpy import load_file
    audio=tmp_path/'audio.wav'
    with wave.open(str(audio),'wb') as f:f.setparams((1,2,16000,3200,'NONE',''));f.writeframes(b'\0\0'*3200)
    v={'<blank>':0,**{f'p{i}':i for i in range(1,428)}};vp=tmp_path/'vocab.json';write_json_new(vp,v)
    x=np.full((8,428),-9.,dtype='float32');x[:4,1]=9.;x[4:,2]=9.;q=torch.from_numpy(x).log_softmax(-1)
    b=torch.full((8,),-8.);b[0]=8.;b[4]=8.;ids=decode_events(q.numpy(),b.numpy())['event_ids']
    fn=lambda *_:(ids,q,b);monkeypatch.setattr(fast,'predict',fn);monkeypatch.setattr(original,'predict',fn)
    head=make_head(torch,{'weight':torch.randn(428,1024),'bias':torch.randn(428)},'E')
    row={'example_id':'r','song_id':'song','audio_role':'original','target_ipa':['p1','p2'],'features':torch.zeros(8,1024),
         'audio_path':str(audio),'audio_metadata':describe_audio_file(audio),'crop':{'effective_start_sec':0.,'effective_end_sec':.2},
         'geometry':{'frame_count':8,'stride_samples':320,'receptive_field_samples':400},'runtime':{}}
    freeze={'vocabulary':{'path':str(vp),'sha256':sha256_file(vp)}}
    a=original.evaluate(torch,head,'E',[row],{},freeze,tmp_path/'old');c=fast.evaluate_fast(torch,head,'E',[row],{},freeze,tmp_path/'new')
    assert a['original']==c['original'] and a['rendered']==c['rendered']
    assert a['details'][0]['prediction']==c['details'][0]['prediction']
    left=load_file(a['details'][0]['scores_path']);right=load_file(c['details'][0]['scores_path'])
    for k in left:np.testing.assert_array_equal(left[k],right[k])
    pp=Path(c['details'][0]['event_path']);assert read_json(pp)['independent_codec_verification_pending'] is True
    assert [asdict(s) for s in SegmentalEventRecognizer(pp).recognize(audio)]==[asdict(s) for s in SegmentalEventRecognizer(pp,fixture=True).recognize(audio)]


def test_no_optimizer_loop_rewritten_by_v2():
    import train_event_head_comparison_v2 as module
    import inspect
    body=inspect.getsource(module.run)
    assert 'old.train(run_dir' in body and 'old.evaluate=evaluate_fast' in body
    assert 'optimizer.step' not in body

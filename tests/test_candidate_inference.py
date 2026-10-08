import wave
from pathlib import Path
import pytest
from any2xsampa.adapters.candidate_inference import audio_request,validate_request,run_candidate
from any2xsampa.cli import build_parser


def wav(path):
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0\0'*32000)


def test_request_is_audio_only_and_hash_bound(tmp_path):
    audio=tmp_path/'a.wav';wav(audio);request=audio_request(audio,0,1)
    validate_request(request)
    with pytest.raises(ValueError):validate_request({**request,'lyrics':'text'})
    audio.write_bytes(audio.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='SHA-256'):validate_request(request)


@pytest.mark.parametrize('start,duration',[(0,21),(0,float('nan')),(-1,1),(0,0),(1,2)])
def test_crop_limits(tmp_path,start,duration):
    audio=tmp_path/'a.wav';wav(audio)
    with pytest.raises(ValueError):audio_request(audio,start,duration)


def test_new_cli_is_explicit_and_does_not_replace_existing_mode():
    p=build_parser();old=p.parse_args(['infer-open','--audio','a.wav','--dialect','cantonese'])
    assert old.recognizer_mode=='fixture'
    new=p.parse_args(['infer-candidate','--audio','a.wav','--bundle','b.json'])
    assert new.duration==8 and new.start==0
    assert not hasattr(new,'lyrics')


def test_refuses_to_overwrite_existing_output(tmp_path):
    with pytest.raises(ValueError,match='new output'):
        run_candidate(audio=tmp_path/'missing',start=0,duration=1,output=tmp_path)

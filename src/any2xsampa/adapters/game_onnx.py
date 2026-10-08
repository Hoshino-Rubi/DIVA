"""Isolated, audio-only GAME ONNX inference. No references or silent fallback."""
from __future__ import annotations
import math
from pathlib import Path
import time
import numpy as np
from any2xsampa.local.common import read, sha, write
from any2xsampa.music.game_notes import POLICY_PATH, GRAPH_NAMES, decode_regions, load_snapshot

EXPECTED_INPUTS={
 'encoder':{'waveform','duration'},
 'segmenter':{'x_seg','language','known_boundaries','prev_boundaries','t','maskT','threshold','radius'},
 'bd2dur':{'boundaries','maskT'},
 'estimator':{'x_est','boundaries','maskT','maskN','threshold'},
}
EXPECTED_OUTPUTS={'encoder':['x_seg','x_est','maskT'],'segmenter':['boundaries'],
                  'bd2dur':['durations','maskN'],'estimator':['presence','scores']}


def checked_request(request: dict) -> dict:
    if set(request)!={'audio_path','audio_sha256','start_sec','end_sec'}:
        raise ValueError('GAME accepts audio path/hash and interval only; no score or reference conditions')
    for key in ('start_sec','end_sec'):
        x=request[key]
        if type(x) not in (int,float) or not math.isfinite(x):raise ValueError('Invalid request time')
    if request['start_sec']<0 or not .1<=request['end_sec']-request['start_sec']<=20:
        raise ValueError('GAME qualification interval must be 0.1 to 20 seconds')
    if not isinstance(request['audio_path'],str) or not isinstance(request['audio_sha256'],str) or len(request['audio_sha256'])!=64:
        raise ValueError('Audio path and SHA256 required')
    return dict(request)


def load_audio(request: dict, destination: Path, sample_rate: int=44100) -> tuple[np.ndarray,dict]:
    import soundfile as sf
    import librosa
    file=Path(request['audio_path'])
    if sha(file)!=request['audio_sha256']:raise ValueError('Source audio hash mismatch')
    with sf.SoundFile(file) as source:
        first=round(request['start_sec']*source.samplerate)
        last=round(request['end_sec']*source.samplerate)
        if last>source.frames:raise ValueError('Requested interval is outside source audio')
        source.seek(first)
        samples=source.read(last-first,dtype='float32',always_2d=True)
        sr=source.samplerate
    if len(samples)!=last-first or not np.isfinite(samples).all():raise ValueError('Truncated or invalid audio')
    mono=samples.mean(axis=1,dtype=np.float32)
    sf.write(destination,mono,sr,subtype='FLOAT')
    if sr!=sample_rate:
        mono=librosa.resample(mono,orig_sr=sr,target_sr=sample_rate,res_type='soxr_hq')
    if sha(file)!=request['audio_sha256']:raise ValueError('Source audio changed while reading')
    info={**request,'effective_start_sec':first/sr,'effective_end_sec':last/sr,
          'source_sample_rate':sr,'source_start_sample':first,'source_end_sample_exclusive':last,
          'model_sample_rate':sample_rate,'model_samples':len(mono),'model_duration_sec':len(mono)/sample_rate,
          'mono_crop_sha256':sha(destination)}
    return np.ascontiguousarray(mono[None,:],dtype=np.float32),info


class GameOnnxRuntime:
    def __init__(self, snapshot: Path, *, device: str, session_factory=None) -> None:
        if device!='cpu':
            raise ValueError('This qualified runtime currently supports explicit CPU only; CUDA validation is pending')
        self.lock,self.config=load_snapshot(snapshot)
        self.policy=read(POLICY_PATH)
        if self.policy['device']!=device:raise ValueError('Device differs from frozen policy')
        self.sessions={};self.calls=[]
        if session_factory is None:
            import onnxruntime as ort
            ort.set_seed(self.policy['seed'])
            def session_factory(file):
                options=ort.SessionOptions()
                options.intra_op_num_threads=4;options.inter_op_num_threads=1
                options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
                options.use_deterministic_compute=True
                return ort.InferenceSession(str(file),sess_options=options,providers=['CPUExecutionProvider'])
        for name in GRAPH_NAMES:
            session=session_factory(snapshot/(name+'.onnx'))
            if session.get_providers()!=['CPUExecutionProvider']:raise ValueError('Unexpected execution provider')
            if {x.name for x in session.get_inputs()}!=EXPECTED_INPUTS[name] or [x.name for x in session.get_outputs()]!=EXPECTED_OUTPUTS[name]:
                raise ValueError('Actual graph signature differs from reviewed author exporter: '+name)
            self.sessions[name]=session

    def _run(self,name: str,values: dict) -> list[np.ndarray]:
        if set(values)!=EXPECTED_INPUTS[name]:raise ValueError('Unexpected graph input')
        start=time.monotonic()
        result=self.sessions[name].run(EXPECTED_OUTPUTS[name],values)
        self.calls.append({'graph':name,'elapsed_seconds':time.monotonic()-start})
        for value in result:
            if not isinstance(value,np.ndarray) or (value.dtype.kind=='f' and not np.isfinite(value).all()):
                raise ValueError('Invalid ONNX output')
        return result

    def infer(self,waveform: np.ndarray) -> dict:
        if waveform.dtype!=np.float32 or waveform.ndim!=2 or waveform.shape[0]!=1 or not np.isfinite(waveform).all():
            raise ValueError('Expected finite mono waveform batch float32[1,L]')
        duration=waveform.shape[1]/self.config['samplerate']
        if not .1<=duration<=self.policy['max_input_seconds']+1e-5:raise ValueError('Unbounded inference rejected')
        x_seg,x_est,mask=self._run('encoder',{'waveform':waveform,'duration':np.array([duration],np.float32)})
        if mask.dtype!=bool or mask.ndim!=2 or mask.shape[0]!=1 or x_seg.shape[:2]!=mask.shape or x_est.shape[:2]!=mask.shape:
            raise ValueError('Encoder dimensions differ')
        known=np.zeros_like(mask,dtype=bool)
        previous=known.copy();iterations=[]
        for ti in self.policy['sample_times']:
            previous,=self._run('segmenter',{'x_seg':x_seg,'language':np.array([self.config['languages']['yue']],np.int64),
                'known_boundaries':known,'prev_boundaries':previous,'t':np.array([ti],np.float32),
                'maskT':mask,'threshold':np.array(self.policy['boundary_threshold'],np.float32),
                'radius':np.array(self.policy['boundary_radius'],np.int64)})
            if previous.dtype!=bool or previous.shape!=mask.shape:raise ValueError('Invalid boundary output')
            iterations.append(previous.copy())
        durations,nmask=self._run('bd2dur',{'boundaries':previous,'maskT':mask})
        presence,scores=self._run('estimator',{'x_est':x_est,'boundaries':previous,'maskT':mask,'maskN':nmask,
            'threshold':np.array(self.policy['presence_threshold'],np.float32)})
        return {'durations':durations,'presence':presence,'scores':scores,'maskN':nmask,'maskT':mask,
                'boundaries':previous,'boundary_iterations':np.stack(iterations),
                'x_seg':x_seg,'x_est':x_est}


def replay_result(source: Path, output: Path) -> dict:
    saved=read(source)
    rawfile=source.parent/saved['raw_evidence']['file']
    if sha(rawfile)!=saved['raw_evidence']['sha256']:raise ValueError('Raw note evidence changed')
    with np.load(rawfile,allow_pickle=False) as arrays:
        decoded=decode_regions(dict(arrays),start_sec=saved['source_start_sec'],
                               duration_sec=saved['input']['model_duration_sec'],timestep=saved['timestep_sec'])
    if decoded['notes']!=saved['notes'] or decoded['regions']!=saved['regions']:
        raise ValueError('Note prediction differs from raw evidence replay')
    proof={'status':'passed','source_result_sha256':sha(source),'note_count':len(decoded['notes']),
           'notes':decoded['notes'],'model_forwards':0,'optimizer_updates':0}
    write(output,proof)
    return proof

"""GAME output codec. Time regions and rest regions are never collapsed by pitch."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from any2xsampa.local.common import read, sha

POLICY_PATH=Path(__file__).resolve().parents[1]/'config/game_note_v1.json'
EXPECTED_ARCHIVE='5b7a21e64c6310efac399f5d12838fffa70565be162436b5a4a65f290721e7d8'
REVISION='475a8ee781fe8cca980b3b12fbe6c80c768a813a'
GRAPH_NAMES=('encoder','segmenter','bd2dur','estimator')


def load_snapshot(directory: Path) -> tuple[dict,dict]:
    lock=read(directory/'snapshot-lock.json')
    if lock.get('schema_version')!='game_onnx_snapshot_v1' or lock.get('revision')!=REVISION or lock.get('archive_sha256')!=EXPECTED_ARCHIVE:
        raise ValueError('Wrong fixed GAME snapshot')
    required={'config.json', *(n+'.onnx' for n in GRAPH_NAMES)}
    seen=set()
    for item in lock['files']:
        relative=Path(item['path']);file=(directory/relative).resolve()
        if not file.is_relative_to(directory.resolve()) or relative.as_posix() in seen:
            raise ValueError('Invalid snapshot member')
        seen.add(relative.as_posix())
        if file.stat().st_size!=item['bytes'] or sha(file)!=item['sha256']:
            raise ValueError('GAME snapshot member hash mismatch')
    if not required<=seen:raise ValueError('Incomplete GAME model archive')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='snapshot-lock.json'}
    if seen!=actual:raise ValueError('Unexpected GAME snapshot files')
    cfg=read(directory/'config.json')
    if cfg.get('samplerate')!=44100 or cfg.get('timestep')!=.01 or cfg.get('loop') is not True:
        raise ValueError('Snapshot frontend differs from the reviewed v1.0.3 interface')
    language=read(POLICY_PATH)['language']
    langs=cfg.get('languages')
    if not isinstance(langs,dict) or language not in langs or type(langs[language]) is not int or langs[language]<=0:
        raise ValueError('Fixed snapshot must explicitly support Cantonese yue; no guessed language fallback')
    return lock,cfg


def _row(values, dtype, name: str) -> np.ndarray:
    value=np.asarray(values)
    if value.ndim!=2 or value.shape[0]!=1:
        raise ValueError(name+' must have batch shape [1,N]')
    if dtype is bool and value.dtype.kind!='b':raise ValueError(name+' must be boolean')
    if dtype is float and (value.dtype.kind not in 'fiu' or not np.isfinite(value).all()):
        raise ValueError(name+' must contain finite numeric values')
    return value[0].astype(dtype)


def decode_regions(raw: dict, *, start_sec: float, duration_sec: float, timestep: float=.01) -> dict:
    if not all(type(x) in (int,float) and math.isfinite(x) for x in (start_sec,duration_sec,timestep)) or start_sec<0 or duration_sec<=0 or timestep<=0:
        raise ValueError('Invalid source time range')
    lengths=_row(raw['durations'],float,'durations')
    presence=_row(raw['presence'],bool,'presence')
    scores=_row(raw['scores'],float,'scores')
    mask=_row(raw['maskN'],bool,'maskN')
    frames=_row(raw['maskT'],bool,'maskT')
    boundaries=_row(raw['boundaries'],bool,'boundaries')
    if not (len(lengths)==len(presence)==len(scores)==len(mask)) or len(boundaries)!=len(frames):
        raise ValueError('GAME tensor length mismatch')
    n=int(mask.sum()); t=int(frames.sum())
    if not n or not t or not np.array_equal(mask,np.arange(len(mask))<n) or not np.array_equal(frames,np.arange(len(frames))<t):
        raise ValueError('Padding masks must be nonempty contiguous prefixes')
    if boundaries[0] or boundaries[t:].any():raise ValueError('Invalid edge/padding boundary')
    cuts=np.r_[0,np.flatnonzero(boundaries[:t]),t]
    expected=np.diff(cuts)*timestep
    if len(expected)!=n or not np.allclose(lengths[:n],expected,atol=1e-6,rtol=1e-6):
        raise ValueError('Durations do not match the predicted frame boundaries')
    if abs(t*timestep-duration_sec)>timestep/2+1e-6:
        raise ValueError('Frame count does not cover the audio duration')
    regions=[];notes=[]
    for i in range(n):
        begin=start_sec+int(cuts[i])*timestep
        end=min(start_sec+duration_sec,start_sec+int(cuts[i+1])*timestep)
        if end<=begin:raise ValueError('Empty note or rest region')
        p=float(scores[i])
        if bool(presence[i]) and not 0<=p<127.5:
            raise ValueError('Out-of-range predicted MIDI pitch; not silently clipped')
        pitch=int(math.floor(p+.5)) if presence[i] else None
        item={'region_index':i,'start_sec':begin,'end_sec':end,'duration_sec':end-begin,
              'presence':bool(presence[i]),'pitch_midi':pitch,'pitch_midi_float':p if presence[i] else None,
              'touches_input_start':i==0,'touches_input_end':i==n-1}
        regions.append(item)
        if presence[i]:notes.append(dict(item))
    return {'schema_version':'note_transcription_v1','notes':notes,'regions':regions,
            'time_basis':'original_audio','timestep_sec':timestep,'source_start_sec':start_sec,
            'source_end_sec':start_sec+duration_sec,'time_quantization_error_bound_sec':timestep/2,
            'known_reference_boundaries_used':False,'automatic_pitch_transpose':False,
            'min_duration_filter_applied':False,'same_pitch_regions_merged':False,
            'quality_validated':False}

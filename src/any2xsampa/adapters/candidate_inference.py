"""Opt-in frozen-candidate execution, isolated from the ordinary environment."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import wave

from any2xsampa.adapters.audio_provenance import describe_audio_file
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
from any2xsampa.training.limited_data_artifacts import finite_number, read_json, sha256_file, verify_file, write_json_new

ROOT = Path(__file__).resolve().parents[3]


def validate_bundle(path: Path) -> dict:
    bundle = read_json(path)
    if bundle.get('schema_version') != 'guarded_candidate_runtime_bundle_v1' or bundle.get('default_replacement') is not False:
        raise ValueError('expected an explicit opt-in candidate bundle')
    for key in ('candidate', 'python', 'worker', 'vocabulary'):
        spec = bundle[key]; verify_file(Path(spec['path']), spec['sha256'])
    if Path(bundle['worker']['path']).resolve() != (ROOT/'tools/run_candidate_audio.py').resolve():
        raise ValueError('bundle worker differs from the dedicated adapter')
    for spec in bundle['code_bindings']:
        verify_file(Path(spec['path']), spec['sha256'])
    return bundle


def audio_request(audio: Path, start: float, duration: float) -> dict:
    first = finite_number(start, 'start', minimum=0)
    seconds = finite_number(duration, 'duration', minimum=0)
    if not .1 <= seconds <= 20:
        raise ValueError('Choose a 0.1 to 20 second crop; long files are read only at the requested interval.')
    info = describe_audio_file(audio)
    if first + seconds > info['duration_sec'] + 1e-6:
        raise ValueError('requested crop extends beyond the WAV file')
    return {'schema_version':'candidate_audio_request_v1','audio_path':str(audio.resolve()),'audio_sha256':info['sha256'],
            'start_sec':first,'end_sec':first+seconds,'contains_labels_or_score_context':False}


def validate_request(request: dict) -> None:
    expected = {'schema_version','audio_path','audio_sha256','start_sec','end_sec','contains_labels_or_score_context'}
    if set(request) != expected or request.get('schema_version') != 'candidate_audio_request_v1' or request.get('contains_labels_or_score_context') is not False:
        raise ValueError('candidate worker accepts audio only, no reference fields')
    start=finite_number(request['start_sec'],'start',minimum=0);end=finite_number(request['end_sec'],'end',minimum=0)
    if not .1 <= end-start <= 20.000001:
        raise ValueError('crop must be 0.1 to 20 seconds')
    verify_file(Path(request['audio_path']),request['audio_sha256'])


@contextmanager
def serial_candidate_job():
    """New candidate entrypoints share one nonblocking local process lock."""
    if os.name != 'nt':
        raise RuntimeError('this isolated runtime bundle is configured for Windows')
    import msvcrt
    from any2xsampa.local.storage import state_dir
    path=state_dir(ROOT)/'candidate-inference.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if not path.stat().st_size:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:raise RuntimeError('another candidate job is running; retry after it completes') from exc
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def replay_result(audio: Path, posterior: Path, output: Path, *, vocabulary: Path | None = None, fixture: bool = False) -> dict:
    segments=PhoneticXeusPosteriorRecognizer(posterior,fixture=fixture,vocabulary_path=vocabulary).recognize(audio)
    payload=read_json(posterior);mapping=read_json(Path(__file__).parents[1]/'config/ipa_to_xsampa.json')
    unsupported=[]
    for segment in segments:
        phone=segment.metadata['raw_label_trace']['label']
        segment.normalized_ipa=phone
        if phone in mapping:segment.xsampa=mapping[phone]
        else:unsupported.append(phone)
    result={'schema_version':'candidate_inference_result_v1','mode':'fixture' if fixture else 'real',
            'candidate_sha256':payload['producer']['model_metadata'].get('guarded_candidate_sha256'),
            'input_audio_path':str(audio.resolve()),'input_audio':payload['producer']['input_audio'],'crop':payload['producer']['crop'],
            'posterior_path':str(posterior.resolve()),'posterior_sha256':sha256_file(posterior),
            'ipa_sequence':[s.normalized_ipa for s in segments],'segments':[asdict(s) for s in segments],
            'xsampa_unmapped':sorted(set(unsupported)),'qualified_recognizer':False,'timings_are_phone_durations':False,'default_replacement':False}
    write_json_new(output/'result.json',result)
    (output/'phones.ipa.txt').write_text(' '.join(result['ipa_sequence'])+'\n',encoding='utf-8')
    (output/'phones.xsampa.txt').write_text(' '.join(s.xsampa or f'[UNMAPPED:{s.normalized_ipa}]' for s in segments)+'\n',encoding='utf-8')
    render_result(result,output)
    return result


def render_result(result: dict, output: Path) -> None:
    crop=result['crop'];audio=Path(result['input_audio_path'])
    with wave.open(str(audio),'rb') as src, wave.open(str(output/'input-crop.wav'),'wb') as dst:
        first,last=crop['source_start_sample'],crop['source_end_sample_exclusive']
        dst.setparams(src.getparams());src.setpos(first);dst.writeframes(src.readframes(last-first))
    rows=''.join('<tr><td>'+html.escape(s['normalized_ipa'])+'</td><td>'+html.escape(s.get('xsampa') or '未映射')+'</td><td>'+f"{s['time_range']['start_sec']:.3f}–{s['time_range']['end_sec']:.3f}"+'</td></tr>' for s in result['segments'])
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>音素识别结果</title>
<style>body{{font:18px/1.7 system-ui;background:#edf4f4;color:#163b40;max-width:1000px;margin:36px auto;padding:24px}}section{{background:white;padding:26px;border-radius:16px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ccd;padding:8px}}a{{color:#006a76}}.phones{{font-size:28px;overflow-wrap:anywhere}}</style>
<h1>音素识别结果</h1><p>候选模型 · {html.escape(result['mode'])} · 原文件 {crop['effective_start_sec']:.3f}–{crop['effective_end_sec']:.3f} 秒</p>
<section><audio controls src="input-crop.wav"></audio><p class="phones">{html.escape(' '.join(result['ipa_sequence']))}</p><p><a href="phones.ipa.txt" download>下载 IPA</a> · <a href="phones.xsampa.txt" download>下载 X-SAMPA</a> · <a href="result.json" download>下载完整结果</a></p></section>
<p>下方时间是 CTC 发射位置，用于查找声音附近；不代表已经确认的音素起止时长。未覆盖的 X-SAMPA 会明确标记。</p>
<section><table><tr><th>IPA</th><th>X-SAMPA</th><th>原文件发射位置（秒）</th></tr>{rows}</table></section></html>'''
    (output/'index.html').write_text(text,encoding='utf-8')


def run_candidate(*, audio: Path, start: float, duration: float, output: Path, bundle_path: Path | None = None,
                  fixture_path: Path | None = None, warm_repeats: int = 0, audio_cache: bool = True) -> dict:
    output=output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'):
        raise ValueError('choose a new output directory beneath .state/runs')
    if fixture_path is not None:
        pp=read_json(fixture_path);crop=pp['producer']['crop']
        rate=pp['producer']['input_audio']['sample_rate']
        if abs(crop['requested_start_sec']-start)>1/rate or abs(crop['requested_end_sec']-(start+duration))>1/rate:
            raise ValueError('fixture crop differs from the requested interval')
        output.mkdir(parents=True)
        return replay_result(audio,fixture_path,output,fixture=True)
    if bundle_path is None:
        raise ValueError('real candidate inference requires a runtime bundle')
    bundle=validate_bundle(bundle_path);request=audio_request(audio,start,duration)
    if warm_repeats not in {0,5}:raise ValueError('benchmark repeats must be zero or five')
    with serial_candidate_job():
        output.mkdir(parents=True);request_path=output/'request.json';digest=write_json_new(request_path,request)
        command=[bundle['python']['path'],'-X','utf8',bundle['worker']['path'],'--bundle',str(bundle_path.resolve()),'--bundle-sha256',sha256_file(bundle_path),
                 '--request',str(request_path),'--request-sha256',digest,'--output',str(output/'worker'),'--warm-repeats',str(warm_repeats),'--audio-cache','on' if audio_cache else 'off']
        record={'command':command,'started_at_utc':datetime.now(timezone.utc).isoformat(),'offline_worker':True}
        write_json_new(output/'COMMAND.json',record)
        start_clock=time.monotonic()
        try:
            with (output/'worker.log').open('w',encoding='utf-8') as log:
                process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,shell=False,timeout=930,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
        except subprocess.TimeoutExpired as exc:
            write_json_new(output/'EXIT.json',{'exit_code':None,'status':'timeout','elapsed_seconds':time.monotonic()-start_clock})
            raise RuntimeError('candidate worker timed out; see worker.log') from exc
        elapsed=time.monotonic()-start_clock
        write_json_new(output/'EXIT.json',{'exit_code':process.returncode,'elapsed_seconds':elapsed,'ended_at_utc':datetime.now(timezone.utc).isoformat()})
        if process.returncode:
            raise RuntimeError(f'candidate worker failed ({process.returncode}); see {output / "worker.log"}')
        worker=read_json(output/'worker/RESULT.json');spec=worker['iterations'][0]['artifact']
        verify_file(Path(spec['posterior_path']),spec['posterior_sha256'])
        result=replay_result(audio,Path(spec['posterior_path']),output)
        write_json_new(output/'PERFORMANCE.json',{'process_wall_seconds':elapsed,'worker':worker,'scope':'new process + model load + audio + heads + serialization + contract verification; warm repeats separately recorded'})
        return result

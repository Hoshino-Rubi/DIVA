"""One reviewed GAME note job in its own offline process; no retries or training."""
from __future__ import annotations
import argparse,os,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.local.common import read,sha,write


def verify_review(review_file: Path,request_file: Path,snapshot: Path,output: Path) -> dict:
    r=read(review_file)
    if r.get('schema_version')!='game_note_launch_review_v1' or r.get('optimizer_updates')!=0 or r.get('device')!='cpu':
        raise ValueError('Expected frozen zero-training CPU review')
    for f,digest in r['code_sha256'].items():
        if sha(ROOT/f)!=digest:raise ValueError('Launch review invalidated: '+f)
    for b in r['validation']:
        if sha(ROOT/b['path'])!=b['sha256']:raise ValueError('Verification evidence changed')
    if sha(snapshot/'snapshot-lock.json')!=r['snapshot_lock_sha256']:raise ValueError('Snapshot lock changed')
    matches=[x for x in r['jobs'] if Path(x['output']).resolve()==output.resolve()]
    if len(matches)!=1 or sha(request_file)!=matches[0]['request_sha256']:
        raise ValueError('Job request/output not authorized by frozen review')
    if len(r['jobs'])>r['maximum_jobs'] or r['maximum_graph_calls_per_job']!=11:
        raise ValueError('Invalid frozen job budget')
    return r


def run(review: Path,request: Path,snapshot: Path,output: Path) -> dict:
    verify_review(review,request,snapshot,output)
    from any2xsampa.adapters.game_onnx import GameOnnxRuntime,checked_request,load_audio
    from any2xsampa.music.game_notes import POLICY_PATH,decode_regions
    req=checked_request(read(request));output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();runtime=None
    write(output/'START.json',{'request_sha256':sha(request),'review_sha256':sha(review),'device':'cpu',
                             'seed':17,'maximum_encoder_calls':1,'maximum_graph_calls':11,'retried':False})
    # Abort only this isolated worker at the safety deadline.
    import threading
    def timeout():
        write(output/'TIMEOUT.json',{'seconds':180,'only_current_process_terminated':True})
        os._exit(73)
    timer=threading.Timer(180,timeout);timer.daemon=True;timer.start()
    try:
        import socket
        def forbidden(*args,**kwargs):raise RuntimeError('Network forbidden during GAME inference')
        socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden;socket.create_connection=forbidden
        waveform,info=load_audio(req,output/'input-crop.wav')
        write(output/'AUDIO_READY.json',info)
        runtime=GameOnnxRuntime(snapshot,device='cpu')
        write(output/'FORWARD_STARTED.json',{'encoder_requests':1,'max_graph_calls':11})
        raw=runtime.infer(waveform)
        import numpy as np
        rawfile=output/'raw-notes.npz'
        with rawfile.open('xb') as f:np.savez_compressed(f,**raw)
        result=decode_regions(raw,start_sec=info['effective_start_sec'],duration_sec=info['model_duration_sec'])
        result.update(input=info,mode='real',model_id='openvpi/GAME-1.0.3-medium-onnx',
            snapshot_lock_sha256=sha(snapshot/'snapshot-lock.json'),policy_sha256=sha(POLICY_PATH),
            device='CPUExecutionProvider',seed=17,language='yue',language_id=3,
            raw_evidence={'file':rawfile.name,'sha256':sha(rawfile)},runtime_calls=runtime.calls)
        if len(runtime.calls)!=11:raise ValueError('Unexpected inference call count')
        if sha(Path(req['audio_path']))!=req['audio_sha256']:raise ValueError('Audio changed during model job')
        write(output/'prediction.json',result)
        status={'status':'completed','prediction':str(output/'prediction.json'),'prediction_sha256':sha(output/'prediction.json'),
                'encoder_calls':1,'graph_calls':len(runtime.calls),'optimizer_updates':0,
                'elapsed_seconds':time.monotonic()-started,'notes':len(result['notes']),'quality_validated':False}
        write(output/'EXIT.json',status);return status
    except BaseException as exc:
        write(output/'EXIT.json',{'status':'failed','error':str(exc),'elapsed_seconds':time.monotonic()-started,
                                 'forward_started':(output/'FORWARD_STARTED.json').exists(),
                                 'completed_graph_calls':runtime.calls if runtime else [],'retried':False})
        raise
    finally:timer.cancel()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('review','request','snapshot','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(__import__('json').dumps(run(a.review,a.request,a.snapshot,a.output)))
if __name__=='__main__':main()

"""Measure local packaging overhead without loading any acoustic model."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import wave

def measure(operation, count=5):
    values=[]
    for _ in range(count):
        start=time.perf_counter()
        operation()
        values.append(time.perf_counter()-start)
    return dict(seconds=values,median_seconds=statistics.median(values),runs=count)

def sha(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def inside(args):
    sys.path.insert(0,str(args.app/"src"))
    from any2xsampa.local.common import crop
    from any2xsampa.local.game_score_tasks import export_saved_notes
    from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline
    audio=args.output/"generated.wav"
    with wave.open(str(audio),"wb") as wav:
        wav.setparams((1,2,44100,0,"NONE","not compressed"))
        wav.writeframes(bytes(44100*2*10))
    count=0
    def crop_one():
        nonlocal count
        crop(audio,1,8,args.output/("crop-"+str(count)+".wav"))
        count+=1
    import numpy as np
    import soxr
    signal=np.zeros(44100*8,dtype=np.float32)
    def resample():
        result=soxr.resample(signal,44100,16000)
        assert len(result)==16000*8 and not np.any(result)
    digest=sha(audio)
    notes=dict(schema_version="note_transcription_v1",mode="fixture",model_id="fixture",
        time_basis="original_audio",source_start_sec=0.,source_end_sec=10.,
        input=dict(audio_path=str(audio),audio_sha256=digest),notes=[
            dict(region_index=0,start_sec=.2,end_sec=1.2,duration_sec=1.,pitch_midi=60,
                 pitch_midi_float=60.,presence=True,touches_input_start=False,touches_input_end=False)])
    file=args.output/"notes.json";file.write_text(json.dumps(notes),encoding="utf8")
    exports=0
    def export():
        nonlocal exports
        output=args.output/("score-"+str(exports)+".vsqx");exports+=1
        proof=export_saved_notes(dict(result_path=str(file),output_path=str(output)))
        assert proof["model_forwards"]==0 and parse_vsqx_timeline(output)["notes"][0]["pitch"]==60
    result=dict(crop=measure(crop_one),resample=measure(resample),vsqx_export=measure(export))
    if args.weight:
        expected=args.weight_sha256
        def weight_hash():
            assert sha(args.weight)==expected,"Weight changed"
        result["full_weight_sha256"]=measure(weight_hash)
        result["weight_bytes"]=args.weight.stat().st_size
    (args.output/"hot.json").write_text(json.dumps(result,indent=2),encoding="utf8")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--app",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--weight",type=Path)
    p.add_argument("--weight-sha256")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    args.app=args.app.resolve();args.output=args.output.resolve()
    if bool(args.weight)!=bool(args.weight_sha256):
        raise ValueError("Weight input must have an expected SHA256")
    if args.inside:
        inside(args);return
    args.output.mkdir(parents=True,exist_ok=False)
    python=args.app/"runtime/core/python.exe"
    env={k:v for k,v in os.environ.items() if not k.startswith(("PYTHON","ANY2XSAMPA_"))}
    env.update(ANY2XSAMPA_HOME=str(args.app),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")
    script="from any2xsampa.local.engine import LocalEngine; import any2xsampa; print(any2xsampa.__version__)"
    def cold():
        done=subprocess.run([str(python),"-B","-X","utf8","-c",script],cwd=args.output,
            env=env,capture_output=True,text=True,encoding="utf8",timeout=30)
        if done.returncode:raise RuntimeError(done.stderr)
    cold_result=measure(cold,3)
    command=[str(python),"-B","-X","utf8",str(Path(__file__).resolve()),
             "--app",str(args.app),"--output",str(args.output),"--inside"]
    if args.weight:command+=["--weight",str(args.weight.resolve()),"--weight-sha256",args.weight_sha256]
    done=subprocess.run(command,cwd=args.output,env=env,capture_output=True,text=True,encoding="utf8",timeout=300)
    (args.output/"worker.log").write_text(done.stdout+done.stderr,encoding="utf8")
    if done.returncode:raise RuntimeError(done.stderr)
    result=dict(schema_version="local_release_benchmark_v1",model_forwards=0,model_loads=0,
        new_process_import=cold_result,hot=json.loads((args.output/"hot.json").read_text()),
        source_sha256=sha(Path(__file__)),
        limitations=["New-process startup, not cold disk-cache startup",
                     "Acoustic model compute/load and GPU memory are not measured",
                     "Synthetic PCM; no recognition-quality comparison"])
    (args.output/"RESULTS.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result),flush=True)

if __name__=="__main__":main()

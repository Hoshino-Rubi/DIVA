"""Offline product regression entry used locally and by Windows CI; no model jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "test_model_install.py", "test_worker_log_ownership.py", "test_local_engine.py",
    "test_local_desktop_layout.py", "test_local_whole_audio.py", "test_local_score.py",
    "test_game_vsqx.py", "test_game_notes.py", "test_portable_release_policy.py", "test_release_build.py",
    "test_release_version.py",
    "test_local_storage.py",
    "test_stream_portable_release.py",
]
def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--skip-desktop-build",action="store_true")
    a=p.parse_args()
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    def command(label, argv, cwd=ROOT, env=None, timeout=600):
        started=time.monotonic()
        done=subprocess.run([str(x) for x in argv],cwd=cwd,env=env,capture_output=True,
                            text=True,encoding="utf-8",errors="replace",timeout=timeout)
        (out/(label+".stdout.log")).write_text(done.stdout,encoding="utf-8")
        (out/(label+".stderr.log")).write_text(done.stderr,encoding="utf-8")
        records.append(dict(label=label,command=[str(x) for x in argv],
                            exit_code=done.returncode,elapsed_seconds=time.monotonic()-started))
        write(out/"COMMANDS.json",records)
        print(label,done.returncode,round(records[-1]["elapsed_seconds"],3),flush=True)
        if done.returncode:raise RuntimeError(label+" failed; see saved logs")
    test_env={**os.environ, "PYTHONPATH":str(ROOT/"src"), "HF_HUB_OFFLINE":"1",
              "TRANSFORMERS_OFFLINE":"1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1"}
    command("python-tests",[sys.executable,"-X","utf8","-m","pytest",
            *["tests/"+x for x in TESTS],"-q","-p","no:cacheprovider",
            "--basetemp="+str(out/"pytest-temp"),"--junitxml="+str(out/"pytest.xml")],env=test_env)
    for name in ("Any2Xsampa.Setup.Tests","Any2Xsampa.LocalTransport.Tests"):
        command(name+"-build",["dotnet","build","apps/"+name+"/"+name+".csproj","-c","Release",
                              "--ignore-failed-sources","-p:NuGetAudit=false"])
    dll=ROOT/"apps/Any2Xsampa.LocalTransport.Tests/bin/Release/net8.0/Any2Xsampa.LocalTransport.Tests.dll"
    command("setup-faults",["dotnet",ROOT/"apps/Any2Xsampa.Setup.Tests/bin/Release/net8.0/Any2Xsampa.Setup.Tests.dll",out/"installer"])
    command("diagnostics",["dotnet",dll,"--diagnostics",out/"diagnostics"])
    command("release-version",["dotnet",dll,"--release"])
    # Fresh path with spaces/Unicode; no repository cwd or inherited model registry.
    clean=out/"clean installation 中文"
    clean.mkdir()
    shutil.copytree(ROOT/"src",clean/"src",ignore=shutil.ignore_patterns("__pycache__","*.pyc","*.egg-info"))
    write(clean/"models.json",dict(schema_version="local_model_registry_v1",models=[],default_model=None))
    write(clean/"local-app.json",dict(root=".",python=sys.executable,models="models.json"))
    audio=clean/"输入.wav"
    with wave.open(str(audio),"wb") as wav:
        wav.setparams((1,2,16000,0,"NONE","not compressed"));wav.writeframes(bytes(16000*2*2))
    fixture=clean/"fixture.json"
    write(fixture,dict(segments=[dict(start_sec=.1,end_sec=.2,confidence=.9,
          candidate_phones=[dict(symbol="i",confidence=.9)])]))
    env={k:v for k,v in os.environ.items() if not k.startswith(("ANY2XSAMPA_","PYTHON"))}
    env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONUTF8="1")
    command("relocated-stdio",["dotnet",dll,clean/"local-app.json",fixture,audio,clean/"result"],cwd=clean,env=env)
    digest=hashlib.sha256(audio.read_bytes()).hexdigest()
    def note(start,end,midi,index):
        return dict(region_index=index,start_sec=start,end_sec=end,duration_sec=end-start,
                    pitch_midi=midi,pitch_midi_float=float(midi),presence=True,
                    touches_input_start=False,touches_input_end=False)
    notes=clean/"notes.json"
    write(notes,dict(schema_version="note_transcription_v1",mode="fixture",model_id="fixture",
          time_basis="original_audio",source_start_sec=0.,source_end_sec=2.,
          input=dict(audio_path=str(audio),audio_sha256=digest),
          notes=[note(.1,.5,60,0),note(.6,1.2,62,1),note(1.4,1.8,62,2)]))
    command("relocated-vsqx",["dotnet",dll,"--game-vsqx",clean/"local-app.json",
            clean/"result/result.json",notes,clean/"score"],cwd=clean,env=env)
    if not a.skip_desktop_build:
        command("desktop-build",["dotnet","build","apps/Any2Xsampa.Desktop/Any2Xsampa.Desktop.csproj",
            "-c","Debug","-p:Platform=x64","-p:WindowsPackageType=None",
            "-p:RestoreLockedMode=true","-p:NuGetAudit=false"])
    write(out/"RESULTS.json",dict(status="passed",model_forwards=0,training_updates=0,
        model_downloads=0,desktop_build=not a.skip_desktop_build,commands=len(records)))
    print("RELEASE_CHECKS_PASSED",flush=True)

if __name__=="__main__":main()

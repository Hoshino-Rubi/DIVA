"""Offline, machine-local WinUI distribution; runtimes/models remain separate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--models",default="local-models.json",help="Model registry path relative to workspace root")
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/"dist"):
        raise ValueError("Choose a new release directory under dist")
    command=["dotnet","publish",str(ROOT/"apps/Any2Xsampa.Desktop/Any2Xsampa.Desktop.csproj"),
             "--no-restore","--self-contained","true","-c","Debug","-p:Platform=x64",
             "-p:WindowsPackageType=None","-p:WindowsAppSDKSelfContained=true",
             "-p:PublishReadyToRun=false","-p:PublishTrimmed=false","-maxcpucount:4","-o",str(out)]
    result=subprocess.run(command,cwd=ROOT,shell=False,capture_output=True,text=True,encoding="utf-8",errors="replace",
                          creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    out.mkdir(parents=True,exist_ok=True)
    (out/"PUBLISH.json").write_text(json.dumps({"command":command,"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr},ensure_ascii=False,indent=2),encoding="utf8")
    if result.returncode:raise SystemExit(result.returncode)
    import os
    relative=os.path.relpath(ROOT,out).replace("\\","/")
    registry=(ROOT/a.models).resolve()
    if not registry.is_relative_to(ROOT) or not registry.is_file():
        raise ValueError("Model registry must already exist beneath the workspace")
    (out/"local-app.json").write_text(json.dumps({"schema_version":"local_desktop_config_v1","root":relative,"python":".venv/Scripts/python.exe","models":a.models},indent=2),encoding="utf8")
    launcher='@echo off\nsetlocal\nset "ANY2XSAMPA_HOME=%~dp0'+relative.replace('/','\\')+'"\nset "PYTHONPATH=%ANY2XSAMPA_HOME%\\src"\nset "ANY2XSAMPA_MODELS=%ANY2XSAMPA_HOME%\\'+a.models.replace('/','\\')+'"\n"%ANY2XSAMPA_HOME%\\.venv\\Scripts\\python.exe" -X utf8 -m any2xsampa.cli %*\nexit /b %errorlevel%\n'
    (out/"any2xsampa.cmd").write_text(launcher,encoding="ascii")
    print(json.dumps({"desktop":str(out/"Any2Xsampa.Desktop.exe"),"cli":str(out/"any2xsampa.cmd"),"runtime_and_models":"external existing local installations","cross_machine_installation_verified":False}))

if __name__=="__main__":main()

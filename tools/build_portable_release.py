"""Build a portable Windows application from verified, existing installations."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from any2xsampa.local.common import sha, read
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.markers import default_environment

PX=ROOT/".state/runs/limited-data-20260906T050251"
GAME=ROOT/".state/runs/note-game-qualification-20261002T113704Z"
FROZEN=ROOT/".state/runs/guarded-retention-20260919T041500/CANDIDATE_FREEZE.json"
CHUNK=1024**3


def write(file,value):
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def kept(relative):
    return not (set(relative.parts)&{"__pycache__","test","tests",".git"} or
                relative.suffix.lower() in {".pyc",".pyo",".pdb",".lib",".whl"} or relative.name=="direct_url.json")


def copy_tree(source,target,exclude=()):
    for file in source.rglob("*"):
        relative=file.relative_to(source)
        if file.is_file() and kept(relative) and not any(relative.parts[0]==x for x in exclude):
            copy(file,target/relative)


def dependencies(site,names,version):
    dists={canonicalize_name(d.metadata["Name"]):d for d in metadata.distributions(path=[str(site)])}
    env=default_environment()
    env.update(python_version=version,python_full_version=version+".0",extra="")
    found=set()
    def visit(name):
        name=canonicalize_name(name)
        if name in found:return
        if name not in dists:raise ValueError("Missing installed dependency: "+name)
        found.add(name)
        for raw in dists[name].requires or []:
            req=Requirement(raw)
            if req.marker is None or req.marker.evaluate(env):
                visit(req.name)
    for name in names:visit(name)
    return [dists[n] for n in sorted(found)]


def runtime(base,site,destination,version,packages=None):
    destination.mkdir(parents=True,exist_ok=False)
    for name in ["python.exe","pythonw.exe","python3.dll","python"+version.replace(".","")+".dll",
                 "vcruntime140.dll","vcruntime140_1.dll","LICENSE.txt"]:
        copy(base/name,destination/name)
    for file in (base/"DLLs").iterdir():
        if file.is_file() and file.suffix.lower() in {".dll",".pyd"} and ".cp314t-" not in file.name and "_d." not in file.name:
            copy(file,destination/"DLLs"/file.name)
    copy_tree(base/"Lib",destination/"Lib",exclude=("site-packages","idlelib","tkinter","turtledemo","ensurepip"))
    target=destination/"Lib/site-packages"
    lock=[]
    if packages is None:
        copy_tree(site,target,exclude=("pip","pip-25.0.1.dist-info"))
        selected=[d for d in metadata.distributions(path=[str(site)]) if d.metadata["Name"].lower()!="pip"]
    else:
        selected=dependencies(site,packages,version)
        seen=set()
        for dist in selected:
            for relative in dist.files or []:
                p=Path(str(relative))
                if p.is_absolute() or ".." in p.parts or not kept(p) or str(p) in seen:continue
                seen.add(str(p))
                file=site/p
                if file.is_file():copy(file,target/p)
    # Isolated executable: no registry, PATH, PYTHONHOME, PYTHONPATH or system site-packages.
    (destination/("python"+version.replace(".","")+"._pth")).write_text(
        ".\nLib\nDLLs\nLib/site-packages\n../../src\n",encoding="utf-8")
    for dist in selected:
        lock.append({"name":dist.metadata["Name"],"version":dist.version})
    write(destination/"DEPENDENCIES.json",lock)
    return lock


def pinned(source,target,row):
    if source.stat().st_size!=row["bytes"] or sha(source)!=row["sha256"]:
        raise ValueError("Source model changed: "+str(source))
    copy(source,target)


def models(out,assets):
    lock=read(PX/"model/snapshot-lock.json")
    specs=[]
    for row in lock["files"]:
        rel="models/phoneticxeus/"+row["file"]
        specs.append({"path":rel,"bytes":row["bytes"],"sha256":row["sha256"]})
        if row["file"]!="model.safetensors":
            pinned(PX/"model"/row["file"],out/rel,row)
    portable_lock={k:lock[k] for k in ("schema_version","model_id","revision","license","files")}
    portable_lock["origin_lock_sha256"]=sha(PX/"model/snapshot-lock.json")
    write(out/"models/phoneticxeus/snapshot-lock.json",portable_lock)
    lp=out/"models/phoneticxeus/snapshot-lock.json"
    specs.append({"path":lp.relative_to(out).as_posix(),"sha256":sha(lp),"bytes":lp.stat().st_size})
    weight=next(r for r in lock["files"] if r["file"]=="model.safetensors")
    source=PX/"model/model.safetensors"
    if sha(source)!=weight["sha256"]:raise ValueError("Original weight changed")
    parts=[]
    with source.open("rb") as stream:
        index=1
        while block:=stream.read(CHUNK):
            name=f"PhoneticXeus-model.safetensors.{index:03d}"
            file=assets/name
            file.write_bytes(block)
            parts.append({"name":name,"bytes":len(block),"sha256":hashlib.sha256(block).hexdigest()})
            index+=1
    download={"url":f"https://huggingface.co/changelinglab/PhoneticXeus/resolve/{lock['revision']}/model.safetensors",
              "offline_name":"model.safetensors","target":"models/phoneticxeus/model.safetensors",
              "sha256":weight["sha256"],"bytes":weight["bytes"],"offline_parts":parts}
    bundles=[{"id":"phoneticxeus","name":"PhoneticXeus 428 类音素模型","license":"Apache-2.0",
              "files":specs,"downloads":[download]}]
    glock=read(GAME/"model/snapshot-lock.json")
    archive=GAME/"downloads/GAME-1.0.3-medium-onnx.zip"
    if sha(archive)!=glock["archive_sha256"]:raise ValueError("GAME archive changed")
    copy(archive,assets/archive.name)
    gfiles=[{"path":"models/game/"+r["path"],"bytes":r["bytes"],"sha256":r["sha256"]} for r in glock["files"]]
    copy(GAME/"model/snapshot-lock.json",out/"models/game/snapshot-lock.json")
    gp=out/"models/game/snapshot-lock.json"
    gfiles.append({"path":"models/game/snapshot-lock.json","sha256":sha(gp),"bytes":gp.stat().st_size})
    members=[]
    with zipfile.ZipFile(archive) as z:
        for r in glock["files"]:
            names=[n for n in z.namelist() if n==r["path"] or n.endswith("/"+r["path"])]
            if len(names)!=1:raise ValueError("Ambiguous GAME member")
            if z.getinfo(names[0]).file_size!=r["bytes"]:raise ValueError("GAME archive size changed")
            members.append({"archive_path":names[0],"target":"models/game/"+r["path"]})
    bundles.append({"id":"game","name":"GAME 粤语音高与起止点模型","license":"CC-BY-NC-SA-4.0",
                    "files":gfiles,"downloads":[{"url":"https://github.com/openvpi/GAME/releases/download/v1.0.3/"+archive.name,
                    "offline_name":archive.name,"bytes":archive.stat().st_size,"sha256":glock["archive_sha256"],"members":members}]})
    write(out/"model-downloads.json",{"schema_version":"local_model_downloads_v1","auto_download":True,
                                    "bundles":bundles,"inference_network":False})
    return bundles


def public_guard_policy(value, original_sha256):
    from any2xsampa.training.limited_data_artifacts import sha256_object
    result={k:v for k,v in value.items() if k not in {"report_bindings","policy_sha256"}}
    result["origin_policy_file_sha256"]=original_sha256
    result["report_binding_sha256s"]=[r["sha256"] for r in value.get("report_bindings",[])]
    result["policy_sha256"]=sha256_object(result)
    return result


def candidate(out):
    frozen=read(FROZEN)
    derived={k:frozen[k] for k in ("schema_version","default_replacement","test_opening","selected_seed","selected_variant")}
    derived["origin_candidate_sha256"]=sha(FROZEN)
    derived["migration"]="File locations made relative; checkpoints unchanged; policy report paths removed without changing routing; evaluation history not distributed"
    for key,name in [("teacher_checkpoint","teacher.safetensors"),("student_checkpoint","student.safetensors"),("policy","policy.json")]:
        row=frozen[key]
        src=Path(row["path"])
        if sha(src)!=row["sha256"]:raise ValueError("Frozen asset mismatch")
        target=out/"models/guarded"/name
        if key=="policy":
            write(target,public_guard_policy(read(src),row["sha256"]))
        else:
            copy(src,target)
        derived[key]={"path":name,"sha256":sha(target),"origin_sha256":row["sha256"]}
    file=out/"models/guarded/candidate.json"
    write(file,derived)
    return file


def package(out,review):
    sources=[p for p in (out/"src/any2xsampa").rglob("*") if p.is_file()]
    sources += [out/"runtime/phonetic/python.exe",out/"models/guarded/candidate.json",
                out/"models/guarded/teacher.safetensors",out/"models/guarded/student.safetensors",
                out/"models/guarded/policy.json",out/"models/phoneticxeus/snapshot-lock.json",
                out/"models/phoneticxeus/src/model/xeusphoneme/resources/ipa_vocab.json"]
    def binding(p):return {"path":p.relative_to(out).as_posix(),"sha256":sha(p)}
    sources += [out/"runtime/core/python314._pth",out/"runtime/phonetic/python312._pth",
                out/"runtime/core/python314.dll",out/"runtime/phonetic/python312.dll",
                out/"runtime/core/DEPENDENCIES.json",out/"runtime/phonetic/DEPENDENCIES.json"]
    write(out/"runtime-review.json",review)
    value={"schema_version":"local_model_package_v1","id":"guarded-candidate-v2","root":".",
           "backend":"guarded-phoneticxeus","quality":"研究候选 / 非质量验收通过",
           "devices":["cuda"],"python":"runtime/phonetic/python.exe","label_count":428,
           "paths":{"snapshot":"models/phoneticxeus","candidate":"models/guarded/candidate.json",
                    "vocabulary":"models/phoneticxeus/src/model/xeusphoneme/resources/ipa_vocab.json"},
           "candidate_sha256":sha(out/"models/guarded/candidate.json"),
           "review":binding(out/"runtime-review.json"),"bindings":[binding(p) for p in sources] + [
               {"path":"models/phoneticxeus/model.safetensors",
                "sha256":next(r["sha256"] for r in read(PX/"model/snapshot-lock.json")["files"] if r["file"]=="model.safetensors")}],
           "default_replacement":False}
    write(out/"guarded-candidate-v2.json",value)
    write(out/"local-models.json",{"schema_version":"local_model_registry_v1","default_model":None,
        "models":[{"id":value["id"],"package":"guarded-candidate-v2.json","sha256":sha(out/"guarded-candidate-v2.json")}]})
    write(out/"local-notes.json",{"schema_version":"local_note_runtime_v1","snapshot":"models/game","python":"runtime/core/python.exe"})
    write(out/"local-app.json",{"schema_version":"local_desktop_config_v1","root":".","python":"runtime/core/python.exe","models":"local-models.json"})


def venv_home(directory):
    settings=dict(line.split(" = ",1) for line in (directory/"pyvenv.cfg").read_text().splitlines() if " = " in line)
    return Path(settings["home"])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--desktop",type=Path,required=True)
    p.add_argument("--review",type=Path,required=True)
    a=p.parse_args()
    out,assets=a.output.resolve(),a.assets.resolve()
    if out.exists() or assets.exists():raise FileExistsError("Use new build directories")
    review=read(a.review)
    for name,digest in review["source_sha256"].items():
        if sha(ROOT/name)!=digest:raise ValueError("Review invalidated: "+name)
    for row in review["verification"]:
        if sha(ROOT/row["path"])!=row["sha256"]:raise ValueError("Evidence changed")
    out.mkdir(parents=True);assets.mkdir(parents=True)
    copy_tree(a.desktop.resolve(),out,exclude=("runtime","models",".state"))
    copy_tree(ROOT/"src",out/"src")
    copy_tree(ROOT/"docs/distribution",out)
    source_code=[x for x in (out/"src").rglob("*.py")]
    if any(str(ROOT).encode() in x.read_bytes() for x in source_code):
        raise ValueError("Product source embeds workspace path")
    print("copy core runtime",flush=True)
    runtime(venv_home(ROOT/".venv"),ROOT/".venv/Lib/site-packages",out/"runtime/core","3.14",
            packages=["onnxruntime-gpu","librosa","soundfile"])
    print("copy phonetic runtime",flush=True)
    runtime(venv_home(PX/"runtime"),PX/"runtime/Lib/site-packages",
            out/"runtime/phonetic","3.12")
    print("models and fixed offline assets",flush=True)
    models(out,assets)
    candidate(out)
    package(out,review)
    copy(ROOT/"docs/distribution/THIRD_PARTY_NOTICES.md",assets/"MODEL_NOTICES.md")
    (out/"any2xsampa.cmd").write_text(
        '@echo off\nsetlocal\nset "ANY2XSAMPA_HOME=%~dp0"\nset "ANY2XSAMPA_MODELS=%~dp0local-models.json"\n'
        'set "ANY2XSAMPA_NOTES="\nset "ANY2XSAMPA_GAME_SNAPSHOT="\nset "PYTHONHOME="\n'
        '"%~dp0runtime\\core\\python.exe" -B -X utf8 -m any2xsampa.cli %*\nexit /b %ERRORLEVEL%\n',
        encoding="ascii")
    print("BUILD_READY",out,flush=True)
    write(out/"BUILD.json",{"schema_version":"portable_build_v1","model_forwards":0,"model_downloads":0,
                           "runtime_source":"existing audited CPython and installed dependencies",
                           "runtime_manifest_files":["runtime/core/DEPENDENCIES.json","runtime/phonetic/DEPENDENCIES.json"],
                           "candidate_origin_sha256":sha(FROZEN),"review_sha256":sha(out/"runtime-review.json"),
                           "runtime_development_static_libs_excluded":True})

if __name__=="__main__":main()

"""Publish NEW reviewed local packages; never rewrite a historical candidate bundle."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from any2xsampa.local.common import read,sha,write

LEGACY=ROOT/".state/runs/decoder-usability-20260920T011000/candidate-bundle-v1.json"
RUNTIME=ROOT/"src/any2xsampa/adapters/phoneticxeus_runtime.py"

def code_paths():
    paths=list((ROOT/"src/any2xsampa/local").glob("*.py"))
    for name in ("candidate_inference.py","phoneticxeus.py","phoneticxeus_runtime.py","phoneticxeus_monitor.py",
                 "guarded_phoneticxeus.py","teacher_guard.py","onnx_espeak.py","audio_provenance.py",
                 "recognizer.py","recognizer_real.py"):
        paths.append(ROOT/"src/any2xsampa/adapters"/name)
    paths += [ROOT/"src/any2xsampa/training"/x for x in ("limited_data_artifacts.py","limited_data_reference.py","phoneticxeus_adaptation.py")]
    paths += [ROOT/"src/any2xsampa/config"/x for x in ("ipa_to_xsampa.json","ipa_to_xsampa_v2.json","ipa_aliases.json","recognizer_onnx_espeak_ipa_map.json")]
    paths += [ROOT/"src/any2xsampa/exporters/ipa_xsampa.py"]
    paths += [ROOT/"src/any2xsampa/core/models.py",ROOT/"src/any2xsampa/pipeline/normalizer.py",ROOT/"src/any2xsampa/cli.py"]
    paths += [ROOT/"src/any2xsampa/config/local_score_v1.json",
              ROOT/"src/any2xsampa/music/pitch_note_analysis.py",
              ROOT/"src/any2xsampa/vocaloid/draft.py",ROOT/"src/any2xsampa/vocaloid/vsqx.py",
              ROOT/"src/any2xsampa/vocaloid/vsqx_alignment.py",ROOT/"src/any2xsampa/vocaloid/phoneme_policy.py",
              ROOT/"examples/accepted/congqian_onnx_miku_v4c.vsqx"]
    paths += [ROOT/"src/any2xsampa/adapters/game_onnx.py",
              ROOT/"src/any2xsampa/music/game_notes.py",ROOT/"src/any2xsampa/config/game_note_v1.json",
              ROOT/"src/any2xsampa/vocaloid/resources/vsq4_empty.vsqx"]
    paths.append(Path(__file__).resolve())
    return paths

def binding(p):
    p=Path(p).resolve()
    return {"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--review",type=Path,required=True)
    parser.add_argument("--install-registry",action="store_true")
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.review=args.review.resolve()
    if not args.output.is_relative_to(ROOT):
        raise ValueError("Runtime package output must be inside the workspace")
    review=read(args.review)
    if review.get("approved_max_smoke_forwards") not in {0, 2} or review.get("new_training_updates")!=0:
        raise ValueError("Expected a hash-bound zero- or two-forward engineering review")
    sources=code_paths()
    for p in sources:
        if review["code_sha256"].get(p.relative_to(ROOT).as_posix())!=sha(p):
            raise ValueError("Review invalidated by code change: "+str(p))
    for evidence in review["verification_evidence"]:
        if sha(ROOT/evidence["path"])!=evidence["sha256"]:
            raise ValueError("Validation evidence changed")
    legacy=read(LEGACY)
    for key in ("candidate","snapshot_lock","python","vocabulary"):
        if sha(Path(legacy[key]["path"])) != legacy[key]["sha256"]:
            raise ValueError("Legacy asset changed: "+key)
    current=RUNTIME.read_text(encoding="utf-8")
    original=re.sub(r"    # Serial checkpoint loads[\s\S]*?torch.set_num_interop_threads\(1\)","    torch.set_num_interop_threads(1)",current,count=1)
    expected=next(x["sha256"] for x in legacy["code_bindings"] if Path(x["path"]).resolve()==RUNTIME.resolve())
    if hashlib.sha256(original.encode()).hexdigest()!=expected:
        raise ValueError("Cannot establish exact legacy source difference")
    for spec in legacy["code_bindings"]:
        if Path(spec["path"]).resolve()!=RUNTIME.resolve() and sha(Path(spec["path"]))!=spec["sha256"]:
            raise ValueError("Additional unreviewed legacy drift")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"LEGACY_RUNTIME.diff").write_text("".join(difflib.unified_diff(original.splitlines(True),current.splitlines(True),fromfile="legacy-exact-hash",tofile="current")),encoding="utf8")
    import os
    base=os.path.relpath(ROOT,args.output).replace("\\","/")
    common={"schema_version":"local_model_package_v1","root":base,"review":binding(args.review),
            "input":{"audio_only":True,"max_duration_seconds":20},"readiness":"hash_verified",
            "default_replacement":False,"capabilities":["PhoneSegment","IPA","explicit_unmapped_xsampa","emission_time_diagnostic"]}
    entries=[]
    def publish(value):
        file=args.output/(value["id"]+".json");write(file,value)
        entries.append({"id":value["id"],"package":file.relative_to(ROOT).as_posix(),"sha256":sha(file)})
    candidate=Path(legacy["candidate"]["path"]);frozen=read(candidate)
    candidate_sources=[candidate,Path(legacy["snapshot_lock"]["path"]),Path(legacy["python"]["path"]),Path(legacy["vocabulary"]["path"])]
    for name in ("teacher_checkpoint","student_checkpoint","policy"):
        s=frozen[name]
        if sha(Path(s["path"]))!=s["sha256"]:raise ValueError("Frozen candidate asset mismatch")
        candidate_sources.append(Path(s["path"]))
    publish({**common,"id":"guarded-candidate-v2","backend":"guarded-phoneticxeus","quality":"研究候选 / 非质量验收通过",
             "devices":["cuda"],"python":Path(legacy["python"]["path"]).relative_to(ROOT).as_posix(),
             "paths":{"snapshot":Path(legacy["snapshot"]).relative_to(ROOT).as_posix(),"candidate":candidate.relative_to(ROOT).as_posix(),
                      "vocabulary":Path(legacy["vocabulary"]["path"]).relative_to(ROOT).as_posix()},
             "candidate_sha256":legacy["candidate"]["sha256"],"label_count":428,
             "bindings":[binding(p) for p in sources+candidate_sources],"legacy_package_sha256":sha(LEGACY),
             "migration":"exactly reconstructed legacy runtime hash; thread-pool idempotency change only; new stdio session adapter",
             "whole_file_validation":"synthetic frame-and-transport tests; no new real whole-file forward in this package review"})
    manifest=read(ROOT/"models/MANIFEST.json")["models"]
    mh=next(x["primary_sha256"] for x in manifest if x["id"]=="wav2vec2-lv-60-espeak-cv-ft-onnx-int8")
    vh=next(x["primary_sha256"] for x in manifest if x["id"]=="wav2vec2-lv-60-espeak-cv-ft-tokenizer")
    model=ROOT/"models/onnx-espeak/model/onnx/model_int8.onnx";vocab=ROOT/"models/onnx-espeak/tokenizer/vocab.json"
    if sha(model)!=mh or sha(vocab)!=vh:raise ValueError("Existing ONNX resources differ")
    publish({**common,"id":"onnx-espeak-int8","backend":"onnx-espeak","quality":"通用基线 / 本轮未做真实运行验收",
             "devices":["cuda","cpu"],"python":".venv/Scripts/python.exe",
             "paths":{"model":model.relative_to(ROOT).as_posix(),"vocabulary":vocab.relative_to(ROOT).as_posix()},
             "model_sha256":mh,"vocabulary_sha256":vh,
             "bindings":[binding(p) for p in sources+[ROOT/".venv/Scripts/python.exe",model,vocab]]})
    registry={"schema_version":"local_model_registry_v1","models":entries,"default_model":None}
    # The release copy records paths relative to workspace; installer only writes a NEW registry.
    write(args.output/"registry-install.json",registry)
    if args.install_registry:write(ROOT/"local-models.json",registry)
    print(json.dumps({"models":[x["id"] for x in entries],"output":str(args.output),"registry_installed":args.install_registry}))

if __name__=="__main__":main()

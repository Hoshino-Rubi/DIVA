"""Serial whole-file task and independent replay; never retry failed chunks."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from .common import ROOT, Cancelled, check_cancel, read, request, sha, write
from .registry import package
from .whole_audio import POLICY, merge_results, plan_chunks, whole_input

def whole_request(params: dict) -> dict:
    if set(params)-{"audio","model","device","output_dir"}:
        raise ValueError("Whole-file recognition accepts audio/model/device/output_dir only; no clip or reference conditions")
    checked = request(params)
    return {k:v for k,v in checked.items() if k not in {"start","duration"}}

def run_whole(engine, params: dict, cancel, progress) -> dict:
    params = whole_request(params)
    check_cancel(cancel)
    spec = package(params["model"])
    if params["device"] not in spec["devices"]:
        raise ValueError("Selected device is not supported by the model")
    info = whole_input(Path(params["audio"]))
    chunks = plan_chunks(info)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(params.get("output_dir") or ROOT/".state/runs"/("whole-audio-"+stamp+"-"+uuid.uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version":"local_whole_evidence_v1", "policy":POLICY,"input":info,"chunks":chunks,
                "backend":spec["backend"],"model_id":params["model"],"package_sha256":spec["package_sha256"],
                "device":params["device"],"completed":[],"reference_conditions_used":False}
    write(output/"request.json", {"command":"recognize-whole","params":params})
    write(output/"PLAN.json", manifest)
    started = time.monotonic()
    try:
        for chunk in chunks:
            check_cancel(cancel)
            if sha(Path(params["audio"])) != info["sha256"]:
                raise ValueError("Whole-file input changed during processing")
            context = {"scope":"whole_file", "chunk_index":chunk["index"]+1,"chunk_total":len(chunks),
                       "processed_seconds":chunk["keep_start_sec"], "total_seconds":info["duration"]}
            def report(event):
                progress({**event, **context})
            report({"stage":"chunk_start"})
            result = engine.execute("recognize", {"audio":params["audio"],"model":params["model"],"device":params["device"],
                "start":chunk["start_sec"],"duration":chunk["end_sec"]-chunk["start_sec"],
                "output_dir":str(output/"chunks"/f'{chunk["index"]:04d}')},cancel,report)
            file = Path(result["result_path"])
            manifest["completed"].append({"index":chunk["index"],"result_path":str(file),"result_sha256":sha(file)})
            write(output/"PROGRESS.json", {"status":"in_progress","completed":manifest["completed"],
                  "chunks_total":len(chunks),"processed_seconds":chunk["keep_end_sec"]},replace=True)
            progress({**context,"stage":"chunk_done","processed_seconds":chunk["keep_end_sec"]})
        check_cancel(cancel)
        if sha(Path(params["audio"])) != info["sha256"]:
            raise ValueError("Whole-file input changed before merge")
        write(output/"whole-evidence.json", manifest)
        progress({"stage":"merging","scope":"whole_file","chunk_index":len(chunks),"chunk_total":len(chunks)})
        evidence = merge_results(manifest,cancel)
        evidence.update(whole_evidence_path=str(output/"whole-evidence.json"),whole_evidence_sha256=sha(output/"whole-evidence.json"))
        check_cancel(cancel)
        result = engine.save_result(output,"real",info,evidence)
        write(output/"EXIT.json", {"status":"completed","elapsed_seconds":time.monotonic()-started,
                                    "model_forwards":len(chunks),"chunks_completed":len(chunks),"chunks_total":len(chunks),"retried":False})
        return {"result_path":str(output/"result.json"),"output_dir":str(output),"mode":"real","scope":"whole_file",
                "phone_count":len(result["segments"]),"model_id":params["model"],"chunk_count":len(chunks),"load_count":evidence["load_count"]}
    except BaseException as exc:
        engine.close()
        status = "cancelled" if isinstance(exc,Cancelled) else "failed"
        write(output/"EXIT.json", {"status":status,"error":str(exc),"elapsed_seconds":time.monotonic()-started,
              "chunks_completed":len(manifest["completed"]),"chunks_total":len(chunks),"retried":False,"partial_artifacts_retained":True})
        write(output/"PROGRESS.json", {"status":status,"completed":manifest["completed"],"chunks_total":len(chunks)},replace=True)
        raise

def replay_whole(engine, params: dict, cancel, progress) -> dict:
    if set(params)-{"fixture","audio","start","duration","output_dir"}:
        raise ValueError("Invalid whole-file replay request")
    source_path = Path(params["fixture"]).resolve()
    saved = read(source_path)
    if saved.get("scope") != "whole_file":
        raise ValueError("Expected whole-file task result")
    info = whole_input(Path(params.get("audio") or saved["input"]["path"]))
    if info["sha256"] != saved["input"]["sha256"] or info["frames"] != saved["input"]["frames"]:
        raise ValueError("Whole-file replay audio mismatch")
    manifest_path = Path(saved["whole_evidence_path"])
    if sha(manifest_path) != saved["whole_evidence_sha256"]:
        raise ValueError("Whole-file evidence manifest changed")
    manifest = read(manifest_path)
    if manifest["input"]["sha256"] != info["sha256"]:
        raise ValueError("Whole evidence belongs to different input")
    check_cancel(cancel)
    progress({"stage":"replaying_whole","chunk_total":len(manifest["chunks"])})
    evidence = merge_results(manifest,cancel)
    def core(rows):
        return [{k:s[k] for k in ("start_sec","end_sec","normalized_ipa","candidate_phones","confidence")} for s in rows]
    if core(saved["segments"]) != core(evidence["segments"]):
        raise ValueError("Whole-file result differs from independent frame evidence replay")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(params.get("output_dir") or ROOT/".state/runs"/("whole-replay-"+stamp)).resolve()
    output.mkdir(parents=True,exist_ok=False)
    evidence.update(whole_evidence_path=str(manifest_path),whole_evidence_sha256=sha(manifest_path),
                    original_mode=saved["mode"],source_result_path=str(source_path),source_result_sha256=sha(source_path))
    engine.save_result(output,"replay",info,evidence)
    write(output/"EXIT.json", {"status":"completed","model_forwards":0})
    return {"result_path":str(output/"result.json"),"output_dir":str(output),"scope":"whole_file","mode":"replay",
            "phone_count":len(evidence["segments"]),"chunk_count":len(manifest["chunks"])}

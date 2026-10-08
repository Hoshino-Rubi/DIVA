"""Local score jobs: explicit CPU analysis or preservation of a supplied VSQX."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
from .common import ROOT, Cancelled, check_cancel, read, sha, write
from .process import WorkerClient
from .score_export import make_score, export_score
from .vsqx_export import export_vsqx_from_result

def _bound_result(params):
    file=Path(params["result_path"]).resolve()
    digest=sha(file)
    if params.get("result_sha256") not in {None,digest}:
        raise ValueError("Result version changed")
    result=read(file)
    if result.get("schema_version")!="local_task_result_v1":raise ValueError("Expected local recognition result")
    if sha(Path(result["input"]["path"]))!=result["input"]["sha256"]:
        raise ValueError("Original audio differs from recognition result")
    return file,digest,result

def _output(params):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=Path(params.get("output_dir") or ROOT/'.state/runs'/('local-score-'+stamp)).resolve()
    output.mkdir(parents=True,exist_ok=False)
    return output

def build_score(params,cancel,progress):
    params = dict(params)
    backend = params.pop("backend", "pyin")
    if backend == "game":
        from .game_score_tasks import build_game_score
        return build_game_score(params, cancel, progress)
    if backend != "pyin":
        raise ValueError("Choose the explicit game or pyin note backend")
    if set(params)-{"result_path","result_sha256","tempo","output_dir"}:raise ValueError("Unexpected score parameters")
    file,digest,result=_bound_result(params)
    tempo=params.get("tempo",120.0)
    if isinstance(tempo,bool) or not isinstance(tempo,(int,float)) or not 20<=tempo<=400 or abs(tempo*100-round(tempo*100))>1e-6:
        raise ValueError("Score grid tempo must be 20–400 BPM with at most two decimal places")
    check_cancel(cancel)
    output=_output(params)
    write(output/'REQUEST.json',{"command":"build_score","params":params,"result_sha256":digest})
    started=time.monotonic()
    try:
        worker=WorkerClient(sys.executable,output/'pitch-worker.log',runtime=False,module='any2xsampa.local.pitch_worker')
        try:
            artifact=worker.call('analyze_pitch',{"result_path":str(file),"output_dir":str(output)},cancel,progress,timeout=2400)
        finally:
            worker.close()
        check_cancel(cancel)
        pitch_path=Path(artifact['pitch_path'])
        if pitch_path!=output/'pitch-analysis.json' or sha(pitch_path)!=artifact['pitch_sha256']:
            raise ValueError("Pitch artifact path/hash mismatch")
        pitch=read(pitch_path)
        if sha(file)!=digest or pitch['result_sha256']!=digest:
            raise ValueError("Source result changed during score analysis")
        score=make_score(result,pitch,tempo)
        score.update(result_path=str(file),result_sha256=digest,pitch_path=str(pitch_path),pitch_sha256=sha(pitch_path))
        write(output/'score.json',score)
        progress({"stage":"exporting_vsqx","note_count":len(score['notes'])})
        export=export_score(output/'score.json',output/'draft.vsqx')
        response={"score_path":str(output/'score.json'),"score_sha256":sha(output/'score.json'),
                  "vsqx_path":export['output'],"vsqx_sha256":sha(Path(export['output'])),
                  "source":score['source'],"note_count":len(score['notes']),
                  "placeholder_note_count":score['placeholder_note_count'],"model_forwards":0}
        write(output/'DELIVERY.json',response)
        write(output/'EXIT.json',{"status":"completed","elapsed_seconds":time.monotonic()-started,"model_forwards":0,"retried":False})
        write(file.parent/'score-link.json',response,replace=True)
        return response
    except BaseException as exc:
        write(output/'EXIT.json',{"status":"cancelled" if isinstance(exc,Cancelled) else "failed", "error":str(exc),
            "elapsed_seconds":time.monotonic()-started,"model_forwards":0,"retried":False})
        raise

def import_score(params,cancel,progress):
    if set(params)-{"result_path","result_sha256","template_path","output_dir"}:raise ValueError("Unexpected import parameters")
    file,digest,result=_bound_result(params)
    template=Path(params['template_path']).resolve()
    raw=template.read_bytes()
    import hashlib
    template_hash=hashlib.sha256(raw).hexdigest()
    from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline
    timeline=parse_vsqx_timeline(template)
    if not timeline['notes']:raise ValueError("VSQX contains no notes")
    if sha(template)!=template_hash:raise ValueError("VSQX changed while reading")
    check_cancel(cancel)
    output=_output(params)
    # Keep the user file byte-for-byte, then create a separate phoneme draft.
    preserved = output/'preserved.vsqx'
    preserved.write_bytes(raw)
    suggested = output/'phone-draft.vsqx'
    suggestion = export_vsqx_from_result(
        result_path=file,
        template_path=template,
        output_path=suggested,
        result_sha256=digest,
        apply_phone_suggestions=True,
    )
    suggestion_rows = read(Path(suggestion["provenance"]))["suggestions"]
    if len(suggestion_rows) != len(timeline["notes"]):
        raise ValueError("Phone suggestion count does not match the VSQX note count")
    score={"schema_version":"local_note_score_v1","source":"supplied_vsqx_phone_draft",
        "source_vsqx_path":str(template),"source_vsqx_sha256":template_hash,
        "preserved_vsqx_path":str(preserved),"preserved_vsqx_sha256":sha(preserved),
        "suggested_vsqx_path":str(suggested),"suggested_vsqx_sha256":sha(suggested),
        "suggestion_provenance_path":suggestion["provenance"],
        "result_path":str(file),"result_sha256":digest,"notes":[
            {"index":i,"start_sec":n['start_sec'],"duration_sec":n['end_sec']-n['start_sec'],
             "pitch":n['pitch'],"lyric":n['lyric'],
             "v4c_phonemes":suggestion_rows[i]["tokens"] if suggestion_rows[i]["applied"] else n['phonemes'],
             "original_v4c_phonemes":n['phonemes'],"proposed_v4c_phonemes":suggestion_rows[i]["tokens"],
             "recognized_ipa":suggestion_rows[i]["phones"],
             "unsupported_phones":suggestion_rows[i]["unsupported_phones"],
             "phone_suggestion_applied":suggestion_rows[i]["applied"],"placeholder":False,
             "start_tick":n['start_tick'],"duration_ticks":n['duration_ticks']}
            for i,n in enumerate(timeline['notes'])],
        "time_basis":"VSQX_project_time_not_confirmed_original_audio_time",
        "tempos":timeline['tempos'],"placeholder_note_count":0,"audio_reestimated":False,
        "phone_suggestion_changed_note_p_count":suggestion["changed_note_p_count"],"model_forwards":0}
    write(output/'score.json',score)
    response={"score_path":str(output/'score.json'),"score_sha256":sha(output/'score.json'),
        "vsqx_path":str(suggested),"vsqx_sha256":sha(suggested),
        "preserved_vsqx_path":str(preserved),"preserved_vsqx_sha256":template_hash,
        "suggestion_provenance_path":suggestion["provenance"],
        "source":score['source'],"note_count":len(score['notes']),
        "phone_suggestion_changed_note_p_count":suggestion["changed_note_p_count"],
        "placeholder_note_count":0,"model_forwards":0}
    write(output/'DELIVERY.json',response)
    write(output/'EXIT.json',{"status":"completed","model_forwards":0,"audio_reestimated":False,
                              "original_vsqx_preserved":True,"phone_draft_created":True})
    write(file.parent/'score-link.json',response,replace=True)
    return response

def export_saved_score(params):
    score_file=Path(params['score_path']).resolve()
    if sha(score_file)!=params['score_sha256']:raise ValueError("Score hash mismatch")
    delivery=read(score_file.parent/'DELIVERY.json')
    if delivery['score_sha256']!=sha(score_file):raise ValueError("Saved score changed")
    source=Path(delivery['vsqx_path'])
    if sha(source)!=delivery['vsqx_sha256']:raise ValueError("VSQX artifact changed")
    output=Path(params['output']).resolve()
    proof=output.with_name(output.name+'.provenance.json')
    if output.exists() or proof.exists():raise FileExistsError("Refusing to overwrite existing VSQX or provenance")
    output.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src, output.open('xb') as dst:shutil.copyfileobj(src,dst)
    write(proof,{"schema_version":"local_vsqx_copy_v1",**delivery,"export_path":str(output),
                 "export_sha256":sha(output),"pitch_and_duration_preserved":True,"model_forwards":0})
    return {"output":str(output),"note_count":delivery['note_count'],"model_forwards":0}

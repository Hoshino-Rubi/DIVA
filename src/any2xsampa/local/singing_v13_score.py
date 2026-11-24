"""V13 preserves V12 seam membership. Pitch ownership and instrumental evidence are separate."""
from pathlib import Path

from ..music.singing_v12_guards import note_evidence, isolated

from .singing_score import clean_rows
from .singing_v11_score import load_features,decisions,continuation
from .singing_v12_score import regroup
from .singing_v9_score import score as decode
from .cantonese_audio import POLICY
from .common import read


HOST_DURATION_RATIO = read(POLICY.with_name("cantonese_singing_v13.json"))["host_duration_ratio"]


def host_id(rows: list, ids: list, current: int, probe) -> int:
    """A short but loud transient cannot win over a sustained, periodically supported target."""
    strongest=max(ids,key=lambda i:rows[i]["duration_sec"])
    def core(i):
        r=rows[i];return probe.core(r["start_sec"],r["end_sec"],r["pitch"])
    d=rows[strongest]["duration_sec"]
    if d>=HOST_DURATION_RATIO*rows[current]["duration_sec"] and core(strongest)[0]>0:
        return strongest
    # A held pitch spanning the preceding note and the first fragment is strong
    # support against a falling termination; never force a 90 ms scoop to win.
    i=ids[0];first=rows[i]
    if (i and first["duration_sec"]>=.18 and first["duration_sec"]>=.85*d
        and first["start_sec"]-rows[i-1]["end_sec"]<=.035
        and rows[i-1]["pitch"]==first["pitch"] and core(i)[0]>0):
        return i
    return current


def restore(grid: list, rows: list, probe, game: dict, edits: list) -> list:
    legacy,_,_=clean_rows(game,len(probe.vocal)/16000,v5=True)
    # A repeated, neighboring V9 anchor protects ambiguous short GAME singleton
    # pitches. Do not restore the old grid or run the V5 blanket flattening rule.
    by_time={round(r["start_sec"],5):i for i,r in enumerate(legacy)}
    by_start={round(e["start"],5):e for e in edits}
    log=[]
    for n in grid:
        old=n["pitch"];key=round(n["start_sec"],5)
        event=by_start.get(key)
        if event:
            i=host_id(rows,n["raw_ids"],event["host"],probe)
            event.update(pitch_host=i,pitch_before=old,pitch=rows[i]["pitch"])
            n["pitch"]=rows[i]["pitch"]
        elif key in by_time:
            k=by_time[key];r=legacy[k]
            if old==r["pitch"] or n["duration_sec"]>.18:continue
            def backed(direction):
                i,j=k+direction,k+2*direction
                if not (0<=j<len(legacy) and 0<=i<len(legacy)):return False
                group=sorted((r,legacy[i],legacy[j]),key=lambda x:x["start_sec"])
                return (legacy[i]["pitch"]==legacy[j]["pitch"]==r["pitch"] and
                        all(b["start_sec"]-a["end_sec"]<=.045 for a,b in zip(group,group[1:])))
            # For an ultrashort event encompassed by the V9 duration, the
            # following long host is evidence only, never permission to move its onset.
            overlap=(n["duration_sec"]<.065 and r["duration_sec"]>=3*n["duration_sec"] and
                     n["index"]+1<len(grid) and grid[n["index"]+1]["pitch"]==r["pitch"] and
                     abs(grid[n["index"]+1]["start_sec"]-n["end_sec"])<=.025)
            if overlap or backed(1) or backed(-1):
                n["pitch"]=r["pitch"]
        if n["pitch"]!=old:
            n["pitch_source"]="GAME_v13_steady_host" if event else "GAME_held_anchor"
            log.append(dict(start=n["start_sec"],old=old,new=n["pitch"],source=n["pitch_source"]))
    return log


def score(audio: Path,vocal,frames,game,checkpoint,model,phrases,vocab):
    onset=[];rows,probe,x=load_features(vocal,frames,game,phrases,checkpoint,vocab,onsets=onset)
    ev=note_evidence(rows,probe,frames,vocab,onset,phrases)
    ghosts=isolated(rows,ev,probe)
    acts={j:k for j,k in decisions(rows,probe,x,model).items() if not ghosts.intersection((j,j+1))}
    grid,cont,audit=regroup(rows,ev,probe,acts,ghosts)
    if not grid:raise ValueError("No supported vocal melody; refusing empty VSQX")
    audit.update(pitch_changes=restore(grid,rows,probe,game,audit["merges"]))
    result=decode(audio,vocal,frames,game,None,checkpoint,True,phrases,
                  prepared=(grid,audit,read(POLICY)),checked_confidence=.70)
    audit["phones_changed"],audit["checked_vetoes"]=continuation(result["notes"],cont)
    for n in result["notes"]:
        if n["recognition_source"]=="v11_continuation":n["recognition_source"]="v13_continuation"
    result.update(recognition_mode="singing_v13",track="AB+guarded_supervised_seam")
    return result

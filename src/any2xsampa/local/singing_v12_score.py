"""V9 A+B and frozen V11 seam model with source-onset-preserving V12 edits."""
from .singing_v11_score import load_features, decisions, continuation
from .singing_v9_score import score as decode
from .cantonese_audio import POLICY
from .common import read
from ..adapters.singing_v11_seams import CFG as V11
from ..music.singing_v12_guards import note_evidence, veto, leading_action, isolated


def regroup(rows,ev,probe,actions,ghosts):
    def core(i):
        r=rows[i];return probe.core(r["start_sec"],r["end_sec"],r["pitch"])
    def rank(i):
        energy,mad,slope,duration=core(i)
        return energy/(1+V11["slope_weight"]*(4*mad+.12*slope)),duration
    guarded={};vetoes=[]
    for j,kind in actions.items():
        a,b=rows[j:j+2]
        why=veto(a,b,core(j),core(j+1),ev[j],ev[j+1])
        # Distinct onsets forbid both a physical merge and a vowel-only relabel.
        if why and (kind=="merge" or why in ("distinct_onsets","new_ctc")):
            vetoes.append(dict(left=j,right=j+1,reason=why));continue
        guarded[j]=kind
    grid=[];cont=set();edits=[]
    def emit(row,src):
        if src[0] and guarded.get(src[0]-1)=="continue" and grid and grid[-1]["raw_ids"][-1]==src[0]-1:
            cont.add(len(grid))
        grid.append(dict(row,index=len(grid),raw_ids=src))
    i=0
    while i<len(rows):
        if i in ghosts:i+=1;continue
        j=i
        while guarded.get(j)=="merge":j+=1
        ids=list(range(i,j+1))
        if j==i or len(ids)>3:
            for k in ids:emit(rows[k],[k])
            i=j+1;continue
        host=max(ids,key=rank);a=rows[i]["start_sec"];b=rows[j]["end_sec"]
        # Apply the pairwise veto to the complete span as well as its original seams.
        # Never bridge two defended long/stable notes transitively via a short fragment.
        if len(ids)>2 and veto(rows[i],rows[j],core(i),core(j),ev[i],ev[j]):
            for k in ids:emit(rows[k],[k])
            i=j+1;continue
        policies=[]
        for k in range(i,host):
            seam=probe.at(round((rows[k]["end_sec"]+rows[k+1]["start_sec"])/2,6))
            policies.append(leading_action(rows[k],ev[k],ev[host],seam.get("dip_db")))
        # Mixed lead policies cannot be represented by a monophonic union.
        if "retain" in policies or len(set(policies))>1:
            for k in ids:emit(rows[k],[k])
            i=j+1;continue
        first=host if "drop" in policies else i
        a=rows[first]["start_sec"]
        emit(dict(rows[host],start_sec=a,end_sec=b,duration_sec=b-a,pitch_source="GAME_steady_fragment"),ids[first-i:])
        edits.append(dict(first=i,last=j,host=host,mode="trim_lead" if first!=i else "union",
                          start=a,end=b,pitch=rows[host]["pitch"],removed=j-i))
        i=j+1
    return grid,cont,dict(input=len(rows),merges=edits,vetoes=vetoes,continuations=len(cont),
                          bleed=[dict(index=j,start=rows[j]["start_sec"],end=rows[j]["end_sec"]) for j in sorted(ghosts)])

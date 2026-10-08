"""Reference-only metrics fixtures, without executing the acoustic model."""
from itertools import product
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_input_native_probe import optimal_statuses, migrate
from input_native_probe_common import a1_gate, SONGS


def brute_statuses(ref,pred):
    paths=[]
    def visit(i,j,cost,status):
        if i==len(ref) and j==len(pred):
            paths.append((cost,status));return
        if i<len(ref):visit(i+1,j,cost+1,status+['deletion'])
        if j<len(pred):visit(i,j+1,cost+1,status)
        if i<len(ref) and j<len(pred):
            same=ref[i]==pred[j]
            visit(i+1,j+1,cost+(not same),status+['match' if same else 'substitution'])
    visit(0,0,0,[])
    best=min(c for c,_ in paths)
    return [{s[i] for c,s in paths if c==best} for i in range(len(ref))]


def test_all_minimum_paths_exhaustive_short_binary_sequences():
    sequences=[list(x) for length in range(4) for x in product('ab',repeat=length)]
    for ref in sequences:
        for pred in sequences:
            assert optimal_statuses(ref,pred)==brute_statuses(ref,pred)


def test_repeated_reference_is_not_fabricated_recovery():
    assert optimal_statuses(['i','i'],['i'])==[{'match','deletion'},{'match','deletion'}]
    result=migrate(['i','i'],['i','i'],['i'])
    assert result['ambiguous_positions']==2
    assert result['net_unambiguous_recovery']==0
    assert migrate(['i','h'],['i','h'],['i'])['net_unambiguous_recovery']==1


def cohort(errors=591, insertions=35):
    rows=[]
    # Exactly 1235 reference phones. Distribute errors evenly among five songs.
    for j in range(20):
        count=62 if j<15 else 61
        deletion=(errors-insertions)//20+(j<(errors-insertions)%20)
        insertion=insertions//20+(j<insertions%20)
        reference=['i']*count
        rows.append({'window_id':str(j),'song_id':SONGS[j%5], 'reference':reference,
            'prediction':reference[:count-deletion]+['z']*insertion})
    # Pure uniform deletion+insertions may align as substitutions; gate's frozen
    # baseline check is separately tested against saved live artifacts.
    return rows


def test_a1_rejects_denominator_change():
    with pytest.raises(ValueError): a1_gate([],[])
    rows=cohort(); other=[dict(r) for r in rows]
    other[0]['reference']=[]
    with pytest.raises(ValueError): a1_gate(rows,other)


def test_no_alignment_claim_for_insertions():
    assert optimal_statuses([],['x'])==[]
    r=migrate(['i'],['i'],['i','h'])
    assert r['net_unambiguous_recovery']==0

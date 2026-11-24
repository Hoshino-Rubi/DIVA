import unittest
from types import SimpleNamespace
from any2xsampa.local.singing_v12_score import regroup
from any2xsampa.music.singing_v12_guards import veto
from any2xsampa.local.singing_v13_score import host_id

def row(a,b,p=65):
    return dict(start_sec=a,end_sec=b,duration_sec=b-a,pitch=p)


def ev(label="",q=.1,db=-22):
    return dict(onset=label,q=q,power_db=db,leading=bool(label),fresh_ctc=False,asr=False,ctc_speech=.01)


class GuardTests(unittest.TestCase):
    def test_long_stable_and_distinct_veto(self):
        e,f=ev("f",.91),ev("n",.88)
        a,b=row(0,.37),row(.37,.73)
        c=(1.,.1,.3,.3)
        self.assertEqual(veto(a,b,c,c,e,f),"distinct_onsets")
        self.assertEqual(veto(a,b,c,c,e,e),"both_long")
        self.assertEqual(veto(row(0,.15),row(.15,.6),c,c,e,e),"stable_span")
        self.assertEqual(veto(row(0,.15),row(.15,.31),c,c,e,dict(e,fresh_ctc=True)),"new_ctc")

    def test_lead_clock_and_uncertainty(self):
        probe=SimpleNamespace(core=lambda a,b,p:(1 if a else .1,.1,4,b-a),at=lambda t:dict(dip_db=1.))
        for end,lead,expected,span in [(.14,ev(db=-28),"trim_lead",.14),(.10,ev(),"trim_lead",.10),
                (.14,ev("m",.85),"union",0.),(.14,ev(),None,0.)]:
            rows=[row(0,end,64),row(end,.45,65)]
            grid,_,audit=regroup(rows,[lead,ev()],probe,{0:"merge"},set())
            self.assertEqual(len(grid),1 if expected else 2)
            self.assertEqual(grid[0]["start_sec"],span)
            if expected:
                self.assertEqual(audit["merges"][0]["mode"],expected)
                self.assertEqual(grid[0]["pitch"],65)
                self.assertEqual(grid[0]["end_sec"],.45)

    def test_duration_lock_both_directions(self):
        pr=SimpleNamespace(core=lambda a,b,p:(1.,.01,1.,b-a))
        for ds,target in (([.08,.28],1),([.37,.09],0)):
            rs=[row(0,ds[0],63),row(ds[0],sum(ds),64)]
            self.assertEqual(host_id(rs,[0,1],1-target,pr),target)

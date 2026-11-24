import unittest
from any2xsampa.exporters.singing_v9_rules import load_rules,allowed,equivalent
from any2xsampa.local.vsqx_export import validated_note_tokens
from any2xsampa.pipeline.singing_v9_decode import match,proposals,combine
from any2xsampa.adapters.singing_v9_fit import classes,FinalResidual
from any2xsampa.adapters.singing_v9_asr import words_from_pieces

class V9Contract(unittest.TestCase):
    def test_inventory(self):
        r=load_rules()
        self.assertTrue(all(allowed(s.split(),r) for s in r["jyutping"].values()))
        self.assertTrue(set(r["extensions"]).issubset(classes(r)[0]))
        for s in ["e o y","f UN","@N","s i k k","s a m k"]:
            with self.assertRaises(ValueError):
                validated_note_tokens(dict(v4c_phonemes=s.split()),r)
    def test_equivalence(self):
        r=load_rules()
        self.assertTrue(equivalent("s UN".split(),"s u@N".split(),r))
        self.assertFalse(equivalent("s UN".split(),"ts u@N".split(),r))
        self.assertFalse(equivalent("a t".split(),"a k".split(),r))
    def test_time_match(self):
        words=[dict(jp="aa",start=t,end=t+.1,confidence=.99) for t in (0.,.2,.5)]
        rows=[dict(start_sec=t,end_sec=t+.12,duration_sec=.12) for t in (0.,.5)]
        m=match(words,rows)
        self.assertEqual([i for i,_,_ in m],[0,1]);self.assertEqual([w["start"] for _,w,_ in m],[0,.5])
        proposals_,_=proposals([dict(a=0,b=1,syllables=[words[0]]*5)],rows,load_rules())
        self.assertFalse(proposals_)
    def test_decoder_and_fallback(self):
        _,out=words_from_pieces([("jyut",.1,.2,.9),(" ming",.3,.5,.95)])
        self.assertEqual([w["jp"] for w in out],["jyut","ming"])
        r=load_rules()
        rows=[dict(start_sec=t,acoustic_prediction=dict(continuation=i>0)) for i,t in enumerate((.1,.23,.8))]
        x=combine(rows,{0:("t a",.9,"daa",.4)},[("p ei",.8)]*3,r,.5,.5)
        self.assertEqual([s[1] for s in x],["asr","melisma","acoustic"]);self.assertEqual(x[1][0],"a")
    def test_low_rank(self):
        net=FinalResidual(224,60)
        self.assertEqual(net.down.out_features,8)
        self.assertEqual(sum(p.numel() for p in net.parameters()),8*(224+60))

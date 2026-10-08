"""Versioned segmental-event decoder and PhoneSegment evidence adapter.

No lyrics, target length, note/pitch context or reference alignment is accepted
by decoding. Segment times are latent model support, not CTC emission times and
not human-verified phone durations. Existing CTC codecs are not modified.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import math
from any2xsampa.training.limited_data_artifacts import read_json,verify_file
from any2xsampa.adapters.audio_provenance import validate_audio_binding
from any2xsampa.core.models import PhoneSegment,PhoneCandidate,TimeRange
from any2xsampa.core.interfaces import UniversalPhoneRecognizer


def _lse(x: Any,axis=None) -> Any:
    import numpy as np
    return np.logaddexp.reduce(x,axis=axis)


class EventScores:
    def __init__(self,frame_log_probs: Any,boundary_logits: Any):
        import numpy as np
        q=np.asarray(frame_log_probs,dtype=np.float64);b=np.asarray(boundary_logits,dtype=np.float64)
        if q.ndim!=2 or len(q)<1 or q.shape[1]<2 or b.shape!=(len(q),):raise ValueError('invalid event geometry')
        if not np.isfinite(q).all() or not np.isfinite(b).all():raise ValueError('non-finite event evidence')
        if not np.allclose(_lse(q,axis=1),0,atol=2e-5,rtol=0):raise ValueError('unnormalized frame evidence')
        self.q=q;self.b=b;self.length=len(q);self.labels=q.shape[1]
        self.prefix=np.vstack((np.zeros((1,q.shape[1])),q.cumsum(0)))
        self.start=-np.logaddexp(0,-b)
        hold=-np.logaddexp(0,b);self.hold=np.r_[0.,hold.cumsum()]
    def ending(self,end: int) -> Any:
        import numpy as np
        if not 1<=end<=self.length:raise ValueError('end outside event sequence')
        d=np.arange(end,0,-1,dtype='float64')
        return ((self.prefix[end,1:]-self.prefix[:end,1:])/d[:,None]
                +self.start[:end,None]+(self.hold[end]-self.hold[1:end+1])[:,None])
    def starting(self,start: int) -> Any:
        import numpy as np
        if not 0<=start<self.length:raise ValueError('start outside event sequence')
        d=np.arange(1,self.length-start+1,dtype='float64')
        return ((self.prefix[start+1:,1:]-self.prefix[start,1:])/d[:,None]
                +self.start[start]+(self.hold[start+1:]-self.hold[start+1])[:,None])
    def segment(self,start: int,end: int) -> Any:
        if not 0<=start<end<=self.length:raise ValueError('invalid segment interval')
        return ((self.prefix[end,1:]-self.prefix[start,1:])/(end-start)
                +self.start[start]+self.hold[end]-self.hold[start+1])


def decode_events(frame_log_probs: Any,boundary_logits: Any) -> dict:
    import numpy as np
    score=EventScores(frame_log_probs,boundary_logits);t=score.length
    best=np.full(t+1,-np.inf);best[0]=0.;alpha=best.copy();back=[]
    for end in range(1,t+1):
        values=score.ending(end)
        paths=best[:end,None]+values
        index=int(paths.argmax());start,k=divmod(index,score.labels-1)
        null=best[end-1]+score.q[end-1,0]
        if null>=paths[start,k]:best[end]=null;back.append((end-1,0))
        else:best[end]=paths[start,k];back.append((start,k+1))
        alpha[end]=np.logaddexp(_lse(alpha[:end]+_lse(values,axis=1)),alpha[end-1]+score.q[end-1,0])
    beta=np.full(t+1,-np.inf);beta[t]=0.
    for start in range(t-1,-1,-1):
        beta[start]=np.logaddexp(score.q[start,0]+beta[start+1],_lse(_lse(score.starting(start),axis=1)+beta[start+1:]))
    if abs(alpha[t]-beta[0])>1e-7:raise ValueError('forward/backward partition mismatch')
    end=t;events=[];arcs=[]
    while end:
        start,k=back[end-1];arcs.append({'start_frame':start,'end_frame_exclusive':end,'label_id':k})
        if k:
            label_scores=score.segment(start,end)
            log_marginal=alpha[start]+label_scores[k-1]+beta[end]-alpha[t]
            if log_marginal>1e-7:raise ValueError('arc marginal exceeds one')
            events.append({'start_frame':start,'end_frame_exclusive':end,'label_id':k,
                'arc_posterior':float(math.exp(min(0.,log_marginal))),
                'conditional_label_probabilities':np.exp(label_scores-_lse(label_scores)).tolist()})
        end=start
    events.reverse();arcs.reverse()
    return {'event_ids':[e['label_id'] for e in events],'events':events,'all_arcs':arcs,
            'best_path_score':float(best[t]),'log_partition':float(alpha[t]),
            'decoder':'exact_segmental_viterbi_v1','duration_cap_frames':None,
            'adjacent_same_label_events_preserved':True,'reference_conditioning':False}


def phone_segments(decoded: dict,vocabulary: dict,crop: dict,geometry: dict) -> list[PhoneSegment]:
    labels={i:p for p,i in vocabulary.items()}
    if len(labels)!=len(vocabulary) or set(labels)!=set(range(len(labels))) or vocabulary.get('<blank>')!=0:
        raise ValueError('invalid event vocabulary')
    begin=float(crop['effective_start_sec']);end=float(crop['effective_end_sec'])
    step=geometry['stride_samples']/16000.;center=(geometry['receptive_field_samples']-1)/32000.
    if not math.isfinite(begin) or not math.isfinite(end) or begin<0 or end<=begin:raise ValueError('invalid source crop')
    result=[]
    for event in decoded['events']:
        k=event['label_id'];symbol=labels[k]
        if symbol.startswith('<'):continue
        start=max(begin,begin+center+(event['start_frame']-.5)*step)
        stop=min(end,begin+center+(event['end_frame_exclusive']-.5)*step)
        if stop<=start:raise ValueError('event support outside source crop')
        p=event['conditional_label_probabilities']
        if len(p)!=len(vocabulary)-1:raise ValueError('conditional label count mismatch')
        ids=sorted((i for i in labels if i and not labels[i].startswith('<')),key=lambda i:(-p[i-1],i))[:8]
        result.append(PhoneSegment(TimeRange(start,stop),[PhoneCandidate(labels[i],p[i-1]) for i in ids],event['arc_posterior'],
            normalized_ipa=symbol,metadata={'source':'segmental_event_v1','raw_label_trace':{'label':symbol,'label_id':k,
                'start_frame':event['start_frame'],'end_frame_exclusive':event['end_frame_exclusive']},
                'timing_kind':'latent_segment_support_not_ctc_or_ground_truth',
                'confidence_kind':'global_segment_arc_posterior_not_accuracy',
                'candidate_confidence_kind':'label_probability_conditional_on_segment',
                'same_label_events_not_collapsed':True}))
    return result


class SegmentalEventRecognizer(UniversalPhoneRecognizer):
    def __init__(self,evidence_path: Path,*,fixture: bool=False):
        self.path=evidence_path;self.fixture=fixture
    def recognize(self,vocal_audio_path: Path,lyrics: str | None=None) -> list[PhoneSegment]:
        if lyrics is not None:raise ValueError('event decoding accepts audio evidence only, no lyrics')
        from safetensors.numpy import load_file
        p=read_json(self.path)
        if (p.get('schema_version')!='segmental_event_evidence_v1' or p.get('reference_conditioning') is not False
                or p.get('scoring_spec')!='mean_logp_start_hold_v1'):
            raise ValueError('unsupported or reference-conditioned event evidence')
        if not validate_audio_binding(p['input_audio'],vocal_audio_path)['verified']:
            raise ValueError('event evidence requires a verified audio binding')
        for key in ('scores','vocabulary'):
            spec=p[key];target=(self.path.parent/spec['file']).resolve()
            if not target.is_relative_to(self.path.parent.resolve()):raise ValueError('unsafe evidence member')
            verify_file(target,spec['sha256'])
        raw=load_file(str(self.path.parent/p['scores']['file']))
        if set(raw)!={'frame_log_probs','boundary_logits'}:raise ValueError('unexpected event score tensors')
        vocab=read_json(self.path.parent/p['vocabulary']['file'])
        if len(vocab)!=428 or raw['frame_log_probs'].shape!=(p['geometry']['frame_count'],428):raise ValueError('event vocabulary/frame geometry mismatch')
        decoded=decode_events(raw['frame_log_probs'],raw['boundary_logits'])
        if decoded['event_ids']!=p['decoded_event_ids']:raise ValueError('saved event path does not replay')
        return phone_segments(decoded,vocab,p['crop'],p['geometry'])

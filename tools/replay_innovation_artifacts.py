"""Independent CPU replay of factor and latent-event evidence; no optimization."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.training.phone_error_metrics import phone_metrics
from any2xsampa.adapters.phoneticxeus import ctc_collapse,PhoneticXeusPosteriorRecognizer
from any2xsampa.adapters.segmental_event import SegmentalEventRecognizer,decode_events


def distance(reference: list,prediction: list) -> int:
    row=list(range(len(prediction)+1))
    for i,a in enumerate(reference,1):
        new=[i]
        for j,b in enumerate(prediction,1):new.append(min(row[j]+1,new[-1]+1,row[j-1]+(a!=b)))
        row=new
    return row[-1]


def replay(run: Path) -> None:
    from safetensors.numpy import load_file
    import numpy as np
    f=read_json(run/'FREEZE.json')
    for p,h in f['bindings'].items():verify_file(Path(p),h)
    vocab=read_json(Path(f['vocabulary']['path']));labels={i:p for p,i in vocab.items()}
    source={r['window_id']:r for r in read_json(run/'FACTOR_INPUTS.json')['rows']}
    report=read_json(run/'factor/RESULT.json');checks=[];maximum_error=0.
    for record in report['artifact_records']:
        pp=Path(record['posterior_path']);verify_file(pp,record['posterior_sha256']);p=read_json(pp)
        raw=pp.parent/p['raw_evidence']['file'];verify_file(raw,p['raw_evidence']['sha256'])
        values=load_file(str(raw))['frame_logits'];row=source[record['window_id']]
        cell=record['cell'];blank=load_file(row['sources'][cell[0]]['artifact']['evidence_path'])['frame_logits'].astype('float64')
        label=load_file(row['sources'][cell[1]]['artifact']['evidence_path'])['frame_logits'].astype('float64')
        # Independent probability-space reconstruction (real logits are not
        # extreme); avoid invoking the production factor-exchange function.
        a=np.exp(blank-blank.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
        q=np.exp(label[:,1:]-label[:,1:].max(-1,keepdims=True));q/=q.sum(-1,keepdims=True)
        wanted=np.c_[a[:,0],a[:,1:].sum(-1,keepdims=True)*q]
        actual=np.exp(values.astype('float64')-values.max(-1,keepdims=True));actual/=actual.sum(-1,keepdims=True)
        error=float(np.max(np.abs(actual-wanted)));maximum_error=max(maximum_error,error)
        if error>2e-6:raise ValueError('independent probability reconstruction differs')
        prediction=[labels[i] for i in ctc_collapse(values.argmax(-1).tolist())]
        if prediction!=record['prediction']:raise ValueError('independent factor sequence changed')
        audio=Path(row['sources'][cell[0]]['input_audio']['path'])
        real=PhoneticXeusPosteriorRecognizer(pp).recognize(audio);fixture=PhoneticXeusPosteriorRecognizer(pp,fixture=True).recognize(audio)
        view=lambda rr:[(asdict(s.time_range),[asdict(x) for x in s.candidate_phones],s.confidence) for s in rr]
        if view(real)!=view(fixture):raise ValueError('factor contract differs')
        count=distance(row['reference'],prediction)
        if count!=phone_metrics([(row['reference'],prediction)])['edit_distance']:raise ValueError('distance mismatch')
        checks.append({'window_id':record['window_id'],'cell':cell,'probability_error':error,'edit_distance':count,'real_fixture_equal':True})
    if len(checks)!=80 or len(list((run/'factor/attempts').glob('*.json')))!=80:raise ValueError('factor accounting mismatch')
    for cell in ('xx','vv','xv','vx'):
        rows=report['details'][cell]
        if phone_metrics([(r['reference'],r['prediction']) for r in rows])!=report['metrics'][cell]:raise ValueError('aggregate differs')
    event=read_json(run/'event-profile/RESULT.json')
    if not event['complete'] or event['optimizer_steps']!=0 or event['head_parameters_before']!=event['head_parameters_after']:
        raise ValueError('prototype profile state invalid')
    if read_json(run/'event-profile-exit.json')['exit_code']!=0:raise ValueError('prototype process failed')
    row=event['records'][0];pp=Path(row['evidence_path']);verify_file(pp,row['evidence_sha256'])
    p=read_json(pp);raw=load_file(str(pp.parent/p['scores']['file']));d=decode_events(raw['frame_log_probs'],raw['boundary_logits'])
    if d['event_ids']!=p['decoded_event_ids']:raise ValueError('event path changed')
    audio=Path(read_json(run/'PROFILE_INPUT.json')['entry']['audio_path'])
    real=SegmentalEventRecognizer(pp).recognize(audio);fixture=SegmentalEventRecognizer(pp,fixture=True).recognize(audio)
    if [asdict(s) for s in real]!=[asdict(s) for s in fixture]:raise ValueError('event contract differs')
    write_json_new(run/'INDEPENDENT_REPLAY.json',{'passed':True,'factor_checks':checks,'factor_fixture_count':80,
        'maximum_probability_error':maximum_error,'event_fixture_count':1,'event_codec_version':'segmental_event_evidence_v1',
        'event_log_partition':d['log_partition'],'event_times_are_not_ground_truth':True,
        'head_parameters_unchanged':True,'quality_metrics_for_untrained_event_head':False,
        'new_acoustic_forwards':0,'optimizer_steps':0,'artifact_bindings':{str(p):sha256_file(p) for p in
            [run/'factor/RESULT.json',run/'event-profile/RESULT.json',pp]}})
    print('INDEPENDENT REPLAY 80 factors + 1 event evidence; max probability error',maximum_error)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    replay(p.parse_args().run.resolve())

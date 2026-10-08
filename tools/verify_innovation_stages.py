"""Final source audit and independent CTC bridge likelihood/stability check."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
from collections import Counter
from pathlib import Path
import argparse
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.training.ctc_position_bridge import position_occupancy
from run_factor_exchange_probe import verify_launch


def compensate_shift(weights,shift):
    """Undo a known one-frame artificial shift only in diagnostic weights.

    Return distributions over common interior frames; never used to decode,
    change audio, relabel a target or construct new trainable examples.
    """
    import numpy as np
    if shift not in (-1,1) or weights.ndim!=2 or len(weights)<3:raise ValueError('invalid shift audit')
    restored=weights[2:] if shift==1 else weights[:-2]
    masses=restored.sum(0)
    if np.any(masses<=0):raise ValueError('no shared interior probability mass')
    return restored/masses, masses


def verify(run: Path) -> None:
    import numpy as np
    import torch
    from safetensors.numpy import load_file
    freeze=verify_launch(run,'bridge_check')
    torch.set_num_threads(4)
    profile=read_json(run/'event-profile/RESULT.json');bridge=read_json(run/'bridge/RESULT.json')
    replay=read_json(run/'INDEPENDENT_REPLAY.json')
    if not replay['passed'] or not profile['complete'] or not bridge['complete']:raise ValueError('stages incomplete')
    out=run/'bridge-check';write_json_new(out/'START.json',{'cached_forward_backward_replays':180,
        'independent_torch_CTC_loss_calls':60,'optimizer_steps':0,'no_new_hypothesis':True})
    base=Path(freeze['singing_run']);cache={r['example_id']:r for r in read_json(base/'CACHE.json')['entries']}
    refs={r['example_id']:r for r in read_json(base/'REFERENCES.json')['rows']}
    for p,h in bridge['source_bindings'].items():verify_file(Path(p),h)
    verified=[];positions=[]
    for wid,old in bridge['soft_position_reports'].items():
        entry=cache[wid];reference=refs[wid]
        values=load_file(entry['path'])['frame_logits'];ids=reference['target_ids']
        baseline=position_occupancy(values,ids)
        q=torch.from_numpy(values.astype('float64')).log_softmax(-1).unsqueeze(1)
        actual=torch.nn.functional.ctc_loss(q,torch.tensor(ids,dtype=torch.long),
            torch.tensor([len(values)]),torch.tensor([len(ids)]),blank=0,reduction='sum',zero_infinity=False)
        error=abs(float(actual)+baseline['log_target_probability'])
        if error>1e-7 or abs(baseline['log_target_probability']-old['log_target_probability'])>1e-10:
            raise ValueError('independent CTC likelihood mismatch')
        weights=baseline['position_weights'];interior=weights[1:-1];interior_mass=interior.sum(0)
        normalized=interior/interior_mass
        tests=[]
        for shift,moved in [(1,np.r_[values[:1],values[:-1]]),(-1,np.r_[values[1:],values[-1:]])]:
            result=position_occupancy(moved,ids)
            restored,mass=compensate_shift(result['position_weights'],shift)
            tests.append((.5*np.abs(normalized-restored).sum(0),mass))
        for i,p in enumerate(baseline['positions']):
            positions.append({'window_id':wid,'song_id':entry['song_id'],'audio_role':entry['audio_role'],
                'position':i,'phone':reference['target_ipa'][i],
                'raw_uncompensated_shift_TV':old['positions'][i]['one_frame_shift_maximum_total_variation'],
                'known_shift_compensated_interior_TV':max(float(v[i]) for v,m in tests),
                'minimum_retained_interior_mass':min(float(interior_mass[i]),*[float(m[i]) for v,m in tests]),
                'not_a_reliability_threshold':True})
        verified.append({'window_id':wid,'independent_torch_ctc_error':error,'positions':len(ids)})
    if len(verified)!=60:raise ValueError('frozen bridge cache coverage changed')
    def distribution(values):
        a=np.asarray(values,dtype='float64')
        return {'count':len(a),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'maximum':float(a.max())}
    source=read_json(Path(freeze['dual_run'])/'PRESERVATION_BEFORE.json')['files']
    protected={p:h for p,h in source.items() if Path(p).suffix.lower() in {'.vsqx','.wav','.json'} or p.endswith('any2xsampa\\cli.py')}
    for p,h in protected.items():verify_file(Path(p),h)
    for phase in ('factor','event','bridge','bridge_check'):
        review=read_json(run/('CODE_REVIEW_'+phase+'.json'))
        for p,h in review['code_bindings'].items():verify_file(Path(p),h)
        for p,h in review.get('data_bindings',{}).items():verify_file(Path(p),h)
    mon=read_json(run/'event-profile/monitor/resource-monitor.json')
    if not mon['completed_without_resource_abort'] or mon['maximum_sampled_total_used_bytes']>14*1024**3:raise ValueError('profile resource limit failed')
    if max(r['peak_reserved_bytes'] for r in profile['records'])>7*1024**3:raise ValueError('process reserved cap exceeded')
    correction={'interpretation':'Raw one-frame-shift TV confounds predictable spike translation with instability. Compensated interior diagnostic is reported alongside, not substituted silently. Neither is human acoustic alignment confirmation.',
        'by_role':{role:{'raw_shift_TV':distribution([p['raw_uncompensated_shift_TV'] for p in positions if p['audio_role']==role]),
            'compensated_TV':distribution([p['known_shift_compensated_interior_TV'] for p in positions if p['audio_role']==role]),
            'retained_mass':distribution([p['minimum_retained_interior_mass'] for p in positions if p['audio_role']==role])} for role in ('original','rendered')},
        'position_records':positions}
    write_json_new(out/'RESULT.json',{'passed':True,'likelihood_checks':verified,'shift_interpretation':correction,
        'ctc_cached_forward_backward_calls_this_verification':180,'independent_torch_ctc_calls':60,
        'total_bridge_numeric_calls_including_first_audit':420,'full_acoustic_forwards':0,'optimizer_steps':0})
    write_json_new(run/'PRESERVATION.json',{'all_verified':True,'protected_originals_mappings_defaults':protected,
        'launch_review_files':[str(run/('CODE_REVIEW_'+x+'.json')) for x in ('factor','event','bridge','bridge_check')],
        'frozen_sources_checked':len(freeze['bindings']),'final_test_predictions_opened':False,'default_replaced':False})
    print('BRIDGE NUMERICS VERIFIED',len(verified),'torch comparisons;',len(positions),'positions')
    print('SHIFT_DIAGNOSTICS',correction['by_role'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    verify(p.parse_args().run.resolve())

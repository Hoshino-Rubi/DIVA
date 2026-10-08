"""Exercise the standalone real and ordinary-environment fixture entrypoints."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,subprocess,json,os,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,verify_file

def check(run:Path)->None:
    seed=read_json(run/'DEVELOPMENT_SELECTION.json')['research_seed']
    verification=read_json(run/'verification-final'/('seed'+str(seed))/'VERIFICATION.json')
    smoke=verification['eight_second_smoke'];payload=read_json(Path(smoke['posterior_path']))
    example=smoke['example_id'].removesuffix('--smoke8')
    row=next(r for r in read_json(run/'AUDIO_REQUESTS.json')['rows'] if r['example_id']==example)
    start=payload['producer']['crop']['requested_start_sec'];duration=8.
    python=read_json(run/'FREEZE.json')['runtime_python'];job=run/'final-training'/('seed'+str(seed))
    real=run/'entry-real';fixture=run/'entry-fixture';commands=[]
    cmd=[python,'-X','utf8','-B',str(ROOT/'tools/infer_singing_domain.py'),'--run',str(run),'--job',str(job),
         '--audio',row['audio_path'],'--start',str(start),'--duration',str(duration),'--output',str(real)]
    def execute(argv:list[str],log:Path)->None:
        began=time.monotonic()
        with log.open('x',encoding='utf-8') as stream:
            p=subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,shell=False,timeout=930,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        commands.append({'argv':argv,'exit_code':p.returncode,'wall_seconds':time.monotonic()-began})
        if p.returncode:raise RuntimeError('inference entry failed; see '+str(log))
    execute(cmd,run/'entry-real.log')
    pp=real/'evidence/posterior.json'
    fixture_cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B',str(ROOT/'tools/infer_singing_domain.py'),
        '--fixture',str(pp),'--audio',row['audio_path'],'--start',str(start),'--duration',str(duration),'--output',str(fixture)]
    execute(fixture_cmd,run/'entry-fixture.log')
    left=read_json(real/'RESULT.json');right=read_json(fixture/'RESULT.json')
    view=lambda result:[{k:segment[k] for k in ('time_range','candidate_phones','confidence')} for segment in result['segments']]
    if view(left)!=view(right):raise ValueError('standalone real/fixture differs')
    from safetensors.numpy import load_file
    import numpy as np
    verify_file(Path(smoke['logits_path']),smoke['logits_sha256'])
    verify_file(pp,left['posterior_sha256'])
    verify_file(real/'evidence/logits.safetensors',read_json(pp)['raw_evidence']['sha256'])
    actual=load_file(str(real/'evidence/logits.safetensors'))['frame_logits']
    prior=load_file(smoke['logits_path'])['frame_logits']
    if not np.isfinite(actual).all() or not np.isfinite(prior).all():raise ValueError('non-finite entry evidence')
    difference=float(np.max(np.abs(actual-prior)))
    if difference>2e-5 or not np.array_equal(actual.argmax(-1),prior.argmax(-1)):raise ValueError('standalone free-audio inference differs from verified full-audio path')
    write_json_new(run/'free-audio-entry-check.json',{'checked_at_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,
        'commands':commands,'real_fixture_equal':True,'free_audio_matches_existing_full_audio':True,'maximum_logit_error':difference,
        'source_audio_sha256':row['audio_sha256'],'duration_seconds':8.,'test_predictions_opened':False,
        'reference_targets_supplied_to_inference':False,'real_result_sha256':sha256_file(real/'RESULT.json'),'fixture_result_sha256':sha256_file(fixture/'RESULT.json')})
    print('Real/fixture entry and full-audio path match; maximum logit error',difference)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);check(p.parse_args().run.resolve())

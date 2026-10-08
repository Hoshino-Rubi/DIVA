"""Seal completed results, successful/failed attempts, caches and implementation versions."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,verify_file,sha256_file


def seal(run:Path)->None:
    completion=run/'completion';summary=read_json(completion/'SUMMARY.json')
    if summary['status']!='completed':raise ValueError('cannot seal an incomplete experiment')
    source=dict(read_json(run/'CODE_FREEZE.json')['files'])
    if (run/'NUMERICAL_FIX_FREEZE.json').exists():source.update(read_json(run/'NUMERICAL_FIX_FREEZE.json')['files'])
    extra=[ROOT/'src/any2xsampa/training/whole_sequence_analysis.py',
        ROOT/'tests/test_whole_sequence_analysis.py',ROOT/'tests/test_whole_sequence_numerical_v2.py']
    extra += [ROOT/'tools'/name for name in ('audit_completed_whole_cache.py','analyze_completed_whole_sequence.py',
        'finalize_completed_whole_sequence.py','regress_completed_whole_sequence.py','diagnose_whole_sequence_numerics.py',
        'verify_recovered_whole_sequence.py','run_whole_sequence_final_v2.py','seal_completed_whole_sequence.py')]
    for path in extra:source[str(path.resolve())]=sha256_file(path)
    for path,digest in source.items():verify_file(Path(path),digest)
    files=set(run.glob('*.json'))
    files.update(p for p in completion.iterdir() if p.is_file() and p.suffix in {'.json','.md','.html','.xml'})
    files.update(p for p in (run/'commands').iterdir() if p.is_file())
    files.update((run/'resource-prechecks').glob('*.json'))
    accepted=read_json(completion/'REGRESSION_ACCEPTED.json')
    files.update([Path(accepted['junit_path']),Path(accepted['run_path'])])
    files.update(p for p in (completion/'regression-collection-failure').glob('*') if p.is_file())
    for entry in read_json(run/'WHOLE_CACHE.json')['entries']:
        files.add(Path(entry['path']));files.add(Path(entry['path']).with_name('entry.json'))
    groups=[Path(read_json(run/'SMOKE_ACCEPTED.json')['result_path']).parent,
        run/'whole-cache',run/'whole-cuda-probe',run/'whole-cuda-probe-v2',run/'numerical-diagnosis',
        run/'checkpoint-verification-screen',run/'checkpoint-verification-final',run/'development']
    groups += list((run/'screen/whole_sequence').glob('*'))
    if (run/'final/whole_sequence').exists():groups+=list((run/'final/whole_sequence').glob('*'))
    for directory in groups:
        if not directory.exists():continue
        for path in directory.rglob('*'):
            if path.is_file() and 'runtime-cache' not in path.parts and path.suffix in {'.json','.safetensors','.diff'}:
                files.add(path)
    delivery={'schema_version':'whole_sequence_completion_delivery_v1',
        'source_files':[{'path':p,'sha256':d} for p,d in sorted(source.items())],
        'artifacts':[{'path':str(p.resolve()),'sha256':sha256_file(p)} for p in sorted(files)],
        'completed_screen_candidates':5,'screen_process_attempts':summary['screen_process_attempts'],
        'failed_attempts_preserved':len(summary['failed_attempts']),
        'old_blocked_delivery_sha256':sha256_file(run/'DELIVERY.json'),'default_model_replaced':False}
    write_json_new(completion/'DELIVERY.json',delivery)
    for row in delivery['source_files']+delivery['artifacts']:verify_file(Path(row['path']),row['sha256'])
    write_json_new(completion/'DELIVERY_CHECK.json',{'delivery_sha256':sha256_file(completion/'DELIVERY.json'),
        'source_files_checked':len(source),'artifacts_checked':len(files),'all_hashes_equal':True})
    print('SEALED',len(source),'sources',len(files),'artifacts',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);seal(p.parse_args().run.resolve())

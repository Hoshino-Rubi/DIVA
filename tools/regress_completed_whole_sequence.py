"""Run full regression after all GPU stages, preserving the actual CUDA environment."""
from pathlib import Path
import argparse,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file


def regression(run:Path,attempt:int=1)->int:
    selection=read_json(run/'ROUTE_SELECTION.json')
    read_json(run/'checkpoint-verification-screen/RESULT.json')
    if selection['passed']:
        read_json(run/'VOCAL_TRANSFER.json');read_json(run/'checkpoint-verification-final/RESULT.json')
    else:read_json(run/'FINAL_TRAINING_NOT_STARTED.json')
    os.environ.update({'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
    import torch
    torch.set_num_threads(4)
    output=run/'completion' if attempt==1 else run/'completion'/('regression-attempt'+str(attempt))
    output.mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('GPU acceptance was restored; CUDA regression must not silently skip it')
    from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,7*2**30/total),0)
    before=check_gpu_budget(torch)
    import pytest
    command=['tests','-q','-p','no:cacheprovider','--basetemp',str(output/'pytest-full-regression'),
        '--junitxml',str(output/'regression.xml')]
    started=time.monotonic()
    # Tests mock subprocess/thread primitives; don't run a watcher in their process.
    code=pytest.main(command)
    after=check_gpu_budget(torch)
    write_json_new(output/'REGRESSION_RUN.json',{'exit_code':int(code),'pytest_arguments':command,
        'elapsed_seconds':time.monotonic()-started,'cuda_available_before_collection':True,
        'cuda_disabled_for_testing':False,'torch_version':torch.__version__,'cpu_threads':torch.get_num_threads(),
        'gpu_before':before,'gpu_after':after,'peak_test_reserved_bytes':torch.cuda.max_memory_reserved()})
    if not code:
        write_json_new(run/'completion/REGRESSION_ACCEPTED.json',{'attempt':attempt,
            'junit_path':str(output/'regression.xml'),'junit_sha256':sha256_file(output/'regression.xml'),
            'run_path':str(output/'REGRESSION_RUN.json'),'run_sha256':sha256_file(output/'REGRESSION_RUN.json'),
            'runner_sha256':sha256_file(Path(__file__))})
    return int(code)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();sys.exit(regression(a.run.resolve(),a.attempt))

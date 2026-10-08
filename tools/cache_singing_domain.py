"""Extract a versioned, audio-only frozen prefix cache in the isolated runtime."""
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file
from any2xsampa.training.singing_domain import load_manifest

def run_cache(run:Path)->None:
    load_manifest(run);freeze=read_json(run/'FREEZE.json')
    rows=read_json(run/'AUDIO_REQUESTS.json')['rows']
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    started=time.monotonic();entries=[]
    with serial_candidate_job(),ProcessResourceMonitor(run/'cache-monitor',timeout_seconds=1800),LockedAudioCache(max_bytes=16*1024**2,max_sources=12) as audio:
        rt=SingingDomainRuntime(Path(freeze['model_snapshot']),run/'cache-runtime',audio_cache=audio)
        write_json_new(run/'ARCHITECTURE.json',rt.identity)
        for i,row in enumerate(sorted(rows,key=lambda r:(r['song_id'],r['audio_role'],r['start_sec'],r['example_id']))):
            entries.append(rt.save_prefix(row,run/'cache'/row['example_id']))
            if (i+1)%10==0:print('cached',i+1,'/',len(rows),flush=True)
        write_json_new(run/'CACHE.json',{'schema_version':'singing_prefix_cache_v1','architecture':rt.identity,
            'audio_requests_sha256':sha256_file(run/'AUDIO_REQUESTS.json'),'entries':entries,
            'all_parameters_frozen':True,'reference_received':False,'peak_reserved_bytes':rt.torch.cuda.max_memory_reserved(),
            'audio_cache':audio.stats(),'elapsed_seconds':time.monotonic()-started})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run_cache(p.parse_args().run.resolve())

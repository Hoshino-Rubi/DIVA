"""Freeze decoder code, then decode both streams without reading phone references."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, write_json_new
from any2xsampa.training.dual_input import load_policy
from any2xsampa.adapters.dual_ctc import dual_decode


def decode(run: Path) -> None:
    load_policy(run); evidence = read_json(run/'EVIDENCE.json')
    if evidence['policy_sha256'] != sha256_file(run/'POLICY.json') or evidence['reference_received'] is not False:
        raise ValueError('evidence policy or reference contamination')
    paths = [ROOT/'src/any2xsampa/adapters/dual_ctc.py', ROOT/'src/any2xsampa/adapters/ctc_prefix.py',
             ROOT/'src/any2xsampa/training/dual_input.py', Path(__file__)]
    write_json_new(run/'DECODER_CODE_LOCK.json', {'files': [{'path': str(p), 'sha256': sha256_file(p)} for p in paths],
        'policy_sha256': sha256_file(run/'POLICY.json'), 'evidence_sha256': sha256_file(run/'EVIDENCE.json'),
        'frozen_before_new_decodes_or_evaluation': True})
    from safetensors.numpy import load_file
    requests = read_json(run/'AUDIO_REQUESTS.json')['requests']
    entries = []; started = time.monotonic()
    for request in requests:
        before = time.monotonic(); inputs = {}
        source_rows = [e for e in evidence['entries'] if e['example_id'] == request['example_id']]
        if len(source_rows) != 2 or {e['condition'] for e in source_rows} != {'mixed','vocal'}:
            raise ValueError('missing or duplicate evidence streams')
        for row in source_rows:
            artifact = row['artifact']; verify_file(Path(artifact['evidence_path']), artifact['evidence_sha256'])
            verify_file(Path(artifact['posterior_path']), artifact['posterior_sha256'])
            inputs[row['condition']] = load_file(artifact['evidence_path'])['frame_logits']
        result = dual_decode(inputs['mixed'], inputs['vocal'])
        result.update(example_id=request['example_id'], song_id=request['song_id'], split=request['split'],
                      evidence_bindings={r['condition']: r['artifact'] for r in source_rows}, elapsed_seconds=time.monotonic()-before)
        path = run/'decoded'/f"{request['example_id']}.json"; write_json_new(path, result)
        entries.append({'example_id': request['example_id'], 'path': str(path), 'sha256': sha256_file(path)})
        print(request['example_id'], 'candidates', len(result['candidates']), 'decode_sec', round(result['elapsed_seconds'],3), flush=True)
    write_json_new(run/'DECODED.json', {'schema_version': 'dual_input_decoded_index_v1',
        'entries': entries, 'code_lock_sha256': sha256_file(run/'DECODER_CODE_LOCK.json'), 'reference_received': False,
        'policy_sha256': sha256_file(run/'POLICY.json'), 'elapsed_seconds': time.monotonic()-started})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    if not a.run.resolve().is_relative_to(ROOT/'.state/runs'): raise ValueError('run outside workspace')
    decode(a.run.resolve())

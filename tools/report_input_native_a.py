"""Reference-only evaluation after fixed audio predictions have been saved."""
from pathlib import Path
import argparse
from input_native_probe_common import aggregate, a1_gate
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, write_json_new, sha256_file


def report(run: Path) -> None:
    f = read_json(run / 'FREEZE.json')
    for path, digest in f['bindings'].items():
        verify_file(Path(path), digest)
    refs = {r['window_id']: r for r in read_json(run / 'REFERENCES.json')['rows']}
    prediction_run = read_json(run / 'A1/RESULT.json')
    if prediction_run['passed'] is not True:
        raise ValueError('inference stage incomplete')
    pools = {'x': [], 'v': [], 'h': []}
    dual = Path(f['dual_run'])
    for wid, ref in refs.items():
        old = read_json(dual / 'evaluation' / (wid + '.json'))
        if old['reference'] != ref['target_ipa']:
            raise ValueError('source reference changed')
        for c, key in [('x', 'mixed_greedy'), ('v', 'vocal_greedy')]:
            pools[c].append({'window_id': wid, 'song_id': ref['song_id'], 'reference': ref['target_ipa'], 'prediction': old['predictions'][key]})
    for record in prediction_run['records']:
        wid = record['task_id'].rsplit('--',1)[0]
        ref = refs[wid]; artifact = record['outputs']['G']
        verify_file(Path(artifact['posterior_path']), artifact['posterior_sha256'])
        pools['h'].append({'window_id': wid, 'song_id': ref['song_id'], 'reference': ref['target_ipa'], 'prediction': artifact['prediction']})
    gate = a1_gate(pools['h'], pools['x'])
    gate.update({'condition_metrics': {c:aggregate(rows) for c,rows in pools.items()}, 'details': pools,
                 'source_result_sha256': sha256_file(run/'A1/RESULT.json'),
                 'reference_sha256': sha256_file(run/'REFERENCES.json'),
                 'A2_status': 'eligible_not_yet_executed' if gate['passed'] else 'not_executed_A1_failed',
                 'no_alpha_or_checkpoint_search': True})
    write_json_new(run/'A1_GATE.json',gate)
    for c,m in gate['condition_metrics'].items():
        print(c,m['edit_distance'],m['reference_phones'],'S/D/I',m['substitutions'],m['deletions'],m['insertions'])
    print('A1 passed',gate['passed'],'failures',gate['failures'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    report(p.parse_args().run.resolve())

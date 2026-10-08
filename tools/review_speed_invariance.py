"""Bind the completed human-agent code review to passing, immutable evidence."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from speed_invariance_common import ROOT, protocol, bound, read_json, write_json_new, sha256_file, slots
from run_speed_worker import caps_for


def bind_review(run: Path) -> None:
    protocol(run)
    if (run / 'EXECUTION_START.json').exists() or (run / 'commands').exists():
        raise ValueError('review must precede the first launch')
    xml = run / 'prelaunch-final.xml'; suite = ET.parse(xml).getroot()
    stats = {k: sum(int(n.attrib.get(k, 0)) for n in suite.iter('testsuite')) for k in ('tests', 'failures', 'errors', 'skipped')}
    if stats != {'tests': 102, 'failures': 0, 'errors': 0, 'skipped': 0}:
        raise ValueError('expected passing executed-code tests')
    fixture = read_json(run / 'runtime-fixture-v2.json')
    if not fixture['passed'] or fixture['cuda_initialized'] or fixture['optimizer_updates'] or fixture['actual_data_neural_requests']:
        raise ValueError('invalid production-runtime fixture evidence')
    counts = Counter()
    for slot in slots(): counts.update(caps_for(run, slot))
    if dict(counts) != {'encoder': 55, 'head': 34206, 'optimizer': 4280}:
        raise ValueError('executed caps differ from authorized budget')
    note = run / 'CODE_REVIEW.md'
    if not note.is_file() or note.stat().st_size < 1000: raise ValueError('actual review narrative missing')
    evidence = run / 'PRELAUNCH_VERIFIED.json'
    write_json_new(evidence, {'passed': True, 'pytest': stats, 'caps_recomputed': dict(counts),
        'artifacts': [bound(p) for p in [xml, run / 'runtime-fixture-v2.json', run / 'PREPARED.json', note]],
        'neural_experiment_budget_consumed': 0, 'actual_training_jobs_started': 0})
    files = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'src').rglob('*.json'))
    files += [p for p in (ROOT / 'tools').glob('*.py') if 'speed' in p.name]
    files += [ROOT / 'tools/verify_acoustic_memory.py', ROOT / 'tools/acoustic_memory_common.py', ROOT / 'tests/test_speed_invariance.py', ROOT / 'AGENTS.md']
    write_json_new(run / 'CODE_REVIEW.json', {'approved_for_launch': True,
        'freeze_sha256': sha256_file(run / 'FREEZE.json'), 'verification': bound(evidence), 'narrative': bound(note),
        'code_bindings': {str(p.resolve()): sha256_file(p) for p in files},
        'review_scope': ['data_isolation', 'same_view_loss', 'initialization', 'sampling_rng', 'update_counts',
            'evaluation_gates', 'resource_limits', 'failure_accounting', 'independent_replay'],
        'authorized_by_user': '现在开始启动新实验', 'no_automatic_retry_or_budget_expansion': True})
    protocol(run, reviewed=True)
    print('Review bound. Budget:', dict(counts), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); bind_review(p.parse_args().run.resolve())

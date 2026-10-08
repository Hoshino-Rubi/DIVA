"""Propose bounded replacements from existing audio; never revive rejected pairs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, sha256_object, write_json_new
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
from any2xsampa.training.replacement_candidates import propose_windows, select_candidates


def build(output: Path) -> None:
    from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
    enforce_offline(output / 'runtime-cache')
    import numpy as np
    from any2xsampa.training.audio_section_correspondence import audio_descriptor, propose_correspondence
    started = time.monotonic()
    bindings = {}

    def load(name: str, path: Path, expected: str | None = None) -> dict:
        digest = verify_file(path, expected) if expected else sha256_file(path)
        bindings[name] = {'path': str(path.resolve()), 'sha256': digest}
        return read_json(path)

    decision = load('proposal', ROOT / '.state/runs/audibility-reference-eval-20260916T013200/NEXT_DECISION.json')
    if decision['proposal_sha256'] != sha256_object({k: v for k, v in decision.items() if k != 'proposal_sha256'}):
        raise ValueError('proposal self hash differs')
    policy = load('policy', ROOT / 'src/any2xsampa/config/training_replacement_policy_v1.json')
    write_json_new(output / 'AUTHORIZATION.json', {'user_statement': '\u5141\u8bb8\u91cd\u5207\u66ff\u6362',
        'received_local_date': '2026-09-17', 'proposal_sha256': decision['proposal_sha256'],
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'starts_training': False})
    write_json_new(output / 'FROZEN_POLICY.json', policy)
    audit = load('audit', Path(decision['previous_audit']['path']), decision['previous_audit']['sha256'])
    bundle = load('bundle', ROOT / '.state/runs/alignment-audit-intake-20260915T132143Z/raw/bundle.json')
    source = bundle['sources']
    reference = load('reference', Path(source['reference']['path']), source['reference']['sha256'])
    evidence = source['mapping_evidence']
    load('mapping_evidence', Path(evidence['path']), evidence['sha256'])
    reference, overlay = apply_original_mapping_overlay(reference, Path(evidence['path']))
    bindings['mapping_overlay'] = {'path': overlay['path'], 'sha256': overlay['sha256']}
    old = load('old_proposals', Path(source['proposals']['path']), source['proposals']['sha256'])
    contexts_spec = old['source_bindings']['contexts']
    context_doc = load('contexts', Path(contexts_spec['path']), contexts_spec['sha256'])
    contexts = [c for c in context_doc['contexts'] if c['split'] == 'train']
    if len(contexts) != 15 or len(set(c['song_id'] for c in contexts)) != 5:
        raise ValueError('must use precisely the fifteen existing training contexts')
    separation = load('separation_index', ROOT / '.state/runs/quality-first8h-20260907T010738/separation-contexts/index.json')
    sep_index = {r['context']['context_id']: r for r in separation['entries']}
    history = {r['review_id']: r for r in audit['rows'] if r['review_id'].startswith('C')}
    retained, rejected = [], []
    for row in bundle['rows']:
        if row['review_id'] not in history:
            continue
        media = next(m for m in row['media'] if m['role'] == 'rendered')
        item = {'review_id': row['review_id'], 'song_id': row['song_id'],
                'start_sec': row['start_sec'], 'end_sec': row['end_sec'],
                'rendered_start_sec': media['source_crop']['requested_start_sec'],
                'decision': history[row['review_id']]['decision'], 'history': history[row['review_id']]}
        (rejected if item['decision'] == 'misaligned' else retained).append(item)
    if len(retained) != 7 or len(rejected) != 13:
        raise ValueError('unexpected reviewed training pool')
    write_json_new(output / 'HISTORY.json', {'retained': retained, 'rejected': rejected, 'old_decisions_unchanged': True})
    pool, failures, alignments = [], [], []
    frame_seconds = 512 / 22050
    factor = policy['descriptor_downsample']
    for song in reference['songs']:
        if song['split'] != 'train':
            continue
        sid = song['song_id']
        for role in ('original', 'rendered'):
            spec = song[role + '_audio']
            verify_file(Path(spec['path']), spec['sha256'])
            bindings[sid + '_' + role] = {'path': spec['path'], 'sha256': spec['sha256']}
        old_match = load(sid + '_descriptor_record', ROOT / f'.state/runs/limited-data-20260906T050251/correspondence-v2/{sid}.json')
        for role in ('original', 'rendered'):
            if old_match[role + '_audio_sha256'] != song[role + '_audio']['sha256']:
                raise ValueError('descriptor source identity differs')
        spec = old_match['descriptor_artifact']
        verify_file(Path(spec['path']), spec['sha256']); bindings[sid + '_descriptors'] = spec
        with np.load(spec['path'], allow_pickle=False) as cache:
            original, rendered = cache['original'], cache['rendered'][:, ::factor]
        for context in [c for c in contexts if c['song_id'] == sid]:
            cid = context['context_id']
            if context['audio_sha256'] != song['original_audio']['sha256']:
                raise ValueError('context source mismatch')
            sep = sep_index[cid]
            record = load(cid + '_separation', Path(sep['record_path']), sep['record_sha256'])
            if record['context'] != context or record['labels_or_score_used'] or not record['clock']['no_time_stretch_or_compensation']:
                raise ValueError('separated context clock or role differs')
            vocal_spec = record['vocal']
            bindings[cid + '_vocal'] = {'path': vocal_spec['path'], 'sha256': vocal_spec['sha256']}
            vocal = audio_descriptor(vocal_spec)[:, ::factor]
            first, last = round(context['start_sec'] / frame_seconds), round(context['end_sec'] / frame_seconds)
            queries = {'mixed': (original[:, first:last:factor], first * frame_seconds),
                       'vocal': (vocal, record['clock']['effective_original_start_sec'])}
            paths = {role: propose_correspondence(x, rendered, query_start_sec=start,
                     frame_seconds=frame_seconds * factor) for role, (x, start) in queries.items()}
            paths_record = {'context': context, 'paths': paths}
            write_json_new(output / 'alignments' / f'{cid}.json', paths_record)
            alignments.append(paths_record)
            if any(not value['candidates'] for value in paths.values()):
                failures.append({'context_id': cid, 'reasons': ['no_audio_path']})
                continue
            selected_paths = {k: v['candidates'][0] for k, v in paths.items()}
            accepted, excluded = propose_windows(context, song, selected_paths, retained, rejected, policy)
            pool.extend(accepted); failures.extend(excluded)
            print(f'{cid}: pool={len(accepted)}; excluded={len(excluded)}', flush=True)
    candidates = select_candidates(pool, retained, context_doc['split']['train'], policy['replacement_limit'])
    available = sorted(rejected, key=lambda r: r['review_id'])
    for i, row in enumerate(candidates, 1):
        same = next((r for r in available if r['song_id'] == row['song_id']), available[0])
        available.remove(same)
        row.update(review_id=f'R{i:02d}', replaces_slot=same['review_id'], candidate_version=1)
        song = next(s for s in reference['songs'] if s['song_id'] == row['song_id'])
        row['original_audio'] = song['original_audio']; row['rendered_audio'] = song['rendered_audio']
    result = {'schema_version': 'training_replacement_candidates_v1', 'policy': policy,
        'proposal_sha256': decision['proposal_sha256'], 'source_bindings': bindings,
        'retained_slots': [r['review_id'] for r in retained], 'rejected_slots_preserved': [r['review_id'] for r in rejected],
        'candidates': candidates, 'retained_count': 7, 'replacement_count': len(candidates),
        'active_count': 7 + len(candidates), 'pool_count': len(pool),
        'song_counts': dict(Counter(r['song_id'] for r in candidates)),
        'unique_new_audio_seconds': len(candidates) * 10, 'failure_counts': dict(Counter(reason for r in failures for reason in r['reasons'])),
        'training_eligible': False, 'test_songs_opened': False, 'elapsed_seconds': time.monotonic() - started}
    result['packet_sha256'] = sha256_object(result)
    write_json_new(output / 'candidate-packet.json', result)
    write_json_new(output / 'excluded-windows.json', {'windows': failures})
    for spec in bindings.values():
        verify_file(Path(spec['path']), spec['sha256'])
    print(f'replacements={len(candidates)}; active={7+len(candidates)}; pool={len(pool)}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / '.state/runs'):
        raise ValueError('output must be new under .state/runs')
    build(output)


if __name__ == '__main__':
    main()

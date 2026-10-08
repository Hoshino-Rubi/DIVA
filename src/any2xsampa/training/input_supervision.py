"""Immutable paired-reference validation and old-control-compatible scheduling."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from any2xsampa.training.limited_data_artifacts import read_json, sha256_object, verify_file
from any2xsampa.training.singing_domain_losses import required_frames

CONFIG = Path(__file__).parents[1] / "config/input_supervision_v2.json"


def policy() -> dict:
    return read_json(CONFIG)


def validate_rows(rows: list[dict], vocabulary: dict, *, enforce_size: bool = True) -> None:
    spec = policy()
    if len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate row identity')
    sources: dict[str, dict] = {}
    for row in rows:
        if row['split'] != 'train' or row['song_id'] not in spec['songs']:
            raise ValueError('nontraining song or role in paired reference pool')
        kind, condition = row['audio_kind'], row['condition']
        if (kind, condition) not in {('original', 'x'), ('original', 'v'), ('rendered', 'rendered')}:
            raise ValueError('invalid source role or input condition')
        if row['reference_nature'] != ('derived_from_confirmed_original_not_separately_audited' if condition == 'v'
                                      else 'confirmed_original' if condition == 'x' else 'rendered_proxy'):
            raise ValueError('reference nature contradicts input role')
        phones, ids = row['target_ipa'], row['target_ids']
        if not phones or any(p not in vocabulary or p.startswith('<') for p in phones):
            raise ValueError('unknown or special reference phone')
        if ids != [vocabulary[p] for p in phones] or any(isinstance(i, bool) or i <= 0 for i in ids):
            raise ValueError('target IDs differ from reference vocabulary')
        if required_frames(ids) > row['frames']:
            raise ValueError('CTC reference cannot fit cached frames')
        if not isinstance(row['source_order'], int) or isinstance(row['source_order'], bool):
            raise ValueError('source order must bind historical cache position')
        a, b = row['source_start_sec'], row['source_end_sec']
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (a, b)) or not 0 <= a < b:
            raise ValueError('invalid source sampling interval')
        if kind == 'original':
            group = sources.setdefault(row['window_id'], {})
            if condition in group:
                raise ValueError('duplicate input view')
            group[condition] = row
    for views in sources.values():
        if set(views) != {'x', 'v'}:
            raise ValueError('missing matched source view')
        for key in ('target_ipa', 'target_ids', 'song_id', 'source_order', 'frames', 'source_audio_sha256', 'source_start_sec', 'source_end_sec'):
            if views['x'][key] != views['v'][key]:
                raise ValueError('paired reference mismatch: ' + key)
    originals = sorted((v['x'] for v in sources.values()), key=lambda r: (r['song_id'], r['source_start_sec']))
    for left, right in zip(originals, originals[1:]):
        if left['song_id'] == right['song_id'] and left['source_audio_sha256'] == right['source_audio_sha256'] and right['source_start_sec'] < left['source_end_sec'] - 1e-6:
            raise ValueError('overlapping original source windows')
    if enforce_size and (len(sources) != spec['source_windows'] or len(rows) != 169 or
                         sum(len(r['target_ipa']) for r in originals) != spec['original_reference_phones'] or
                         sum(r['audio_kind'] == 'rendered' for r in rows) != spec['rendered_windows']):
        raise ValueError('fixed source coverage or denominator changed')


def supervision_slots(rows: list[dict], held_song: str | None, variant: str) -> list[dict]:
    spec = policy()
    if variant not in spec['variants'] or (held_song is not None and held_song not in spec['songs']):
        raise ValueError('unplanned variant or held song')
    if any(r['split'] != 'train' or r['song_id'] not in spec['songs'] for r in rows):
        raise ValueError('song split leakage')
    lookup = {(r['window_id'], r['condition']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('duplicate source/view')
    anchors = sorted((r for r in rows if r['condition'] in {'x', 'rendered'} and r['song_id'] != held_song),
                     key=lambda r: r['source_order'])
    if len({r['source_order'] for r in anchors}) != len(anchors):
        raise ValueError('ambiguous historical source order')
    ordered = []
    for row in anchors:
        views = ['rendered'] if row['audio_kind'] == 'rendered' else spec['slots'][variant]
        for slot, view in enumerate(views):
            mate = lookup[(row['window_id'], view)]
            if any(mate[k] != row[k] for k in ('song_id', 'target_ipa', 'target_ids', 'source_order')):
                raise ValueError('slot view changed reference identity')
            ordered.append({**mate, 'slot_id': f"{row['window_id']}:{slot}"})
    return ordered


def shuffle_epoch(slots: list[dict], unused_unlabeled_ids: list[str], rng: Any) -> list[dict]:
    ordered = list(slots)
    rng.shuffle(ordered)
    # Old baseline consumed this shuffle even though it never trained these rows.
    ignored = list(unused_unlabeled_ids)
    rng.shuffle(ignored)
    return ordered


def load_manifest(run: Path) -> tuple[dict, dict, dict]:
    freeze = read_json(run / 'FREEZE.json')
    if freeze.get('schema_version') != 'input_supervision_freeze_v2':
        raise ValueError('legacy v1 schedule is not a valid matched-control experiment')
    for binding in freeze['bindings'].values():
        verify_file(Path(binding['path']), binding['sha256'])
    for path, digest in freeze['code_bindings'].items():
        verify_file(Path(path), digest)
    if read_json(run / 'POLICY.json') != policy():
        raise ValueError('frozen training policy changed')
    manifest, index = read_json(run / 'ROWS.json'), read_json(run / 'CACHE_BINDINGS.json')
    vocabulary = read_json(Path(freeze['vocabulary_path']))
    validate_rows(manifest['rows'], vocabulary)
    entries = index['entries']
    if (len(entries) != len(manifest['rows']) or len({e['cache_id'] for e in entries}) != len(entries)
            or index['labels_received'] is not False or index['all_parameters_frozen'] is not True):
        raise ValueError('cache coverage or provenance changed')
    return freeze, manifest, index

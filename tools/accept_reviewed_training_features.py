"""Audio-only requests and label-side acceptance for reviewed original windows."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, sha256_object, verify_file, write_json_new
from any2xsampa.training.limited_data_reference import SPLITS, load_vocabulary
from any2xsampa.training.phoneticxeus_adaptation import required_ctc_frames
from tools.reconcile_replacement_edges import build_reference, load_verified_packet
from tools.run_phoneticxeus_worker import AUDIO_KEYS, validate_requests


def verified_reference(run: Path) -> dict:
    inputs = read_json(run / 'INPUTS.json')
    for name in ('packet', 'answers'):
        verify_file(Path(inputs[name]['path']), inputs[name]['sha256'])
    packet_path = Path(inputs['packet']['path'])
    if read_json(packet_path).get('schema_version') == 'free_location_review_packet_v1':
        from tools.free_location_review import assemble, verify_packet
        packet = read_json(packet_path)
        verify_packet(packet_path.parent, packet)
        rebuilt = assemble(packet, read_json(Path(inputs['answers']['path'])))
    elif read_json(packet_path).get('schema_version') == 'batch_training_review_packet_v1':
        from tools.batch_training_review import assemble, verify_sources_and_media
        packet = read_json(packet_path)
        verified = verify_sources_and_media(packet_path.parent, packet)
        rebuilt = assemble(packet, read_json(Path(inputs['answers']['path'])), verified)
    else:
        packet = load_verified_packet(packet_path)
        rebuilt = build_reference(packet, read_json(Path(inputs['answers']['path'])))
    if read_json(run / 'reviewed-reference.json') != rebuilt:
        raise ValueError('reviewed reference cannot be reproduced from bound answers')
    return rebuilt


def requests_for(reference: dict, source_hash: str) -> dict:
    rows = []
    for row in reference['examples']:
        if row['status'] != 'human_confirmed':
            continue
        if row['split'] != 'train' or row['audio_role'] != 'original':
            raise ValueError('only confirmed original training references')
        audio = {key: row[key] for key in AUDIO_KEYS}
        audio['example_id'] = row['example_id'].lower()
        rows.append(audio)
    result = {'schema_version': 'phoneticxeus_audio_requests_v1', 'purpose': 'original-reviewed-v2',
        'source_dataset_hash': reference['manifest_hash'], 'source_manifest_file_sha256': source_hash,
        'split_sha256': sha256_object(SPLITS), 'rows': rows, 'contains_labels_or_score_context': False}
    validate_requests(result)
    return result


def accept_row(row: dict, entry: dict, vocabulary: dict, feature_shape: tuple, logit_shape: tuple) -> dict:
    if row['status'] != 'human_confirmed' or row['split'] != 'train' or row['audio_role'] != 'original':
        raise ValueError('unconfirmed or non-training reference')
    for key in ('song_id', 'split', 'audio_role', 'audio_sha256', 'start_sec', 'end_sec'):
        if row[key] != entry[key]:
            raise ValueError('cache/reference binding mismatch: ' + key)
    if entry['example_id'] != row['example_id'].lower() or entry['test_outputs_sealed']:
        raise ValueError('wrong or sealed cache')
    if len(vocabulary) != 428 or vocabulary.get('<blank>') != 0:
        raise ValueError('wrong vocabulary')
    ids = [vocabulary.get(p['phone'], -1) for p in row['phones']]
    if not ids or any(i <= 0 for i in ids):
        raise ValueError('unknown or blank reference phone')
    frames = entry['frames']
    if feature_shape != (frames, 1024) or logit_shape != (frames, 428):
        raise ValueError('cache shape mismatch')
    minimum = required_ctc_frames(ids)
    if frames < minimum:
        raise ValueError('insufficient CTC frames including repeated labels')
    return {'example_id': row['example_id'], 'song_id': row['song_id'], 'split': 'train',
            'status': 'data_accepted', 'audio_role': 'original', 'audio_sha256': row['audio_sha256'],
            'start_sec': row['start_sec'], 'end_sec': row['end_sec'],
            'model_target_ids': ids, 'reference_phones': len(ids), 'ctc_minimum_frames': minimum,
            'actual_frames': frames, 'feature_shape': list(feature_shape), 'logit_shape': list(logit_shape),
            'feature_binding': entry, 'dataset_training_gate_passed': False}


def finish(reference_run: Path, output: Path, *, write: bool = True) -> dict:
    import numpy as np
    from safetensors.numpy import load_file

    reference = verified_reference(reference_run)
    requests_path = output / 'requests/original-reviewed.json'
    requests = read_json(requests_path)
    expected = requests_for(reference, sha256_file(reference_run / 'reviewed-reference.json'))
    if requests != expected:
        raise ValueError('audio requests changed')
    index_path = output / 'features/index.json'
    index = read_json(index_path)
    if index['requests_sha256'] != sha256_file(requests_path) or index['source_dataset_hash'] != reference['manifest_hash']:
        raise ValueError('feature index binding mismatch')
    if index['all_parameters_frozen'] is not True or index['no_labels_or_score_received'] is not True:
        raise ValueError('worker did not retain audio-only frozen contract')
    expected_ids = {r['example_id'].lower() for r in reference['examples'] if r['status'] == 'human_confirmed'}
    if {e['example_id'] for e in index['entries']} != expected_ids or len(index['entries']) != len(expected_ids):
        raise ValueError('feature cache entries differ from accepted references')
    rows = []
    for entry in index['entries']:
        verify_file(Path(entry['evidence_path']), entry['evidence_sha256'])
        verify_file(Path(entry['posterior_path']), entry['posterior_sha256'])
        payload = read_json(Path(entry['posterior_path']))
        vocab_path = Path(entry['posterior_path']).parent / 'vocabulary.json'
        verify_file(vocab_path, entry['vocabulary_sha256'])
        vocabulary = load_vocabulary(vocab_path)
        tensors = load_file(entry['evidence_path'])
        if set(tensors) != {'encoder_features', 'frame_logits'} or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in tensors.values()):
            raise ValueError('invalid feature or logit tensors')
        row = next(
            (r for r in reference['examples']
             if r.get('example_id') and r['example_id'].lower() == entry['example_id']),
            None,
        )
        if row is None:
            raise ValueError('feature cache entry is not a confirmed reference: ' + entry['example_id'])
        crop = payload['producer']['crop']
        if crop['requested_start_sec'] != row['start_sec'] or crop['requested_end_sec'] != row['end_sec']:
            raise ValueError('posterior crop changed')
        if payload['producer']['input_audio']['sha256'] != row['audio_sha256']:
            raise ValueError('posterior audio hash changed')
        accepted = accept_row(row, entry, vocabulary, tensors['encoder_features'].shape, tensors['frame_logits'].shape)
        accepted['sample_crop'] = crop
        rows.append(accepted)
    result = {'schema_version': 'reviewed_training_data_acceptance_v1', 'rows': rows,
        'reviewed_reference_sha256': sha256_file(reference_run / 'reviewed-reference.json'),
        'feature_index_sha256': sha256_file(index_path), 'valid_window_count': len(rows),
        'valid_training_song_count': len({r['song_id'] for r in rows}),
        'training_gate_passed': False, 'training_started': False,
        'required_windows': 20, 'required_training_songs': 3}
    if write:
        write_json_new(output / 'DATA_ACCEPTANCE.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'finish'])
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / '.state/runs'):
        parser.error('output outside workspace runs')
    if args.stage == 'prepare':
        reference = verified_reference(args.reference_run.resolve())
        path = output / 'requests/original-reviewed.json'
        print(write_json_new(path, requests_for(reference, sha256_file(args.reference_run / 'reviewed-reference.json'))))
    else:
        result = finish(args.reference_run.resolve(), output)
        print([(r['example_id'], r['reference_phones'], r['ctc_minimum_frames'], r['actual_frames']) for r in result['rows']])


if __name__ == '__main__':
    main()

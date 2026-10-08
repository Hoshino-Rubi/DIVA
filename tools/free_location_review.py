"""One manual training slot, freely selected from the complete score catalog."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, sha256_file, sha256_object, verify_file, finite_number
from any2xsampa.training.limited_data_reference import SONG_SPLIT
from any2xsampa.training.original_edge_review import _validate_answer
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay
from any2xsampa.training.replacement_candidates import overlaps
from tools.build_original_edge_review import note_view
from tools.reconcile_replacement_edges import assemble_row

PACKET_SCHEMA = 'free_location_review_packet_v1'
ANSWER_SCHEMA = 'free_location_review_answers_v1'
RUNS = ROOT / '.state/runs'
REPAIR = RUNS / 'correspondence-repair-20260918T234000'


def validate_packet(packet: dict) -> None:
    if packet.get('schema_version') != PACKET_SCHEMA or packet.get('packet_sha256') != sha256_object({k:v for k,v in packet.items() if k != 'packet_sha256'}):
        raise ValueError('free-location packet hash/schema mismatch')
    if SONG_SPLIT.get(packet['song_id']) != 'train' or packet['split'] != 'train':
        raise ValueError('only training song allowed')
    if packet.get('maximum_accepted_windows') != 1 or packet.get('active_slots_after') != 20:
        raise ValueError('one remaining training slot only')
    if packet.get('machine_correspondence_passed') is not False or packet.get('training_started') is not False:
        raise ValueError('manual location cannot claim machine confirmation or training')
    if len(packet['windows']) != 2 or len({w['window_id'] for w in packet['windows']}) != 2:
        raise ValueError('expected two alternatives for one slot')
    for w in packet['windows']:
        start = finite_number(w['start_sec'], 'start', minimum=0)
        end = finite_number(w['end_sec'], 'end', minimum=0)
        if abs(end - start - 10) > 1e-6:
            raise ValueError('original windows must be ten seconds')
    notes = packet['catalog']
    if not notes or len({n['note_id'] for n in notes}) != len(notes):
        raise ValueError('invalid full note catalog')


def validate_answers(packet: dict, payload: dict) -> dict:
    validate_packet(packet)
    if payload.get('schema_version') != ANSWER_SCHEMA or payload.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('answers belong to a different manual location packet')
    reviewer = payload.get('reviewer')
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 120 or payload.get('training_started') is not False:
        raise ValueError('reviewer required; training must remain false')
    if 'answers' in payload:
        raise ValueError('only one answer is permitted for this slot')
    answer = payload.get('answer')
    if not isinstance(answer, dict) or answer.get('status') not in {'confirmed', 'pending', 'unavailable'}:
        raise ValueError('one answer with explicit status required')
    if not isinstance(answer.get('comment'), str) or len(answer['comment']) > 4000:
        raise ValueError('invalid comment')
    if answer.get('window_id') not in {w['window_id'] for w in packet['windows']}:
        raise ValueError('unknown original window')
    if answer['status'] != 'confirmed':
        if answer.get('edges') != {} or answer.get('sequence_confirmed') is not False:
            raise ValueError('pending answer cannot assert phone reference')
        return answer
    if answer.get('sequence_confirmed') is not True or set(answer.get('edges', {})) != {'start', 'end'}:
        raise ValueError('explicit sequence and both edges required')
    for side in ('start', 'end'):
        edge = answer['edges'][side]
        if edge.get('review_id') != answer['window_id'] + ('-S' if side == 'start' else '-E') or edge.get('status') != 'confirmed' or edge.get('observation_schema') != 'phone_audibility_v2':
            raise ValueError('invalid edge envelope')
        _validate_answer({'side': side, 'notes': packet['catalog']}, edge, reviewer)
        note = next(n for n in packet['catalog'] if n['note_id'] == edge['note_id'])
        if [p['phone_index'] for p in edge['phones']] != list(range(len(note['phones']))):
            raise ValueError('every edge phone requires explicit observation')
    positions = {n['note_id']: i for i,n in enumerate(packet['catalog'])}
    if positions[answer['edges']['start']['note_id']] > positions[answer['edges']['end']['note_id']]:
        raise ValueError('start note follows end note')
    return answer


def assemble(packet: dict, payload: dict) -> dict:
    answer = validate_answers(packet, payload)
    if answer['status'] != 'confirmed':
        row = {'review_id': 'M01', 'status': 'pending', 'phones': [], 'training_eligible': False,
               'raw_answer': deepcopy(answer)}
    else:
        window = next(w for w in packet['windows'] if w['window_id'] == answer['window_id'])
        clip = dict(example_id='manual-' + window['window_id'].lower(), song_id=packet['song_id'], split='train',
                    audio_path=packet['original_audio']['path'], audio_sha256=packet['original_audio']['sha256'],
                    source_vsqx_sha256=packet['source_vsqx']['sha256'],
                    reference_manifest_sha256=packet['source_bindings']['reference']['sha256'],
                    start_sec=window['start_sec'], end_sec=window['end_sec'], notes=packet['catalog'],
                    source_sequence_approval_bound=packet['source_sequence_approval_bound'])
        row = assemble_row(clip, answer['edges'], payload['reviewer'])
        row.update(review_id='M01', raw_answer=deepcopy(answer),
                   correspondence_source='human_selected_full_catalog_without_machine_choice')
    result = dict(schema_version='batch_reviewed_training_reference_v1', packet_sha256=packet['packet_sha256'],
                  answers_sha256=sha256_object(payload), examples=[row], training_started=False,
                  confirmed_count=int(row['status'] == 'human_confirmed'))
    result['manifest_hash'] = sha256_object(result)
    return result


def verify_packet(directory: Path, packet: dict) -> None:
    import wave

    validate_packet(packet)
    for spec in packet['source_bindings'].values():
        verify_file(Path(spec['path']), spec['sha256'])
    sources = packet['source_bindings']
    reference, _ = apply_original_mapping_overlay(read_json(Path(sources['reference']['path'])), Path(sources['mapping_evidence']['path']))
    song = next(s for s in reference['songs'] if s['song_id'] == packet['song_id'] and s['split'] == 'train')
    expected = [{**note_view(n), 'start_sec': n['start_sec'], 'end_sec': n['end_sec']} for n in song['notes']]
    if packet['catalog'] != expected:
        raise ValueError('catalog must equal the complete bound song; no machine truncation')
    for key, role in [('original_audio', 'original_audio'), ('rendered_audio', 'rendered_audio'), ('source_vsqx', 'human_vsqx')]:
        if packet[key] != song[role]:
            raise ValueError('song source mismatch')
        verify_file(Path(packet[key]['path']), packet[key]['sha256'])
    claim = next(r for r in read_json(Path(sources['reference_triplets']['path']))['items'] if r['song_id'] == packet['song_id'])
    bound = claim['human_vsqx']['sha256'] == song['human_vsqx']['sha256'] and claim['human_claims'].get('phoneme_correctness') == 'user_asserted_correct'
    if packet['source_sequence_approval_bound'] != bound or not bound:
        raise ValueError('source phone approval not bound')
    history = read_json(Path(sources['history']['path']))
    location = read_json(Path(sources['location_review']['path']))
    protected = history['older_reviews']['retained'] + history['older_reviews']['rejected'] + history['batch_reviews'] + location['items']
    cumulative = read_json(Path(sources['cumulative']['path']))
    if cumulative['manifest_hash'] != sha256_object({k:v for k,v in cumulative.items() if k != 'manifest_hash'}) or cumulative['valid_windows'] != 19:
        raise ValueError('expected nineteen existing windows')
    protected += cumulative['rows']
    contexts = read_json(Path(sources['contexts']['path']))['contexts']
    media = [(packet['rendered_preview'], song['rendered_audio'], 0., song['rendered_audio']['duration_sec'])]
    for w in packet['windows']:
        row = {**w, 'song_id': packet['song_id']}
        if set(w['edges']) != {'start', 'end'}:
            raise ValueError('both original edge previews required')
        if any(overlaps(row, p) for p in protected):
            raise ValueError('new manual question repeats reviewed audio')
        if not any(c['context_id'] == w['context_id'] and c['song_id'] == packet['song_id'] and c['split'] == 'train' and c['start_sec'] <= w['start_sec'] < w['end_sec'] <= c['end_sec'] for c in contexts):
            raise ValueError('window outside authorized training contexts')
        media.append((w['original'], song['original_audio'], w['start_sec'], w['end_sec']))
        for side, m in w['edges'].items():
            media.append((m, song['original_audio'], w[side + '_sec'] - 1.25, w[side + '_sec'] + 1.25))
    for m, source, start, end in media:
        crop = m['source_crop']
        if m['source_sha256'] != source['sha256'] or crop['requested_start_sec'] != start or crop['requested_end_sec'] != end:
            raise ValueError('media source clock mismatch')
        for kind in ('audio', 'waveform'):
            path = (directory / 'media' / m[kind + '_file']).resolve()
            if not path.is_relative_to((directory / 'media').resolve()):
                raise ValueError('unsafe media path')
            verify_file(path, m[kind + '_sha256'])
        with wave.open(str(directory / 'media' / m['audio_file']), 'rb') as wav:
            if abs(wav.getnframes() / wav.getframerate() - (end - start)) > 1 / wav.getframerate():
                raise ValueError('media sample duration mismatch')


def render(directory: Path, packet: dict) -> None:
    verify_packet(directory, packet)
    audio = [packet['rendered_preview']]
    for w in packet['windows']:
        audio += [w['original'], *w['edges'].values()]
    data = {m['audio_file']: base64.b64encode((directory / 'media' / m['audio_file']).read_bytes()).decode('ascii') for m in audio}
    assets = ROOT / 'tools/free_location_review_gui'
    encode = lambda obj: json.dumps(obj, ensure_ascii=True).replace('<', r'\u003c')
    html = (assets / 'index.html').read_text(encoding='utf-8')
    values = {'PACKET': encode(packet), 'AUDIO': encode(data),
              'CSS': (assets / 'style.css').read_text(encoding='utf-8'),
              'STATE': (assets / 'state.js').read_text(encoding='utf-8'),
              'APP': (assets / 'app.js').read_text(encoding='utf-8')}
    for name, value in values.items():
        html = html.replace('/* INLINE_' + name + ' */', value)
    with (directory / 'index.html').open('x', encoding='utf-8') as stream:
        stream.write(html)


def reference_preview(source: dict, media: Path) -> dict:
    """Full score audition only, assembled from bounded audio-reader crops."""
    import numpy as np
    import soundfile as sf
    from any2xsampa.adapters.phoneticxeus_runtime import read_audio_crop
    from tools.build_limited_data_review import waveform_png

    blocks, crops = [], []
    start, end = 0., source['duration_sec']
    while start < end:
        stop = min(start + 20, end)
        samples, _, crop = read_audio_crop(Path(source['path']), source['sha256'], start, stop)
        blocks.append(samples); crops.append(crop); start = stop
    samples = np.concatenate(blocks)
    gain = min(1., .98 / float(np.max(np.abs(samples)))) if np.any(samples) else 1.
    samples *= gain
    wav, png = media / 'full-reference.wav', media / 'full-reference.png'
    sf.write(wav, samples, 16000, subtype='PCM_16')
    waveform_png(samples, len(samples) / 16000, png, '#b24758')
    return dict(role='rendered', audio_file=wav.name, waveform_file=png.name,
                audio_sha256=sha256_file(wav), waveform_sha256=sha256_file(png),
                source_path=source['path'], source_sha256=source['sha256'],
                source_crop={'requested_start_sec':0., 'requested_end_sec':end},
                duration_sec=len(samples)/16000, preview_gain=gain,
                preview_transform='bounded_crop_concatenation_mono16khz_pcm16_audition_only',
                component_crops=crops, model_inference=False)


def build(output: Path) -> dict:
    from tools.prepare_original_training_candidates import _preview

    if output.exists() or not output.is_relative_to(RUNS):
        raise ValueError('new run under .state/runs required')
    prior = read_json(RUNS / 'final-training-window-followup-20260918T223000/packet.json')
    sources = {k: deepcopy(prior['source_bindings'][k]) for k in ('reference', 'mapping_evidence', 'mapping_overlay', 'reference_triplets', 'contexts', 'location_review')}
    for name, path in [('history', REPAIR / 'diagnostic-vocal/HISTORY.json'),
                       ('fresh_audio', REPAIR / 'FRESH_AUDIO_COVERAGE.json'),
                       ('machine_failures', REPAIR / 'FRESH_CANDIDATE_CHECKS.json'),
                       ('cumulative', RUNS / 'reviewed-batch-followup-features-20260918T141000/CUMULATIVE_DATA.json')]:
        sources[name] = {'path': str(path), 'sha256': sha256_file(path)}
    for spec in sources.values():
        verify_file(Path(spec['path']), spec['sha256'])
    ref, _ = apply_original_mapping_overlay(read_json(Path(sources['reference']['path'])), Path(sources['mapping_evidence']['path']))
    song = next(s for s in ref['songs'] if s['song_id'] == 'queen-road-east')
    # Fixed chronological order among this song's unseen windows; no PER or
    # prediction ranking. The third extends beyond currently validated paths.
    unseen = read_json(Path(sources['fresh_audio']['path']))['unreviewed_windows']
    candidates = sorted((r for r in unseen if r['song_id'] == song['song_id']), key=lambda r:r['start_sec'])[:2]
    if len(candidates) != 2:
        raise ValueError('two unreviewed alternatives unavailable')
    output.mkdir(parents=True)
    media = output / 'media'; media.mkdir()
    windows = []
    for i, row in enumerate(candidates):
        wid = 'M01-' + chr(65+i)
        w = {**row, 'window_id': wid}
        w['original'] = _preview(song['original_audio'], row, media, wid + '-original', 'original')
        w['edges'] = {side: _preview(song['original_audio'], {'start_sec':row[side+'_sec']-1.25,
                      'end_sec':row[side+'_sec']+1.25}, media, wid+'-'+side, 'original') for side in ('start','end')}
        windows.append(w)
    rendered = reference_preview(song['rendered_audio'], media)
    packet = dict(schema_version=PACKET_SCHEMA, source_bindings=sources, song_id=song['song_id'], song_name='皇后大道東', split='train',
                  windows=windows, catalog=[{**note_view(n), 'start_sec':n['start_sec'], 'end_sec':n['end_sec']} for n in song['notes']],
                  original_audio=song['original_audio'], rendered_audio=song['rendered_audio'], source_vsqx=song['human_vsqx'],
                  rendered_preview=rendered, source_sequence_approval_bound=True,
                  maximum_accepted_windows=1, active_slots_after=20, machine_correspondence_passed=False,
                  training_started=False, purpose='human_free_location_and_edges_in_one_pass',
                  policy={'no_machine_note_selection':True,'full_catalog_available':True,'one_of_two_original_windows':True,
                          'no_new_mapping_aliases':True,'test_songs_opened':False})
    packet['packet_sha256'] = sha256_object(packet)
    write_json_new(output / 'packet.json', packet)
    render(output, packet)
    write_json_new(output / 'DELIVERY.json', {'packet_sha256':packet['packet_sha256'], 'html_sha256':sha256_file(output/'index.html'),
                   'answers_filename':'free-location-answers-'+packet['packet_sha256'][:12]+'.json',
                   'training_started':False, 'one_manual_slot':True})
    return packet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--answers', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.build:
        p = build(args.build.resolve()); print('packet', p['packet_sha256'])
    else:
        if not all((args.packet,args.answers,args.output)):
            parser.error('--packet --answers --output required for intake')
        out=args.output.resolve()
        if out.exists() or not out.is_relative_to(RUNS):
            parser.error('new output directory required')
        p=read_json(args.packet); verify_packet(args.packet.resolve().parent,p)
        a=read_json(args.answers); result=assemble(p,a);out.mkdir(parents=True)
        write_json_new(out/'raw-answers.json',a);write_json_new(out/'reviewed-reference.json',result)
        write_json_new(out/'INPUTS.json',{name:{'path':str(path.resolve()),'sha256':sha256_file(path)}
                       for name,path in [('packet',args.packet),('answers',args.answers)]})
        print('confirmed references',result['confirmed_count'],'training not started')

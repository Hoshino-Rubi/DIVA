"""Preserve blind human answers and propose explicit, opt-in proxy mappings."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, sha256_file, sha256_object, verify_file
from any2xsampa.training.limited_data_reference import load_mapping, load_vocabulary, project_tokens
from any2xsampa.training.original_reference_mapping import documented_compound_mapping, OVERLAY_PATH
from any2xsampa.training.original_edge_review import _validate_answer
from tools.final_test_reference_review import assemble, bind, validate_answers, verify_packet as verify_source_packet
from tools.reconcile_replacement_edges import assemble_row

Json = dict[str, Any]
SCHEMA = 'final_test_proxy_mapping_review_packet_v1'
ANSWERS = 'final_test_proxy_mapping_review_answers_v1'
CONFIG = ROOT / 'src/any2xsampa/config/final_proxy_mapping_proposals_v1.json'
TOKENS = {'7', 'UN', 'iE_r', 'iUN', 'uo'}
CONSENT = 'engine_proxy_diagnostic_not_original_ipa_accuracy'


def selected_notes(packet: Json, answer: Json) -> tuple[Json, Json, list[Json]]:
    item = next(i for i in packet['items'] if i['window_id'] == answer['window_id'])
    song = next(s for s in packet['songs'] if s['song_id'] == item['song_id'])
    if answer['status'] not in {'confirmed', 'located'}:
        return item, song, []
    ids = [n['note_id'] for n in song['catalog']]
    lo, hi = [ids.index(answer['edges'][s]['note_id']) for s in ('start', 'end')]
    return item, song, song['catalog'][lo:hi + 1]


def edge_observation(packet: Json, payload: Json, answer: Json, side: str) -> Json:
    """Recover the GUI's preserved known edge without inventing unknown phones."""
    _, song, _ = selected_notes(packet, answer)
    edge = answer['edges'][side]
    note = next(n for n in song['catalog'] if n['note_id'] == edge['note_id'])
    origin = 'submitted_answer'
    if answer['status'] == 'located':
        draft = payload.get('unconfirmed_drafts', {}).get(answer['window_id'])
        if draft:
            same = (draft.get('window_id') == answer['window_id'] and
                    draft.get('status') == 'confirmed' and draft.get('sequence_confirmed') is True and
                    all(draft.get('edges', {}).get(s, {}).get('note_id') == answer['edges'][s]['note_id'] for s in ('start', 'end')))
            if not same:
                raise ValueError('preserved confirmed draft does not match submitted location')
            edge = draft['edges'][side]
            origin = 'preserved_explicitly_confirmed_draft'
    result = dict(note_id=note['note_id'], lyric=note['lyric'], source=origin,
                  raw=deepcopy(edge), phones=[], whole_note_audibility=edge.get('whole_note_audibility'))
    if note['mapping_status'] != 'resolved_proxy' or not edge.get('phones'):
        return result
    observed = deepcopy(edge)
    observed.update(review_id=answer['window_id'] + ('-S' if side == 'start' else '-E'),
                    status='confirmed', observation_schema='phone_audibility_v2',
                    audible_identity_confirmed=True,
                    statement='Retained from the same hash-bound human-confirmed answer and draft.')
    _validate_answer({'side': side, 'notes': song['catalog']}, observed, payload['reviewer'])
    if [p['phone_index'] for p in observed['phones']] != list(range(len(note['phones']))):
        raise ValueError('preserved known edge must cover every phone')
    result.update(phones=deepcopy(observed['phones']), validated_edge=observed)
    return result


def audit_intake(packet: Json, payload: Json) -> Json:
    validate_answers(packet, payload)
    assembled = assemble(packet, payload)
    examples = {r['window_id']: r for r in assembled['examples']}
    rows = []
    for answer in payload['answers']:
        item, song, notes = selected_notes(packet, answer)
        edges = {s: edge_observation(packet, payload, answer, s) for s in ('start', 'end')} if notes else {}
        rows.append(dict(window_id=answer['window_id'], song_id=song['song_id'], song_name=song['song_name'],
                         example_id=item['source_example_id'], start_sec=item['start_sec'], end_sec=item['end_sec'],
                         source_answer_status=answer['status'], location_confirmed=bool(notes),
                         notes=deepcopy(notes), edges=edges,
                         unresolved_notes=[deepcopy(n) for n in notes if n['mapping_status'] != 'resolved_proxy'],
                         reference_status=examples[answer['window_id']]['status'],
                         blocking_reasons=examples[answer['window_id']].get('blocking_reasons', [])))
    states = Counter(p['edge_status'] for r in rows for e in r['edges'].values() for p in e['phones'])
    notes = [n for r in rows for n in r['notes']]
    mapped = sum(n['mapping_status'] == 'resolved_proxy' for n in notes)
    return dict(schema_version='final_test_manual_audit_v1', rows=rows,
                coverage=dict(total_windows=len(rows), location_confirmed_windows=sum(r['location_confirmed'] for r in rows),
                              complete_reference_windows=sum(r['reference_status'] == 'human_confirmed' for r in rows),
                              total_seconds=sum(r['end_sec'] - r['start_sec'] for r in rows),
                              selected_notes=len(notes), mapped_notes=mapped, unresolved_notes=len(notes)-mapped,
                              mapped_note_fraction=mapped/len(notes) if notes else None,
                              known_edge_observations=dict(states)),
                mapping_responses=deepcopy(payload['mapping_responses']), test_predictions_opened=False,
                training_allowed=False, original_ipa_ground_truth_created=False)


def validate_review_packet(packet: Json) -> None:
    body = {k: v for k, v in packet.items() if k != 'packet_sha256'}
    if packet.get('schema_version') != SCHEMA or packet.get('packet_sha256') != sha256_object(body):
        raise ValueError('mapping review packet hash mismatch')
    rules = packet.get('rules', [])
    if len(rules) != 5 or {r['token'] for r in rules} != TOKENS:
        raise ValueError('exactly five mapping rules required')
    if packet.get('training_allowed') is not False or packet.get('test_predictions_opened') is not False:
        raise ValueError('proposal must remain blind and non-training')
    vocab = set(packet['allowed_ipa'])
    for rule in rules:
        ids = [o['id'] for o in rule['options']]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('duplicate or empty rule options')
        for option in rule['options']:
            if not option['ipa'] or not set(option['ipa']) <= vocab:
                raise ValueError('unsupported proposed IPA')
        if rule['bulk_option_id'] is not None and rule['bulk_option_id'] not in ids:
            raise ValueError('invalid bulk option')
    if next(r for r in rules if r['token'] == 'UN')['bulk_option_id'] is not None:
        raise ValueError('conflicting UN convention cannot be bulk approved')


def validate_decisions(packet: Json, payload: Json) -> list[Json]:
    validate_review_packet(packet)
    if payload.get('schema_version') != ANSWERS or payload.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('decisions belong to another mapping review')
    if not isinstance(payload.get('reviewer'), str) or not 0 < len(payload['reviewer'].strip()) <= 120:
        raise ValueError('reviewer required')
    if payload.get('training_started') is not False or payload.get('test_predictions_opened') is not False:
        raise ValueError('decisions must remain prediction blind')
    decisions = payload.get('decisions', [])
    if len(decisions) != 5 or {d.get('token') for d in decisions} != TOKENS:
        raise ValueError('five unique decisions required')
    for decision in decisions:
        rule = next(r for r in packet['rules'] if r['token'] == decision['token'])
        if decision.get('status') not in {'approve_proxy', 'custom_proxy', 'defer'}:
            raise ValueError('invalid proxy decision')
        if not isinstance(decision.get('comment'), str) or len(decision['comment']) > 4000:
            raise ValueError('invalid comment')
        ipa = decision.get('ipa')
        if decision['status'] == 'defer':
            if ipa != [] or decision.get('option_id') is not None:
                raise ValueError('deferred rule cannot assert IPA')
        elif decision['status'] == 'approve_proxy':
            option = next((o for o in rule['options'] if o['id'] == decision.get('option_id')), None)
            if option is None or ipa != option['ipa']:
                raise ValueError('selected proxy differs from frozen proposal')
        elif (decision.get('option_id') is not None or not isinstance(ipa, list) or not 1 <= len(ipa) <= 8 or
              any(not isinstance(p, str) or p not in packet['allowed_ipa'] for p in ipa)):
            raise ValueError('custom IPA must use supported nonblank labels')
    approved = any(d['status'] != 'defer' for d in decisions)
    if approved and payload.get('consent') != CONSENT:
        raise ValueError('explicit proxy-only consent required')
    if payload.get('authorize_frozen_proxy_comparison') is not approved:
        raise ValueError('proxy comparison authorization must match explicit decisions')
    return decisions


def assemble_proxy_reference(review: Json, payload: Json, source: Json, original_answers: Json,
                             mapping: Json, vocabulary: dict[str, int]) -> Json:
    decisions = validate_decisions(review, payload)
    audit = audit_intake(source, original_answers)
    if audit != review['manual_audit']:
        raise ValueError('upstream human audit changed')
    overlay = {d['token']: {'ipa': d['ipa']} for d in decisions if d['status'] != 'defer'}
    new_mapping = deepcopy(mapping)
    if set(overlay) & set(new_mapping['tokens']):
        raise ValueError('proxy proposal cannot overwrite resolved base tokens')
    new_mapping['tokens'].update(overlay)
    examples = []
    for row in audit['rows']:
        song = next(s for s in source['songs'] if s['song_id'] == row['song_id'])
        catalog = deepcopy(song['catalog'])
        for note in catalog:
            projected = project_tokens(note['raw_v4c_tokens'], new_mapping, vocabulary)
            note.update(phones=projected['model_target_ipa'], mapping_status=projected['mapping_status'])
        by_id = {n['note_id']: n for n in catalog}
        edges: Json = {}
        for side, observation in row['edges'].items():
            if 'validated_edge' in observation:
                edges[side] = deepcopy(observation['validated_edge'])
            elif observation['whole_note_audibility'] == 'full' and by_id[observation['note_id']]['phones']:
                note = by_id[observation['note_id']]
                edges[side] = dict(note_id=note['note_id'], review_id=row['window_id']+('-S' if side=='start' else '-E'),
                                   status='confirmed', observation_schema='phone_audibility_v2', audible_identity_confirmed=True,
                                   statement='Human confirmed the whole note fully audible; proxy units expanded under explicit mapping consent.',
                                   evidence_scope='derived_proxy_units_from_whole_note_full',
                                   phones=[dict(phone_index=i, edge_status='full') for i in range(len(note['phones']))])
        clip = dict(example_id=row['example_id'], song_id=row['song_id'], split='test',
                    audio_path=song['original_audio']['path'], audio_sha256=song['original_audio']['sha256'],
                    source_vsqx_sha256=song['source_vsqx']['sha256'],
                    reference_manifest_sha256=source['source_bindings']['reference']['sha256'],
                    start_sec=row['start_sec'], end_sec=row['end_sec'], notes=catalog,
                    source_sequence_approval_bound=song['source_sequence_approval_bound'])
        reference = assemble_row(clip, {s: edges.get(s) for s in ('start', 'end')}, payload['reviewer'])
        ready = reference['status'] == 'human_confirmed'
        reference.update(window_id=row['window_id'], status='proxy_reference_ready' if ready else 'pending',
                         proxy_evaluation_eligible=ready, original_ipa_ground_truth=False,
                         reference_nature='approved_engine_proxy_on_human_located_original_audio',
                         raw_location_answer=next(deepcopy(a) for a in original_answers['answers'] if a['window_id']==row['window_id']))
        examples.append(reference)
    result = dict(schema_version='approved_final_test_proxy_reference_v1', examples=examples, mapping_overlay=overlay,
                  review_packet_sha256=review['packet_sha256'], decision_sha256=sha256_object(payload),
                  candidate_binding=source['source_bindings']['candidate'], approved_decisions=deepcopy(decisions),
                  comparison_scope=CONSENT, formal_original_ipa_accuracy_allowed=False,
                  training_allowed=False, test_predictions_opened=False, default_replacement=False,
                  coverage=dict(total_windows=len(examples), proxy_ready_windows=sum(r['proxy_evaluation_eligible'] for r in examples),
                                reference_phones=sum(len(r['phones']) for r in examples if r['proxy_evaluation_eligible'])))
    result['manifest_hash'] = sha256_object(result)
    return result


def verify_review_packet(path: Path) -> Json:
    packet = read_json(path)
    validate_review_packet(packet)
    for binding in packet['source_bindings'].values():
        verify_file(Path(binding['path']), binding['sha256'])
    bindings = packet['source_bindings']
    original = read_json(Path(bindings['original_packet']['path']))
    verify_source_packet(Path(bindings['original_packet']['path']).parent, original)
    if audit_intake(original, read_json(Path(bindings['original_answers']['path']))) != packet['manual_audit']:
        raise ValueError('manual audit differs from source answers')
    proposals = read_json(Path(bindings['proposals']['path']))
    if [{k:v for k,v in r.items() if k not in {'human_statement','affected'}} for r in packet['rules']] != proposals['rules']:
        raise ValueError('proposal rules differ from bound configuration')
    vocabulary = load_vocabulary(Path(bindings['vocabulary']['path']))
    if packet['allowed_ipa'] != sorted(p for p in vocabulary if not p.startswith('<')):
        raise ValueError('allowed IPA differs from frozen vocabulary')
    return packet


def build(packet_path: Path, intake: Path, output: Path) -> Json:
    source = read_json(packet_path)
    verify_source_packet(packet_path.parent, source)
    inputs = read_json(intake/'INPUTS.json')
    verify_file(packet_path, inputs['packet']['sha256'])
    verify_file(intake/'raw-answers.json', inputs['answers']['sha256'])
    payload = read_json(intake/'raw-answers.json')
    if assemble(source, payload) != read_json(intake/'reviewed-reference.json'):
        raise ValueError('intake does not replay exactly')
    audit = audit_intake(source, payload)
    config = read_json(CONFIG)
    reference = read_json(Path(source['source_bindings']['reference']['path']))
    vocabulary_path = Path(reference['bindings']['vocabulary']['path'])
    verify_file(vocabulary_path, reference['bindings']['vocabulary']['sha256'])
    vocabulary = load_vocabulary(vocabulary_path)
    rules = deepcopy(config['rules'])
    for rule in rules:
        rule['human_statement'] = next(m['description'] for m in payload['mapping_responses'] if m['token']==rule['token'])
        rule['affected'] = [dict(window_id=r['window_id'], song_name=r['song_name'], note_id=n['note_id'], lyric=n['lyric'])
                            for r in audit['rows'] for n in r['unresolved_notes'] if rule['token'] in n['unresolved_tokens']]
    bindings = dict(original_packet=bind(packet_path), original_answers=bind(intake/'raw-answers.json'),
                    intake_reference=bind(intake/'reviewed-reference.json'), candidate=source['source_bindings']['candidate'],
                    vocabulary=bind(vocabulary_path), base_mapping=reference['bindings']['mapping'],
                    mapping_overlay=source['source_bindings']['mapping_overlay'],
                    manual_evidence=source['source_bindings']['mapping_evidence'],
                    legacy_v3=bind(ROOT/'src/any2xsampa/config/v4c_to_recognizer_contract_v3.json'))
    output.mkdir(parents=True)
    shutil.copyfile(CONFIG, output/'mapping-proposals.json')
    bindings['proposals'] = bind(output/'mapping-proposals.json')
    manual = read_json(Path(bindings['manual_evidence']['path']))
    for page, spec in zip((168,169), manual['page_images']):
        verify_file(Path(spec['path']), spec['sha256'])
        bindings['manual_page_'+str(page)] = bind(Path(spec['path']))
    packet = dict(schema_version=SCHEMA, source_bindings=bindings, manual_audit=audit,
                  rules=rules, allowed_ipa=sorted(p for p in vocabulary if not p.startswith('<')),
                  previous_reviewer=payload['reviewer'], consent_scope=CONSENT,
                  training_allowed=False, test_predictions_opened=False, original_ipa_ground_truth_created=False,
                  prediction_based_rule_selection=False, generated_utc=datetime.now(timezone.utc).isoformat())
    packet['packet_sha256'] = sha256_object(packet)
    validate_review_packet(packet)
    write_json_new(output/'packet.json', packet)
    write_json_new(output/'AUDIT.json', audit)
    return packet


def run_import(packet_path: Path, answers_path: Path, output: Path) -> Json:
    review = verify_review_packet(packet_path)
    bindings = review['source_bindings']
    source = read_json(Path(bindings['original_packet']['path']))
    original_answers = read_json(Path(bindings['original_answers']['path']))
    mapping = documented_compound_mapping(load_mapping(Path(bindings['base_mapping']['path'])), read_json(OVERLAY_PATH))
    vocabulary = load_vocabulary(Path(bindings['vocabulary']['path']))
    answers = read_json(answers_path)
    result = assemble_proxy_reference(review, answers, source, original_answers, mapping, vocabulary)
    output.mkdir(parents=True)
    shutil.copyfile(answers_path, output/'raw-mapping-decisions.json')
    write_json_new(output/'proxy-reference.json', result)
    write_json_new(output/'INPUTS.json', dict(packet=bind(packet_path), answers=bind(answers_path), predictions_read=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--intake', type=Path)
    parser.add_argument('--answers', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.state/runs'):
        raise ValueError('new run directory required')
    if bool(args.intake) == bool(args.answers):
        raise ValueError('choose intake preparation or answer import')
    result = build(args.packet.resolve(), args.intake.resolve(), output) if args.intake else run_import(args.packet.resolve(), args.answers.resolve(), output)
    print(result.get('packet_sha256', result.get('coverage')))


if __name__ == '__main__':
    main()

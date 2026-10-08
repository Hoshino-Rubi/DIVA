"""Aggregate listening preferences without promoting them to phonetic truth."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from any2xsampa.training.phone_diagnostic_review import validate_answers


def summarize_answers(packet: dict, answers: dict) -> dict:
    responses = validate_answers(packet, answers)
    items = {item['review_id']: item for item in packet['items']}
    categories: dict[str, Counter] = defaultdict(Counter)
    tokens: dict[str, list[dict]] = defaultdict(list)
    rows = []
    for response in responses:
        item = items[response['review_id']]
        outcome = response['selected_source'] or response['decision']
        categories[item['category']][outcome] += 1
        row = {**response, 'category': item['category'], 'lyric': item['lyric'],
               'song_name': item['song_name'], 'choices': item['choices'],
               'choice_sources': item['choice_sources'], 'outcome': outcome,
               'interpretation_scope': 'comparative_preference_only',
               'reference_update_allowed': False}
        rows.append(row)
        for token in dict.fromkeys(item['raw_v4c_tokens']):
            tokens[token].append({'review_id': response['review_id'], 'outcome': outcome})
    return {
        'schema_version': 'phone_diagnostic_analysis_v1',
        'packet_sha256': packet['packet_sha256'], 'items': rows,
        'counts': dict(Counter(row['outcome'] for row in rows)),
        'counts_by_category': {key: dict(value) for key, value in categories.items()},
        'raw_token_observations': dict(tokens),
        'sample_selection': 'purposive_disagreement_sample_not_random_accuracy_sample',
        'formal_reference_created': False, 'global_mapping_update_allowed': False,
        'preference_accuracy_claim_allowed': False,
        'reference_counts_or_training_labels_changed': False,
    }


def contiguous_note_prediction(example: dict, note_id: str, system: str) -> dict:
    positions = [i for i, p in enumerate(example['source_phones']) if p['note_id'] == note_id]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError('note reference positions are missing or noncontiguous')
    selected = [e for e in example['edit_operations'][system]
                if e['reference_index'] in positions and e['prediction_index'] is not None]
    indices = [e['prediction_index'] for e in selected]
    # Preserve inserted predictions between the first/last aligned token.
    # This remains an edit-alignment hypothesis, not an acoustic word boundary.
    phones = example['predictions'][system][min(indices):max(indices)+1] if indices else []
    return {'reference': [example['reference'][i] for i in positions],
            'prediction': phones, 'reference_indices': positions,
            'predicted_range': [min(indices), max(indices)+1] if indices else None,
            'acoustic_boundaries_confirmed': False}


def frame_strata(np: Any, logits: Any, labels: list[str], radius: int = 2) -> dict:
    if radius != 2 or logits.ndim != 2 or logits.shape[1] != len(labels) or len(labels) != 428:
        raise ValueError('expected all 428 labels and fixed radius 2')
    if len(logits) == 0 or not np.isfinite(logits).all():
        raise ValueError('empty or nonfinite teacher frames')
    ids = logits.argmax(-1)
    masks = {}
    for name, selected in [('aspirated', [i for i, x in enumerate(labels) if '\u02b0' in x]),
                           ('glottal_h', [i for i, x in enumerate(labels) if x == 'h'])]:
        raw = np.isin(ids, selected)
        expanded = raw.copy()
        for offset in range(1, radius+1):
            expanded[offset:] |= raw[:-offset]
            expanded[:-offset] |= raw[offset:]
        masks[name] = expanded
    # Mutually exclusive masks: aspiration takes precedence at overlaps.
    masks['glottal_h'] &= ~masks['aspirated']
    masks['other'] = ~(masks['aspirated'] | masks['glottal_h'])
    return masks

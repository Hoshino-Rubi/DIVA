"""Whole-source sequence supervision; no inferred target timestamps."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Callable, Iterator
import math
import time

from any2xsampa.training.singing_domain_losses import required_frames, sequence_ctc


def plan_chunks(song_id: str, audio: dict, spec: dict) -> dict:
    rate, samples = audio['sample_rate'], audio['frame_count']
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (rate, samples)):
        raise ValueError('invalid whole-source sample geometry')
    core, context = spec['core_seconds'], spec['context_seconds']
    if core != 16. or context != 2. or spec['maximum_encoder_seconds'] != 20.:
        raise ValueError('unsupported frozen whole-sequence chunk geometry')
    duration = samples / rate
    model_samples = (samples * 16000 + rate - 1) // rate
    total_frames = (model_samples - 400) // 320 + 1
    if total_frames < 1:
        raise ValueError('whole source is too short')
    chunks = []
    next_frame = 0
    for index in range(math.ceil(duration / core)):
        left, right = index * core, min((index+1)*core, duration)
        first = round(max(0., left-context)*rate)
        last = round(min(duration, right+context)*rate)
        crop_start = first / rate
        model_start = crop_start*16000
        if abs(model_start/320 - round(model_start/320)) > 1e-7:
            raise ValueError('chunk origin not on the global encoder grid')
        model_count = ((last-first)*16000 + rate-1) // rate
        frames = (model_count-400)//320+1
        offset = round(model_start/320)
        global_end = min(total_frames, math.ceil((right*16000-199.5)/320))
        if global_end == next_frame and chunks and chunks[-1]['source_end_sample_exclusive'] == samples:
            chunks[-1]['core_end_sec'] = right
            continue
        # All global frontend frame centers are kept once, including repeats.
        lo, hi = next_frame-offset, global_end-offset
        if not 0 <= lo < hi <= frames:
            raise ValueError('invalid whole-song frame ownership')
        chunks.append({'example_id':f'whole-{song_id}-{index:03d}',
            'core_start_sec':left, 'core_end_sec':right,
            'source_start_sample':first, 'source_end_sample_exclusive':last,
            'start_sec':crop_start, 'end_sec':last/rate, 'expected_frames':frames,
            'keep_start_frame':lo, 'keep_end_frame':hi,
            'global_start_frame':next_frame, 'global_end_frame':global_end})
        next_frame = global_end
    if next_frame != total_frames:
        raise ValueError('whole-source global frame coverage differs')
    return {'source_duration_sec':duration, 'source_samples':samples,
        'model_samples':model_samples, 'total_frames':total_frames, 'chunks':chunks,
        'stride_samples':320, 'receptive_field_samples':400,
        'audio_coverage':'entire original file, including pauses and instrumental sections',
        'context_is_discarded_from_loss_once_owned_by_adjacent_core':True,
        'chunk_boundary_is_not_a_phone_boundary':True}


def validate_chunks(chunks: list[dict], total_frames: int) -> None:
    next_frame = 0
    for row in chunks:
        lo, hi, frames = row['keep_start_frame'], row['keep_end_frame'], row['expected_frames']
        if any(isinstance(x,bool) or not isinstance(x,int) for x in (lo,hi,frames,row['global_start_frame'],row['global_end_frame'])):
            raise ValueError('frame ownership must use integers')
        if not 0 <= lo < hi <= frames or row['global_start_frame'] != next_frame:
            raise ValueError('frame gap, overlap or invalid selection')
        next_frame += hi-lo
        if row['global_end_frame'] != next_frame:
            raise ValueError('frame ownership endpoint mismatch')
    if not chunks or next_frame != total_frames:
        raise ValueError('whole sequence has incomplete frame coverage')


def accepted_reference(song: dict, approval: dict, vocabulary: dict) -> dict:
    if song['split'] != 'train' or approval['song_id'] != song['song_id']:
        raise ValueError('whole-song training split/source mismatch')
    if (approval['status'] != 'sequence_confirmed'
            or approval['original_answer'].get('explicit_sequence_scope_confirmed') is not True
            or approval.get('phone_timing_confirmed') is not False
            or approval.get('mapping_extended') is not False):
        raise ValueError('whole source requires explicit sequence-only approval')
    for role in ('original_audio', 'human_vsqx'):
        if any(approval['source_bindings'][role][k] != song[role][k] for k in ('path','sha256')):
            raise ValueError('approved whole-source hash/path mismatch')
    ids, phones, identity = [], [], []
    for i, note in enumerate(song['notes'], 1):
        if note['note_index'] != i or note['note_id'] != f"{song['song_id']}-n{i:03d}":
            raise ValueError('whole-source notes are not continuous')
        if note['mapping_status'] != 'resolved_proxy' or note['unresolved_tokens'] or not note['model_target_ids']:
            raise ValueError('complete song contains unresolved mapping')
        if [vocabulary.get(p) for p in note['model_target_ipa']] != note['model_target_ids']:
            raise ValueError('IPA/vocabulary projection mismatch')
        ids.extend(note['model_target_ids']); phones.extend(note['model_target_ipa'])
        identity.extend({'note_id':note['note_id'], 'phone_index':j, 'phone':p}
                        for j,p in enumerate(note['model_target_ipa']))
    if not ids or any(isinstance(i,bool) or not isinstance(i,int) or i<=0 for i in ids):
        raise ValueError('invalid or blank whole-sequence labels')
    return {'song_id':song['song_id'], 'split':'train', 'audio_role':'original',
        'audio_path':song['original_audio']['path'], 'audio_sha256':song['original_audio']['sha256'],
        'source_vsqx':song['human_vsqx'], 'review_id':approval['review_id'],
        'target_ids':ids, 'target_ipa':phones, 'source_phones':identity,
        'note_count':len(song['notes']), 'minimum_ctc_frames':required_frames(ids),
        'reference_nature':'human_confirmed_whole_sequence_with_derived_engine_IPA',
        'phone_timing_confirmed':False, 'mapping_extended':False,
        'automatic_clock_paths_used':False, 'per_chunk_targets_created':False}


@contextmanager
def deterministic_checkpointed_tail(tail: Any) -> Iterator[None]:
    """Keep tail activation checkpointing, disable stochastic child layers only here."""
    modules = list(tail.modules())
    modes = [m.training for m in modules]
    try:
        tail.training = True
        for module in modules[1:]:
            module.training = False
        yield
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode


def backward_whole_sequence(torch: Any, tail: Any, rows: list[dict], target_ids: list[int],
                            forward: Callable[[dict], Any], *, coefficient: float,
                            deadline: float, device: Any) -> dict:
    """Exact chain rule via logit gradients; one live chunk activation graph at a time."""
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('whole-sequence coefficient must be positive finite')
    validate_chunks(rows, rows[-1]['global_end_frame'] if rows else 0)
    start = time.monotonic()
    pieces = []
    def budget() -> None:
        if time.monotonic() > deadline:
            raise TimeoutError('whole sequence exceeded the fixed training job budget')
    with deterministic_checkpointed_tail(tail):
        with torch.no_grad():
            for row in rows:
                budget()
                values = forward(row)
                if values.ndim != 3 or values.shape[:2] != (1,row['expected_frames']) or values.shape[-1] < 2:
                    raise ValueError('whole chunk output shape differs from its cache')
                part = values[:,row['keep_start_frame']:row['keep_end_frame']]
                if not torch.isfinite(part).all():
                    raise ValueError('non-finite whole chunk logits')
                pieces.append(part.detach().cpu().clone())
        full = torch.cat(pieces,1).to(device).requires_grad_(True)
        loss = sequence_ctc(torch, full, target_ids)
        if not torch.isfinite(loss):
            raise ValueError('non-finite whole CTC objective')
        grad, = torch.autograd.grad(coefficient*loss, full)
        gradient = grad.detach().cpu()
        if not torch.isfinite(gradient).all():
            raise ValueError('non-finite whole CTC logit gradient')
        value = float(loss.detach())
        del full, grad, loss
        maximum = 0.
        for row, expected in zip(rows, pieces):
            budget()
            raw = forward(row)
            part = raw[:,row['keep_start_frame']:row['keep_end_frame']]
            difference = float((part.detach().cpu()-expected).abs().max())
            maximum = max(maximum,difference)
            if difference > 2e-5:
                raise ValueError('whole CTC recomputation logits changed before optimizer step')
            left, right = row['global_start_frame'], row['global_end_frame']
            part.backward(gradient[:,left:right].to(device))
            del raw, part
    return {'ctc':value, 'coefficient':coefficient, 'frames':gradient.shape[1],
        'target_phones':len(target_ids), 'chunk_count':len(rows),
        'maximum_recompute_logit_difference':maximum,
        'elapsed_seconds':time.monotonic()-start, 'independent_chunk_ctc_used':False,
        'reference_used_only_in_global_loss':True, 'whole_branch_dropout':False}

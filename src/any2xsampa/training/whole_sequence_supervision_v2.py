"""Version 2: identical grad-enabled forward paths in both whole-CTC passes.

The first pass discards each activation graph after saving detached logits.
The loss, coefficients, frame ownership and numerical tolerance are unchanged.
"""
from __future__ import annotations
from typing import Any,Callable
import math,time
from any2xsampa.training.singing_domain_losses import sequence_ctc
from any2xsampa.training.whole_sequence_supervision import validate_chunks,deterministic_checkpointed_tail

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
        with torch.enable_grad():
            for row in rows:
                budget()
                values = forward(row)
                if values.ndim != 3 or values.shape[:2] != (1,row['expected_frames']) or values.shape[-1] < 2:
                    raise ValueError('whole chunk output shape differs from its cache')
                part = values[:,row['keep_start_frame']:row['keep_end_frame']]
                if not torch.isfinite(part).all():
                    raise ValueError('non-finite whole chunk logits')
                pieces.append(part.detach().cpu().clone())
                del values, part
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
                raise ValueError(f"whole CTC recomputation changed: example={row['example_id']} max_absolute_difference={difference:.12g}")
            left, right = row['global_start_frame'], row['global_end_frame']
            part.backward(gradient[:,left:right].to(device))
            del raw, part
    return {'ctc':value, 'coefficient':coefficient, 'frames':gradient.shape[1],
        'target_phones':len(target_ids), 'chunk_count':len(rows),
        'maximum_recompute_logit_difference':maximum,
        'elapsed_seconds':time.monotonic()-start, 'independent_chunk_ctc_used':False,
        'reference_used_only_in_global_loss':True, 'whole_branch_dropout':False}

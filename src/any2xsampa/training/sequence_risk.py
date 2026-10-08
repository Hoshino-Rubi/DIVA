"""Differentiable CTC candidate risk; references enter the loss, never decoding."""
from __future__ import annotations
from typing import Any, Sequence
import math

from any2xsampa.adapters.dual_ctc import prefix_nbest
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.training.singing_domain_losses import required_frames, sequence_ctc


def weighted_phone_distance(reference: Sequence[str], prediction: Sequence[str], spec: dict) -> float:
    extra = float(spec["extra_aspiration_error_cost"])
    ins = lambda p: float(spec["insertion_cost"]) + extra * ("ʰ" in p)
    delete = lambda p: float(spec["deletion_cost"]) + extra * ("ʰ" in p)
    previous = [0.0]
    for p in prediction:
        previous.append(previous[-1] + ins(p))
    for expected in reference:
        current = [previous[0] + delete(expected)]
        for j, actual in enumerate(prediction, 1):
            substitution = 0.0 if expected == actual else float(spec["substitution_cost"]) + extra * ("ʰ" in expected or "ʰ" in actual)
            current.append(min(previous[j] + delete(expected), current[j-1] + ins(actual), previous[j-1] + substitution))
        previous = current
    return previous[-1]


def candidate_pool(logits: Any, target_ids: Sequence[int], labels: Sequence[str], spec: dict) -> dict:
    """The acoustic beam is computed before its separate training-reference entry."""
    import numpy as np
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != len(labels) or not np.isfinite(values).all():
        raise ValueError("invalid candidate logits")
    origins: dict[tuple[int, ...], list[str]] = {}
    for item in prefix_nbest(values, beam_width=spec["beam_width"]):
        origins.setdefault(item.ids, []).append("audio_only_beam")
    origins.setdefault(tuple(ctc_collapse(values.argmax(-1).tolist())), []).append("audio_only_greedy")
    origins.setdefault(tuple(target_ids), []).append("training_reference_loss_only")
    reference = [labels[i] for i in target_ids]
    if not reference or any(i <= 0 or i >= len(labels) for i in target_ids):
        raise ValueError("invalid reference targets")
    rows = []
    for ids, source in sorted(origins.items()):
        if required_frames(ids) > len(values):
            raise ValueError("infeasible CTC loss candidate")
        distance = weighted_phone_distance(reference, [labels[i] for i in ids], spec)
        rows.append({"ids": list(ids), "cost": distance / len(reference), "origins": source})
    return {"candidates": rows, "target_ids": list(target_ids), "frame_count": len(values),
        "decode_received_reference": False, "reference_appended_only_for_training_loss": True,
        "risk_is_not_decoding_confidence": True}


def ctc_candidate_risk(torch: Any, logits: Any, pool: dict, *, temperature: float = 2.0,
                       valid_frames: int | None = None) -> tuple[Any, dict]:
    if logits.ndim != 3 or logits.shape[0] != 1 or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("risk requires one CTC sequence and positive finite temperature")
    frames = logits.shape[1] if valid_frames is None else valid_frames
    if isinstance(frames, bool) or not isinstance(frames, int) or not 0 < frames <= logits.shape[1]:
        raise ValueError("invalid risk frame mask")
    valid = logits[:, :frames]
    if not torch.isfinite(valid).all() or pool["frame_count"] != frames:
        raise ValueError("invalid/stale candidate frame count")
    rows = pool["candidates"]
    if not rows or len(rows) > 10 or len({tuple(r["ids"]) for r in rows}) != len(rows):
        raise ValueError("invalid candidate budget or duplicate sequence")
    for row in rows:
        if (any(isinstance(i, bool) or not isinstance(i, int) or not 0 < i < logits.shape[-1] for i in row["ids"])
                or required_frames(row["ids"]) > frames or not math.isfinite(row["cost"]) or row["cost"] < 0):
            raise ValueError("invalid candidate labels, repetition or risk")
    sequences = [row["ids"] for row in rows]
    log_probs = valid.log_softmax(-1).transpose(0, 1).expand(-1, len(rows), -1).contiguous()
    targets = torch.tensor([i for seq in sequences for i in seq], dtype=torch.long, device=logits.device)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=logits.device)
    input_lengths = torch.full((len(rows),), frames, dtype=torch.long, device=logits.device)
    nll = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, lengths,
        blank=0, reduction="none", zero_infinity=False)
    if not torch.isfinite(nll).all():
        raise ValueError("non-finite exact CTC candidate probabilities")
    posterior = (-nll / temperature).softmax(0)
    costs = logits.new_tensor([r["cost"] for r in rows])
    risk = (posterior * costs).sum()
    return risk, {"expected_weighted_phone_risk": float(risk.detach()), "candidate_count": len(rows),
        "minimum_candidate_cost": float(costs.min()), "maximum_candidate_cost": float(costs.max()),
        "exact_sequence_log_probabilities": (-nll.detach()).cpu().tolist()}


def whole_sequence_ctc(torch: Any, logits: Sequence[Any], selections: Sequence[dict], targets: Sequence[int]) -> Any:
    """Preserve a gap-free global emission sequence; no per-chunk target truncation."""
    if len(logits) != len(selections) or not logits:
        raise ValueError("missing whole-sequence chunks")
    pieces, next_frame = [], 0
    for values, row in zip(logits, selections):
        lo, hi = row["keep_start_frame"], row["keep_end_frame"]
        if (values.ndim != 3 or values.shape[0] != 1 or not 0 <= lo < hi <= values.shape[1]
                or row["global_start_frame"] != next_frame):
            raise ValueError("whole-sequence chunk gap, overlap or invalid selection")
        pieces.append(values[:, lo:hi])
        next_frame += hi-lo
    return sequence_ctc(torch, torch.cat(pieces, dim=1), targets)

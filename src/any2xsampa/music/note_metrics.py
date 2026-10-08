"""One-to-one note evaluation, with no timing shift or reference-driven repair."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class NoteEvent:
    start_sec: float
    end_sec: float
    pitch_midi: int

    def __post_init__(self) -> None:
        for x in (self.start_sec, self.end_sec):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError("Note times must be finite numbers")
        if self.start_sec < 0 or self.end_sec <= self.start_sec:
            raise ValueError("Note must have a nonnegative start and positive duration")
        if isinstance(self.pitch_midi, bool) or not isinstance(self.pitch_midi, int) or not 0 <= self.pitch_midi <= 127:
            raise ValueError("Note pitch must be integer MIDI 0..127")


def _maximum_matching(edges: list[list[int]], n_predictions: int) -> list[tuple[int, int]]:
    """Iterative augmenting paths avoid greedy undercount and recursion limits."""
    matched_prediction: dict[int, int] = {}
    matched_reference: dict[int, int] = {}
    for origin in range(len(edges)):
        queue = [origin]
        seen_ref = {origin}
        parent: dict[int, int] = {}
        target = None
        for ref in queue:
            for pred in edges[ref]:
                if pred in parent:
                    continue
                parent[pred] = ref
                if pred not in matched_prediction:
                    target = pred
                    break
                next_ref = matched_prediction[pred]
                if next_ref not in seen_ref:
                    seen_ref.add(next_ref)
                    queue.append(next_ref)
            if target is not None:
                break
        while target is not None:
            ref = parent[target]
            previous = matched_reference.get(ref)
            matched_prediction[target] = ref
            matched_reference[ref] = target
            target = previous
    return sorted(matched_reference.items())


def match_notes(reference: Sequence[NoteEvent], prediction: Sequence[NoteEvent], *,
                onset_tolerance: float = .05, offset_tolerance: float | None = .05,
                require_pitch: bool = True) -> list[tuple[int, int]]:
    for value in (onset_tolerance, offset_tolerance):
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid boundary tolerance")
    edges = []
    for ref in reference:
        candidates = [j for j, pred in enumerate(prediction)
                      if (not require_pitch or ref.pitch_midi == pred.pitch_midi)
                      and abs(ref.start_sec - pred.start_sec) <= onset_tolerance + 1e-9
                      and (offset_tolerance is None or abs(ref.end_sec - pred.end_sec) <= offset_tolerance + 1e-9)]
        candidates.sort(key=lambda j: (abs(reference[len(edges)].start_sec - prediction[j].start_sec), j))
        edges.append(candidates)
    return _maximum_matching(edges, len(prediction))


def _statistics(pairs: list[tuple[int, int]], n_ref: int, n_pred: int) -> dict:
    true = len(pairs)
    precision = true / n_pred if n_pred else 0.0
    recall = true / n_ref if n_ref else 0.0
    return {"matches": true, "false_positives": n_pred - true, "false_negatives": n_ref - true,
            "reference_notes": n_ref, "predicted_notes": n_pred,
            "precision": precision, "recall": recall,
            "f1": 2 * true / (n_ref + n_pred) if n_ref + n_pred else 0.0,
            "pairs": [{"reference_index": r, "prediction_index": p} for r, p in pairs]}


def _errors(values: list[float]) -> dict:
    ordered = sorted(abs(v) for v in values)
    if not ordered:
        return {"count": 0, "median_abs_sec": None, "p90_abs_sec": None, "maximum_abs_sec": None}
    def percentile(q: float) -> float:
        index = (len(ordered) - 1) * q
        low = math.floor(index)
        return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)
    return {"count": len(values), "median_abs_sec": percentile(.5), "p90_abs_sec": percentile(.9),
            "maximum_abs_sec": ordered[-1], "signed_errors_sec": values}


def evaluate_notes(reference: Sequence[NoteEvent], prediction: Sequence[NoteEvent], *,
                   reference_clock_confirmed: bool = False) -> dict:
    onset_pairs = match_notes(reference, prediction, offset_tolerance=None)
    full_pairs = match_notes(reference, prediction)
    boundary_pairs = match_notes(reference, prediction, require_pitch=False)
    full = _statistics(full_pairs, len(reference), len(prediction))
    # Errors on onset-matched notes include bad offsets; joint-matched-only
    # errors would conceal duration failures by selecting successes.
    onsets = [prediction[p].start_sec - reference[r].start_sec for r, p in onset_pairs]
    offsets = [prediction[p].end_sec - reference[r].end_sec for r, p in onset_pairs]
    durations = [offset - onset for onset, offset in zip(onsets, offsets, strict=True)]
    return {"schema_version": "strict_note_evaluation_v1", "pitch_rule": "integer_midi_exact",
            "onset_tolerance_sec": .05, "offset_tolerance_sec": .05,
            "automatic_shift_or_warp": False, "reference_clock_confirmed": reference_clock_confirmed,
            "evaluation_kind": "formal" if reference_clock_confirmed else "conditional_clock_diagnostic",
            "pitch_onset": _statistics(onset_pairs, len(reference), len(prediction)),
            "pitch_onset_offset": full,
            "boundaries_without_pitch": _statistics(boundary_pairs, len(reference), len(prediction)),
            "octave_errors_on_boundary_matches": sum(abs(reference[r].pitch_midi-prediction[p].pitch_midi) == 12
                                                       for r, p in boundary_pairs),
            "timing_on_pitch_onset_matches": {"onset": _errors(onsets), "offset": _errors(offsets),
                                                "duration": _errors(durations)},
            "quality_target_passed": bool(reference_clock_confirmed and reference and prediction
                                           and full["precision"] >= .95 and full["recall"] >= .95)}

"""Bounded whole-audio GAME note assembly. No lyrics, reference notes or phone input."""
from __future__ import annotations

import copy

from .whole_audio import plan_chunks


def note_windows(start: float, end: float, rate: int) -> list[dict]:
    first, last = round(start * rate), round(end * rate)
    windows = plan_chunks({"sample_rate": rate, "frames": last - first})
    for w in windows:
        for name in ("start_sec", "end_sec", "keep_start_sec", "keep_end_sec"):
            w[name] += first / rate
    return windows


def stitch_note_windows(predictions: list[dict], windows: list[dict]) -> list[dict]:
    """Own each time interval once; join only a note crossing an artificial seam.

    An actual repeated-pitch boundary is retained. Different seam predictions
    are kept as two notes, with the deterministic ownership cut recorded.
    """
    rows: list[dict] = []
    for prediction, window in zip(predictions, windows, strict=True):
        left, right = window["keep_start_sec"], window["keep_end_sec"]
        for raw in prediction["notes"]:
            begin, end = max(left, raw["start_sec"]), min(right, raw["end_sec"])
            if end <= begin:
                continue
            row = copy.deepcopy(raw)
            evidence = {"window_index": window["index"], "region_index": raw["region_index"],
                        "start_sec": raw["start_sec"], "end_sec": raw["end_sec"]}
            row.update(start_sec=begin, end_sec=end, duration_sec=end-begin,
                       source_regions=[evidence], clipped_at_window_seam=(
                           begin != raw["start_sec"] or end != raw["end_sec"]))
            previous = rows[-1] if rows else None
            crosses_left = raw["start_sec"] < left < raw["end_sec"]
            can_join = (previous is not None and abs(previous["end_sec"] - left) < 1e-8
                        and previous["source_regions"][-1]["window_index"] != window["index"]
                        and previous["source_regions"][-1]["start_sec"] < left
                        and previous["source_regions"][-1]["end_sec"] > left
                        and crosses_left and previous["pitch_midi"] == raw["pitch_midi"])
            if can_join:
                previous["end_sec"] = end
                previous["duration_sec"] = end - previous["start_sec"]
                previous["source_regions"].append(evidence)
                previous["joined_across_artificial_seam"] = True
                previous["touches_input_end"] = raw.get("touches_input_end", False)
            else:
                rows.append(row)
    for i, row in enumerate(rows):
        row["region_index"] = i
    return rows

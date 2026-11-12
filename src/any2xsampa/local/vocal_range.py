"""Explicit singer-register adaptation at export; acoustic evidence stays raw."""
from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any
from xml.etree import ElementTree as ET

from .common import read, sha

POLICY = Path(__file__).resolve().parents[1] / "config/vocal_ranges_v1.json"
PITCH_MODES = frozenset({"original", "miku_v4c_octave", "manual"})


def _distribution(pitches: list[int], low: int, high: int) -> dict[str, Any]:
    below = sum(pitch < low for pitch in pitches)
    above = sum(pitch > high for pitch in pitches)
    return {
        "note_count": len(pitches),
        "minimum_midi": min(pitches) if pitches else None,
        "maximum_midi": max(pitches) if pitches else None,
        "median_midi": median(pitches) if pitches else None,
        "below_range_note_count": below,
        "above_range_note_count": above,
        "in_range_note_count": len(pitches) - below - above,
    }


def _octave_shift(pitches: list[int], durations: list[int], profile: dict) -> int:
    low, high = profile["recommended_min_midi"], profile["recommended_max_midi"]
    limit = profile["maximum_transpose_semitones"]
    step = profile["octave_step_semitones"]
    candidates = [shift for shift in range(-limit, limit + 1, step)
                  if all(0 <= pitch + shift <= 127 for pitch in pitches)]

    def rank(shift: int) -> tuple[int, int, int]:
        cost = sum(duration * max(low - pitch - shift, pitch + shift - high, 0) ** 2
                   for pitch, duration in zip(pitches, durations, strict=True))
        return cost, abs(shift), shift

    return min(candidates, key=rank)


def apply_export_pitch(
    tree: ET.ElementTree, *, mode: str = "original",
    transpose_semitones: int | None = None,
) -> dict[str, Any]:
    """Change only note/n in this export tree, using one shift for every note."""
    if mode not in PITCH_MODES:
        raise ValueError("Unknown export pitch mode")
    profile = read(POLICY)["profiles"]["miku_v4c"]
    limit = profile["maximum_transpose_semitones"]
    if mode == "manual":
        if type(transpose_semitones) is not int or not -limit <= transpose_semitones <= limit:
            raise ValueError(f"Manual transpose must be an integer between {-limit} and {limit}")
    elif transpose_semitones is not None:
        raise ValueError("Explicit transpose is allowed only in manual pitch mode")
    root = tree.getroot()
    namespace = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    notes = root.findall(".//" + namespace + "note")
    nodes = [note.find(namespace + "n") for note in notes]
    if any(node is None for node in nodes):
        raise ValueError("VSQX note is missing its MIDI pitch")
    pitches = [int(node.text) for node in nodes]
    durations = [int(note.findtext(namespace + "dur", "0")) for note in notes]
    if any(not 0 <= pitch <= 127 for pitch in pitches) or any(value <= 0 for value in durations):
        raise ValueError("VSQX notes must have valid MIDI pitches and positive durations")
    shift = (transpose_semitones if mode == "manual" else
             _octave_shift(pitches, durations, profile) if mode == "miku_v4c_octave" else 0)
    adapted = [pitch + shift for pitch in pitches]
    if any(not 0 <= pitch <= 127 for pitch in adapted):
        raise ValueError("Transpose exceeds MIDI range; notes were not clipped")
    low, high = profile["recommended_min_midi"], profile["recommended_max_midi"]
    report = {
        "schema_version": "vsqx_pitch_adaptation_v1", "mode": mode,
        "profile": "miku_v4c", "policy_sha256": sha(POLICY),
        "recommended_min_midi": low, "recommended_max_midi": high,
        "transpose_semitones": shift, "note_count": len(pitches),
        "before": _distribution(pitches, low, high),
        "after": _distribution(adapted, low, high),
        "pitch_intervals_preserved": True,
        "individual_notes_clamped_or_octave_folded": False,
        "raw_model_evidence_modified": False, "accuracy_improvement_claimed": False,
    }
    if shift:
        for node, pitch in zip(nodes, adapted, strict=True):
            node.text = str(pitch)
    return report


__all__ = ["apply_export_pitch"]

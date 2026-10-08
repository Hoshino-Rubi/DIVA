from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from any2xsampa.music.pitch_note_analysis import midi_to_note_name


@dataclass(frozen=True, slots=True)
class PitchRenderPolicy:
    """Policy for separating extracted pitch evidence from VSQX render pitch.

    The extracted melody is evidence.  The render melody is what we place in a
    Miku V4C draft/probe VSQX so the candidate remains audible enough for human
    listening.  This intentionally records the semitone shift instead of
    pretending the analyzer produced the shifted value.
    """

    enabled: bool = True
    floor_midi: int = 64
    strategy: str = "phrase_minimum_floor_transpose"


@dataclass(frozen=True, slots=True)
class PitchRenderDecision:
    extracted_midi: int | None
    extracted_note_name: str | None
    render_midi: int
    render_note_name: str
    transpose_semitones: int
    safety_applied: bool
    source: str

    def to_provenance(self) -> dict[str, Any]:
        return {
            "extracted_midi": self.extracted_midi,
            "extracted_note_name": self.extracted_note_name,
            "render_midi": self.render_midi,
            "render_note_name": self.render_note_name,
            "pitch_transpose_semitones": self.transpose_semitones,
            "pitch_safety_applied": self.safety_applied,
            "pitch_source": self.source,
        }


def build_pitch_render_policy_summary(policy: PitchRenderPolicy | None) -> dict[str, Any]:
    if policy is None or not policy.enabled:
        return {
            "enabled": False,
            "floor_midi": None,
            "floor_note_name": None,
            "strategy": "disabled_preserve_extracted_or_fallback_pitch",
            "reason": "render_pitch_safety_disabled",
        }
    return {
        "enabled": True,
        "floor_midi": policy.floor_midi,
        "floor_note_name": midi_to_note_name(policy.floor_midi),
        "strategy": policy.strategy,
        "reason": "miku_v4c_probe_render_audibility_floor",
    }


def build_pitch_render_decisions(
    *,
    note_candidates: Sequence[Mapping[str, Any] | None],
    fallback_pitches: Sequence[int],
    note_count: int,
    policy: PitchRenderPolicy | None = None,
) -> list[PitchRenderDecision]:
    """Return per-note render decisions while retaining extracted evidence."""

    if note_count <= 0:
        return []
    if not fallback_pitches:
        raise ValueError("fallback_pitches must contain at least one MIDI pitch")

    extracted_values: list[int | None] = []
    sources: list[str] = []
    for index in range(note_count):
        candidate = note_candidates[index] if index < len(note_candidates) else None
        extracted = _note_midi(candidate)
        extracted_values.append(extracted)
        if isinstance(candidate, Mapping):
            sources.append(str(candidate.get("source") or "pitch_note_analysis"))
        else:
            sources.append("fallback_contour")

    base_values = [
        extracted if extracted is not None else int(fallback_pitches[index % len(fallback_pitches)])
        for index, extracted in enumerate(extracted_values)
    ]
    transpose = _policy_transpose(base_values, policy)
    safety_applied = bool(policy and policy.enabled and transpose != 0)

    decisions: list[PitchRenderDecision] = []
    for index, base_midi in enumerate(base_values):
        render_midi = int(base_midi + transpose)
        extracted = extracted_values[index]
        decisions.append(
            PitchRenderDecision(
                extracted_midi=extracted,
                extracted_note_name=midi_to_note_name(extracted) if extracted is not None else None,
                render_midi=render_midi,
                render_note_name=midi_to_note_name(render_midi),
                transpose_semitones=transpose,
                safety_applied=safety_applied,
                source=sources[index],
            )
        )
    return decisions


def _policy_transpose(base_values: Sequence[int], policy: PitchRenderPolicy | None) -> int:
    if policy is None or not policy.enabled or not base_values:
        return 0
    if policy.strategy != "phrase_minimum_floor_transpose":
        raise ValueError(f"unsupported pitch render strategy: {policy.strategy!r}")
    floor = int(policy.floor_midi)
    minimum = min(int(value) for value in base_values)
    if minimum >= floor:
        return 0
    return floor - minimum


def _note_midi(note_candidate: Mapping[str, Any] | None) -> int | None:
    if note_candidate is None:
        return None
    for key in ("midi", "pitch", "pitch_midi", "note_midi"):
        if key not in note_candidate:
            continue
        try:
            value = int(note_candidate[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(float(value)):
            return value
    return None

"""Explicit song clocks and conservative crop membership, without phone decoding."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

from any2xsampa.training.limited_data_artifacts import finite_number, sha256_object


def rational(value: Any, name: str) -> Fraction:
    return Fraction(str(finite_number(value, name)))


@dataclass(frozen=True)
class ClockOffset:
    """A named unit-rate transform: target seconds = source seconds + offset."""

    source: str
    target: str
    offset: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not isinstance(self.target, str) or not self.source or not self.target:
            raise ValueError("clock names are required")
        if self.source == self.target or not isinstance(self.offset, Fraction):
            raise ValueError("clock transform needs distinct names and an exact rational offset")

    def at(self, seconds: float) -> Fraction:
        return rational(seconds, "clock time") + self.offset

    def inverse(self) -> ClockOffset:
        return ClockOffset(self.target, self.source, -self.offset)

    def then(self, next_clock: ClockOffset) -> ClockOffset:
        if self.target != next_clock.source:
            raise ValueError("clock chain has mismatched domains")
        return ClockOffset(self.source, next_clock.target, self.offset + next_clock.offset)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "scale": 1,
                "offset_numerator": self.offset.numerator, "offset_denominator": self.offset.denominator,
                "offset_sec": float(self.offset)}


def exact_sample(seconds: float, sample_rate: int, *, side: str) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample rate must be a positive integer")
    value = rational(seconds, "sample time") * sample_rate
    if value < 0 or side not in {"floor", "ceil"}:
        raise ValueError("invalid sample time or rounding policy")
    return value.numerator // value.denominator if side == "floor" else -((-value.numerator) // value.denominator)


def score_clocks(path: Path) -> dict[str, Any]:
    """Read only the constant-tempo, single-signature VSQ4 clock used by Shalong."""
    root = ET.parse(path).getroot()
    namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
    if namespace != "http://www.yamaha.co.jp/vocaloid/schema/vsq4/":
        raise ValueError("this clock rule requires VSQ4; other clock schemas need a separate version")
    ns = {"v": namespace}
    master = root.find("v:masterTrack", ns)
    if master is None:
        raise ValueError("VSQX master clock is missing")
    resolution = int(master.findtext("v:resolution", namespaces=ns) or "0")
    premeasure = int(master.findtext("v:preMeasure", namespaces=ns) or "-1")
    tempos, signatures = master.findall("v:tempo", ns), master.findall("v:timeSig", ns)
    if resolution <= 0 or premeasure < 0 or len(tempos) != 1 or len(signatures) != 1:
        raise ValueError("unsupported or missing constant-tempo score clock")
    tempo, signature = tempos[0], signatures[0]
    if int(tempo.findtext("v:t", namespaces=ns) or "-1") != 0 or int(signature.findtext("v:m", namespaces=ns) or "-1") != 0:
        raise ValueError("tempo and signature must start at zero")
    bpm_x100 = int(tempo.findtext("v:v", namespaces=ns) or "0")
    numerator = int(signature.findtext("v:nu", namespaces=ns) or "0")
    denominator = int(signature.findtext("v:de", namespaces=ns) or "0")
    if min(bpm_x100, numerator, denominator) <= 0:
        raise ValueError("non-positive score tempo or signature")
    tick_sec = Fraction(6000, resolution * bpm_x100)
    premeasure_ticks = Fraction(premeasure * numerator * 4 * resolution, denominator)
    premeasure_sec = premeasure_ticks * tick_sec
    notes = []
    for part in root.findall(".//v:vsPart", ns):
        part_tick = int(part.findtext("v:t", namespaces=ns) or "0")
        for note in part.findall("v:note", ns):
            start = part_tick + int(note.findtext("v:t", namespaces=ns) or "0")
            duration = int(note.findtext("v:dur", namespaces=ns) or "0")
            if start < 0 or duration <= 0:
                raise ValueError("invalid score note interval")
            end = start + duration
            notes.append({"start_tick": start, "end_tick": end,
                          "absolute_start_sec": float(start * tick_sec), "absolute_end_sec": float(end * tick_sec),
                          "display_start_sec": float(start * tick_sec - premeasure_sec),
                          "display_end_sec": float(end * tick_sec - premeasure_sec)})
    notes.sort(key=lambda row: (row["start_tick"], row["end_tick"]))
    if not notes:
        raise ValueError("empty VSQX notes")
    return {"schema_version": "score_clock_vsq4_constant_v1", "resolution": resolution,
            "bpm_x100": bpm_x100, "premeasure_count": premeasure,
            "tick_seconds": {"numerator": tick_sec.numerator, "denominator": tick_sec.denominator},
            "premeasure_seconds": float(premeasure_sec), "premeasure_ticks": float(premeasure_ticks),
            "absolute_to_display": ClockOffset("score_absolute", "score_display", -premeasure_sec).to_dict(),
            "notes": notes, "render_audio_origin_inferred_from_metadata": False}


def bind_score_notes(score: Mapping[str, Any], notes: Sequence[Mapping[str, Any]]) -> None:
    if len(score["notes"]) != len(notes):
        raise ValueError("score/reference note count mismatch")
    for exact, row in zip(score["notes"], notes):
        for side in ("start", "end"):
            if abs(exact[f"absolute_{side}_sec"] - finite_number(row[f"{side}_sec"], "reference time")) > 1e-6:
                raise ValueError("reference time is not the VSQX absolute clock")


def boundary_audit(notes: Sequence[Mapping[str, Any]], *, start: float, end: float, offset_low: float, offset_high: float, uncertainty: float = 0.0) -> dict[str, Any]:
    start = finite_number(start, "crop start", minimum=0)
    end = finite_number(end, "crop end", minimum=0)
    low = finite_number(offset_low, "offset low")
    high = finite_number(offset_high, "offset high")
    pad = finite_number(uncertainty, "edge uncertainty", minimum=0)
    if start >= end or low > high:
        raise ValueError("invalid crop or clock support interval")
    rows = []
    for note in notes:
        a = finite_number(note["start_sec"], "note start", minimum=0)
        b = finite_number(note["end_sec"], "note end", minimum=0)
        if a >= b:
            raise ValueError("non-positive note interval")
        if a < end + high + pad and b > start + low - pad:
            rows.append(note)
    full = [r["note_id"] for r in rows if r["start_sec"] >= start + high + pad and r["end_sec"] <= end + low - pad]
    edges = [r["note_id"] for r in rows if r["note_id"] not in set(full)]
    unresolved = sorted({token for r in rows for token in r["unresolved_tokens"]})
    mapped = bool(rows) and all(r["mapping_status"] == "resolved_proxy" for r in rows)
    reasons = []
    if not rows:
        reasons.append("no_complete_reference_sequence")
    if edges:
        reasons.append("crop_may_cut_or_include_partial_edge_notes")
    if rows and not mapped:
        reasons.append("whole_window_contains_unresolved_mapping")
    return {"start_sec": start, "end_sec": end, "offset_support_sec": [low, high],
            "edge_uncertainty_sec": pad, "all_potential_note_ids": [r["note_id"] for r in rows],
            "fully_contained_note_ids": full, "partial_or_uncertain_edge_note_ids": edges,
            "unresolved_tokens": unresolved, "mapping_complete_for_all_potential_notes": mapped,
            "boundary_complete": bool(rows) and not edges, "exclusion_reasons": reasons,
            "partial_notes_removed_to_improve_score": False,
            "human_clip_approval_inferred": False}


def clock_binding_record(*, transforms: Sequence[ClockOffset], source_bindings: Mapping[str, Any], order_evidence: Mapping[str, Any], diagnostics: Mapping[str, Any], clips: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [c.source for c in transforms] != ["original", "rendered"] or [c.target for c in transforms] != ["rendered", "score_absolute"]:
        raise ValueError("expected the original -> rendered -> score_absolute clock chain")
    record = {"schema_version": "original_reference_clock_rule_v1", "song_id": "shalong", "split": "dev",
              "source_bindings": dict(source_bindings), "order_evidence": dict(order_evidence),
              "transforms": [c.to_dict() for c in transforms], "composed": transforms[0].then(transforms[1]).to_dict(),
              "clock_validation": dict(diagnostics), "clips": list(clips),
              "versioned_rule_is_accuracy_acceptance": False, "human_reviewed_clock": False,
              "official_clip_bindings_created": 0, "training_or_test_access_allowed": False}
    record["rule_hash"] = sha256_object(record)
    return record


def safe_expanding_crop(notes: Sequence[Mapping[str, Any]], *, start: float, end: float, offset_low: float, offset_high: float, uncertainty: float, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Suggest rest-midpoint boundaries without removing any old crop content."""
    start, end = finite_number(start, "crop start", minimum=0), finite_number(end, "crop end", minimum=0)
    low, high = finite_number(offset_low, "offset low"), finite_number(offset_high, "offset high")
    pad = finite_number(uncertainty, "edge uncertainty", minimum=0)
    if start >= end or low > high:
        raise ValueError("invalid crop proposal interval")
    expansion = finite_number(policy["maximum_endpoint_expansion_sec"], "expansion budget", minimum=0)
    max_duration = finite_number(policy["maximum_duration_sec"], "duration budget", minimum=0)
    rests = []
    for a, b in zip(notes, notes[1:]):
        safe_low = a["end_sec"] + pad - low
        safe_high = b["start_sec"] - pad - high
        if safe_high > safe_low:
            rests.append({"original_midpoint_sec": (safe_low + safe_high) / 2,
                          "safe_original_interval_sec": [safe_low, safe_high],
                          "neighbor_note_ids": [a["note_id"], b["note_id"]]})
    before = [r for r in rests if max(0, start - expansion) <= r["original_midpoint_sec"] <= start]
    after = [r for r in rests if end <= r["original_midpoint_sec"] <= end + expansion]
    if not before or not after:
        return {"status": "no_safe_expanding_rest_pair", "start_boundary_found": bool(before),
                "end_boundary_found": bool(after), "old_audio_content_discarded": False,
                "inference_started": False, "official_evaluation_eligible": False}
    left = max(before, key=lambda r: r["original_midpoint_sec"])
    right = min(after, key=lambda r: r["original_midpoint_sec"])
    a, b = left["original_midpoint_sec"], right["original_midpoint_sec"]
    audit = boundary_audit(notes, start=a, end=b, offset_low=low, offset_high=high, uncertainty=pad)
    reasons = list(audit["exclusion_reasons"])
    if b - a > max_duration:
        reasons.append("expanded_crop_exceeds_locked_runtime_duration")
    return {"status": "candidate_only" if not reasons else "not_eligible", "start_sec": a, "end_sec": b,
            "duration_sec": b - a, "start_rest": left, "end_rest": right, "boundary_audit": audit,
            "exclusion_reasons": reasons, "old_audio_content_discarded": False,
            "inference_started": False, "official_evaluation_eligible": False}


def membership_probe_offsets(notes: Sequence[Mapping[str, Any]], *, start: float, end: float, low: float, high: float) -> list[float]:
    """All overlap-membership regimes, including their exact boundary points."""
    start_q, end_q = rational(start, "crop start"), rational(end, "crop end")
    low_q, high_q = rational(low, "offset low"), rational(high, "offset high")
    if start_q >= end_q or low_q > high_q:
        raise ValueError("invalid membership sensitivity interval")
    events = {low_q, high_q}
    for note in notes:
        for value in (rational(note["start_sec"], "note start") - end_q, rational(note["end_sec"], "note end") - start_q):
            if low_q < value < high_q:
                events.add(value)
    ordered = sorted(events)
    points = events | {(a + b) / 2 for a, b in zip(ordered, ordered[1:])}
    return [float(x) for x in sorted(points)]

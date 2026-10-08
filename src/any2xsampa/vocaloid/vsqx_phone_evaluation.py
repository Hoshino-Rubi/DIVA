"""Reference VSQX phone identity scoring without re-decoding audio."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .vsqx_alignment import parse_vsqx_timeline


def _phone_candidates(segment: dict[str, Any]) -> list[str]:
    candidates = segment.get("candidate_phones", [])
    values = [str(item.get("symbol")) for item in candidates if isinstance(item, dict) and item.get("symbol") is not None]
    winner = segment.get("normalized_ipa")
    if winner is not None and str(winner) not in values:
        values.insert(0, str(winner))
    return values


def _tokens(note: dict[str, Any], known: set[str]) -> list[str]:
    values = [str(item) for item in note.get("phonemes", [])]
    return [token for token in values if token in known]


def _edit_distance(expected: list[str], actual: list[str]) -> int:
    previous = list(range(len(actual) + 1))
    for row, expected_token in enumerate(expected, start=1):
        current = [row]
        for column, actual_token in enumerate(actual, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected_token != actual_token),
            ))
        previous = current
    return previous[-1]


def build_vsqx_phone_identity_evaluation(phone_segments_path: str | Path, vsqx_path: str | Path, *, profile: str = "strict") -> dict[str, Any]:
    phone_payload = json.loads(Path(phone_segments_path).read_text(encoding="utf-8-sig"))
    timeline = parse_vsqx_timeline(vsqx_path)
    segments = [item for item in phone_payload.get("segments", []) if isinstance(item, dict)]
    known = {"sil", "sp", "a", "aI", "e", "ei", "i", "o", "u", "u@N", "m", "n", "N", "k", "t", "p", "x", "f", "l", "s", "s\\", "s`", "@", "@N", "@`"}
    notes: list[dict[str, Any]] = []
    top1_hits = top3_hits = 0
    unknown_counts: Counter[str] = Counter()
    configured_tokens = mapped_tokens = 0
    substitutions: Counter[str] = Counter()
    for index, note in enumerate(timeline.get("notes", [])):
        reference = list(note.get("phonemes", []))
        configured = [token for token in reference if token in known]
        unknown = [token for token in reference if token not in known]
        unknown_counts.update(unknown)
        configured_tokens += len(reference)
        mapped_tokens += len(configured)
        winner_sequence: list[str] = []
        top3_sequence: list[list[str]] = []
        start, end = float(note["start_sec"]), float(note["end_sec"])
        for segment in segments:
            ss, ee = float(segment.get("start_sec", 0.0)), float(segment.get("end_sec", 0.0))
            if max(0.0, min(ee, end) - max(ss, start)) <= 0:
                continue
            candidates = _phone_candidates(segment)
            if candidates:
                winner_sequence.append(candidates[0])
                top3_sequence.append(candidates[:3])
        expected = configured
        if expected:
            top1_hits += sum(token in winner_sequence for token in expected)
            top3_hits += sum(any(token in candidates for candidates in top3_sequence) for token in expected)
        winner_text = " ".join(winner_sequence)
        exact = (winner_sequence == expected) if expected and not unknown else None
        if expected and winner_sequence:
            for reference_token, winner in zip(expected, winner_sequence):
                if reference_token != winner:
                    substitutions[f"{reference_token}->{winner}"] += 1
        notes.append({
            "index": index,
            "lyric": note.get("lyric", ""),
            "reference_tokens": reference,
            "configured_reference_tokens": configured,
            "unknown_reference_tokens": unknown,
            "predicted_winner_sequence": winner_sequence,
            "predicted_top3_sequence": top3_sequence,
            "fully_mapped_reference": len(configured) == len(reference),
            "exact_winner_sequence_match": exact,
        })
    total = sum(len(item["configured_reference_tokens"]) for item in notes)
    sequence_errors = sum(
        _edit_distance(item["configured_reference_tokens"], item["predicted_winner_sequence"])
        for item in notes
        if not item["unknown_reference_tokens"]
    )
    summary = {
        "note_count": len(notes),
        "micro_top1_reference_phone_presence_ratio": round(top1_hits / total, 6) if total else 0.0,
        "micro_top3_reference_phone_presence_ratio": round(top3_hits / total, 6) if total else 0.0,
        "sequence_token_error_rate": round(sequence_errors / total, 6) if total else 0.0,
        "exact_winner_sequence_note_ratio": round(sum(item["exact_winner_sequence_match"] is True for item in notes) / len(notes), 6) if notes else 0.0,
        "top_substitutions": dict(substitutions),
        "unknown_reference_token_counts": dict(unknown_counts),
        "configured_reference_token_ratio": round(mapped_tokens / configured_tokens, 6) if configured_tokens else 0.0,
    }
    return {
        "schema_version": "vsqx_phone_identity_evaluation_v1",
        "profile": profile,
        "summary": summary,
        "notes": notes,
        "recognizer_redecoded": False,
        "vsqx_used_as_recognizer_input": False,
        "lyrics_used_as_recognizer_input": False,
    }


__all__ = ["build_vsqx_phone_identity_evaluation"]

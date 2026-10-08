"""Replace pathological MFA phone ratios with explicit review-only priors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from any2xsampa.adapters.forced_aligner import (
    MACHINE_REVIEW_DRAFT_STATUS,
    PENDING_HUMAN_PHONE_REVIEW_STATUS,
)
from any2xsampa.lyrics.timeline import (
    LyricsTimeline,
    canonical_timeline_sha256,
    load_lyrics_timeline,
    sha256_file,
)

PRIORS_RESOURCE = "cantonese_singing_phone_review_priors.json"
PRIORS_SCHEMA_VERSION = "cantonese_singing_phone_review_priors_v1"
TRANSFORM_NAME = "cantonese_singing_syllable_role_prior_v1"
_SHA256_LENGTH = 64


class RebalanceError(ValueError):
    """A review draft cannot be safely rebalanced."""


def _load_priors() -> tuple[dict[str, Any], str]:
    with resources.files("any2xsampa.config").joinpath(PRIORS_RESOURCE).open(
        "rb"
    ) as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PRIORS_SCHEMA_VERSION:
        raise RebalanceError("singing phone review priors have an invalid schema")
    return payload, hashlib.sha256(raw).hexdigest()


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RebalanceError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RebalanceError(f"{label} must be finite")
    return parsed


def _bounded_duration(duration: float, config: Mapping[str, Any], prefix: str) -> float:
    fraction = _finite(config.get(f"{prefix}_fraction"), label=f"{prefix}_fraction")
    minimum = _finite(config.get(f"{prefix}_min_sec"), label=f"{prefix}_min_sec")
    maximum = _finite(config.get(f"{prefix}_max_sec"), label=f"{prefix}_max_sec")
    if not 0.0 < fraction < 1.0 or not 0.0 < minimum <= maximum:
        raise RebalanceError(f"invalid {prefix} duration prior")
    return min(max(duration * fraction, minimum), maximum)


def _phone_durations(count: int, duration: float, priors: Mapping[str, Any]) -> list[float]:
    minimum = _finite(priors.get("min_phone_duration_sec"), label="min_phone_duration_sec")
    if duration < count * minimum:
        raise RebalanceError("lyric item is too short for its phone inventory")
    if count == 1:
        return [duration]
    key = "two_phone" if count == 2 else "three_phone"
    config = priors.get(key)
    if not isinstance(config, Mapping) or count not in {2, 3}:
        raise RebalanceError(f"unsupported phone count per Cantonese syllable: {count}")
    nucleus_minimum = _finite(config.get("nucleus_min_sec"), label="nucleus_min_sec")
    onset = _bounded_duration(duration, config, "onset")
    if count == 2:
        onset = min(onset, duration - max(nucleus_minimum, minimum))
        return [onset, duration - onset]
    coda = _bounded_duration(duration, config, "coda")
    if duration - onset - coda < nucleus_minimum:
        edge_budget = duration - max(nucleus_minimum, minimum)
        edge_total = onset + coda
        onset = edge_budget * onset / edge_total
        coda = edge_budget - onset
    return [onset, duration - onset - coda, coda]


def _group_segments(raw: object) -> list[tuple[int, list[dict[str, Any]]]]:
    if not isinstance(raw, list) or not raw:
        raise RebalanceError("review draft has no phone segments")
    groups: list[tuple[int, list[dict[str, Any]]]] = []
    for index, segment in enumerate(raw):
        if not isinstance(segment, Mapping):
            raise RebalanceError(f"phone segment {index} must be an object")
        metadata = segment.get("metadata")
        if not isinstance(metadata, Mapping):
            raise RebalanceError(f"phone segment {index} has no metadata")
        lyric_index = metadata.get("lyric_item_index")
        if isinstance(lyric_index, bool) or not isinstance(lyric_index, int):
            raise RebalanceError(f"phone segment {index} has no lyric item index")
        copied = dict(segment)
        if groups and groups[-1][0] == lyric_index:
            groups[-1][1].append(copied)
        elif any(existing_index == lyric_index for existing_index, _ in groups):
            raise RebalanceError("phone segments for a lyric item must be contiguous")
        else:
            groups.append((lyric_index, [copied]))
    return groups


def rebalance_singing_phone_review_draft(
    payload: Mapping[str, Any],
    *,
    timeline: LyricsTimeline,
    source_alignment_sha256: str,
) -> dict[str, Any]:
    """Return a pending review draft with bounded syllable-role suggestions."""

    if payload.get("schema_version") != "forced_phone_alignment_v1":
        raise RebalanceError("input must be forced_phone_alignment_v1")
    if payload.get("model_output") is not True or payload.get("product_accepted") is not False:
        raise RebalanceError("input must be an unaccepted machine review draft")
    if str(payload.get("lyrics_timeline_sha256") or "").lower() != canonical_timeline_sha256(
        timeline
    ):
        raise RebalanceError("review draft is bound to a different lyrics timeline")
    source_hash = str(source_alignment_sha256 or "").lower()
    if len(source_hash) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in source_hash):
        raise RebalanceError("source alignment SHA-256 is invalid")

    priors, priors_hash = _load_priors()
    timeline_items = {item.index: item for item in timeline.items}
    transformed: list[dict[str, Any]] = []
    for lyric_index, group in _group_segments(payload.get("segments")):
        item = timeline_items.get(lyric_index)
        if item is None:
            raise RebalanceError(f"lyrics timeline has no item {lyric_index}")
        item_duration = item.end_sec - item.start_sec
        durations = _phone_durations(len(group), item_duration, priors)
        cursor = item.start_sec
        for phone_index, (segment, phone_duration) in enumerate(zip(group, durations, strict=True)):
            previous_start = _finite(segment.get("start_sec"), label="prior phone start")
            previous_end = _finite(segment.get("end_sec"), label="prior phone end")
            end = item.end_sec if phone_index == len(group) - 1 else cursor + phone_duration
            metadata = dict(segment.get("metadata"))
            metadata.update(
                {
                    "prior_review_suggestion_start_sec": previous_start,
                    "prior_review_suggestion_end_sec": previous_end,
                    "lyric_item_index": lyric_index,
                    "direct_overlap_sec": end - cursor,
                    "phone_overlap_ratio": 1.0,
                    "boundary_transform_applied": True,
                    "boundary_transform": TRANSFORM_NAME,
                    "machine_suggestion": True,
                    "review_status": MACHINE_REVIEW_DRAFT_STATUS,
                }
            )
            updated = dict(segment)
            updated.update({"start_sec": cursor, "end_sec": end, "metadata": metadata})
            transformed.append(updated)
            cursor = end

    provenance = dict(payload.get("provenance")) if isinstance(payload.get("provenance"), Mapping) else {}
    provenance.update(
        {
            "status": MACHINE_REVIEW_DRAFT_STATUS,
            "product_accepted": False,
            "machine_suggestion": True,
            "human_review_required": True,
            "previous_review_draft_sha256": source_hash,
            "boundary_transform_applied": True,
            "boundary_transform": TRANSFORM_NAME,
            "boundary_transform_acoustic_truth": False,
            "review_priors_schema": PRIORS_SCHEMA_VERSION,
            "review_priors_sha256": priors_hash,
            "phone_boundaries_source": "review_only_cantonese_singing_syllable_role_priors",
        }
    )
    result = dict(payload)
    result.update(
        {
            "alignment_source": f"{payload.get('alignment_source', 'machine')}_{TRANSFORM_NAME}",
            "segments": transformed,
            "status": MACHINE_REVIEW_DRAFT_STATUS,
            "product_accepted": False,
            "machine_suggestion": True,
            "human_review_required": True,
            "boundary_transform_applied": True,
            "provenance": provenance,
        }
    )
    review = dict(payload.get("review")) if isinstance(payload.get("review"), Mapping) else {}
    review.update({"status": PENDING_HUMAN_PHONE_REVIEW_STATUS, "machine_suggestion": True})
    result["review"] = review
    return result


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RebalanceError(f"refusing to overwrite existing output: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--lyrics-timeline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.alignment.read_text(encoding="utf-8-sig"))
        if not isinstance(source, Mapping):
            raise RebalanceError("alignment JSON must be an object")
        result = rebalance_singing_phone_review_draft(
            source,
            timeline=load_lyrics_timeline(args.lyrics_timeline),
            source_alignment_sha256=sha256_file(args.alignment),
        )
        report = _write_exclusive(args.output.expanduser().resolve(), result)
    except (OSError, UnicodeError, json.JSONDecodeError, RebalanceError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

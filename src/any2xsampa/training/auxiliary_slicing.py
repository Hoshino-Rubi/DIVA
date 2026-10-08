"""Conservative word-gap proposals: auxiliary evidence, never phone truth."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import unicodedata

from any2xsampa.training.limited_data_artifacts import finite_number, require_digest
from any2xsampa.training.limited_data_reference import SONG_SPLIT


CONTEXT_KEYS = {"context_id", "example_id", "song_id", "split", "audio_path", "audio_sha256",
                "start_sec", "end_sec", "old_start_sec", "old_end_sec", "contains_labels_or_score_context"}


def validate_context(row: Mapping[str, Any]) -> None:
    if set(row) != CONTEXT_KEYS or row["contains_labels_or_score_context"] is not False:
        raise ValueError("auxiliary acoustic input cannot contain labels or score context")
    if SONG_SPLIT.get(row["song_id"]) != row["split"] or row["split"] not in {"train", "dev"}:
        raise ValueError("sealed or mismatched song split")
    require_digest(row["audio_sha256"], "audio_sha256")
    start, end, old_start, old_end = [finite_number(row[k], k, minimum=0) for k in ("start_sec", "end_sec", "old_start_sec", "old_end_sec")]
    if not start <= old_start < old_end <= end or not 0 < end - start <= 30.000001:
        raise ValueError("context must contain its old clip and remain within thirty seconds")
    if abs(old_end - old_start - 18) > 1e-6:
        raise ValueError("context must bind an original frozen eighteen-second clip")
    if not row["context_id"] == row["example_id"] + "-context":
        raise ValueError("unexpected context identity")


def normalize_word(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("word text must be a string")
    return "".join(c for c in unicodedata.normalize("NFC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def audit_timestamps(items: Sequence[Mapping[str, Any]], duration: float) -> dict[str, Any]:
    duration = finite_number(duration, "audio duration", minimum=0)
    rows = []
    for index, raw in enumerate(items):
        reasons = []
        text = raw.get("text")
        word = normalize_word(text) if isinstance(text, str) else ""
        if not word:
            reasons.append("empty_word")
        start, end = raw.get("start_sec"), raw.get("end_sec")
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in (start, end)):
            reasons.append("non_finite_or_non_numeric_timestamp")
        else:
            if end <= start:
                reasons.append("non_positive_duration")
            if start < 0 or end > duration + 1e-6:
                reasons.append("timestamp_outside_audio")
        bad_number = "non_finite_or_non_numeric_timestamp" in reasons
        rows.append({"index": index, "text": text if isinstance(text, str) else None, "normalized_word": word,
                     "start_sec": None if bad_number else start, "end_sec": None if bad_number else end,
                     "invalid_value_repr": [repr(start), repr(end)] if bad_number else None, "reasons": reasons})
    # Check every valid pair, not only the immediate predecessor: nested spans
    # must not make an overlapping later word appear safe.
    for i, row in enumerate(rows):
        if row["reasons"]:
            continue
        for previous in rows[:i]:
            if any(x in previous["reasons"] for x in ("non_finite_or_non_numeric_timestamp", "non_positive_duration", "timestamp_outside_audio")):
                continue
            if row["start_sec"] < previous["end_sec"]:
                row["reasons"].append("overlap_or_out_of_order")
                if "overlap_or_out_of_order" not in previous["reasons"]:
                    previous["reasons"].append("overlap_or_out_of_order")
    for row in rows:
        row["valid"] = not row["reasons"]
    return {"rows": rows, "invalid_indices": [r["index"] for r in rows if not r["valid"]],
            "empty_output": not rows, "repaired_timestamps": False}


def matched_word_pairs(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Deterministic exact-word LCS, preserving unmatched words in later audits."""
    n, m = len(left), len(right)
    if n * m > 250000:
        raise ValueError("word matching exceeds the bounded context limit")
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            same = left[i]["normalized_word"] and left[i]["normalized_word"] == right[j]["normalized_word"]
            dp[i][j] = 1 + dp[i + 1][j + 1] if same else max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    pairs = []
    while i < n and j < m:
        if left[i]["normalized_word"] and left[i]["normalized_word"] == right[j]["normalized_word"]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def gap_candidates(mixed: Mapping[str, Any], vocal: Mapping[str, Any], energy: Sequence[Mapping[str, float]], policy: Mapping[str, Any]) -> dict[str, Any]:
    left, right = mixed["rows"], vocal["rows"]
    matches = matched_word_pairs(left, right)
    candidates = []
    for (a, b), (c, d) in zip(matches, matches[1:]):
        reasons = []
        if c != a + 1 or d != b + 1:
            reasons.append("unmatched_words_between_gap_anchors")
        anchors = (left[a], right[b], left[c], right[d])
        if not all(r["valid"] for r in anchors):
            reasons.append("invalid_anchor_timestamp")
        record = {"mixed_indices": [a, c], "vocal_indices": [b, d], "words": [left[a]["text"], left[c]["text"]], "reasons": reasons}
        if reasons:
            candidates.append({**record, "eligible": False})
            continue
        start = max(left[a]["end_sec"], right[b]["end_sec"])
        end = min(left[c]["start_sec"], right[d]["start_sec"])
        discrepancy = max(abs(left[a]["end_sec"] - right[b]["end_sec"]), abs(left[c]["start_sec"] - right[d]["start_sec"]))
        if discrepancy > policy["maximum_boundary_disagreement_sec"] + 1e-9:
            reasons.append("boundary_disagreement")
        if end - start < policy["minimum_gap_sec"] - 1e-9:
            reasons.append("common_gap_too_short")
        cut = (start + end) / 2
        guard = policy["cut_guard_sec"]
        samples = [e for e in energy if e["start_sec"] <= cut + guard and e["end_sec"] >= cut - guard]
        covered_until = cut - guard
        covered = bool(samples)
        for sample in sorted(samples, key=lambda e: e["start_sec"]):
            if not all(math.isfinite(sample[k]) for k in ("start_sec", "end_sec", "relative_peak_db")) or sample["end_sec"] <= sample["start_sec"]:
                covered = False
                break
            if sample["start_sec"] > covered_until + 1e-9:
                covered = False
                break
            covered_until = max(covered_until, sample["end_sec"])
        covered = covered and covered_until >= cut + guard - 1e-9
        if not covered or any(e["relative_peak_db"] > policy["maximum_gap_rms_relative_peak_db"] for e in samples):
            reasons.append("vocal_energy_does_not_support_pause")
        candidates.append({**record, "start_sec": start, "end_sec": end, "cut_sec": cut,
            "boundary_disagreement_sec": discrepancy, "energy_frames": len(samples), "eligible": not reasons})
    return {"matches": [list(p) for p in matches], "candidates": candidates,
            "unmatched_mixed_indices": sorted(set(range(len(left))) - {a for a, _ in matches}),
            "unmatched_vocal_indices": sorted(set(range(len(right))) - {b for _, b in matches})}


def select_window(gaps: Mapping[str, Any], mixed: Mapping[str, Any], vocal: Mapping[str, Any], *, old_start: float, old_end: float, policy: Mapping[str, Any]) -> dict[str, Any]:
    safe = [r for r in gaps["candidates"] if r["eligible"]]
    choices, rejected = [], []
    for i, first in enumerate(safe):
        for last in safe[i + 1:]:
            start, end = first["cut_sec"], last["cut_sec"]
            duration = end - start
            if not policy["minimum_window_sec"] <= duration <= policy["maximum_window_sec"]:
                continue
            a, c = first["mixed_indices"][1], last["mixed_indices"][0]
            b, d = first["vocal_indices"][1], last["vocal_indices"][0]
            lrows, rrows = mixed["rows"][a:c + 1], vocal["rows"][b:d + 1]
            reasons = []
            if len(lrows) < policy["minimum_internal_matched_words"] or [r["normalized_word"] for r in lrows] != [r["normalized_word"] for r in rrows]:
                reasons.append("internal_transcripts_disagree")
            if not all(r["valid"] for r in (*lrows, *rrows)):
                reasons.append("invalid_internal_timestamps")
            candidate = {"start_sec": start, "end_sec": end, "duration_sec": duration,
                "mixed_word_indices": list(range(a, c + 1)), "vocal_word_indices": list(range(b, d + 1)), "reasons": reasons}
            if reasons:
                rejected.append(candidate)
                continue
            candidate["selection_key"] = [abs(duration - policy["preferred_window_sec"]), abs((start + end - old_start - old_end) / 2), start]
            candidate["old_interval_coverage_sec"] = max(0.0, min(end, old_end) - max(start, old_start))
            choices.append(candidate)
    choices.sort(key=lambda r: r["selection_key"])
    return {"selected": choices[0] if choices else None, "eligible_pairs": choices, "rejected_pairs": rejected,
        "status": "machine_supported_candidate" if choices else "no_safe_window",
        "phone_predictions_used": False, "human_approved": False, "official_original_accuracy_claim_allowed": False}

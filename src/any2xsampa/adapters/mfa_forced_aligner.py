"""Convert MFA phone-level JSON into the project's forced-phone contract.

MFA remains an external, replaceable aligner. This adapter binds its output to
the exact vocal stem and lyrics timeline and refuses unknown phone symbols or
text/timing mismatches. The configured phone-set projection is an export-neutral
internal IPA normalization; the original MFA phone is retained in metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from any2xsampa.adapters.forced_aligner import (
    MACHINE_REVIEW_DRAFT_STATUS,
    PENDING_HUMAN_PHONE_REVIEW_STATUS,
    ForcedAlignmentResult,
    ForcedPhoneAlignerError,
    lyrics_timeline_sha256,
    parse_forced_phone_alignment,
)
from any2xsampa.cantonese.g2p import ResolvedLyricsItem
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.lyrics.timeline import LyricsTimeline, LyricsTimelineItem, sha256_file

MFA_SCHEMA_VERSION = "mfa_alignment_json_v1"
MFA_MANIFEST_SCHEMA_VERSION = "mfa_alignment_manifest_v1"
_MAPPING_RESOURCE = "mfa_cv19_v3_phone_map.json"
_ALIGNMENT_SOURCE = "mfa_hkcantonese_cv19_v3"
_MODEL_NAME = "HKCantonese_models_cv19_v3"
_PHONE_SET = "IPA_cv19_v3"
_REPOSITORY_LICENSE = "MIT"
_MODEL_DATA_LICENSE_STATUS = "local_only_unverified_model_data_rights"
_TIMING_TOLERANCE_SEC = 1e-6
MFA_TIMING_POLICY_SCHEMA_VERSION = "mfa_timing_policy_v1"
_SILENCE_WORD_LABELS = frozenset({"<eps>", "sil", "O"})
_UNKNOWN_WORD_LABELS = frozenset({"<unk>", "unk", "spn"})
_TIMELINE_CONSTRAINED_MANIFEST_KIND = "trusted_timeline_constrained_mfa_alignment"
_TIMELINE_CONSTRAINT = "manual_lyrics_item_bounds_with_mfa_internal_phone_boundaries"
_BOUNDARY_POLICY = "close_first_phone_to_item_start_and_last_phone_to_item_end"
_INTERPOLATION_POLICY = "edge_closure_only_explicitly_recorded"
_ManifestInput = Mapping[str, Any] | Path | str
MFA_PHONE_REVIEW_DRAFT_SCHEMA_VERSION = "mfa_phone_review_draft_v1"


@dataclass(frozen=True, slots=True)
class MfaTimingPolicy:
    """The only timing comparison policy used by the MFA adapter.

    MFA word intervals are evidence that must already be on the supplied
    lyrics timeline.  We intentionally do not infer a global offset, stretch,
    or warp from the differences: a correction would need its own
    hash-bound, reviewed evidence before it could be applied to product data.
    The one-microsecond bound only accommodates JSON/float representation
    noise and is not an acoustic alignment tolerance.
    """

    schema_version: str = MFA_TIMING_POLICY_SCHEMA_VERSION
    name: str = "strict_lyrics_timeline"
    boundary_tolerance_sec: float = _TIMING_TOLERANCE_SEC
    transform: str | None = None
    allow_offset_transform: bool = False
    allow_warp_transform: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "boundary_tolerance_sec": self.boundary_tolerance_sec,
            "transform": self.transform,
            "allow_offset_transform": self.allow_offset_transform,
            "allow_warp_transform": self.allow_warp_transform,
        }


MFA_TIMING_POLICY = MfaTimingPolicy()


def _model_metadata(*, revision: str, model_hash: str) -> dict[str, Any]:
    return {
        "name": _MODEL_NAME,
        "revision": revision,
        "hash": model_hash.lower(),
        "repository_license": _REPOSITORY_LICENSE,
        "license_status": _MODEL_DATA_LICENSE_STATUS,
        "distribution_allowed": False,
        "phone_set": _PHONE_SET,
    }


def _mapping_payload() -> dict[str, Any]:
    try:
        with resources.files("any2xsampa.config").joinpath(_MAPPING_RESOURCE).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map cannot be read") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "mfa_cv19_v3_phone_map_v1":
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map has an invalid schema")
    if payload.get("model") != _MODEL_NAME:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map has an unexpected model")
    phones = payload.get("phones")
    silence = payload.get("silence")
    oov = payload.get("oov")
    if not isinstance(phones, Mapping) or not isinstance(silence, list) or not isinstance(oov, list):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map is malformed")
    if any(not isinstance(key, str) or not key.strip() for key in phones):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map has an empty source phone")
    if any(not isinstance(value, str) or not value.strip() for value in phones.values()):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map has an empty canonical phone")
    if any(not isinstance(value, str) or not value.strip() for value in silence):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map has an invalid silence phone")
    if any(not isinstance(value, str) or not value.strip() for value in oov):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map has an invalid OOV phone")
    if set(silence) & set(oov):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map overlaps silence and OOV phones")
    if set(phones) & (set(silence) | set(oov)):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone map overlaps phones and non-IPA labels")
    return payload


def phone_map_sha256() -> str:
    payload = _mapping_payload()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def alignment_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a parsed alignment with the manifest's canonical JSON encoding."""
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", "MFA alignment cannot be canonicalized"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _manifest_payload(manifest: _ManifestInput | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    if isinstance(manifest, Mapping):
        return dict(manifest)
    path = Path(manifest)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", f"invalid MFA alignment manifest: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", "MFA alignment manifest must be an object"
        )
    return dict(payload)


def _manifest_alias(
    manifest: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    label: str,
    required: bool,
) -> Any:
    present = [(key, manifest[key]) for key in aliases if key in manifest]
    if not present:
        if required:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence", f"MFA manifest is missing {label}"
            )
        return None
    first_key, first_value = present[0]
    for key, value in present[1:]:
        equal = (
            value.lower() == first_value.lower()
            if "sha256" in label and isinstance(value, str) and isinstance(first_value, str)
            else value == first_value
        )
        if not equal:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA manifest has conflicting {label} fields",
                details={"fields": [first_key, key]},
            )
    return first_value


def _manifest_nested_alias(
    manifest: Mapping[str, Any],
    container_name: str,
    aliases: Sequence[str],
    *,
    label: str,
    required: bool,
) -> Any:
    nested = manifest.get(container_name)
    if nested is None:
        return None
    if not isinstance(nested, Mapping):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", f"MFA manifest {container_name} must be an object"
        )
    present = [(f"{container_name}.{key}", nested[key]) for key in aliases if key in nested]
    if not present:
        if required:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence", f"MFA manifest is missing {label}"
            )
        return None
    first_key, first_value = present[0]
    for key, value in present[1:]:
        equal = (
            value.lower() == first_value.lower()
            if "sha256" in label and isinstance(value, str) and isinstance(first_value, str)
            else value == first_value
        )
        if not equal:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA manifest has conflicting {label} fields",
                details={"fields": [first_key, key]},
            )
    return first_value


def _manifest_value(
    manifest: Mapping[str, Any],
    *,
    flat_aliases: Sequence[str],
    nested_container: str | None,
    nested_aliases: Sequence[str],
    label: str,
    required: bool,
) -> Any:
    flat = _manifest_alias(manifest, flat_aliases, label=label, required=False)
    nested = (
        _manifest_nested_alias(
            manifest,
            nested_container,
            nested_aliases,
            label=label,
            required=False,
        )
        if nested_container
        else None
    )
    equal = (
        flat.lower() == nested.lower()
        if "sha256" in label and isinstance(flat, str) and isinstance(nested, str)
        else flat == nested
    )
    if flat is not None and nested is not None and not equal:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            f"MFA manifest has conflicting {label} fields",
            details={"flat": flat, "nested": nested},
        )
    value = flat if flat is not None else nested
    if value is None and required:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", f"MFA manifest is missing {label}"
        )
    return value


def _manifest_hash(
    manifest: Mapping[str, Any],
    *,
    flat_aliases: Sequence[str],
    nested_container: str | None,
    nested_aliases: Sequence[str],
    label: str,
    required: bool,
) -> str | None:
    value = _manifest_value(
        manifest,
        flat_aliases=flat_aliases,
        nested_container=nested_container,
        nested_aliases=nested_aliases,
        label=label,
        required=required,
    )
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", f"MFA manifest {label} must be SHA-256"
        )
    return value.lower()


def _validate_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    trusted: bool,
    alignment_hash: str,
    vocal_stem_hash: str,
    timeline_hash: str,
    model_hash: str,
    model_revision: str,
    dictionary_hash: str | None,
    map_schema: str,
    map_hash: str,
) -> dict[str, Any] | None:
    """Validate the signed-by-convention manifest against all supplied evidence."""
    if manifest is None:
        if trusted:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                "trusted MFA alignment requires an mfa_alignment_manifest_v1 manifest",
            )
        return None
    if manifest.get("schema_version") != MFA_MANIFEST_SCHEMA_VERSION:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            f"MFA manifest schema must be {MFA_MANIFEST_SCHEMA_VERSION}",
        )
    declared_alignment_hash = _manifest_hash(
        manifest,
        flat_aliases=("alignment_json_sha256", "alignment_sha256"),
        nested_container="alignment",
        nested_aliases=("json_sha256", "sha256"),
        label="alignment_json_sha256",
        required=True,
    )
    declared_stem_hash = _manifest_hash(
        manifest,
        flat_aliases=("vocal_stem_sha256", "audio_sha256"),
        nested_container="vocal_stem",
        nested_aliases=("sha256",),
        label="vocal_stem_sha256",
        required=True,
    )
    declared_timeline_hash = _manifest_hash(
        manifest,
        flat_aliases=("lyrics_timeline_sha256", "lyrics_timeline_canonical_sha256"),
        nested_container="lyrics_timeline",
        nested_aliases=("canonical_sha256", "sha256"),
        label="lyrics_timeline_sha256",
        required=True,
    )
    declared_model_hash = _manifest_hash(
        manifest,
        flat_aliases=("model_sha256", "model_hash"),
        nested_container="model",
        nested_aliases=("sha256", "hash"),
        label="model_sha256",
        required=True,
    )
    declared_dictionary_hash = _manifest_hash(
        manifest,
        flat_aliases=("dictionary_sha256", "dictionary_hash"),
        nested_container="dictionary",
        nested_aliases=("sha256", "hash"),
        label="dictionary_sha256",
        required=True,
    )
    declared_map_hash = _manifest_hash(
        manifest,
        flat_aliases=("phone_map_sha256",),
        nested_container="phone_map",
        nested_aliases=("sha256",),
        label="phone_map_sha256",
        required=True,
    )
    declared_revision = _manifest_value(
        manifest,
        flat_aliases=("model_revision",),
        nested_container="model",
        nested_aliases=("revision",),
        label="model_revision",
        required=True,
    )
    declared_map_schema = _manifest_value(
        manifest,
        flat_aliases=("phone_map_schema",),
        nested_container="phone_map",
        nested_aliases=("schema", "schema_version"),
        label="phone_map_schema",
        required=True,
    )
    if not isinstance(declared_revision, str) or not declared_revision.strip():
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA manifest model_revision must be non-empty")
    if not isinstance(declared_map_schema, str) or not declared_map_schema.strip():
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA manifest phone_map_schema must be non-empty")
    checks = (
        ("alignment_json_sha256", declared_alignment_hash, alignment_hash),
        ("vocal_stem_sha256", declared_stem_hash, vocal_stem_hash),
        ("lyrics_timeline_sha256", declared_timeline_hash, timeline_hash),
        ("model_sha256", declared_model_hash, model_hash),
        ("phone_map_sha256", declared_map_hash, map_hash),
    )
    for label, declared, actual in checks:
        if declared != actual.lower():
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA manifest {label} does not match supplied evidence",
                details={"declared": declared, "actual": actual.lower()},
            )
    if declared_revision.strip() != model_revision:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", "MFA manifest model_revision does not match supplied model"
        )
    if declared_map_schema != map_schema:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", "MFA manifest phone_map_schema does not match checked-in map"
        )
    if dictionary_hash is not None and declared_dictionary_hash != dictionary_hash.lower():
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "MFA manifest dictionary_sha256 does not match supplied dictionary",
            details={"declared": declared_dictionary_hash, "actual": dictionary_hash.lower()},
        )
    return dict(manifest)


def _alignment_transform_provenance(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    *,
    trusted: bool,
) -> dict[str, Any] | None:
    raw_provenance = payload.get("provenance")
    provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
    declared = (
        provenance.get("timeline_constraint"),
        provenance.get("boundary_policy"),
        provenance.get("interpolation_policy"),
    )
    manifest_kind = manifest.get("manifest_kind") if manifest is not None else None
    constrained_manifest = manifest_kind == _TIMELINE_CONSTRAINED_MANIFEST_KIND
    if not any(value is not None for value in declared) and not constrained_manifest:
        return None
    expected = (_TIMELINE_CONSTRAINT, _BOUNDARY_POLICY, _INTERPOLATION_POLICY)
    if declared != expected:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "timeline-constrained MFA provenance must declare the complete supported boundary policy",
        )
    if trusted and not constrained_manifest:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "trusted timeline-constrained MFA output requires a matching manifest_kind",
        )
    if constrained_manifest:
        validation = manifest.get("validation") if manifest is not None else None
        if not isinstance(validation, Mapping):
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                "timeline-constrained MFA manifest requires a validation object",
            )
        if (
            validation.get("exact_lyric_order") is not True
            or validation.get("timeline_constrained") is not True
            or validation.get("boundary_policy") != _BOUNDARY_POLICY
        ):
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                "timeline-constrained MFA manifest validation does not match the supported boundary policy",
            )
    return {
        "schema_version": "mfa_alignment_transform_v1",
        "timeline_constraint": _TIMELINE_CONSTRAINT,
        "boundary_policy": _BOUNDARY_POLICY,
        "interpolation_policy": _INTERPOLATION_POLICY,
        "manifest_kind": manifest_kind,
        "word_boundaries_source": "hash_bound_lyrics_timeline_v1",
        "internal_phone_boundaries_source": "mfa_per_lyric_item_json",
        "uniform_duration_inference": False,
        "boundary_transform_applied": True,
        "review_status": "mechanically_validated_pending_listening_review",
    }


def _normalise_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {label} must be SHA-256")
    return value.lower()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {label} is not numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {label} is not finite")
    return parsed


def _entries(payload: object, tier: str) -> list[tuple[float, float, str]]:
    if not isinstance(payload, Mapping):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA alignment JSON must be an object")
    tiers = payload.get("tiers")
    tier_payload = tiers.get(tier) if isinstance(tiers, Mapping) else None
    if not isinstance(tiers, Mapping) or not isinstance(tier_payload, Mapping):
        raise ForcedPhoneAlignerError("missing_phone_evidence", f"MFA alignment has no {tier} tier")
    tier_type = tier_payload.get("type")
    if tier_type != "IntervalTier":
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {tier} tier must be an IntervalTier")
    raw_entries = tier_payload.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {tier} tier entries are malformed")
    result: list[tuple[float, float, str]] = []
    previous_end = -1.0
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Sequence) or len(raw) != 3:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {tier} entry {index} is malformed")
        start = _finite(raw[0], f"{tier} entry {index} start")
        end = _finite(raw[1], f"{tier} entry {index} end")
        if not isinstance(raw[2], str):
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {tier} entry {index} label is not text")
        label = raw[2].strip()
        if start < 0.0 or end <= start or start < previous_end - 1e-9 or not label:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {tier} entry {index} has invalid timing")
        result.append((start, end, label))
        previous_end = end
    return result


def _validate_resolved_timeline(
    timeline: LyricsTimeline,
    resolved_items: Sequence[ResolvedLyricsItem],
) -> None:
    timeline_items = {item.index: item for item in timeline.items}
    resolved_by_index = {item.index: item for item in resolved_items}
    if set(resolved_by_index) != set(timeline_items):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "resolved G2P items must cover exactly the lyrics timeline items",
        )
    for item_index, item in resolved_by_index.items():
        timeline_item = timeline_items[item_index]
        if (
            item.text != timeline_item.text
            or abs(item.start_sec - timeline_item.start_sec) > 1e-9
            or abs(item.end_sec - timeline_item.end_sec) > 1e-9
        ):
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"resolved G2P timing/text does not match timeline item {item_index}",
            )


def _timing_comparison(
    words: Sequence[tuple[float, float, str]],
    resolved_items: Sequence[ResolvedLyricsItem | LyricsTimelineItem],
    *,
    source_offset_sec: float,
) -> dict[str, Any]:
    """Compare MFA word bounds without applying an inferred correction.

    The complete row list is retained so a rejected alignment can be audited
    without rerunning MFA.  ``actual_*`` values are the MFA values after only
    the caller-declared source offset; no offset or warp is estimated here.
    """
    rows: list[dict[str, Any]] = []
    for item, (start, end, _label) in zip(resolved_items, words):
        actual_start = start + source_offset_sec
        actual_end = end + source_offset_sec
        start_delta = actual_start - item.start_sec
        end_delta = actual_end - item.end_sec
        duration_delta = (actual_end - actual_start) - (item.end_sec - item.start_sec)
        within_tolerance = (
            abs(start_delta) <= MFA_TIMING_POLICY.boundary_tolerance_sec
            and abs(end_delta) <= MFA_TIMING_POLICY.boundary_tolerance_sec
        )
        rows.append(
            {
                "lyric_item_index": item.index,
                "expected_start_sec": item.start_sec,
                "expected_end_sec": item.end_sec,
                "actual_start_sec": actual_start,
                "actual_end_sec": actual_end,
                "start_delta_sec": start_delta,
                "end_delta_sec": end_delta,
                "duration_delta_sec": duration_delta,
                "within_tolerance": within_tolerance,
            }
        )

    start_deltas = [abs(float(row["start_delta_sec"])) for row in rows]
    end_deltas = [abs(float(row["end_delta_sec"])) for row in rows]
    duration_deltas = [abs(float(row["duration_delta_sec"])) for row in rows]
    mismatches = [row for row in rows if not row["within_tolerance"]]

    def _maximum(values: Sequence[float]) -> float:
        return max(values) if values else 0.0

    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "schema_version": "mfa_timing_comparison_v1",
        "policy": MFA_TIMING_POLICY.to_dict(),
        "source_offset_sec": source_offset_sec,
        "transform_applied": False,
        "word_count": len(rows),
        "mismatch_count": len(mismatches),
        "mismatch_lyric_item_indices": [row["lyric_item_index"] for row in mismatches],
        "within_policy": not mismatches,
        "max_abs_start_delta_sec": _maximum(start_deltas),
        "max_abs_end_delta_sec": _maximum(end_deltas),
        "max_abs_duration_delta_sec": _maximum(duration_deltas),
        "mean_abs_start_delta_sec": _mean(start_deltas),
        "mean_abs_end_delta_sec": _mean(end_deltas),
        "mean_abs_duration_delta_sec": _mean(duration_deltas),
        "rows": rows,
    }


def compare_mfa_timing(
    words: Sequence[tuple[float, float, str]],
    resolved_items: Sequence[ResolvedLyricsItem],
    *,
    source_offset_sec: float = 0.0,
) -> dict[str, Any]:
    """Return the strict MFA-to-lyrics timing audit without applying a fix."""
    offset = _finite(source_offset_sec, "source_offset_sec")
    usable = [entry for entry in words if entry[2] not in _SILENCE_WORD_LABELS]
    return _timing_comparison(usable, resolved_items, source_offset_sec=offset)


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (EOFError, OSError, wave.Error):
        return None


def _word_matches(
    words: Sequence[tuple[float, float, str]],
    resolved_items: Sequence[ResolvedLyricsItem],
    *,
    source_offset_sec: float = 0.0,
    timing_comparison: Mapping[str, Any] | None = None,
) -> list[tuple[int, float, float]]:
    indexes = [item.index for item in resolved_items]
    if indexes != list(range(len(resolved_items))):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "resolved G2P items must be ordered and indexed from zero",
        )
    usable: list[tuple[float, float, str]] = []
    for index, (start, end, label) in enumerate(words):
        if label in _UNKNOWN_WORD_LABELS:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA word tier contains an unknown token at entry {index}: {label}",
            )
        if label not in _SILENCE_WORD_LABELS:
            usable.append((start, end, label))
    expected_count = len(resolved_items)
    if len(usable) < expected_count:
        raise ForcedPhoneAlignerError(
            "missing_phone_evidence",
            f"MFA word tier has {len(usable)} usable entries for {expected_count} lyric items",
            details={"usable_word_count": len(usable), "lyric_item_count": expected_count},
        )
    if len(usable) > expected_count:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            f"MFA word tier has {len(usable) - expected_count} extra usable token(s)",
            details={"extra_word_count": len(usable) - expected_count},
        )
    matches: list[tuple[int, float, float]] = []
    for item, (start, end, label) in zip(resolved_items, usable):
        if label != item.text:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA word token for lyric item {item.index} is {label!r}, expected {item.text!r}",
                details={"lyric_item_index": item.index, "expected": item.text, "actual": label},
            )
        matches.append((item.index, start, end))
    comparison = (
        dict(timing_comparison)
        if timing_comparison is not None
        else _timing_comparison(usable, resolved_items, source_offset_sec=source_offset_sec)
    )
    if not bool(comparison.get("within_policy")):
        mismatch_rows = comparison.get("rows", [])
        first_mismatch = next(
            (
                row
                for row in mismatch_rows
                if isinstance(row, Mapping) and not bool(row.get("within_tolerance"))
            ),
            None,
        )
        item_index = (
            first_mismatch.get("lyric_item_index")
            if isinstance(first_mismatch, Mapping)
            else None
        )
        details = {
            "timing_comparison": comparison,
            "timing_policy": MFA_TIMING_POLICY.to_dict(),
            "source_offset_sec": source_offset_sec,
        }
        if isinstance(first_mismatch, Mapping):
            details.update(dict(first_mismatch))
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            (
                "MFA word timing does not match lyric item "
                f"{item_index if item_index is not None else 'timeline'}"
            ),
            details=details,
        )
    return matches


def _alignment_bounds(payload: Mapping[str, Any]) -> tuple[float, float] | None:
    """Validate optional MFA document bounds and return them when present."""
    has_start = "start" in payload
    has_end = "end" in payload
    if has_start != has_end:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "MFA alignment must provide both start and end when either is present",
        )
    if not has_start:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "MFA alignment must provide finite start and end bounds",
        )
    start = _finite(payload.get("start"), "alignment start")
    end = _finite(payload.get("end"), "alignment end")
    if start < 0.0 or end <= start:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA alignment bounds are invalid")
    return start, end


def _validate_optional_hash(value: object, *, label: str, expected: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA {label} must be SHA-256")
    if value.lower() != expected.lower():
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            f"MFA {label} does not match the supplied evidence",
            details={"declared": value.lower(), "actual": expected.lower()},
        )


def _validate_input_metadata(
    payload: Mapping[str, Any],
    *,
    vocal_stem_hash: str,
    timeline_hash: str,
    model_hash: str,
    model_revision: str,
    map_schema: str,
    map_hash: str,
    source_offset_sec: float,
) -> None:
    if "schema_version" in payload and payload.get("schema_version") != MFA_SCHEMA_VERSION:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA schema must be {MFA_SCHEMA_VERSION}")
    for key, expected in (
        ("vocal_stem_sha256", vocal_stem_hash),
        ("audio_sha256", vocal_stem_hash),
        ("stem_sha256", vocal_stem_hash),
        ("lyrics_timeline_sha256", timeline_hash),
        ("model_hash", model_hash),
    ):
        if key in payload:
            _validate_optional_hash(payload[key], label=key, expected=expected)
    if "model_revision" in payload and payload["model_revision"] != model_revision:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA model_revision does not match the supplied model")
    if "source_offset_sec" in payload:
        declared_offset = _finite(payload["source_offset_sec"], "source_offset_sec")
        if abs(declared_offset - source_offset_sec) > _TIMING_TOLERANCE_SEC:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                "MFA source offset does not match the supplied offset",
                details={
                    "timing_policy": MFA_TIMING_POLICY.to_dict(),
                    "transform_applied": False,
                    "declared_source_offset_sec": declared_offset,
                    "supplied_source_offset_sec": source_offset_sec,
                    "source_offset_delta_sec": declared_offset - source_offset_sec,
                },
            )
    if "alignment_source" in payload and payload["alignment_source"] != _ALIGNMENT_SOURCE:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA alignment_source is not the configured adapter source")
    if "phone_map_schema" in payload and payload["phone_map_schema"] != map_schema:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA phone_map_schema does not match the checked-in map")
    if "phone_map_sha256" in payload:
        _validate_optional_hash(payload["phone_map_sha256"], label="phone_map_sha256", expected=map_hash)
    model_payload = payload.get("model")
    if model_payload is not None:
        if not isinstance(model_payload, Mapping):
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA model metadata must be an object")
        for key, expected in (("name", _MODEL_NAME), ("phone_set", _PHONE_SET), ("revision", model_revision)):
            if key in model_payload and model_payload[key] != expected:
                raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"MFA model.{key} does not match the configured model")
        if "hash" in model_payload:
            _validate_optional_hash(model_payload["hash"], label="model.hash", expected=model_hash)
    provenance = payload.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA provenance metadata must be an object")
        if "mfa_schema_version" in provenance and provenance["mfa_schema_version"] != MFA_SCHEMA_VERSION:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA provenance schema does not match the alignment schema")
        if "phone_map_schema" in provenance and provenance["phone_map_schema"] != map_schema:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA provenance phone map schema does not match")
        if "phone_map_sha256" in provenance:
            _validate_optional_hash(provenance["phone_map_sha256"], label="provenance.phone_map_sha256", expected=map_hash)
        if "source_offset_sec" in provenance:
            declared_offset = _finite(provenance["source_offset_sec"], "provenance.source_offset_sec")
            if abs(declared_offset - source_offset_sec) > _TIMING_TOLERANCE_SEC:
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    "MFA provenance source offset does not match the supplied offset",
                    details={
                        "timing_policy": MFA_TIMING_POLICY.to_dict(),
                        "transform_applied": False,
                        "declared_source_offset_sec": declared_offset,
                        "supplied_source_offset_sec": source_offset_sec,
                        "source_offset_delta_sec": declared_offset - source_offset_sec,
                    },
                )
        if "source_audio_hash_bound" in provenance and provenance["source_audio_hash_bound"] is not True:
            raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA provenance must declare source_audio_hash_bound=true")


def _phone_owners(
    words: Sequence[tuple[float, float, str]],
    phones: Sequence[tuple[float, float, str]],
    *,
    alignment_bounds: tuple[float, float] | None,
    mapping: Mapping[str, Any],
) -> tuple[dict[int, list[tuple[float, float, str, str, float]]], dict[int, int]]:
    """Map every non-silence MFA phone to exactly one usable word interval."""
    owners: dict[int, list[tuple[float, float, str, str, float]]] = {}
    assigned: dict[int, int] = {}
    usable_words: list[tuple[float, float, str]] = []
    for word_index, word in enumerate(words):
        if word[2] in _UNKNOWN_WORD_LABELS:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA word tier contains an unknown token at entry {word_index}: {word[2]}",
            )
        if word[2] not in _SILENCE_WORD_LABELS:
            usable_words.append(word)
    for phone_index, (start, end, source_phone) in enumerate(phones):
        if alignment_bounds is not None:
            bound_start, bound_end = alignment_bounds
            if start < bound_start - _TIMING_TOLERANCE_SEC or end > bound_end + _TIMING_TOLERANCE_SEC:
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    f"MFA phone entry {phone_index} lies outside alignment bounds",
                )
        canonical = _canonical_phone(source_phone, mapping)
        if canonical is None:
            if any(
                min(end, word_end) - max(start, word_start) > _TIMING_TOLERANCE_SEC
                for word_start, word_end, _label in usable_words
            ):
                raise ForcedPhoneAlignerError(
                    "missing_phone_evidence",
                    f"MFA silence/OOV phone {source_phone} overlaps a lyric word",
                    details={"phone_index": phone_index, "source_phone": source_phone},
                )
            continue
        overlapping_words = [
            (word_index, word_start, word_end)
            for word_index, (word_start, word_end, _label) in enumerate(usable_words)
            if min(end, word_end) - max(start, word_start) > _TIMING_TOLERANCE_SEC
        ]
        if len(overlapping_words) != 1:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA phone {source_phone} is not contained in exactly one lyric word",
                details={"phone_index": phone_index, "source_phone": source_phone, "overlapping_word_count": len(overlapping_words)},
            )
        word_index, word_start, word_end = overlapping_words[0]
        if start < word_start - _TIMING_TOLERANCE_SEC or end > word_end + _TIMING_TOLERANCE_SEC:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA phone {source_phone} crosses a lyric word boundary",
                details={"phone_index": phone_index, "word_index": word_index, "source_phone": source_phone},
            )
        overlap = max(0.0, min(end, word_end) - max(start, word_start))
        ratio = overlap / (end - start)
        owners.setdefault(word_index, []).append((start, end, source_phone, canonical, ratio))
        assigned[phone_index] = word_index
    return owners, assigned


def _canonical_phone(source_phone: str, mapping: Mapping[str, Any]) -> str | None:
    silence = {str(value) for value in mapping.get("silence", [])}
    oov = {str(value) for value in mapping.get("oov", [])}
    if source_phone in silence or source_phone in oov:
        return None
    phones = mapping.get("phones", {})
    canonical = phones.get(source_phone) if isinstance(phones, Mapping) else None
    if not isinstance(canonical, str) or not canonical.strip():
        raise ForcedPhoneAlignerError(
            "unsupported_forced_phone_set",
            f"MFA phone is not present in the checked-in phone map: {source_phone}",
        )
    return canonical.strip()


def _review_draft_word_matches(
    words: Sequence[tuple[float, float, str]],
    timeline: LyricsTimeline,
) -> tuple[list[tuple[int, float, float]], list[tuple[float, float, str]]]:
    usable: list[tuple[float, float, str]] = []
    for word_index, entry in enumerate(words):
        if entry[2] in _UNKNOWN_WORD_LABELS:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA word tier contains an unknown token at entry {word_index}: {entry[2]}",
            )
        if entry[2] not in _SILENCE_WORD_LABELS:
            usable.append(entry)
    if len(usable) != len(timeline.items):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "MFA word tier must cover the complete lyrics timeline in review-draft mode",
            details={
                "usable_word_count": len(usable),
                "lyric_item_count": len(timeline.items),
            },
        )
    matches: list[tuple[int, float, float]] = []
    for item, (start, end, label) in zip(timeline.items, usable, strict=True):
        if label != item.text:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"MFA word token for lyric item {item.index} is {label!r}, expected {item.text!r}",
                details={
                    "lyric_item_index": item.index,
                    "expected": item.text,
                    "actual": label,
                },
            )
        matches.append((item.index, start, end))
    return matches, usable


def build_mfa_full_song_review_draft(
    payload: Mapping[str, Any],
    *,
    vocal_stem: Path,
    timeline: LyricsTimeline,
    model_hash: str,
    model_revision: str,
    dictionary_path: Path | str | None = None,
    alignment_sha256: str | None = None,
) -> dict[str, Any]:
    """Build hash-bound machine evidence that still requires human review.

    Unlike :func:`convert_mfa_alignment`, this path deliberately accepts MFA
    word timing drift from the lyrics timeline. It preserves the acoustic
    phone boundaries byte-for-byte as review suggestions and marks the result
    as ineligible for product use until a detached human acceptance exists.
    """

    if not isinstance(payload, Mapping):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", "MFA alignment JSON must be an object"
        )
    stem = Path(vocal_stem)
    if not stem.is_file():
        raise ForcedPhoneAlignerError(
            "missing_vocal_stem_evidence", f"missing vocal stem: {stem}"
        )
    normalized_model_hash = _normalise_sha256(model_hash, label="model_hash")
    revision = str(model_revision or "").strip()
    if not revision:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", "MFA model_revision must be non-empty"
        )
    duration = _wav_duration(stem)
    if duration is None:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence", "vocal stem must be a readable PCM WAV"
        )

    mapping = _mapping_payload()
    stem_hash = sha256_file(stem)
    timeline_hash = lyrics_timeline_sha256(timeline)
    map_schema = str(mapping.get("schema_version"))
    map_hash = phone_map_sha256()
    raw_alignment_hash = (
        _normalise_sha256(alignment_sha256, label="alignment_sha256")
        if alignment_sha256 is not None
        else alignment_json_sha256(payload)
    )
    dictionary_hash: str | None = None
    if dictionary_path is not None:
        dictionary = Path(dictionary_path)
        if not dictionary.is_file():
            raise ForcedPhoneAlignerError(
                "missing_phone_evidence", f"missing MFA dictionary: {dictionary}"
            )
        dictionary_hash = sha256_file(dictionary)

    _validate_input_metadata(
        payload,
        vocal_stem_hash=stem_hash,
        timeline_hash=timeline_hash,
        model_hash=normalized_model_hash,
        model_revision=revision,
        map_schema=map_schema,
        map_hash=map_hash,
        source_offset_sec=0.0,
    )
    alignment_bounds = _alignment_bounds(payload)
    bound_start, bound_end = alignment_bounds
    if abs(bound_start) > _TIMING_TOLERANCE_SEC or abs(bound_end - duration) > 1e-3:
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "full-song MFA review draft must cover the complete vocal stem",
            details={
                "alignment_start_sec": bound_start,
                "alignment_end_sec": bound_end,
                "vocal_stem_duration_sec": duration,
            },
        )

    words = _entries(payload, "words")
    phones = _entries(payload, "phones")
    for tier_name, entries in (("word", words), ("phone", phones)):
        for entry_index, (start, end, _label) in enumerate(entries):
            if start < bound_start - _TIMING_TOLERANCE_SEC or end > bound_end + _TIMING_TOLERANCE_SEC:
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    f"MFA {tier_name} entry {entry_index} lies outside alignment bounds",
                )
    matches, usable_words = _review_draft_word_matches(words, timeline)
    timing_comparison = _timing_comparison(
        usable_words,
        timeline.items,
        source_offset_sec=0.0,
    )
    owners, _assigned_phone_indexes = _phone_owners(
        words,
        phones,
        alignment_bounds=alignment_bounds,
        mapping=mapping,
    )
    alignment_source = f"{_ALIGNMENT_SOURCE}_machine_review_draft"
    segments: list[dict[str, Any]] = []
    for word_index, (lyric_index, word_start, word_end) in enumerate(matches):
        owned_phones = owners.get(word_index, [])
        if not owned_phones:
            raise ForcedPhoneAlignerError(
                "missing_phone_evidence",
                f"MFA phone tier has no usable phone overlap for lyric item {lyric_index}",
            )
        for start, end, source_phone, canonical, direct_ratio in owned_phones:
            overlap = max(0.0, min(end, word_end) - max(start, word_start))
            segments.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "normalized_ipa": canonical,
                    "candidate_phones": [{"symbol": canonical, "confidence": 0.0}],
                    "confidence": 0.0,
                    "metadata": {
                        "alignment_source": alignment_source,
                        "lyric_item_index": lyric_index,
                        "direct_overlap_sec": overlap,
                        "phone_overlap_ratio": direct_ratio,
                        "alignment_confidence": 0.0,
                        "model_hash": normalized_model_hash,
                        "source_mode": "real",
                        "truth_layer": "phones_ipa_articulatory_features",
                        "interpolation_reason": None,
                        "source_phone": source_phone,
                        "source_phone_set": _MODEL_NAME,
                        "phone_map_schema": map_schema,
                        "phone_map_sha256": map_hash,
                        "alignment_confidence_source": "MFA_JSON_has_no_posterior_confidence",
                        "review_status": MACHINE_REVIEW_DRAFT_STATUS,
                    },
                }
            )

    normalized_inventory = list(dict.fromkeys(str(item["normalized_ipa"]) for item in segments))
    source_inventory = list(
        dict.fromkeys(str(item["metadata"]["source_phone"]) for item in segments)
    )
    return {
        "schema_version": "forced_phone_alignment_v1",
        "vocal_stem_sha256": stem_hash,
        "lyrics_timeline_sha256": timeline_hash,
        "alignment_source": alignment_source,
        "model_output": True,
        "model": _model_metadata(revision=revision, model_hash=normalized_model_hash),
        "model_hash": normalized_model_hash,
        "segments": segments,
        "status": MACHINE_REVIEW_DRAFT_STATUS,
        "product_accepted": False,
        "machine_suggestion": True,
        "human_review_required": True,
        "boundary_transform_applied": False,
        "review": {
            "status": PENDING_HUMAN_PHONE_REVIEW_STATUS,
            "scope": "full_song",
            "reviewed_range_sec": [0.0, duration],
            "machine_suggestion": True,
        },
        "provenance": {
            "schema_version": MFA_PHONE_REVIEW_DRAFT_SCHEMA_VERSION,
            "status": MACHINE_REVIEW_DRAFT_STATUS,
            "product_accepted": False,
            "machine_suggestion": True,
            "human_review_required": True,
            "source_audio_hash_bound": True,
            "raw_mfa_alignment_sha256": raw_alignment_hash,
            "dictionary_sha256": dictionary_hash,
            "phone_map_schema": map_schema,
            "phone_map_sha256": map_hash,
            "source_phone_inventory": source_inventory,
            "normalized_phone_inventory": normalized_inventory,
            "word_sequence_exact": True,
            "word_count": len(matches),
            "phone_count": len(segments),
            "timing_comparison": timing_comparison,
            "boundary_transform_applied": False,
            "phone_boundaries_source": "unmodified_mfa_phone_interval_tier",
        },
    }


def build_mfa_timeline_constrained_review_draft(
    payload: Mapping[str, Any],
    *,
    vocal_stem: Path,
    timeline: LyricsTimeline,
    model_hash: str,
    model_revision: str,
    dictionary_path: Path | str | None = None,
    alignment_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a machine-only draft with timeline-bounded phone suggestions.

    Each raw MFA phone keeps its source symbol and ordering, while its interval
    is mapped by the source-word/target-lyric-item affine transform.  The raw
    boundaries remain in segment metadata for audit; the result is still an
    explicit pending human-review artifact and is never accepted product truth.
    """

    draft = build_mfa_full_song_review_draft(
        payload,
        vocal_stem=vocal_stem,
        timeline=timeline,
        model_hash=model_hash,
        model_revision=model_revision,
        dictionary_path=dictionary_path,
        alignment_sha256=alignment_sha256,
    )
    words = _entries(payload, "words")
    phones = _entries(payload, "phones")
    mapping = _mapping_payload()
    alignment_bounds = _alignment_bounds(payload)
    matches, _usable_words = _review_draft_word_matches(words, timeline)
    owners, _assigned_phone_indexes = _phone_owners(
        words,
        phones,
        alignment_bounds=alignment_bounds,
        mapping=mapping,
    )
    timeline_items = {item.index: item for item in timeline.items}
    raw_segments = draft.get("segments")
    if not isinstance(raw_segments, list):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "machine review draft has no segment list",
        )

    transformed_segments: list[dict[str, Any]] = []
    transformed_word_rows: list[tuple[float, float, str]] = []
    segment_index = 0
    previous_mapped_end = -1.0
    for word_index, (lyric_index, source_word_start, source_word_end) in enumerate(matches):
        target_item = timeline_items.get(lyric_index)
        if target_item is None:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"timeline has no lyric item {lyric_index}",
            )
        source_duration = source_word_end - source_word_start
        target_duration = target_item.end_sec - target_item.start_sec
        if source_duration <= 0.0 or target_duration <= 0.0:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence",
                f"cannot transform lyric item {lyric_index} with a non-positive duration",
            )
        transformed_word_rows.append((target_item.start_sec, target_item.end_sec, target_item.text))
        owned_phones = owners.get(word_index, [])
        if not owned_phones:
            raise ForcedPhoneAlignerError(
                "missing_phone_evidence",
                f"MFA phone tier has no usable phone overlap for lyric item {lyric_index}",
            )
        for raw_start, raw_end, _source_phone, _canonical, _direct_ratio in owned_phones:
            if segment_index >= len(raw_segments):
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    "machine review draft segment count does not match MFA phone ownership",
                )
            raw_segment = raw_segments[segment_index]
            if not isinstance(raw_segment, Mapping):
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    f"machine review segment {segment_index} is malformed",
                )
            relative_start = (raw_start - source_word_start) / source_duration
            relative_end = (raw_end - source_word_start) / source_duration
            mapped_start = target_item.start_sec + relative_start * target_duration
            mapped_end = target_item.start_sec + relative_end * target_duration
            if mapped_start < target_item.start_sec - _TIMING_TOLERANCE_SEC or mapped_end > target_item.end_sec + _TIMING_TOLERANCE_SEC:
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    f"transformed phone {segment_index} falls outside lyric item {lyric_index}",
                )
            if mapped_start < previous_mapped_end - _TIMING_TOLERANCE_SEC:
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    f"transformed phone {segment_index} overlaps the previous phone",
                )
            previous_mapped_end = mapped_end
            transformed = dict(raw_segment)
            metadata = dict(raw_segment.get("metadata")) if isinstance(raw_segment.get("metadata"), Mapping) else {}
            metadata.update(
                {
                    "raw_start_sec": float(raw_start),
                    "raw_end_sec": float(raw_end),
                    "source_start_sec": float(raw_start),
                    "source_end_sec": float(raw_end),
                    "raw_word_start_sec": float(source_word_start),
                    "raw_word_end_sec": float(source_word_end),
                    "lyric_item_index": lyric_index,
                    "direct_overlap_sec": mapped_end - mapped_start,
                     "phone_overlap_ratio": 1.0,
                     "boundary_transform_applied": True,
                     "machine_suggestion": True,
                     "review_status": MACHINE_REVIEW_DRAFT_STATUS,
                 }
            )
            transformed.update(
                {
                    "start_sec": mapped_start,
                    "end_sec": mapped_end,
                    "metadata": metadata,
                }
            )
            transformed_segments.append(transformed)
            segment_index += 1
    if segment_index != len(raw_segments):
        raise ForcedPhoneAlignerError(
            "invalid_forced_phone_evidence",
            "machine review draft contains unowned phone segments",
        )

    provenance = dict(draft.get("provenance")) if isinstance(draft.get("provenance"), Mapping) else {}
    raw_timing_comparison = provenance.get("timing_comparison")
    transformed_timing_comparison = _timing_comparison(
        transformed_word_rows,
        timeline.items,
        source_offset_sec=0.0,
    )
    transformed_timing_comparison.update(
        {
            "transform_applied": True,
            "transform": "source_word_relative_affine_to_lyrics_item",
        }
    )
    alignment_transform = {
        "schema_version": "mfa_alignment_transform_v1",
        "timeline_constraint": _TIMELINE_CONSTRAINT,
        "boundary_policy": "source_word_relative_affine_to_target_lyrics_item",
        "interpolation_policy": _INTERPOLATION_POLICY,
        "manifest_kind": None,
        "word_boundaries_source": "hash_bound_lyrics_timeline_v1",
        "internal_phone_boundaries_source": "mfa_per_lyric_item_json",
         "uniform_duration_inference": False,
         "boundary_transform_applied": True,
         "machine_suggestion": True,
         "review_status": MACHINE_REVIEW_DRAFT_STATUS,
    }
    raw_alignment_hash = provenance.get("raw_mfa_alignment_sha256")
    provenance.update(
        {
            "status": MACHINE_REVIEW_DRAFT_STATUS,
            "product_accepted": False,
            "raw_timing_comparison": raw_timing_comparison,
            "timing_comparison": raw_timing_comparison,
            "target_timing_comparison": transformed_timing_comparison,
            "alignment_transform": alignment_transform,
            "boundary_transform_applied": True,
            "machine_suggestion": True,
            "human_review_required": True,
            "phone_boundaries_source": "timeline_constrained_machine_suggestion",
            "raw_alignment_sha256": raw_alignment_hash,
        }
    )
    draft["alignment_source"] = f"{_ALIGNMENT_SOURCE}_timeline_constrained_machine_review_draft"
    draft["segments"] = transformed_segments
    draft["boundary_transform_applied"] = True
    draft["machine_suggestion"] = True
    draft["human_review_required"] = True
    draft["status"] = MACHINE_REVIEW_DRAFT_STATUS
    draft["product_accepted"] = False
    draft["provenance"] = provenance
    review = dict(draft.get("review")) if isinstance(draft.get("review"), Mapping) else {}
    review.update({"status": PENDING_HUMAN_PHONE_REVIEW_STATUS, "machine_suggestion": True})
    draft["review"] = review
    return draft


def convert_mfa_alignment_review_draft(
    payload: Mapping[str, Any],
    *,
    vocal_stem: Path,
    timeline: LyricsTimeline,
    model_hash: str,
    model_revision: str,
    dictionary_path: Path | str | None = None,
    alignment_sha256: str | None = None,
    timeline_constrained: bool = False,
) -> dict[str, Any]:
    """Convert raw MFA JSON into an explicit pending human-review artifact.

    ``timeline_constrained=False`` preserves raw MFA phone intervals.  The
    constrained variant maps each phone inside its hash-bound lyric item while
    retaining raw boundaries in provenance.
    """

    builder = (
        build_mfa_timeline_constrained_review_draft
        if timeline_constrained
        else build_mfa_full_song_review_draft
    )
    return builder(
        payload,
        vocal_stem=vocal_stem,
        timeline=timeline,
        model_hash=model_hash,
        model_revision=model_revision,
        dictionary_path=dictionary_path,
        alignment_sha256=alignment_sha256,
    )


def convert_mfa_alignment(
    payload: Mapping[str, Any],
    *,
    vocal_stem: Path,
    timeline: LyricsTimeline,
    resolved_items: Sequence[ResolvedLyricsItem],
    model_hash: str,
    model_revision: str,
    source_offset_sec: float = 0.0,
    manifest: _ManifestInput | None = None,
    trusted: bool | None = None,
    dictionary_path: Path | str | None = None,
    alignment_sha256: str | None = None,
) -> ForcedAlignmentResult:
    """Convert one MFA JSON result without inventing timing or phone evidence.

    A manifest (or an explicit ``trusted=True`` request) is required for
    product/model output.  Calls without a manifest remain available for
    fixture/debug use, but are returned as untrusted fixture evidence.
    """
    if not isinstance(payload, Mapping):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA alignment JSON must be an object")
    if not vocal_stem.is_file():
        raise ForcedPhoneAlignerError("missing_vocal_stem_evidence", f"missing vocal stem: {vocal_stem}")
    if not isinstance(model_hash, str) or len(model_hash) != 64 or any(c not in "0123456789abcdefABCDEF" for c in model_hash):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA model_hash must be SHA-256")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA model_revision must be non-empty")
    revision = model_revision.strip()
    offset = _finite(source_offset_sec, "source_offset_sec")
    mapping = _mapping_payload()
    stem_hash = sha256_file(vocal_stem)
    timeline_hash = lyrics_timeline_sha256(timeline)
    map_schema = str(mapping.get("schema_version"))
    map_hash = phone_map_sha256()
    alignment_hash = (
        _normalise_sha256(alignment_sha256, label="alignment_sha256")
        if alignment_sha256 is not None
        else alignment_json_sha256(payload)
    )
    dictionary_hash: str | None = None
    if dictionary_path is not None:
        dictionary = Path(dictionary_path)
        if not dictionary.is_file():
            raise ForcedPhoneAlignerError(
                "missing_phone_evidence", f"missing MFA dictionary: {dictionary}"
            )
        dictionary_hash = sha256_file(dictionary)
    manifest_payload = _manifest_payload(manifest)
    trusted_mode = bool(trusted) if trusted is not None else manifest_payload is not None
    _validate_manifest(
        manifest_payload,
        trusted=trusted_mode,
        alignment_hash=alignment_hash,
        vocal_stem_hash=stem_hash,
        timeline_hash=timeline_hash,
        model_hash=model_hash.lower(),
        model_revision=revision,
        dictionary_hash=dictionary_hash,
        map_schema=map_schema,
        map_hash=map_hash,
    )
    _validate_input_metadata(
        payload,
        vocal_stem_hash=stem_hash,
        timeline_hash=timeline_hash,
        model_hash=model_hash,
        model_revision=revision,
        map_schema=map_schema,
        map_hash=map_hash,
        source_offset_sec=offset,
    )
    alignment_transform = _alignment_transform_provenance(
        payload, manifest_payload, trusted=trusted_mode
    )
    alignment_bounds = _alignment_bounds(payload)
    words = _entries(payload, "words")
    phones = _entries(payload, "phones")
    if alignment_bounds is not None:
        bound_start, bound_end = alignment_bounds
        for word_index, (word_start, word_end, _label) in enumerate(words):
            if word_start < bound_start - _TIMING_TOLERANCE_SEC or word_end > bound_end + _TIMING_TOLERANCE_SEC:
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    f"MFA word entry {word_index} lies outside alignment bounds",
                )
    timing_comparison = _timing_comparison(
        [entry for entry in words if entry[2] not in _SILENCE_WORD_LABELS],
        resolved_items,
        source_offset_sec=offset,
    )
    matches = _word_matches(
        words,
        resolved_items,
        source_offset_sec=offset,
        timing_comparison=timing_comparison,
    )
    _validate_resolved_timeline(timeline, resolved_items)
    owners, _assigned_phone_indexes = _phone_owners(
        words,
        phones,
        alignment_bounds=alignment_bounds,
        mapping=mapping,
    )
    segments: list[dict[str, Any]] = []
    for word_index, (lyric_index, word_start, word_end) in enumerate(matches):
        owned_phones = owners.get(word_index, [])
        if not owned_phones:
            raise ForcedPhoneAlignerError(
                "missing_phone_evidence",
                f"MFA phone tier has no usable phone overlap for lyric item {lyric_index}",
            )
        for start, end, source_phone, canonical, direct_ratio in owned_phones:
            absolute_start = start + offset
            absolute_end = end + offset
            overlap = max(0.0, min(end, word_end) - max(start, word_start))
            segments.append(
                {
                    "start_sec": absolute_start,
                    "end_sec": absolute_end,
                    "normalized_ipa": canonical,
                    "candidate_phones": [{"symbol": canonical, "confidence": 0.0}],
                    "confidence": 0.0,
                    "metadata": {
                        "alignment_source": _ALIGNMENT_SOURCE,
                        "lyric_item_index": lyric_index,
                        "direct_overlap_sec": overlap,
                        "phone_overlap_ratio": direct_ratio,
                        "alignment_confidence": 0.0,
                        "model_hash": model_hash.lower(),
                        "source_mode": "real" if trusted_mode else "fixture",
                        "truth_layer": "phones_ipa_articulatory_features",
                        "interpolation_reason": None,
                        "source_phone": source_phone,
                        "source_phone_set": _MODEL_NAME,
                        "phone_map_schema": map_schema,
                        "phone_map_sha256": map_hash,
                        "alignment_confidence_source": "MFA_JSON_has_no_posterior_confidence",
                    },
                }
            )
    forced_provenance = {
        "mfa_schema_version": MFA_SCHEMA_VERSION,
        "timing_policy": MFA_TIMING_POLICY.to_dict(),
        "timing_comparison": timing_comparison,
        "phone_map_schema": map_schema,
        "phone_map_sha256": map_hash,
        "source_offset_sec": offset,
        "source_audio_hash_bound": True,
        "alignment_json_sha256": alignment_hash,
        "dictionary_sha256": (
            dictionary_hash
            or (_manifest_hash(
                manifest_payload,
                flat_aliases=("dictionary_sha256", "dictionary_hash"),
                nested_container="dictionary",
                nested_aliases=("sha256", "hash"),
                label="dictionary_sha256",
                required=False,
            ) if manifest_payload is not None else None)
        ),
        "manifest_schema": (
            MFA_MANIFEST_SCHEMA_VERSION if manifest_payload is not None else None
        ),
        "alignment_trust": "trusted_manifest" if trusted_mode else "untrusted_fixture",
    }
    if alignment_transform is not None:
        forced_provenance["alignment_transform"] = alignment_transform
    forced_payload = {
        "schema_version": "forced_phone_alignment_v1",
        "vocal_stem_sha256": stem_hash,
        "lyrics_timeline_sha256": timeline_hash,
        "alignment_source": _ALIGNMENT_SOURCE,
        "model_output": trusted_mode,
        "model": _model_metadata(revision=revision, model_hash=model_hash),
        "model_hash": model_hash.lower(),
        "provenance": forced_provenance,
        "timing_policy": MFA_TIMING_POLICY.to_dict(),
        "segments": segments,
    }
    if not trusted_mode:
        fixture_segments: list[PhoneSegment] = []
        duration = _wav_duration(vocal_stem)
        if duration is None:
            raise ForcedPhoneAlignerError(
                "invalid_forced_phone_evidence", "vocal stem must be a readable PCM WAV"
            )
        previous_end = -1.0
        for raw_segment in segments:
            start_sec = float(raw_segment["start_sec"])
            end_sec = float(raw_segment["end_sec"])
            if start_sec < previous_end or end_sec > duration + _TIMING_TOLERANCE_SEC:
                raise ForcedPhoneAlignerError(
                    "invalid_forced_phone_evidence",
                    "MFA fixture segment has invalid or overlapping timing",
                )
            raw_metadata = dict(raw_segment["metadata"])
            raw_metadata.update(
                {
                    "source_mode": "fixture",
                    "model_output": False,
                    "trusted": False,
                }
            )
            fixture_segments.append(
                PhoneSegment(
                    time_range=TimeRange(start_sec, end_sec),
                    candidate_phones=[PhoneCandidate(symbol=str(raw_segment["normalized_ipa"]), confidence=0.0)],
                    confidence=0.0,
                    normalized_ipa=str(raw_segment["normalized_ipa"]),
                    metadata=raw_metadata,
                )
            )
            previous_end = end_sec
        fixture_metadata = {
            "alignment_source": _ALIGNMENT_SOURCE,
            "source_mode": "fixture",
            "timing_policy": MFA_TIMING_POLICY.to_dict(),
            "timing_comparison": timing_comparison,
            "model_output": False,
            "trusted": False,
            "alignment_trust": "untrusted_fixture",
            "manifest_present": manifest_payload is not None,
            "vocal_stem_sha256": stem_hash,
            "lyrics_audio_sha256": timeline.audio_sha256,
            "lyrics_timeline_sha256": timeline_hash,
            "model": _model_metadata(revision=revision, model_hash=model_hash),
            "model_hash": model_hash.lower(),
            "segment_count": len(fixture_segments),
            "lyric_item_count": len(resolved_items),
            "covered_lyric_item_count": len({
                int(segment.metadata["lyric_item_index"]) for segment in fixture_segments
            }),
            "truth_layer": "phones_ipa_articulatory_features",
            "provenance": dict(forced_payload["provenance"]),
        }
        return ForcedAlignmentResult(tuple(fixture_segments), fixture_metadata)
    result = parse_forced_phone_alignment(
        forced_payload,
        vocal_stem=vocal_stem,
        timeline=timeline,
        resolved_items=resolved_items,
    )
    metadata = {
        **dict(result.metadata),
        "timing_policy": MFA_TIMING_POLICY.to_dict(),
        "timing_comparison": timing_comparison,
        "trusted": True,
        "model_output": True,
        "alignment_trust": "trusted_manifest",
        "manifest": dict(manifest_payload or {}),
    }
    return ForcedAlignmentResult(result.segments, metadata)


def load_and_convert_mfa_alignment(
    alignment_path: Path | str,
    **kwargs: Any,
) -> ForcedAlignmentResult:
    path = Path(alignment_path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", f"invalid MFA alignment: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ForcedPhoneAlignerError("invalid_forced_phone_evidence", "MFA alignment JSON must be an object")
    kwargs.setdefault("alignment_sha256", hashlib.sha256(raw).hexdigest())
    return convert_mfa_alignment(payload, **kwargs)


class MfaForcedPhoneAligner:
    """ForcedPhoneAligner implementation for an explicit MFA JSON artifact."""

    def __init__(
        self,
        alignment_path: Path | str,
        *,
        model_hash: str,
        model_revision: str,
        source_offset_sec: float = 0.0,
        manifest: _ManifestInput | None = None,
        trusted: bool | None = True,
        dictionary_path: Path | str | None = None,
    ) -> None:
        self.alignment_path = Path(alignment_path)
        self.model_hash = model_hash
        self.model_revision = model_revision
        self.source_offset_sec = source_offset_sec
        self.manifest = manifest
        self.trusted = trusted
        self.dictionary_path = Path(dictionary_path) if dictionary_path is not None else None

    def align(
        self,
        vocal_stem: Path,
        timeline: LyricsTimeline,
        resolved_items: Sequence[ResolvedLyricsItem],
    ) -> ForcedAlignmentResult:
        return load_and_convert_mfa_alignment(
            self.alignment_path,
            vocal_stem=vocal_stem,
            timeline=timeline,
            resolved_items=resolved_items,
            model_hash=self.model_hash,
            model_revision=self.model_revision,
            source_offset_sec=self.source_offset_sec,
            manifest=self.manifest,
            trusted=self.trusted,
            dictionary_path=self.dictionary_path,
        )


__all__ = [
    "MFA_MANIFEST_SCHEMA_VERSION",
    "MFA_PHONE_REVIEW_DRAFT_SCHEMA_VERSION",
    "MFA_SCHEMA_VERSION",
    "MFA_TIMING_POLICY",
    "MFA_TIMING_POLICY_SCHEMA_VERSION",
    "MfaForcedPhoneAligner",
    "MfaTimingPolicy",
    "alignment_json_sha256",
    "build_mfa_full_song_review_draft",
    "build_mfa_timeline_constrained_review_draft",
    "compare_mfa_timing",
    "convert_mfa_alignment",
    "convert_mfa_alignment_review_draft",
    "load_and_convert_mfa_alignment",
    "phone_map_sha256",
]

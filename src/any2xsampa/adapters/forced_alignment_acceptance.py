"""Acceptance evidence for product forced-phone alignment.

The alignment artifact and its acceptance record are deliberately separate.
This prevents an aligner from declaring its own output trusted.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from any2xsampa.core.models import PhoneSegment
from any2xsampa.lyrics.timeline import sha256_file

SCHEMA_VERSION = "forced_phone_acceptance_v1"
RESOURCE_MANIFEST_SCHEMA_VERSION = "forced_phone_model_manifest_v1"
QUALIFIED_MODEL_ROUTE = "qualified_cantonese_singing_model"
HUMAN_REVIEW_ROUTE = "human_reviewed_phone_alignment"
SHORT_CLIP_SCOPE = "short_clip"
FULL_SONG_SCOPE = "full_song"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANTONESE_NAMES = {"cantonese", "hong kong cantonese", "yue", "yue-hant-hk"}
_SINGING_CLAIM = "trained_or_finetuned_on_cantonese_singing"


class ForcedPhoneAcceptanceError(ValueError):
    """A strict product alignment has no qualifying acceptance evidence."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.error_kind = "unqualified_forced_phone_evidence"
        self.details = dict(details or {})


def canonical_phone_segments_sha256(segments: Sequence[PhoneSegment]) -> str:
    payload = [segment.to_dict() for segment in segments]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_phone_inventory_sha256(symbols: Sequence[str]) -> str:
    encoded = json.dumps(
        list(symbols),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_forced_phone_acceptance(
    path: Path,
    *,
    alignment_path: Path,
    vocal_stem: Path,
    model_hash: str,
    segments: Sequence[PhoneSegment],
) -> dict[str, Any]:
    """Load and validate a detached acceptance record for one alignment."""

    acceptance_path = Path(path)
    try:
        payload = json.loads(acceptance_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ForcedPhoneAcceptanceError(f"missing forced-phone acceptance record: {acceptance_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForcedPhoneAcceptanceError(f"invalid forced-phone acceptance record: {acceptance_path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ForcedPhoneAcceptanceError(f"acceptance schema must be {SCHEMA_VERSION}")

    route = _required_text(payload.get("route"), label="acceptance route")
    normalized = _validate_common(
        payload,
        acceptance_path=acceptance_path,
        alignment_path=Path(alignment_path),
        vocal_stem=Path(vocal_stem),
        segments=segments,
        require_alignment_path=route == HUMAN_REVIEW_ROUTE,
    )
    if route == QUALIFIED_MODEL_ROUTE:
        normalized["model"] = _validate_model_route(
            payload.get("model"),
            model_hash=model_hash,
            acceptance_path=acceptance_path,
            segments=segments,
            phone_inventory=normalized["phone_inventory"],
        )
    elif route == HUMAN_REVIEW_ROUTE:
        normalized["review"] = _validate_human_route(
            payload.get("review"),
            payload=payload,
            acceptance_path=acceptance_path,
            vocal_stem=Path(vocal_stem),
            segments=segments,
        )
    else:
        raise ForcedPhoneAcceptanceError(f"unsupported forced-phone acceptance route: {route}")
    normalized.update(
        {
            "schema_version": SCHEMA_VERSION,
            "route": route,
            "acceptance_artifact_path": str(acceptance_path),
            "acceptance_artifact_sha256": sha256_file(acceptance_path),
        }
    )
    return normalized


def validate_short_clip_predecessor(
    path: Path | str,
    *,
    vocal_stem_sha256: str,
) -> dict[str, Any]:
    """Validate a short human acceptance before opening full-song review."""

    acceptance_path = Path(path).expanduser().resolve()
    stem_hash = _sha(vocal_stem_sha256, label="vocal_stem_sha256")
    if not acceptance_path.is_file():
        raise ForcedPhoneAcceptanceError("approved short-clip predecessor is missing")
    _validate_short_clip_predecessor(
        {
            "acceptance_path": str(acceptance_path),
            "acceptance_sha256": sha256_file(acceptance_path),
        },
        current_payload={"vocal_stem_sha256": stem_hash},
        current_acceptance_path=acceptance_path.parent / "full_song_acceptance.json",
    )
    try:
        payload = json.loads(acceptance_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForcedPhoneAcceptanceError("short-clip predecessor is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ForcedPhoneAcceptanceError("short-clip predecessor must be an object")
    return dict(payload)


def _validate_common(
    payload: Mapping[str, Any],
    *,
    acceptance_path: Path,
    alignment_path: Path,
    vocal_stem: Path,
    segments: Sequence[PhoneSegment],
    require_alignment_path: bool,
) -> dict[str, Any]:
    actual_alignment_hash = sha256_file(alignment_path)
    declared_alignment_hash = _sha(payload.get("alignment_artifact_sha256"), label="alignment_artifact_sha256")
    if declared_alignment_hash != actual_alignment_hash:
        raise ForcedPhoneAcceptanceError(
            "acceptance record is bound to a different alignment artifact",
            details={"declared": declared_alignment_hash, "actual": actual_alignment_hash},
        )
    declared_alignment_path = payload.get("alignment_artifact_path")
    normalized_alignment_path: str | None = None
    if declared_alignment_path is not None or require_alignment_path:
        raw_alignment_path = _required_text(
            declared_alignment_path,
            label="alignment_artifact_path",
        )
        referenced_alignment = Path(raw_alignment_path)
        if not referenced_alignment.is_absolute():
            referenced_alignment = acceptance_path.parent / referenced_alignment
        try:
            referenced_alignment = referenced_alignment.resolve()
            supplied_alignment = alignment_path.resolve()
        except OSError as exc:
            raise ForcedPhoneAcceptanceError(
                "acceptance alignment artifact path cannot be resolved"
            ) from exc
        if referenced_alignment != supplied_alignment:
            raise ForcedPhoneAcceptanceError(
                "acceptance record points to a different alignment artifact"
            )
        normalized_alignment_path = str(referenced_alignment)
    actual_audio_hash = sha256_file(vocal_stem)
    declared_audio_hash = _sha(payload.get("vocal_stem_sha256"), label="vocal_stem_sha256")
    if declared_audio_hash != actual_audio_hash:
        raise ForcedPhoneAcceptanceError(
            "acceptance record is bound to a different vocal stem",
            details={"declared": declared_audio_hash, "actual": actual_audio_hash},
        )
    actual_segments_hash = canonical_phone_segments_sha256(segments)
    declared_segments_hash = _sha(payload.get("phone_segments_sha256"), label="phone_segments_sha256")
    if declared_segments_hash != actual_segments_hash:
        raise ForcedPhoneAcceptanceError(
            "acceptance record is bound to different parsed phone boundaries",
            details={"declared": declared_segments_hash, "actual": actual_segments_hash},
        )
    inventory = _validate_inventory(payload.get("phone_inventory"), segments=segments)
    result = {
        "alignment_artifact_sha256": actual_alignment_hash,
        "vocal_stem_sha256": actual_audio_hash,
        "phone_segments_sha256": actual_segments_hash,
        "phone_inventory": inventory,
    }
    if normalized_alignment_path is not None:
        result["alignment_artifact_path"] = normalized_alignment_path
    return result


def _validate_inventory(raw: object, *, segments: Sequence[PhoneSegment]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ForcedPhoneAcceptanceError("phone_inventory must be an object")
    raw_symbols = raw.get("symbols")
    if not isinstance(raw_symbols, Sequence) or isinstance(raw_symbols, (str, bytes)):
        raise ForcedPhoneAcceptanceError("phone_inventory.symbols must be a list")
    symbols = [_required_text(value, label="phone inventory symbol") for value in raw_symbols]
    if not symbols or len(set(symbols)) != len(symbols):
        raise ForcedPhoneAcceptanceError("phone_inventory.symbols must be non-empty and unique")
    declared_hash = _sha(raw.get("sha256"), label="phone_inventory.sha256")
    actual_hash = canonical_phone_inventory_sha256(symbols)
    if declared_hash != actual_hash:
        raise ForcedPhoneAcceptanceError("phone_inventory.sha256 does not match its symbols")
    used = {segment.normalized_ipa.strip() for segment in segments}
    missing = sorted(used - set(symbols))
    if missing:
        raise ForcedPhoneAcceptanceError(
            "parsed phones are missing from the declared phone inventory",
            details={"missing_phones": missing},
        )
    return {
        "symbols": symbols,
        "sha256": actual_hash,
        "source": _required_text(raw.get("source"), label="phone_inventory.source"),
    }


def _validate_model_route(
    raw: object,
    *,
    model_hash: str,
    acceptance_path: Path,
    segments: Sequence[PhoneSegment],
    phone_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ForcedPhoneAcceptanceError("qualified model route requires model evidence")
    declared_hash = _sha(raw.get("sha256"), label="model.sha256")
    if declared_hash != str(model_hash).strip().lower():
        raise ForcedPhoneAcceptanceError("acceptance model SHA-256 does not match alignment provenance")
    revision = _required_text(raw.get("revision"), label="model.revision")
    if raw.get("revision_is_immutable") is not True:
        raise ForcedPhoneAcceptanceError("model revision must explicitly be immutable")
    license_evidence = _validate_model_license(raw.get("license"))
    training = _validate_cantonese_singing_training(raw.get("cantonese_singing_training"))
    manifest_path = _resolve_file(
        raw.get("resource_manifest_path"),
        base_dir=acceptance_path.parent,
        label="model.resource_manifest_path",
    )
    manifest_hash = _sha(
        raw.get("resource_manifest_sha256"),
        label="model.resource_manifest_sha256",
    )
    actual_manifest_hash = sha256_file(manifest_path)
    if manifest_hash != actual_manifest_hash:
        raise ForcedPhoneAcceptanceError(
            "qualified model resource manifest SHA-256 does not match",
            details={"declared": manifest_hash, "actual": actual_manifest_hash},
        )
    resource_manifest = _load_resource_manifest(
        manifest_path,
        model_hash=declared_hash,
        segments=segments,
    )
    manifest_inventory = resource_manifest["phone_inventory"]
    if (
        manifest_inventory["symbols"] != phone_inventory["symbols"]
        or manifest_inventory["sha256"] != phone_inventory["sha256"]
    ):
        raise ForcedPhoneAcceptanceError(
            "qualified model resource manifest phone inventory does not match acceptance evidence"
        )
    manifest_model = resource_manifest["artifacts"]["model_weights"]
    if manifest_model["revision"] != revision:
        raise ForcedPhoneAcceptanceError(
            "qualified model resource manifest revision does not match model evidence"
        )
    manifest_training = resource_manifest["training_evidence"]
    training_fields = (
        "claim",
        "language",
        "domain",
        "datasets",
        "source_type",
        "source",
        "source_revision",
        "source_sha256",
    )
    if any(manifest_training.get(field) != training.get(field) for field in training_fields):
        raise ForcedPhoneAcceptanceError(
            "qualified model resource manifest training evidence does not match model evidence"
        )
    manifest_license = manifest_model["license"]
    license_fields = (
        "identifier",
        "applies_to",
        "noncommercial_restriction",
        "commercial_use",
        "source",
        "source_sha256",
    )
    if any(manifest_license.get(field) != license_evidence.get(field) for field in license_fields):
        raise ForcedPhoneAcceptanceError(
            "qualified model resource manifest license evidence does not match model evidence"
        )
    return {
        "name": _required_text(raw.get("name"), label="model.name"),
        "revision": revision,
        "revision_is_immutable": True,
        "sha256": declared_hash,
        "license": license_evidence,
        "cantonese_singing_training": training,
        "resource_manifest_path": str(manifest_path),
        "resource_manifest_sha256": actual_manifest_hash,
        "resource_manifest": resource_manifest,
    }


def _resolve_file(value: object, *, base_dir: Path, label: str) -> Path:
    raw = _required_text(value, label=label)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ForcedPhoneAcceptanceError(f"{label} must point to an existing file") from exc
    if not resolved.is_file():
        raise ForcedPhoneAcceptanceError(f"{label} must point to an existing file")
    return resolved


def _load_resource_manifest(
    path: Path,
    *,
    model_hash: str,
    segments: Sequence[PhoneSegment],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForcedPhoneAcceptanceError("qualified model resource manifest is invalid") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != RESOURCE_MANIFEST_SCHEMA_VERSION:
        raise ForcedPhoneAcceptanceError(
            f"resource manifest schema must be {RESOURCE_MANIFEST_SCHEMA_VERSION}"
        )
    raw_required = payload.get("required_artifacts")
    if not isinstance(raw_required, Sequence) or isinstance(raw_required, (str, bytes)):
        raise ForcedPhoneAcceptanceError("resource manifest required_artifacts must be a list")
    required = [str(value).strip() for value in raw_required]
    if len(set(required)) != len(required) or any(not value for value in required):
        raise ForcedPhoneAcceptanceError("resource manifest required_artifacts must be unique")
    mandatory = {"model_weights", "dictionary", "phone_map"}
    if not mandatory.issubset(required):
        raise ForcedPhoneAcceptanceError(
            "resource manifest must require model_weights, dictionary, and phone_map"
        )
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ForcedPhoneAcceptanceError("resource manifest artifacts must be an object")
    artifacts: dict[str, Any] = {}
    for kind in required:
        raw_artifact = raw_artifacts.get(kind)
        if not isinstance(raw_artifact, Mapping):
            raise ForcedPhoneAcceptanceError(f"resource manifest is missing artifact: {kind}")
        artifact = _validate_resource_artifact(
            raw_artifact,
            kind=kind,
            base_dir=path.parent,
        )
        artifacts[kind] = artifact
    if artifacts["model_weights"]["sha256"] != model_hash:
        raise ForcedPhoneAcceptanceError(
            "resource manifest model_weights SHA-256 does not match model evidence"
        )
    inventory = _validate_inventory(payload.get("phone_inventory"), segments=segments)
    phone_map = artifacts["phone_map"]
    if phone_map.get("phone_inventory_sha256") != inventory["sha256"]:
        raise ForcedPhoneAcceptanceError(
            "phone-map artifact is not bound to the declared phone inventory"
        )
    training = _validate_training_evidence_file(
        payload.get("training_evidence"),
        base_dir=path.parent,
    )
    return {
        "schema_version": RESOURCE_MANIFEST_SCHEMA_VERSION,
        "required_artifacts": required,
        "artifacts": artifacts,
        "phone_inventory": inventory,
        "training_evidence": training,
    }


def _validate_resource_artifact(
    raw: Mapping[str, Any],
    *,
    kind: str,
    base_dir: Path,
) -> dict[str, Any]:
    artifact_path = _resolve_file(raw.get("path"), base_dir=base_dir, label=f"artifact.{kind}.path")
    digest = _sha(raw.get("sha256"), label=f"artifact.{kind}.sha256")
    actual = sha256_file(artifact_path)
    if digest != actual:
        raise ForcedPhoneAcceptanceError(
            f"artifact.{kind}.sha256 does not match the file",
            details={"declared": digest, "actual": actual},
        )
    revision = _required_text(raw.get("revision"), label=f"artifact.{kind}.revision")
    if raw.get("revision_is_immutable") is not True:
        raise ForcedPhoneAcceptanceError(f"artifact.{kind}.revision must be immutable")
    license_evidence = _validate_resource_license(
        raw.get("license"),
        kind=kind,
        base_dir=base_dir,
    )
    normalized: dict[str, Any] = {
        "path": str(artifact_path),
        "sha256": actual,
        "revision": revision,
        "revision_is_immutable": True,
        "license": license_evidence,
    }
    if kind == "phone_map":
        normalized["phone_inventory_sha256"] = _sha(
            raw.get("phone_inventory_sha256"),
            label="artifact.phone_map.phone_inventory_sha256",
        )
    return normalized


def _validate_resource_license(
    raw: object,
    *,
    kind: str,
    base_dir: Path,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ForcedPhoneAcceptanceError(f"artifact.{kind}.license is required")
    identifier = _required_text(raw.get("identifier"), label=f"artifact.{kind}.license.identifier")
    lowered = identifier.casefold()
    if "noncommercial" in lowered or re.search(r"(^|[-_ ])nc($|[-_ ])", lowered):
        raise ForcedPhoneAcceptanceError(f"artifact.{kind} license contains a NonCommercial restriction")
    if raw.get("noncommercial_restriction") is not False:
        raise ForcedPhoneAcceptanceError(
            f"artifact.{kind} license must explicitly declare no NC restriction"
        )
    if str(raw.get("commercial_use") or "").strip().lower() != "allowed":
        raise ForcedPhoneAcceptanceError(f"artifact.{kind} license must allow commercial use")
    if str(raw.get("applies_to") or "").strip().lower() != kind:
        raise ForcedPhoneAcceptanceError(f"artifact.{kind} license must apply to {kind}")
    source = _required_text(raw.get("source"), label=f"artifact.{kind}.license.source")
    source_hash = _sha(
        raw.get("source_sha256"),
        label=f"artifact.{kind}.license.source_sha256",
    )
    source_path = _resolve_file(
        raw.get("source_path"),
        base_dir=base_dir,
        label=f"artifact.{kind}.license.source_path",
    )
    actual_source_hash = sha256_file(source_path)
    if source_hash != actual_source_hash:
        raise ForcedPhoneAcceptanceError(
            f"artifact.{kind} license source SHA-256 does not match the file"
        )
    return {
        "identifier": identifier,
        "applies_to": kind,
        "noncommercial_restriction": False,
        "commercial_use": "allowed",
        "source": source,
        "source_path": str(source_path),
        "source_sha256": actual_source_hash,
    }


def _validate_training_evidence_file(raw: object, *, base_dir: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ForcedPhoneAcceptanceError("resource manifest training_evidence is required")
    path = _resolve_file(raw.get("path"), base_dir=base_dir, label="training_evidence.path")
    digest = _sha(raw.get("sha256"), label="training_evidence.sha256")
    actual = sha256_file(path)
    if digest != actual:
        raise ForcedPhoneAcceptanceError("training evidence SHA-256 does not match the file")
    normalized = _validate_cantonese_singing_training(raw)
    source_path = _resolve_file(
        raw.get("source_path"),
        base_dir=base_dir,
        label="training_evidence.source_path",
    )
    source_hash = _sha(raw.get("source_sha256"), label="training_evidence.source_sha256")
    if source_hash != sha256_file(source_path):
        raise ForcedPhoneAcceptanceError(
            "training evidence primary source SHA-256 does not match the file"
        )
    normalized["path"] = str(path)
    normalized["sha256"] = actual
    normalized["source_path"] = str(source_path)
    normalized["source_sha256"] = source_hash
    return normalized


def _validate_model_license(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ForcedPhoneAcceptanceError("model license evidence is required")
    identifier = _required_text(raw.get("identifier"), label="model.license.identifier")
    lowered = identifier.casefold()
    if "noncommercial" in lowered or re.search(r"(^|[-_ ])nc($|[-_ ])", lowered):
        raise ForcedPhoneAcceptanceError("model license contains a NonCommercial restriction")
    if raw.get("noncommercial_restriction") is not False:
        raise ForcedPhoneAcceptanceError("model license must explicitly declare no NC restriction")
    if str(raw.get("commercial_use") or "").strip().lower() != "allowed":
        raise ForcedPhoneAcceptanceError("model license must explicitly allow commercial use")
    if str(raw.get("applies_to") or "").strip().lower() != "model_weights":
        raise ForcedPhoneAcceptanceError("license evidence must explicitly apply to model weights")
    return {
        "identifier": identifier,
        "applies_to": "model_weights",
        "noncommercial_restriction": False,
        "commercial_use": "allowed",
        "source": _required_text(raw.get("source"), label="model.license.source"),
        "source_sha256": _sha(raw.get("source_sha256"), label="model.license.source_sha256"),
    }


def _validate_cantonese_singing_training(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ForcedPhoneAcceptanceError("Cantonese singing training evidence is required")
    claim = str(raw.get("claim") or "").strip().lower()
    language = str(raw.get("language") or "").strip().lower()
    domain = str(raw.get("domain") or "").strip().lower()
    if claim != _SINGING_CLAIM or language not in _CANTONESE_NAMES or domain != "singing_voice":
        raise ForcedPhoneAcceptanceError("model training evidence must explicitly establish Cantonese singing training")
    if str(raw.get("source_type") or "").strip().lower() != "primary":
        raise ForcedPhoneAcceptanceError("Cantonese singing training evidence must be a primary source")
    datasets = raw.get("datasets")
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)) or not datasets:
        raise ForcedPhoneAcceptanceError("Cantonese singing training evidence must name its dataset(s)")
    return {
        "claim": _SINGING_CLAIM,
        "language": language,
        "domain": "singing_voice",
        "datasets": [_required_text(value, label="training dataset") for value in datasets],
        "source_type": "primary",
        "source": _required_text(raw.get("source"), label="training evidence source"),
        "source_revision": _required_text(raw.get("source_revision"), label="training evidence source_revision"),
        "source_sha256": _sha(raw.get("source_sha256"), label="training evidence source_sha256"),
    }


def _validate_human_route(
    raw: object,
    *,
    payload: Mapping[str, Any],
    acceptance_path: Path,
    vocal_stem: Path,
    segments: Sequence[PhoneSegment],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ForcedPhoneAcceptanceError("human-reviewed route requires review evidence")
    if str(raw.get("review_status") or "").strip().lower() != "approved":
        raise ForcedPhoneAcceptanceError("human phone-boundary review must be approved")
    if str(raw.get("reviewer_kind") or "").strip().lower() != "human":
        raise ForcedPhoneAcceptanceError("phone-boundary reviewer must explicitly be human")
    if raw.get("phone_boundaries_reviewed") is not True:
        raise ForcedPhoneAcceptanceError("every phone boundary must be explicitly reviewed")
    reviewed_at = _review_timestamp(raw.get("reviewed_at"))
    start, end = _review_range(raw.get("reviewed_range_sec"))
    if not segments or min(segment.time_range.start_sec for segment in segments) < start - 1e-6:
        raise ForcedPhoneAcceptanceError("phone segments start outside the reviewed range")
    if max(segment.time_range.end_sec for segment in segments) > end + 1e-6:
        raise ForcedPhoneAcceptanceError("phone segments end outside the reviewed range")
    scope = _required_text(raw.get("scope"), label="review.scope")
    if scope == SHORT_CLIP_SCOPE:
        duration = end - start
        if duration < 10.0 or duration > 20.0:
            raise ForcedPhoneAcceptanceError("initial human-reviewed clip must be 10-20 seconds")
    elif scope == FULL_SONG_SCOPE:
        duration = _wav_duration(vocal_stem)
        if start > 0.05 or end < duration - 0.05:
            raise ForcedPhoneAcceptanceError("full-song review range does not cover the complete vocal stem")
        _validate_short_clip_predecessor(
            raw.get("short_clip_predecessor"),
            current_payload=payload,
            current_acceptance_path=acceptance_path,
        )
    else:
        raise ForcedPhoneAcceptanceError(f"unsupported human review scope: {scope}")
    return {
        "review_status": "approved",
        "reviewer_kind": "human",
        "reviewer_id": _required_text(raw.get("reviewer_id"), label="review.reviewer_id"),
        "reviewed_at": reviewed_at,
        "review_method": _required_text(raw.get("review_method"), label="review.review_method"),
        "phone_boundaries_reviewed": True,
        "scope": scope,
        "reviewed_range_sec": [start, end],
    }


def _validate_short_clip_predecessor(
    raw: object,
    *,
    current_payload: Mapping[str, Any],
    current_acceptance_path: Path,
) -> None:
    if not isinstance(raw, Mapping):
        raise ForcedPhoneAcceptanceError("full-song review requires an approved short-clip predecessor")
    raw_path = _required_text(raw.get("acceptance_path"), label="short_clip_predecessor.acceptance_path")
    predecessor_path = Path(raw_path)
    if not predecessor_path.is_absolute():
        predecessor_path = current_acceptance_path.parent / predecessor_path
    expected_hash = _sha(raw.get("acceptance_sha256"), label="short_clip_predecessor.acceptance_sha256")
    if not predecessor_path.is_file() or sha256_file(predecessor_path) != expected_hash:
        raise ForcedPhoneAcceptanceError("short-clip acceptance artifact is missing or has the wrong SHA-256")
    try:
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForcedPhoneAcceptanceError("short-clip acceptance artifact is invalid") from exc
    review = predecessor.get("review") if isinstance(predecessor, Mapping) else None
    if (
        not isinstance(predecessor, Mapping)
        or predecessor.get("schema_version") != SCHEMA_VERSION
        or predecessor.get("route") != HUMAN_REVIEW_ROUTE
        or predecessor.get("vocal_stem_sha256") != current_payload.get("vocal_stem_sha256")
        or not isinstance(review, Mapping)
        or str(review.get("review_status") or "").strip().lower() != "approved"
        or review.get("scope") != SHORT_CLIP_SCOPE
    ):
        raise ForcedPhoneAcceptanceError("predecessor is not an approved short-clip review for this vocal stem")
    start, end = _review_range(review.get("reviewed_range_sec"))
    if end - start < 10.0 or end - start > 20.0:
        raise ForcedPhoneAcceptanceError("predecessor review is not a 10-20 second clip")
    _validate_predecessor_alignment(
        predecessor,
        predecessor_path=predecessor_path,
        current_payload=current_payload,
    )


def _validate_predecessor_alignment(
    predecessor: Mapping[str, Any],
    *,
    predecessor_path: Path,
    current_payload: Mapping[str, Any],
) -> None:
    raw_alignment_path = _required_text(
        predecessor.get("alignment_artifact_path"),
        label="predecessor alignment_artifact_path",
    )
    alignment_path = Path(raw_alignment_path)
    if not alignment_path.is_absolute():
        alignment_path = predecessor_path.parent / alignment_path
    expected_alignment_hash = _sha(
        predecessor.get("alignment_artifact_sha256"),
        label="predecessor alignment_artifact_sha256",
    )
    if not alignment_path.is_file() or sha256_file(alignment_path) != expected_alignment_hash:
        raise ForcedPhoneAcceptanceError(
            "short-clip predecessor alignment artifact is missing or has the wrong SHA-256"
        )
    try:
        alignment = json.loads(alignment_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForcedPhoneAcceptanceError(
            "short-clip predecessor alignment artifact is invalid"
        ) from exc
    raw_segments = alignment.get("segments") if isinstance(alignment, Mapping) else None
    if (
        not isinstance(alignment, Mapping)
        or alignment.get("schema_version") != "forced_phone_alignment_v1"
        or alignment.get("vocal_stem_sha256") != current_payload.get("vocal_stem_sha256")
        or not isinstance(raw_segments, Sequence)
        or isinstance(raw_segments, (str, bytes))
        or not raw_segments
    ):
        raise ForcedPhoneAcceptanceError(
            "short-clip predecessor alignment artifact has invalid schema or audio binding"
        )
    normalized_segments = [
        {key: value for key, value in raw.items() if key != "reviewed"}
        if isinstance(raw, Mapping)
        else raw
        for raw in raw_segments
    ]
    encoded = json.dumps(
        normalized_segments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != predecessor.get("phone_segments_sha256"):
        raise ForcedPhoneAcceptanceError(
            "short-clip predecessor alignment has different phone boundaries"
        )


def _review_range(raw: object) -> tuple[float, float]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ForcedPhoneAcceptanceError("reviewed_range_sec must contain start and end")
    try:
        start, end = float(raw[0]), float(raw[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ForcedPhoneAcceptanceError("reviewed_range_sec must be numeric") from exc
    if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
        raise ForcedPhoneAcceptanceError("reviewed_range_sec is invalid")
    return start, end


def _review_timestamp(raw: object) -> str:
    value = _required_text(raw, label="review.reviewed_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForcedPhoneAcceptanceError("review.reviewed_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ForcedPhoneAcceptanceError("review.reviewed_at must include a timezone")
    return value


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                raise ForcedPhoneAcceptanceError("vocal stem sample rate must be positive")
            return handle.getnframes() / rate
    except (EOFError, OSError, wave.Error) as exc:
        raise ForcedPhoneAcceptanceError("vocal stem must be a readable PCM WAV") from exc


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ForcedPhoneAcceptanceError(f"{label} is required")
    return text


def _sha(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ForcedPhoneAcceptanceError(f"{label} must be a SHA-256 digest")
    return digest


__all__ = [
    "FULL_SONG_SCOPE",
    "HUMAN_REVIEW_ROUTE",
    "QUALIFIED_MODEL_ROUTE",
    "RESOURCE_MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SHORT_CLIP_SCOPE",
    "ForcedPhoneAcceptanceError",
    "canonical_phone_inventory_sha256",
    "canonical_phone_segments_sha256",
    "load_forced_phone_acceptance",
    "validate_short_clip_predecessor",
]

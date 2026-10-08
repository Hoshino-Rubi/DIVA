"""Validation for auditable phone-level training truth."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


HUMAN_LABEL_SCHEMA = "human_phone_label_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def validated_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def human_phone_truth(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated human provenance and phone truth from a corpus row."""

    provenance = row.get("label_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("training-ready human row requires label_provenance")
    if provenance.get("schema_version") != HUMAN_LABEL_SCHEMA:
        raise ValueError(f"human label schema must be {HUMAN_LABEL_SCHEMA}")
    if str(provenance.get("kind") or "").strip().lower() != "human":
        raise ValueError("human label provenance kind must be human")
    if str(provenance.get("review_status") or "").strip().lower() != "approved":
        raise ValueError("human label provenance must be approved")
    normalized_provenance = {
        "schema_version": HUMAN_LABEL_SCHEMA,
        "kind": "human",
        "source_id": required_text(provenance.get("source_id"), label="label source_id"),
        "source_revision": required_text(
            provenance.get("source_revision"), label="label source_revision"
        ),
        "annotation_sha256": validated_sha256(
            provenance.get("annotation_sha256"), label="annotation_sha256"
        ),
        "policy_version": required_text(
            provenance.get("policy_version"), label="label policy_version"
        ),
        "annotator_id": required_text(
            provenance.get("annotator_id"), label="label annotator_id"
        ),
        "review_status": "approved",
    }
    phones = _tokens(row.get("target_phone_tokens"), label="target_phone_tokens")
    ipa = _tokens(row.get("internal_ipa_tokens"), label="internal_ipa_tokens")
    raw_features = row.get("articulatory_features")
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        raise ValueError("articulatory_features must be a list")
    features = [dict(value) for value in raw_features if isinstance(value, Mapping) and value]
    if len(phones) != len(ipa) or len(phones) != len(features):
        raise ValueError("phones, IPA, and articulatory features must have equal non-zero length")
    return {
        "label_provenance": normalized_provenance,
        "target_phone_tokens": phones,
        "internal_ipa_tokens": ipa,
        "articulatory_features": features,
    }


def stable_identities(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: required_text(row.get(name), label=name)
        for name in ("singer_id", "song_id", "phrase_id")
    }


def optional_sha256(row: Mapping[str, Any], name: str) -> str | None:
    value = row.get(name)
    if value is None or not str(value).strip():
        return None
    return validated_sha256(value, label=name)


def _tokens(value: object, *, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list")
    tokens = [str(item).strip() for item in value]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{label} must contain non-empty values")
    return tokens


__all__ = [
    "HUMAN_LABEL_SCHEMA",
    "human_phone_truth",
    "optional_sha256",
    "required_text",
    "stable_identities",
    "validated_sha256",
]

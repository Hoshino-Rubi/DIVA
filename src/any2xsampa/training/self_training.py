"""Auditable pseudo-label selection and resumable self-training batches."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence
import uuid


SCHEMA_VERSION = "any2xsampa_self_training_batch_v1"
PSEUDO_LABEL_SCHEMA = "any2xsampa_pseudo_phone_label_v1"


class SelfTrainingError(ValueError):
    """Raised when a batch cannot satisfy truth/provenance constraints."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _digest(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise SelfTrainingError(f"{label} must be a SHA-256 digest")
    return text


def _confidence(value: object) -> float:
    if isinstance(value, bool):
        raise SelfTrainingError("pseudo-label confidence must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SelfTrainingError("pseudo-label confidence must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SelfTrainingError("pseudo-label confidence must be finite and between zero and one")
    return result


def _phones(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result = [str(token).strip() for token in value if str(token).strip()]
    return result if len(result) == len(value) else []


def _identity(candidate: Mapping[str, Any]) -> tuple[str, str, str]:
    singer = str(candidate.get("singer_id") or "").strip()
    song = str(candidate.get("song_id") or "").strip()
    phrase = str(candidate.get("phrase_id") or "").strip()
    if not singer or not song or not phrase:
        raise SelfTrainingError("self-training candidates require singer_id, song_id, and phrase_id")
    return singer, song, phrase


def _model_provenance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    value = candidate.get("model_provenance")
    if not isinstance(value, Mapping):
        raise SelfTrainingError("pseudo-label candidate is missing model_provenance")
    model_id = str(value.get("model_id") or "").strip()
    revision = str(value.get("revision") or "").strip()
    source = str(value.get("source") or "").strip()
    checkpoint_sha256 = _digest(value.get("checkpoint_sha256"), label="checkpoint_sha256")
    inference_config_sha256 = _digest(
        value.get("inference_config_sha256"), label="inference_config_sha256"
    )
    input_audio_sha256 = _digest(
        value.get("input_audio_sha256"), label="input_audio_sha256"
    )
    vocabulary_sha256 = _digest(
        value.get("vocabulary_sha256"), label="vocabulary_sha256"
    )
    calibration_sha256 = _digest(
        value.get("calibration_sha256"), label="calibration_sha256"
    )
    policy_version = str(value.get("policy_version") or "").strip()
    generated_at_utc = str(value.get("generated_at_utc") or "").strip()
    if not model_id or not revision or not source or not policy_version or not generated_at_utc:
        raise SelfTrainingError(
            "model_provenance requires model_id, revision, source, policy_version, and generated_at_utc"
        )
    return {
        "model_id": model_id,
        "revision": revision,
        "source": source,
        "checkpoint_sha256": checkpoint_sha256,
        "inference_config_sha256": inference_config_sha256,
        "input_audio_sha256": input_audio_sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "calibration_sha256": calibration_sha256,
        "policy_version": policy_version,
        "generated_at_utc": generated_at_utc,
    }


def _has_human_truth(candidate: Mapping[str, Any]) -> bool:
    if candidate.get("human_label") is not None:
        return True
    if any(
        key in candidate and candidate.get(key) is not None
        for key in (
            "manual_label",
            "manual_training_label",
            "manual_positive",
            "manual_negative",
        )
    ):
        return True
    for key in (
        "human_authored_annotation_evidence",
        "manual_label_locked",
        "phoneme_lock",
    ):
        value = candidate.get(key)
        if value is True or value == 1:
            return True
        if isinstance(value, str) and value.strip().casefold() in {
            "1",
            "true",
            "yes",
            "locked",
        }:
            return True
    return False


def _decision(candidate: Mapping[str, Any], *, min_confidence: float) -> tuple[str, dict[str, Any] | None]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        return "missing_candidate_id", None
    try:
        singer, song, phrase = _identity(candidate)
    except SelfTrainingError:
        return "missing_identity", None
    label_source = str(candidate.get("label_source") or "").strip().casefold()
    label_provenance = candidate.get("label_provenance")
    provenance_kind = (
        str(label_provenance.get("kind") or "").strip().casefold()
        if isinstance(label_provenance, Mapping)
        else ""
    )
    if (
        _has_human_truth(candidate)
        or label_source in {"human", "manual", "human_authored"}
        or provenance_kind in {"human", "manual", "human_authored"}
    ):
        return "human_truth_protected", None
    if candidate.get("model_output") is not True:
        return "missing_model_output", None
    phones = _phones(candidate.get("phone_hypothesis"))
    if not phones:
        return "missing_phone_hypothesis", None
    try:
        confidence = _confidence(candidate.get("confidence"))
    except SelfTrainingError:
        return "invalid_confidence", None
    if confidence < min_confidence:
        return "confidence_below_threshold", None
    try:
        provenance = _model_provenance(candidate)
    except SelfTrainingError:
        return "missing_model_provenance", None
    evidence = candidate.get("acoustic_evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        return "missing_acoustic_evidence", None
    normalized_ipa = _phones(candidate.get("normalized_ipa"))
    if len(normalized_ipa) != len(phones):
        return "invalid_truth_fields", None
    articulatory = candidate.get("articulatory_features")
    if (
        not isinstance(articulatory, list)
        or len(articulatory) != len(phones)
        or any(not isinstance(item, Mapping) or not item for item in articulatory)
    ):
        return "invalid_truth_fields", None
    label = {
        "schema_version": PSEUDO_LABEL_SCHEMA,
        "candidate_id": candidate_id,
        "example_id": candidate_id,
        "singer_id": singer,
        "song_id": song,
        "phrase_id": phrase,
        "group_id": f"{singer}/{song}/{phrase}",
        "phone_tokens": phones,
        "target_phone_tokens": phones,
        "normalized_ipa": normalized_ipa,
        "internal_ipa_tokens": normalized_ipa,
        "articulatory_features": [dict(item) for item in articulatory],
        "confidence": confidence,
        "label_source": "pseudo",
        "model_output": True,
        "pseudo_label": True,
        "model_provenance": provenance,
        "acoustic_evidence": dict(evidence),
        "asr_transcript": candidate.get("asr_transcript"),
        "asr_transcript_is_phone_truth": False,
        "human_truth_overwritten": False,
        "source_candidate_sha256": _sha(candidate),
    }
    label["pseudo_label_sha256"] = _sha(label)
    return "accepted", label


def _load_resume(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise SelfTrainingError("resume manifest has an unsupported schema")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise SelfTrainingError("resume manifest decisions must be a list")
    expected = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if expected != _sha(unsigned):
        raise SelfTrainingError("resume manifest SHA-256 mismatch")
    candidate_ids = [str(row.get("candidate_id") or "") for row in decisions if isinstance(row, Mapping)]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SelfTrainingError("resume manifest contains duplicate decisions")
    accepted_count = sum(row.get("status") == "accepted" for row in decisions if isinstance(row, Mapping))
    if payload.get("candidate_count") != len(decisions) or payload.get("accepted_count") != accepted_count:
        raise SelfTrainingError("resume manifest decision counts are inconsistent")
    if payload.get("rejected_count") != len(decisions) - accepted_count:
        raise SelfTrainingError("resume manifest rejection count is inconsistent")
    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in decisions:
        if not isinstance(row, Mapping):
            raise SelfTrainingError("resume manifest decisions must be objects")
        if row.get("status") == "accepted":
            if row.get("reason") is not None:
                raise SelfTrainingError("accepted resume decision cannot have a failure reason")
            label = _validate_pseudo_label(row.get("pseudo_label"), decision=row)
            accepted.append(label)
        else:
            reason = str(row.get("reason") or "").strip()
            if not reason or row.get("pseudo_label") is not None:
                raise SelfTrainingError("rejected resume decision is inconsistent")
            failures.append({"candidate_id": row.get("candidate_id"), "reason": reason})
        _digest(row.get("source_candidate_sha256"), label="source_candidate_sha256")
    accepted.sort(key=lambda row: str(row["candidate_id"]))
    failures.sort(key=lambda row: (str(row.get("candidate_id") or ""), str(row.get("reason") or "")))
    if payload.get("accepted_sha256") != _sha(accepted):
        raise SelfTrainingError("resume manifest accepted SHA-256 is inconsistent")
    if payload.get("failure_reasons") != failures:
        raise SelfTrainingError("resume manifest failure reasons are inconsistent")
    return payload


def _validate_pseudo_label(
    value: object,
    *,
    decision: Mapping[str, Any] | None = None,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != PSEUDO_LABEL_SCHEMA:
        raise SelfTrainingError("resume pseudo label has an unsupported schema")
    payload = dict(value)
    expected = _digest(payload.pop("pseudo_label_sha256", None), label="pseudo_label_sha256")
    if _sha(payload) != expected:
        raise SelfTrainingError("resume pseudo label SHA-256 mismatch")
    phones = _phones(payload.get("phone_tokens"))
    ipa = _phones(payload.get("normalized_ipa"))
    features = payload.get("articulatory_features")
    if (
        not phones
        or len(phones) != len(ipa)
        or not isinstance(features, list)
        or len(features) != len(phones)
        or any(not isinstance(item, Mapping) or not item for item in features)
    ):
        raise SelfTrainingError("resume pseudo label has incomplete phone truth")
    if (
        payload.get("label_source") != "pseudo"
        or payload.get("model_output") is not True
        or payload.get("pseudo_label") is not True
        or payload.get("human_truth_overwritten") is not False
    ):
        raise SelfTrainingError("resume pseudo label source is invalid")
    if payload.get("target_phone_tokens") != phones or payload.get("internal_ipa_tokens") != ipa:
        raise SelfTrainingError("resume pseudo label truth aliases disagree")
    candidate_id = str(payload.get("candidate_id") or "").strip()
    if not candidate_id or str(payload.get("example_id") or "").strip() != candidate_id:
        raise SelfTrainingError("resume pseudo label identity is invalid")
    singer, song, phrase = _identity(payload)
    if payload.get("group_id") != f"{singer}/{song}/{phrase}":
        raise SelfTrainingError("resume pseudo label group identity is invalid")
    source_sha256 = _digest(
        payload.get("source_candidate_sha256"), label="source_candidate_sha256"
    )
    if decision is not None and (
        str(decision.get("candidate_id") or "").strip() != candidate_id
        or decision.get("source_candidate_sha256") != source_sha256
    ):
        raise SelfTrainingError("resume pseudo label does not match its decision")
    _model_provenance({"model_provenance": payload.get("model_provenance")})
    confidence = _confidence(payload.get("confidence"))
    if min_confidence is not None and confidence < min_confidence:
        raise SelfTrainingError("resume pseudo label is below the active confidence threshold")
    return dict(value)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_self_training_batch(
    *,
    candidates: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    min_confidence: float = 0.9,
    seed: int = 1337,
    resume_manifest: str | Path | None = None,
) -> dict[str, Any]:
    if not 0.0 <= min_confidence <= 1.0:
        raise SelfTrainingError("min_confidence must be between zero and one")
    resume = _load_resume(Path(resume_manifest) if resume_manifest is not None else None)
    if resume is not None and (resume.get("seed") != seed or resume.get("min_confidence") != min_confidence):
        raise SelfTrainingError("resume seed and confidence threshold must match")
    previous = {str(row["candidate_id"]): dict(row) for row in (resume or {}).get("decisions", []) if isinstance(row, Mapping)}
    rows = [dict(candidate) for candidate in candidates]
    identifiers = [str(row.get("candidate_id") or "").strip() for row in rows]
    duplicates = {value for value in identifiers if value and identifiers.count(value) > 1}
    if duplicates:
        raise SelfTrainingError("duplicate candidate IDs: " + ", ".join(sorted(duplicates)))
    missing_resumed = sorted(set(previous) - set(identifiers))
    if missing_resumed:
        raise SelfTrainingError(
            "resume manifest contains candidates absent from the current input: "
            + ", ".join(missing_resumed)
        )
    for candidate_id, prior in previous.items():
        current = next(row for row in rows if str(row.get("candidate_id") or "").strip() == candidate_id)
        if prior.get("source_candidate_sha256") != _sha(current):
            raise SelfTrainingError(f"resume candidate changed: {candidate_id}")
        if prior.get("status") == "accepted":
            _validate_pseudo_label(
                prior.get("pseudo_label"),
                decision=prior,
                min_confidence=min_confidence,
            )
    randomizer = random.Random(seed)
    ordered = sorted(rows, key=lambda row: (_sha(row), str(row.get("candidate_id") or "")))
    randomizer.shuffle(ordered)
    decisions = list(previous.values())
    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in decisions:
        if row.get("status") == "accepted" and isinstance(row.get("pseudo_label"), Mapping):
            accepted.append(dict(row["pseudo_label"]))
        elif row.get("status") != "accepted":
            failures.append({"candidate_id": row.get("candidate_id"), "reason": row.get("reason")})
    for candidate in ordered:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if candidate_id and candidate_id in previous:
            continue
        reason, label = _decision(candidate, min_confidence=min_confidence)
        decision = {
            "candidate_id": candidate_id,
            "status": reason,
            "reason": None if reason == "accepted" else reason,
            "source_candidate_sha256": _sha(candidate),
            "pseudo_label": label,
        }
        decisions.append(decision)
        if label is None:
            failures.append({"candidate_id": candidate_id, "reason": reason})
        else:
            accepted.append(label)
    decisions.sort(key=lambda row: (str(row.get("candidate_id") or ""), str(row.get("source_candidate_sha256") or "")))
    accepted.sort(key=lambda row: str(row["candidate_id"]))
    failures.sort(key=lambda row: (str(row.get("candidate_id") or ""), str(row.get("reason") or "")))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "min_confidence": min_confidence,
        "candidate_count": len(decisions),
        "accepted_count": len(accepted),
        "rejected_count": len(failures),
        "decisions": decisions,
        "accepted_sha256": _sha(accepted),
        "failure_reasons": failures,
        "truth_policy": {
            "phones_ipa_articulatory_are_truth": True,
            "asr_transcript_is_phone_truth": False,
            "human_truth_is_immutable": True,
            "pseudo_labels_are_separate_records": True,
        },
        "selection_policy": {
            "schema_version": "self_training_selection_policy_v1",
            "min_confidence": min_confidence,
            "seed": seed,
            "requires_complete_phone_ipa_articulatory_truth": True,
            "requires_content_and_configuration_hashes": True,
        },
        "resume_parent_sha256": (resume or {}).get("manifest_sha256"),
    }
    manifest["manifest_sha256"] = _sha(manifest)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "accepted_pseudo_labels.jsonl", output / "failures.jsonl", output / "self_training_manifest.json"]
    completion = output / "batch_complete.json"
    if resume is None and any(path.exists() for path in [*targets, completion]):
        raise SelfTrainingError("self-training output already exists; provide a validated resume manifest")
    _atomic_write(targets[0], "".join(_canonical(row) + "\n" for row in accepted))
    _atomic_write(targets[1], "".join(_canonical(row) + "\n" for row in failures))
    _atomic_write(targets[2], json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(
        completion,
        json.dumps(
            {
                "schema_version": "any2xsampa_self_training_complete_v1",
                "manifest_sha256": manifest["manifest_sha256"],
                "accepted_file_sha256": hashlib.sha256(targets[0].read_bytes()).hexdigest(),
                "failures_file_sha256": hashlib.sha256(targets[1].read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    return manifest


__all__ = [
    "PSEUDO_LABEL_SCHEMA",
    "SCHEMA_VERSION",
    "SelfTrainingError",
    "build_self_training_batch",
]

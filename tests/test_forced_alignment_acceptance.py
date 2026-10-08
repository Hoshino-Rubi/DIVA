from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from any2xsampa.adapters.forced_alignment_acceptance import (
    FULL_SONG_SCOPE,
    HUMAN_REVIEW_ROUTE,
    QUALIFIED_MODEL_ROUTE,
    SCHEMA_VERSION,
    SHORT_CLIP_SCOPE,
    ForcedPhoneAcceptanceError,
    canonical_phone_inventory_sha256,
    canonical_phone_segments_sha256,
    load_forced_phone_acceptance,
)
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.lyrics.timeline import sha256_file


def _wav(path: Path, *, duration_sec: float = 12.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * duration_sec))
    return path


def _segments() -> tuple[PhoneSegment, ...]:
    return (
        PhoneSegment(
            time_range=TimeRange(0.25, 5.5),
            candidate_phones=[PhoneCandidate("e", 0.95)],
            confidence=0.95,
            normalized_ipa="e",
            metadata={"lyric_item_index": 0},
        ),
        PhoneSegment(
            time_range=TimeRange(5.5, 11.75),
            candidate_phones=[PhoneCandidate("u", 0.94)],
            confidence=0.94,
            normalized_ipa="u",
            metadata={"lyric_item_index": 1},
        ),
    )


def _common(alignment: Path, stem: Path, segments: tuple[PhoneSegment, ...]) -> dict[str, object]:
    symbols = ["e", "u"]
    return {
        "schema_version": SCHEMA_VERSION,
        "alignment_artifact_sha256": sha256_file(alignment),
        "vocal_stem_sha256": sha256_file(stem),
        "phone_segments_sha256": canonical_phone_segments_sha256(segments),
        "phone_inventory": {
            "symbols": symbols,
            "sha256": canonical_phone_inventory_sha256(symbols),
            "source": "fixture-phones.txt@fixture-revision",
        },
    }


def _model_acceptance(
    alignment: Path,
    stem: Path,
    segments: tuple[PhoneSegment, ...],
) -> dict[str, object]:
    root = alignment.parent
    model_file = root / "model.bin"
    dictionary_file = root / "dictionary.txt"
    tokenizer_file = root / "tokenizer.json"
    phone_map_file = root / "phone_map.json"
    license_file = root / "LICENSE.txt"
    training_file = root / "training-evidence.json"
    model_file.write_bytes(b"fixture model weights")
    dictionary_file.write_text("e e\nu u\n", encoding="utf-8")
    tokenizer_file.write_text('{"tokens":["e","u"]}\n', encoding="utf-8")
    phone_map_file.write_text('{"e":"e","u":"u"}\n', encoding="utf-8")
    license_file.write_text("Apache License 2.0\n", encoding="utf-8")
    training_file.write_text(
        '{"claim":"trained_or_finetuned_on_cantonese_singing",'
        '"language":"Cantonese","domain":"singing_voice"}\n',
        encoding="utf-8",
    )

    def digest(path: Path) -> str:
        return sha256_file(path)

    model_hash = digest(model_file)
    license_hash = digest(license_file)
    training_hash = digest(training_file)
    symbols = ["e", "u"]
    inventory_hash = canonical_phone_inventory_sha256(symbols)

    def artifact(kind: str, path: Path, *, inventory: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": path.name,
            "sha256": digest(path),
            "revision": "fixture-resource-revision",
            "revision_is_immutable": True,
            "license": {
                "identifier": "Apache-2.0",
                "applies_to": kind,
                "noncommercial_restriction": False,
                "commercial_use": "allowed",
                "source": "fixture-license.txt",
                "source_path": license_file.name,
                "source_sha256": license_hash,
            },
        }
        if inventory is not None:
            payload["phone_inventory_sha256"] = inventory
        return payload

    manifest = {
        "schema_version": "forced_phone_model_manifest_v1",
        "required_artifacts": ["model_weights", "dictionary", "tokenizer", "phone_map"],
        "artifacts": {
            "model_weights": artifact("model_weights", model_file),
            "dictionary": artifact("dictionary", dictionary_file),
            "tokenizer": artifact("tokenizer", tokenizer_file),
            "phone_map": artifact("phone_map", phone_map_file, inventory=inventory_hash),
        },
        "phone_inventory": {
            "symbols": symbols,
            "sha256": inventory_hash,
            "source": "fixture-phone-inventory.txt",
        },
        "training_evidence": {
            "path": training_file.name,
            "sha256": training_hash,
            "claim": "trained_or_finetuned_on_cantonese_singing",
            "language": "Cantonese",
            "domain": "singing_voice",
            "datasets": ["fixture-cantonese-singing-corpus"],
            "source_type": "primary",
            "source": "fixture-training-report.json",
            "source_path": training_file.name,
            "source_revision": "fixture-training-revision",
            "source_sha256": training_hash,
        },
    }
    manifest_path = root / "resource-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        **_common(alignment, stem, segments),
        "route": QUALIFIED_MODEL_ROUTE,
        "model": {
            "name": "fixture-cantonese-singing-aligner",
            "revision": "fixture-resource-revision",
            "revision_is_immutable": True,
            "sha256": model_hash,
            "resource_manifest_path": manifest_path.name,
            "resource_manifest_sha256": digest(manifest_path),
            "license": {
                "identifier": "Apache-2.0",
                "applies_to": "model_weights",
                "noncommercial_restriction": False,
                "commercial_use": "allowed",
                "source": "fixture-license.txt",
                "source_sha256": license_hash,
            },
            "cantonese_singing_training": {
                "claim": "trained_or_finetuned_on_cantonese_singing",
                "language": "Cantonese",
                "domain": "singing_voice",
                "datasets": ["fixture-cantonese-singing-corpus"],
                "source_type": "primary",
                "source": "fixture-training-report.json",
                "source_revision": "fixture-training-revision",
                "source_sha256": training_hash,
            },
        },
    }


def _human_acceptance(
    alignment: Path,
    stem: Path,
    segments: tuple[PhoneSegment, ...],
    *,
    scope: str = SHORT_CLIP_SCOPE,
) -> dict[str, object]:
    return {
        **_common(alignment, stem, segments),
        "alignment_artifact_path": alignment.name,
        "route": HUMAN_REVIEW_ROUTE,
        "review": {
            "review_status": "approved",
            "reviewer_kind": "human",
            "reviewer_id": "fixture-reviewer",
            "reviewed_at": "2026-08-15T12:00:00+08:00",
            "review_method": "auditory_acoustic_phone_boundary_review",
            "phone_boundaries_reviewed": True,
            "scope": scope,
            "reviewed_range_sec": [0.0, 12.0],
        },
    }


def _load(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    alignment: Path,
    stem: Path,
    segments: tuple[PhoneSegment, ...],
    name: str = "acceptance.json",
) -> dict[str, object]:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_forced_phone_acceptance(
        path,
        alignment_path=alignment,
        vocal_stem=stem,
        model_hash=str(payload["model"]["sha256"]) if isinstance(payload.get("model"), dict) else "",
        segments=segments,
    )


def test_accepts_complete_qualified_model_route(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    alignment = tmp_path / "alignment.json"
    alignment.write_text("{}", encoding="utf-8")
    segments = _segments()

    result = _load(
        tmp_path,
        _model_acceptance(alignment, stem, segments),
        alignment=alignment,
        stem=stem,
        segments=segments,
    )

    assert result["route"] == QUALIFIED_MODEL_ROUTE
    assert result["model"]["license"]["commercial_use"] == "allowed"  # type: ignore[index]
    manifest = result["model"]["resource_manifest"]  # type: ignore[index]
    assert manifest["schema_version"] == "forced_phone_model_manifest_v1"  # type: ignore[index]
    assert set(manifest["required_artifacts"]) == {  # type: ignore[index]
        "model_weights",
        "dictionary",
        "tokenizer",
        "phone_map",
    }


@pytest.mark.parametrize(
    "failure",
    [
        "missing_manifest",
        "tampered_model",
        "dictionary_nc",
        "mutable_phone_map",
        "inventory_binding",
        "missing_training_file",
    ],
)
def test_rejects_unverifiable_qualified_resource_manifest(tmp_path: Path, failure: str) -> None:
    stem = _wav(tmp_path / "stem.wav")
    alignment = tmp_path / "alignment.json"
    alignment.write_text("{}", encoding="utf-8")
    segments = _segments()
    payload = _model_acceptance(alignment, stem, segments)
    model = payload["model"]
    assert isinstance(model, dict)
    manifest_path = tmp_path / str(model["resource_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if failure == "missing_manifest":
        model["resource_manifest_path"] = "missing-manifest.json"
    elif failure == "tampered_model":
        (tmp_path / "model.bin").write_bytes(b"tampered")
    elif failure == "dictionary_nc":
        manifest["artifacts"]["dictionary"]["license"]["identifier"] = "CC-BY-NC-4.0"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        model["resource_manifest_sha256"] = sha256_file(manifest_path)
    elif failure == "mutable_phone_map":
        manifest["artifacts"]["phone_map"]["revision_is_immutable"] = False
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        model["resource_manifest_sha256"] = sha256_file(manifest_path)
    elif failure == "inventory_binding":
        manifest["artifacts"]["phone_map"]["phone_inventory_sha256"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        model["resource_manifest_sha256"] = sha256_file(manifest_path)
    else:
        training_path = tmp_path / "training-evidence.json"
        training_path.unlink()

    with pytest.raises(ForcedPhoneAcceptanceError):
        _load(tmp_path, payload, alignment=alignment, stem=stem, segments=segments)


@pytest.mark.parametrize("failure", ["nc_license", "missing_singing_training", "mutable_revision"])
def test_rejects_incomplete_model_route(tmp_path: Path, failure: str) -> None:
    stem = _wav(tmp_path / "stem.wav")
    alignment = tmp_path / "alignment.json"
    alignment.write_text("{}", encoding="utf-8")
    segments = _segments()
    payload = _model_acceptance(alignment, stem, segments)
    model = payload["model"]
    assert isinstance(model, dict)
    if failure == "nc_license":
        model["license"]["identifier"] = "CC-BY-NC-4.0"  # type: ignore[index]
    elif failure == "missing_singing_training":
        model.pop("cantonese_singing_training")
    else:
        model["revision_is_immutable"] = False

    with pytest.raises(ForcedPhoneAcceptanceError) as exc_info:
        _load(tmp_path, payload, alignment=alignment, stem=stem, segments=segments)
    assert exc_info.value.error_kind == "unqualified_forced_phone_evidence"


def test_accepts_approved_twelve_second_human_review(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    alignment = tmp_path / "alignment.json"
    alignment.write_text("{}", encoding="utf-8")
    segments = _segments()

    result = _load(
        tmp_path,
        _human_acceptance(alignment, stem, segments),
        alignment=alignment,
        stem=stem,
        segments=segments,
    )

    assert result["route"] == HUMAN_REVIEW_ROUTE
    assert result["review"]["scope"] == SHORT_CLIP_SCOPE  # type: ignore[index]


@pytest.mark.parametrize("failure", ["pending", "too_short", "wrong_audio", "wrong_segments"])
def test_rejects_unqualified_human_review(tmp_path: Path, failure: str) -> None:
    stem = _wav(tmp_path / "stem.wav")
    alignment = tmp_path / "alignment.json"
    alignment.write_text("{}", encoding="utf-8")
    segments = _segments()
    payload = _human_acceptance(alignment, stem, segments)
    review = payload["review"]
    assert isinstance(review, dict)
    if failure == "pending":
        review["review_status"] = "pending"
    elif failure == "too_short":
        review["reviewed_range_sec"] = [0.0, 9.99]
    elif failure == "wrong_audio":
        payload["vocal_stem_sha256"] = "f" * 64
    else:
        payload["phone_segments_sha256"] = "f" * 64

    with pytest.raises(ForcedPhoneAcceptanceError):
        _load(tmp_path, payload, alignment=alignment, stem=stem, segments=segments)


def test_full_song_requires_real_approved_short_clip_predecessor(tmp_path: Path) -> None:
    stem = _wav(tmp_path / "stem.wav")
    alignment = tmp_path / "alignment.json"
    segments = _segments()
    alignment.write_text(
        json.dumps(
            {
                "schema_version": "forced_phone_alignment_v1",
                "vocal_stem_sha256": sha256_file(stem),
                "segments": [segment.to_dict() for segment in segments],
            }
        ),
        encoding="utf-8",
    )
    full_payload = _human_acceptance(alignment, stem, segments, scope=FULL_SONG_SCOPE)

    with pytest.raises(ForcedPhoneAcceptanceError, match="short-clip predecessor"):
        _load(tmp_path, full_payload, alignment=alignment, stem=stem, segments=segments)

    short_alignment = tmp_path / "short-alignment.json"
    short_alignment.write_bytes(alignment.read_bytes())
    short_path = tmp_path / "short.json"
    short_path.write_text(
        json.dumps(_human_acceptance(short_alignment, stem, segments)),
        encoding="utf-8",
    )
    review = full_payload["review"]
    assert isinstance(review, dict)
    review["short_clip_predecessor"] = {
        "acceptance_path": short_path.name,
        "acceptance_sha256": sha256_file(short_path),
    }

    result = _load(
        tmp_path,
        full_payload,
        alignment=alignment,
        stem=stem,
        segments=segments,
        name="full.json",
    )
    assert result["review"]["scope"] == FULL_SONG_SCOPE  # type: ignore[index]

    short_alignment.write_text("{}", encoding="utf-8")
    with pytest.raises(ForcedPhoneAcceptanceError, match="predecessor alignment artifact"):
        _load(
            tmp_path,
            full_payload,
            alignment=alignment,
            stem=stem,
            segments=segments,
            name="full-after-short-alignment-change.json",
        )

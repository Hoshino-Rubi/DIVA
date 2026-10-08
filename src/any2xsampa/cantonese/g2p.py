"""Replaceable Cantonese G2P adapters for timed lyric evidence.

The resolver returns canonical phones/IPA candidates.  It never treats the
export phoneme spelling as recognizer truth and it never downloads a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Protocol, Sequence

from any2xsampa.cantonese.resolver import CantonesePhonemeResolver, load_lexicon_file
from any2xsampa.lyrics.timeline import LyricsTimeline, LyricsTimelineItem
from any2xsampa.vocaloid.phoneme_mapper import MikuV4CPhonemeMapper


class CantoneseG2PError(ValueError):
    """G2P evidence is absent, unverifiable, or contains an OOV item."""

    def __init__(self, error_kind: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class CantoneseG2PCandidate:
    original_text: str
    jyutping: str
    internal_phones: tuple[str, ...]
    ipa: tuple[str, ...]
    v4c_phonemes: tuple[str, ...]
    confidence: float
    source: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.original_text.strip() or not self.jyutping.strip() or not self.source.strip():
            raise CantoneseG2PError("invalid_g2p_evidence", "G2P text, Jyutping, and source must be non-empty")
        if not self.internal_phones or not self.ipa:
            raise CantoneseG2PError("invalid_g2p_evidence", "G2P candidates require internal phones and IPA")
        if any(not str(value).strip() for value in (*self.internal_phones, *self.ipa, *self.v4c_phonemes)):
            raise CantoneseG2PError("invalid_g2p_evidence", "G2P phone sequences contain an empty token")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise CantoneseG2PError("invalid_g2p_evidence", "G2P confidence must be finite and between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "jyutping": self.jyutping,
            "internal_phones": list(self.internal_phones),
            "ipa": list(self.ipa),
            "v4c_phonemes": list(self.v4c_phonemes),
            "confidence": self.confidence,
            "source": self.source,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ResolvedLyricsItem:
    index: int
    text: str
    start_sec: float
    end_sec: float
    candidate: CantoneseG2PCandidate

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "index": self.index,
            "text": self.text,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }
        payload.update(self.candidate.to_dict())
        return payload


class CantoneseG2PAdapter(Protocol):
    """Minimal adapter contract used by the product runner."""

    def resolve(self, text: str, *, item: LyricsTimelineItem) -> Sequence[CantoneseG2PCandidate]: ...

    def metadata(self) -> Mapping[str, Any]: ...


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class ReviewedLexiconCantoneseG2PAdapter:
    """Use the checked-in, manually reviewed lexicon and mapper."""

    def __init__(self, lexicon_path: Path | str | None = None) -> None:
        self.lexicon_path = Path(lexicon_path) if lexicon_path else None
        lexicon = load_lexicon_file(self.lexicon_path) if self.lexicon_path else None
        self._resolver = CantonesePhonemeResolver(lexicon=lexicon)

    def resolve(self, text: str, *, item: LyricsTimelineItem) -> Sequence[CantoneseG2PCandidate]:
        overrides = item.jyutping
        try:
            resolution = self._resolver.resolve_syllable(text, index=item.index, jyutping_override=overrides)
        except (TypeError, ValueError) as exc:
            raise CantoneseG2PError(
                "unsupported_lyrics_evidence",
                f"invalid Jyutping evidence for timeline item {item.index}: {overrides or text}",
                details={"index": item.index, "text": text, "jyutping": overrides},
            ) from exc
        candidates: list[CantoneseG2PCandidate] = []
        for candidate in resolution.candidates:
            candidates.append(
                CantoneseG2PCandidate(
                    original_text=text,
                    jyutping=candidate.jyutping,
                    internal_phones=tuple(candidate.internal_phones),
                    ipa=tuple(candidate.ipa),
                    v4c_phonemes=tuple(candidate.v4c_phonemes),
                    confidence=float(candidate.score),
                    source="reviewed_lexicon" if not overrides else "timeline_jyutping",
                    notes=tuple(candidate.notes),
                )
            )
        if overrides and not candidates:
            raise CantoneseG2PError(
                "unsupported_lyrics_evidence",
                f"timeline Jyutping override is not a valid Cantonese syllable: {overrides}",
                details={"index": item.index, "text": text, "jyutping": overrides},
            )
        return candidates

    def metadata(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "adapter": "reviewed_lexicon",
            "license_status": "project_reviewed_resources",
            "model_revision": "checked_in_recovered_lexicon",
            "lexicon_sha256": "d2232eb78a811faf85d13323984f3449813a4e47ce771546bfbe75171647773e",
            "model_hash": "d2232eb78a811faf85d13323984f3449813a4e47ce771546bfbe75171647773e",
        }
        if self.lexicon_path and self.lexicon_path.is_file():
            payload["lexicon_path"] = str(self.lexicon_path)
            payload["lexicon_sha256"] = _digest(self.lexicon_path)
            payload["model_hash"] = payload["lexicon_sha256"]
        return payload


class PyCantoneseG2PAdapter:
    """Licensed local PyCantonese adapter; imports lazily and never downloads."""

    PACKAGE_VERSION = "5.0.0"
    LICENSE_ID = "MIT"
    WHEEL_SHA256 = "8f7bdc1e9a37b5e164cd57f22aecae8532d4d0b74ff5b6f5e28480a657fa53f9"

    def __init__(self) -> None:
        self._backend = None
        self._mapper = MikuV4CPhonemeMapper.for_cantonese()
        try:
            import pycantonese
        except ImportError as exc:
            raise CantoneseG2PError(
                "g2p_dependency_unavailable",
                f"PyCantonese {self.PACKAGE_VERSION} is not installed; no model is downloaded automatically",
            ) from exc
        version = str(getattr(pycantonese, "__version__", ""))
        if version != self.PACKAGE_VERSION:
            raise CantoneseG2PError(
                "g2p_version_mismatch",
                f"PyCantonese version {version or 'unknown'} does not match locked {self.PACKAGE_VERSION}",
            )
        self._backend = pycantonese

    @classmethod
    def for_timeline(cls, timeline: LyricsTimeline) -> CantoneseG2PAdapter:
        """Use the local mapper only when every item already has Jyutping."""
        if all(item.jyutping for item in timeline.items):
            return ReviewedLexiconCantoneseG2PAdapter()
        return cls()

    def resolve(self, text: str, *, item: LyricsTimelineItem) -> Sequence[CantoneseG2PCandidate]:
        if item.jyutping:
            jyutping_values = [item.jyutping]
            source = "timeline_jyutping"
        else:
            if self._backend is None:
                raise CantoneseG2PError(
                    "g2p_dependency_unavailable",
                    "PyCantonese is required when a timeline item has no explicit Jyutping",
                )
            raw = self._backend.characters_to_jyutping([text])
            jyutping_values = []
            if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], tuple) and raw[0][1]:
                values = [value for value in str(raw[0][1]).split() if value]
                if len(values) == 1:
                    jyutping_values = values
            source = "pycantonese_5.0.0"
        candidates: list[CantoneseG2PCandidate] = []
        for jyutping in jyutping_values:
            try:
                mapped = self._mapper.map_jyutping(jyutping)
            except (TypeError, ValueError):
                continue
            candidates.append(
                CantoneseG2PCandidate(
                    original_text=text,
                    jyutping=str(mapped["jyutping"]),
                    internal_phones=tuple(str(value) for value in mapped["internal_phones"]),
                    ipa=tuple(str(value) for value in mapped["ipa"]),
                    v4c_phonemes=tuple(str(value) for value in mapped["v4c_phonemes"]),
                    confidence=1.0 if item.jyutping else 0.9,
                    source=source,
                    notes=tuple(str(value) for value in mapped.get("notes", [])),
                )
            )
        return candidates

    def metadata(self) -> Mapping[str, Any]:
        return {
            "adapter": "pycantonese",
            "package_version": self.PACKAGE_VERSION,
            "model_revision": "pycantonese-5.0.0-bundled-rime-data",
            "license_id": self.LICENSE_ID,
            "bundled_data_licenses": ["CC-BY", "CC-BY-4.0", "GPL-3.0", "MPL-2.0", "CC0-1.0"],
            "wheel_sha256": self.WHEEL_SHA256,
            "model_hash": self.WHEEL_SHA256,
            "automatic_download": False,
        }


class ExternalCommandCantoneseG2PAdapter:
    """Call an explicitly provisioned, license-approved G2P executable.

    The command receives one JSON request on stdin and must return a JSON
    list of candidate objects.  This keeps third-party packages/models behind
    a replaceable boundary and makes licensing provenance mandatory.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        license_id: str,
        model_path: Path | str,
        model_revision: str,
        expected_model_sha256: str | None = None,
        timeout_sec: float = 30.0,
    ) -> None:
        if not command:
            raise CantoneseG2PError("g2p_configuration_error", "external G2P command is empty")
        license_text = str(license_id).strip()
        normalized_license = license_text.casefold().replace("_", "-")
        if (
            not license_text
            or normalized_license in {"unknown", "unclear", "nc", "non-commercial"}
            or "non-commercial" in normalized_license
            or "-nc" in normalized_license
            or normalized_license.endswith("nc")
        ):
            raise CantoneseG2PError("g2p_license_unverified", "external G2P license must be explicitly non-NC")
        self.command = tuple(str(part) for part in command)
        self.model_path = Path(model_path)
        self.license_id = license_text
        self.model_revision = str(model_revision).strip()
        self.timeout_sec = float(timeout_sec)
        if not self.model_path.exists() or not self.model_path.is_file():
            raise CantoneseG2PError("g2p_model_missing", f"external G2P model file is missing: {self.model_path}")
        self.model_hash = _digest(self.model_path)
        if expected_model_sha256 and self.model_hash.lower() != str(expected_model_sha256).lower():
            raise CantoneseG2PError("g2p_model_hash_mismatch", "external G2P model hash does not match the expected digest")

    def resolve(self, text: str, *, item: LyricsTimelineItem) -> Sequence[CantoneseG2PCandidate]:
        request = {"text": text, "index": item.index, "start_sec": item.start_sec, "end_sec": item.end_sec}
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                check=True,
                timeout=self.timeout_sec,
                shell=False,
            )
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise CantoneseG2PError("g2p_runtime_error", f"external G2P failed for item {item.index}: {exc}") from exc
        if not isinstance(payload, list):
            raise CantoneseG2PError("g2p_runtime_error", "external G2P output must be a candidate list")
        candidates: list[CantoneseG2PCandidate] = []
        for raw in payload:
            if not isinstance(raw, Mapping):
                continue
            try:
                candidates.append(
                    CantoneseG2PCandidate(
                        original_text=text,
                        jyutping=str(raw["jyutping"]),
                        internal_phones=tuple(str(value) for value in raw["internal_phones"]),
                        ipa=tuple(str(value) for value in raw["ipa"]),
                        v4c_phonemes=tuple(str(value) for value in raw.get("v4c_phonemes", [])),
                        confidence=float(raw.get("confidence", raw.get("score", 0.0))),
                        source=str(raw.get("source", "external_g2p")),
                        notes=tuple(str(value) for value in raw.get("notes", [])),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CantoneseG2PError("g2p_runtime_error", f"invalid external G2P candidate: {exc}") from exc
        return candidates

    def metadata(self) -> Mapping[str, Any]:
        return {
            "adapter": "external_command",
            "command": list(self.command),
            "license_id": self.license_id,
            "model_path": str(self.model_path),
            "model_revision": self.model_revision,
            "model_hash": self.model_hash,
        }


def resolve_lyrics_timeline(
    timeline: LyricsTimeline,
    *,
    adapter: CantoneseG2PAdapter | None = None,
) -> tuple[list[ResolvedLyricsItem], dict[str, Any]]:
    selected = adapter or PyCantoneseG2PAdapter.for_timeline(timeline)
    resolved: list[ResolvedLyricsItem] = []
    failures: list[dict[str, Any]] = []
    for item in timeline.items:
        candidates = list(selected.resolve(item.text, item=item))
        valid = [candidate for candidate in candidates if candidate.jyutping and candidate.internal_phones and candidate.ipa]
        if not valid:
            failures.append({"index": item.index, "text": item.text, "reason": "no_verifiable_jyutping_phone_candidate"})
            continue
        resolved.append(ResolvedLyricsItem(item.index, item.text, item.start_sec, item.end_sec, valid[0]))
    if failures:
        raise CantoneseG2PError(
            "unsupported_lyrics_evidence",
            f"{len(failures)} lyric timeline item(s) lack verifiable G2P evidence",
            details={"oov_items": failures, "adapter": dict(selected.metadata())},
        )
    return resolved, {"adapter": dict(selected.metadata()), "item_count": len(resolved), "oov_items": []}


__all__ = [
    "CantoneseG2PAdapter",
    "CantoneseG2PCandidate",
    "CantoneseG2PError",
    "ExternalCommandCantoneseG2PAdapter",
    "ResolvedLyricsItem",
    "PyCantoneseG2PAdapter",
    "ReviewedLexiconCantoneseG2PAdapter",
    "resolve_lyrics_timeline",
]

"""Functional Miku V4C ranker dataset reconstruction.

The historical training module is evidence-poor, so this candidate keeps the
boundary deliberately small: candidate JSON is normalized to phone/IPA truth,
human and pseudo labels retain separate provenance, and split assignment is a
hash-stable operation over identity groups.  X-SAMPA/Vocaloid fields are never
used to create the phone truth sequence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from any2xsampa.data.deterministic_split import SplitError, SplitRecord, split_records


SCHEMA_VERSION = "miku-v4c-ranker-dataset-v2"
VOCAB_SCHEMA_VERSION = "miku-v4c-ranker-vocab-v2"
SPLIT_ALGORITHM = "connected-component-balanced-v1"
_TOKEN_SPLIT = re.compile(r"\s+")


class DatasetBuildError(ValueError):
    """Raised when a ranker candidate cannot satisfy the dataset contract."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize JSON with stable ordering for manifests and hashes."""

    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _as_bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "positive", "pos"}:
            return True
        if normalized in {"0", "false", "no", "n", "negative", "neg"}:
            return False
    return default


def _non_empty(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token for token in _TOKEN_SPLIT.split(value.strip()) if token]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _normalise_token(value: Any) -> str:
    if isinstance(value, Mapping):
        value = _first(value, "phone", "phone_id", "token", "phoneme", "symbol", "ipa")
    return _non_empty(value)


def _phone_truth(candidate: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract canonical phone tokens and IPA/articulatory evidence.

    The source candidate formats evolved over several probes.  The adapter
    accepts the observed aliases but deliberately ignores X-SAMPA/export-only
    fields when selecting the truth sequence.
    """

    raw_segments = _first(candidate, "phone_segments", "segments", "phones")
    segment_values = _sequence(raw_segments)
    tokens: list[str] = []
    truth: list[dict[str, Any]] = []
    if segment_values and all(isinstance(item, Mapping) for item in segment_values):
        for item in segment_values:
            assert isinstance(item, Mapping)
            token = _normalise_token(item)
            if not token:
                continue
            ipa = _non_empty(_first(item, "ipa", "ipa_symbol", "symbol")) or None
            features = _first(item, "features", "articulatory_features")
            if isinstance(features, Mapping) and "values" in features:
                features = features["values"]
            if not isinstance(features, Mapping):
                features = {}
            tokens.append(token)
            truth.append({"phone": token, "ipa": ipa, "features": dict(features)})
    else:
        raw_tokens = _first(candidate, "phoneme_tokens", "phone_tokens", "phonemes", "phoneme_sequence")
        if raw_tokens is None and segment_values:
            raw_tokens = segment_values
        if raw_tokens is None and isinstance(_first(candidate, "phonemes_by_note"), Sequence):
            flattened: list[Any] = []
            for note in _first(candidate, "phonemes_by_note") or []:
                if isinstance(note, Mapping):
                    flattened.extend(_sequence(_first(note, "phonemes", "phone_tokens")))
            raw_tokens = flattened
        tokens = [_normalise_token(item) for item in _sequence(raw_tokens)]
        tokens = [token for token in tokens if token]
        raw_ipa = _sequence(_first(candidate, "ipa_tokens", "ipa_sequence", "ipa"))
        raw_features = _first(candidate, "articulatory_features", "features")
        for index, token in enumerate(tokens):
            ipa = _non_empty(raw_ipa[index]) if index < len(raw_ipa) else ""
            features: Mapping[str, Any] = {}
            if isinstance(raw_features, Sequence) and not isinstance(raw_features, (str, bytes, bytearray)):
                if index < len(raw_features) and isinstance(raw_features[index], Mapping):
                    features = raw_features[index]
            elif isinstance(raw_features, Mapping):
                features = raw_features
            truth.append({"phone": token, "ipa": ipa or None, "features": dict(features)})
    return tokens, truth


def identity_group(candidate: Mapping[str, Any]) -> str:
    """Return a stable singer/song/phrase identity key for leakage prevention."""

    explicit = _first(candidate, "identity_group", "group_id", "identity_id")
    if explicit is not None:
        return _non_empty(explicit)
    singer = _first(candidate, "singer_id", "singer", "voice", "speaker_id", "speaker")
    song = _first(candidate, "song_id", "song", "track_id", "track", "title")
    phrase = _first(candidate, "phrase_id", "phrase", "utterance_id", "segment_id")
    parts = [_non_empty(value) for value in (singer, song, phrase)]
    if all(parts):
        return "/".join(parts)
    raise DatasetBuildError(
        "candidate requires identity_group/group_id or complete singer, song, and phrase identity fields"
    )


def _split_record(candidate: Mapping[str, Any]) -> SplitRecord:
    candidate_id = _non_empty(_first(candidate, "candidate_id", "record_id", "id"))
    if not candidate_id:
        raise DatasetBuildError("candidate requires candidate_id")
    singer = _non_empty(_first(candidate, "singer_id", "singer", "voice", "speaker_id", "speaker"))
    song = _non_empty(_first(candidate, "song_id", "song", "track_id", "track", "title"))
    phrase = _non_empty(_first(candidate, "phrase_id", "phrase", "utterance_id", "segment_id"))
    if not all((singer, song, phrase)):
        explicit = _non_empty(_first(candidate, "identity_group", "group_id", "identity_id"))
        if not explicit:
            raise DatasetBuildError(
                "candidate requires complete singer, song, and phrase identities or an explicit identity group"
            )
        singer = song = phrase = explicit
    source_hashes: set[str] = set()
    for key in (
        "content_hash",
        "audio_sha256",
        "source_audio_sha256",
        "input_audio_sha256",
        "rendered_wav_sha256",
        "source_vsqx_sha256",
        "source_corpus_sha256",
    ):
        value = _non_empty(candidate.get(key))
        if value:
            source_hashes.add(value.casefold())
    for nested_key in ("source", "model_provenance", "label_provenance"):
        nested = candidate.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        for key in ("sha256", "input_audio_sha256", "source_sha256"):
            value = _non_empty(nested.get(key))
            if value:
                source_hashes.add(value.casefold())
    return SplitRecord(
        example_id=candidate_id,
        singer_id=singer,
        song_id=song,
        phrase_id=phrase,
        content_hash=min(source_hashes) if source_hashes else None,
        metadata={"source_hashes": sorted(source_hashes)},
    )


def _label_provenance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    value = _first(candidate, "label_provenance", "provenance", "annotation_provenance")
    provenance = dict(value) if isinstance(value, Mapping) else {}
    kind = _first(candidate, "label_kind", "label_source", "annotation_kind")
    if kind is None:
        kind = provenance.get("kind")
    if isinstance(kind, str) and kind.lower() in {"human", "manual"}:
        kind = "human"
    elif isinstance(kind, str) and kind.lower() in {"pseudo", "model", "automatic", "auto"}:
        kind = "pseudo"
    else:
        kind = "human" if any(key in candidate for key in ("manual_label", "manual_positive", "manual_negative")) else "pseudo"
    provenance["kind"] = kind
    provenance.setdefault("policy", "manual-review-v1" if kind == "human" else "model-pseudo-v1")
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        source = {}
    provenance["source"] = dict(source)
    if kind == "human" and _first(candidate, "annotator_ref", "reviewer") is not None:
        provenance.setdefault("annotator_ref", _first(candidate, "annotator_ref", "reviewer"))
    elif kind == "pseudo":
        if _first(candidate, "model_id", "recognizer_id") is not None:
            provenance.setdefault("model_id", _first(candidate, "model_id", "recognizer_id"))
        if _first(candidate, "model_revision", "model_version") is not None:
            provenance.setdefault("model_revision", _first(candidate, "model_revision", "model_version"))
        if _first(candidate, "generated_at_utc", "created_at") is not None:
            provenance.setdefault("generated_at_utc", _first(candidate, "generated_at_utc", "created_at"))
    return _jsonable(provenance)


def _provenance_missing(candidate: Mapping[str, Any], provenance: Mapping[str, Any]) -> list[str]:
    """Return required provenance fields absent from a label stream."""

    missing: list[str] = []
    kind = str(provenance.get("kind", "")).lower()
    if kind in {"human", "manual"}:
        if not _non_empty(provenance.get("annotator_ref")) and not _non_empty(_first(candidate, "annotator_ref", "reviewer")):
            missing.append("annotator_ref")
    elif kind == "pseudo":
        if not _non_empty(provenance.get("model_id")) and not _non_empty(_first(candidate, "model_id", "recognizer_id")):
            missing.append("model_id")
        if not _non_empty(provenance.get("model_revision")) and not _non_empty(_first(candidate, "model_revision", "model_version")):
            missing.append("model_revision")
        if not _non_empty(provenance.get("generated_at_utc")) and not _non_empty(_first(candidate, "generated_at_utc", "created_at")):
            missing.append("generated_at_utc")
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        source = {}
    for key in ("source_id", "revision", "sha256"):
        if not _non_empty(source.get(key)):
            missing.append(f"source.{key}")
    return missing


def _all_provenance_missing(candidate: Mapping[str, Any], primary: Mapping[str, Any], *, score: float | None, manual: bool | None) -> list[str]:
    missing: list[str] = []
    streams: list[tuple[str, Mapping[str, Any]]] = [("label", primary)]
    human = _first(candidate, "human_provenance", "manual_provenance", "human_label_provenance")
    pseudo = _first(candidate, "pseudo_provenance", "model_provenance", "pseudo_label_provenance")
    if isinstance(human, Mapping):
        streams.append(("human", dict(human, kind="human")))
    elif manual is not None and str(primary.get("kind", "")).lower() not in {"human", "manual"}:
        streams.append(("human", {"kind": "human"}))
    if isinstance(pseudo, Mapping):
        streams.append(("pseudo", dict(pseudo, kind="pseudo")))
    elif score is not None and str(primary.get("kind", "")).lower() not in {"pseudo", "model", "automatic"}:
        streams.append(("pseudo", {"kind": "pseudo"}))
    for stream_name, provenance in streams:
        for field in _provenance_missing(candidate, provenance):
            missing.append(f"{stream_name}.{field}")
    return sorted(set(missing))


def _label_streams(candidate: Mapping[str, Any], *, score: float | None, manual: bool | None) -> list[dict[str, Any]]:
    """Keep manual and model-derived annotations as separate audit streams."""

    streams: list[dict[str, Any]] = []
    human_value = _first(candidate, "human_label", "manual_label", "manual_training_label", "manual_positive", "manual_negative")
    human_provenance = _first(candidate, "human_provenance", "manual_provenance", "human_label_provenance")
    pseudo_provenance = _first(candidate, "pseudo_provenance", "model_provenance", "pseudo_label_provenance")
    main_provenance = _first(candidate, "label_provenance", "provenance", "annotation_provenance")
    main_kind = str(main_provenance.get("kind", "")).lower() if isinstance(main_provenance, Mapping) else ""
    if human_value is not None or isinstance(human_provenance, Mapping) or main_kind in {"human", "manual"}:
        provenance = (
            dict(human_provenance)
            if isinstance(human_provenance, Mapping)
            else dict(main_provenance)
            if isinstance(main_provenance, Mapping) and main_kind in {"human", "manual"}
            else {}
        )
        provenance.setdefault("kind", "human")
        provenance.setdefault("policy", "manual-review-v1")
        if _first(candidate, "annotator_ref", "reviewer") is not None:
            provenance.setdefault("annotator_ref", _first(candidate, "annotator_ref", "reviewer"))
        streams.append({"kind": "human", "label": manual, "provenance": _jsonable(provenance)})
    if (
        isinstance(pseudo_provenance, Mapping)
        or main_kind in {"pseudo", "model", "automatic"}
        or (human_value is not None and score is not None)
        or (human_value is None and not isinstance(human_provenance, Mapping) and main_kind not in {"human", "manual"})
    ):
        provenance = dict(pseudo_provenance) if isinstance(pseudo_provenance, Mapping) else dict(main_provenance) if isinstance(main_provenance, Mapping) else {}
        provenance.setdefault("kind", "pseudo")
        provenance.setdefault("policy", "model-pseudo-v1")
        if _first(candidate, "model_id", "recognizer_id") is not None:
            provenance.setdefault("model_id", _first(candidate, "model_id", "recognizer_id"))
        if _first(candidate, "model_revision", "model_version") is not None:
            provenance.setdefault("model_revision", _first(candidate, "model_revision", "model_version"))
        streams.append({"kind": "pseudo", "label": None if score is None else score, "provenance": _jsonable(provenance)})
    return streams


def _render_issues(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = _first(candidate, "render_chain_audit_issues", "audit_issues")
    if isinstance(value, Mapping):
        return [dict(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) if isinstance(item, Mapping) else {"kind": str(item)} for item in value]


def _contains_hung1(candidate: Mapping[str, Any]) -> bool:
    for key in ("jyutping", "target_syllables"):
        value = candidate.get(key)
        for item in _sequence(value):
            if isinstance(item, Mapping):
                item = _first(item, "jyutping", "syllable", "text")
            if str(item).lower() == "hung1":
                return True
    for key in ("candidate_id", "phrase_id", "group_id", "source_probe_id", "probe_id"):
        if "hung1" in str(candidate.get(key, "")).lower():
            return True
    return False


def infer_failure_kind(
    candidate: Mapping[str, Any],
    *,
    phoneme_tokens: Sequence[str],
    positive_threshold: float,
) -> str:
    explicit = _non_empty(_first(candidate, "failure_kind", "failure", "status"))
    issues = _render_issues(candidate)
    if issues and (not explicit or explicit == "ok"):
        issue_kind = _non_empty(_first(issues[0], "kind", "failure_kind"))
        if issue_kind:
            return issue_kind
    if _as_bool(candidate.get("dummy_lyric"), False) and (not explicit or explicit == "ok"):
        return "dummy_lyric"
    if _contains_hung1(candidate) and (not explicit or explicit == "ok"):
        return "collapsed_hung1_family"
    rendered = _first(candidate, "rendered_wav", "rendered_audio", "rendered_path", "wav")
    if rendered is None and isinstance(candidate.get("artifacts"), Mapping):
        rendered = _first(candidate["artifacts"], "rendered_wav", "rendered_audio", "rendered_path", "wav")
    if rendered is None or not _non_empty(rendered):
        if not explicit or explicit == "ok":
            return "missing_rendered_wav"
    recognized = next(
        (candidate[key] for key in ("qwen_recognized_text", "recognized_text", "transcript") if key in candidate),
        None,
    )
    if recognized is not None and not _non_empty(recognized):
        if not explicit or explicit == "ok":
            return "empty_recognized_text"
    score_value = _first(candidate, "qwen_score", "score", "confidence")
    if score_value is not None:
        try:
            if float(score_value) < positive_threshold and (not explicit or explicit == "ok"):
                return "qwen_low_score"
        except (TypeError, ValueError):
            pass
    mapping_status = _first(candidate, "phoneme_mapping_status", "mapping_status")
    if isinstance(mapping_status, str) and mapping_status.lower() not in {"ok", "valid", "complete"}:
        if not explicit or explicit == "ok":
            return "bad_phoneme_mapping"
    mapping_error = _first(candidate, "phoneme_mapping_error", "mapping_error")
    if mapping_error is not None or _as_bool(candidate.get("mapping_valid"), True) is False:
        if not explicit or explicit == "ok":
            return "bad_phoneme_mapping"
    # ``c`` is a known evidence-poor legacy token in the V4C probe fixtures;
    # retain it as a rejected mapping rather than promote it to phone truth.
    if any(token.lower() == "c" for token in phoneme_tokens) and (not explicit or explicit == "ok"):
        return "bad_phoneme_mapping"
    if not phoneme_tokens and (not explicit or explicit == "ok"):
        return "bad_phoneme_mapping"
    return explicit or "ok"


def _manual_label(candidate: Mapping[str, Any]) -> bool | None:
    for key in ("manual_label", "manual_training_label", "human_label"):
        if key in candidate:
            return _as_bool(candidate.get(key))
    if "manual_positive" in candidate:
        value = _as_bool(candidate.get("manual_positive"))
        if value is not None:
            return value
    if "manual_negative" in candidate:
        value = _as_bool(candidate.get("manual_negative"))
        if value is not None:
            return not value
    return None


def _score(candidate: Mapping[str, Any]) -> float | None:
    value = _first(candidate, "qwen_score", "score", "confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_bucket(group: str, *, seed: int, algorithm: str) -> float:
    # Match the surviving historical helper: seed and group are the hash input;
    # algorithm remains explicit metadata so future split revisions can coexist.
    _ = algorithm
    digest = hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def grouped_split(
    groups: Iterable[str],
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 0,
    algorithm: str = SPLIT_ALGORITHM,
) -> dict[str, str]:
    """Assign each identity group exactly once using a stable hash threshold."""

    if not 0.0 <= val_ratio <= 1.0 or not 0.0 <= test_ratio <= 1.0:
        raise DatasetBuildError("val_ratio and test_ratio must be between 0 and 1")
    if val_ratio + test_ratio > 1.0:
        raise DatasetBuildError("val_ratio + test_ratio must not exceed 1")
    result: dict[str, str] = {}
    test_limit = test_ratio
    val_limit = test_ratio + val_ratio
    for group in sorted({_non_empty(value) for value in groups}):
        bucket = _group_bucket(group, seed=seed, algorithm=algorithm)
        result[group] = "test" if bucket < test_limit else "val" if bucket < val_limit else "train"
    return result


def split_manifest(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    leakage_guard: Mapping[str, Any] | None = None,
    algorithm: str = SPLIT_ALGORITHM,
) -> dict[str, Any]:
    groups = sorted(assignments)
    candidates = sorted(
        [
            {
                "candidate_id": _non_empty(_first(row, "candidate_id", "record_id", "id")),
                "identity_group": identity_group(row),
                "split": assignments[identity_group(row)],
            }
            for row in rows
        ],
        key=lambda item: (item["identity_group"], item["candidate_id"]),
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": algorithm,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "groups": [{"identity_group": group, "split": assignments[group]} for group in groups],
        "candidates": candidates,
    }
    if leakage_guard is not None:
        payload["leakage_guard"] = dict(leakage_guard)
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


@dataclass(frozen=True)
class RankerExample:
    candidate_id: str
    identity_group: str
    phrase_id: str
    lyrics: str
    phoneme_tokens: tuple[str, ...]
    phones: tuple[dict[str, Any], ...]
    label: int
    split: str
    failure_kind: str
    positive_gate_passed: bool
    positive_gate_failures: tuple[str, ...]
    score: float | None
    manual_label: bool | None
    label_provenance: dict[str, Any]
    label_streams: tuple[dict[str, Any], ...]
    provenance_missing: tuple[str, ...]
    source: dict[str, Any] = field(default_factory=dict)

    def to_row(self, *, split_algorithm: str, seed: int, manifest_sha256: str) -> dict[str, Any]:
        row = dict(self.source)
        rendered = _first(row, "rendered_wav", "rendered_audio", "rendered_path", "wav")
        if rendered is None and isinstance(row.get("artifacts"), Mapping):
            rendered = _first(row["artifacts"], "rendered_wav", "rendered_audio", "rendered_path", "wav")
        if rendered is not None:
            row["rendered_wav"] = rendered
        row.update(
            {
                "candidate_id": self.candidate_id,
                "identity_group": self.identity_group,
                "group_id": self.identity_group,
                "phrase_id": self.phrase_id,
                "lyrics": self.lyrics,
                "phoneme_tokens": list(self.phoneme_tokens),
                "phones": [dict(item) for item in self.phones],
                "truth_layer": "phones-ipa-articulatory",
                "truth_ipa_complete": bool(self.phones) and all(bool(item.get("ipa")) for item in self.phones),
                "truth_features_complete": bool(self.phones) and all(bool(item.get("features")) for item in self.phones),
                "label": self.label,
                "split": self.split,
                "failure_kind": self.failure_kind,
                "positive_gate_passed": self.positive_gate_passed,
                "positive_gate_failures": list(self.positive_gate_failures),
                "score": self.score,
                "manual_label": self.manual_label,
                "label_provenance": dict(self.label_provenance),
                "label_provenance_status": "complete" if not self.provenance_missing else "incomplete",
                "label_provenance_missing": list(self.provenance_missing),
                "label_streams": [dict(item) for item in self.label_streams],
                "split_algorithm": split_algorithm,
                "split_seed": seed,
                "split_manifest_sha256": manifest_sha256,
            }
        )
        return _jsonable(row)


def normalise_candidate(
    candidate: Mapping[str, Any],
    *,
    split: str,
    positive_threshold: float,
    trust_score_label: bool = False,
) -> RankerExample:
    candidate_id = _non_empty(_first(candidate, "candidate_id", "record_id", "id"))
    if not candidate_id:
        raise DatasetBuildError("candidate requires candidate_id")
    group = identity_group(candidate)
    phrase = _non_empty(_first(candidate, "phrase_id", "phrase", "utterance_id")) or group
    lyrics = _non_empty(_first(candidate, "lyrics", "lyric", "text"))
    tokens, phones = _phone_truth(candidate)
    failure_kind = infer_failure_kind(candidate, phoneme_tokens=tokens, positive_threshold=positive_threshold)
    issues = _render_issues(candidate)
    training_candidate = _as_bool(candidate.get("training_data_candidate"), False) is True
    training_candidate_missing = "training_data_candidate" not in candidate
    dummy = _as_bool(candidate.get("dummy_lyric"), False) is True
    collapsed = failure_kind == "collapsed_hung1_family" or _contains_hung1(candidate)
    label_provenance = _label_provenance(candidate)
    manual = _manual_label(candidate)
    score = _score(candidate)
    provenance_missing = tuple(_all_provenance_missing(candidate, label_provenance, score=score, manual=manual))
    gate_failures: list[str] = []
    if failure_kind != "ok":
        gate_failures.append("failure_kind_not_ok")
    if issues:
        gate_failures.append("render_chain_audit_issues_present")
    if training_candidate_missing:
        gate_failures.append("training_data_candidate_missing")
    elif not training_candidate:
        gate_failures.append("training_data_candidate_not_true")
    if provenance_missing:
        gate_failures.append("label_provenance_incomplete")
    if not phones or any(not item.get("ipa") for item in phones):
        gate_failures.append("phone_truth_ipa_incomplete")
    if not phones or any(not item.get("features") for item in phones):
        gate_failures.append("phone_truth_features_incomplete")
    if dummy:
        gate_failures.append("dummy_lyric")
    if collapsed:
        gate_failures.append("collapsed_hung1_family")
    gate_passed = not gate_failures
    if "score_label" in candidate:
        score_label = _as_bool(candidate.get("score_label"), False) is True
    elif score is not None:
        score_label = score >= positive_threshold
    else:
        score_label = _as_bool(candidate.get("label"), False) is True
    requested = manual if manual is not None else score_label
    # trust_score_label is retained as audit metadata; it never bypasses gates.
    _ = trust_score_label
    label = int(bool(requested and gate_passed))
    return RankerExample(
        candidate_id=candidate_id,
        identity_group=group,
        phrase_id=phrase,
        lyrics=lyrics,
        phoneme_tokens=tuple(tokens),
        phones=tuple(phones),
        label=label,
        split=split,
        failure_kind=failure_kind,
        positive_gate_passed=gate_passed,
        positive_gate_failures=tuple(gate_failures),
        score=score,
        manual_label=manual,
        label_provenance=label_provenance,
        label_streams=tuple(_label_streams(candidate, score=score, manual=manual)),
        provenance_missing=provenance_missing,
        source=_jsonable(
            {
                key: value
                for key, value in candidate.items()
                if key
                not in {
                    "xsampa",
                    "x_sampa",
                    "vocaloid_phonemes",
                    "openutau_alias",
                    "engine_phoneme",
                }
            }
        ),
    )


def vocabulary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chars = sorted({char for row in rows for char in _non_empty(row.get("lyrics"))})
    tokens = sorted({token for row in rows for token in _sequence(row.get("phoneme_tokens")) if _non_empty(token)})
    char_to_id = {"<pad>": 0, "<unk>": 1}
    phone_to_id = {"<pad>": 0, "<unk>": 1}
    for value in chars:
        if value not in char_to_id:
            char_to_id[value] = len(char_to_id)
    for value in tokens:
        value = _non_empty(value)
        if value not in phone_to_id:
            phone_to_id[value] = len(phone_to_id)
    return {
        "schema_version": VOCAB_SCHEMA_VERSION,
        "special_tokens": {"pad": "<pad>", "unk": "<unk>"},
        "chars": chars,
        "phoneme_tokens": tokens,
        "char_to_id": char_to_id,
        "phoneme_to_id": phone_to_id,
    }


def load_candidates(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetBuildError(f"cannot read candidate input: {source}") from exc
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not all(isinstance(row, Mapping) for row in rows):
            raise DatasetBuildError("JSONL candidate rows must be objects")
        return "jsonl", [dict(row) for row in rows]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return "jsonl", [dict(row) for row in rows]
    if isinstance(payload, list):
        return "list", [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        raise DatasetBuildError("candidate JSON must be an object or array")
    for key in ("scored_candidates", "candidates", "rows", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return key, [dict(row) for row in value if isinstance(row, Mapping)]
    raise DatasetBuildError("candidate JSON has no scored_candidates/candidates/rows list")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def build_dataset(
    scored_candidates: str | Path,
    output_root: str | Path,
    *,
    positive_threshold: float = 0.9,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 0,
    trust_score_label: bool = False,
) -> tuple[int, dict[str, Any]]:
    if not 0.0 <= positive_threshold <= 1.0:
        raise DatasetBuildError("positive_threshold must be between 0 and 1")
    input_key, rows = load_candidates(scored_candidates)
    rows = sorted(rows, key=lambda row: (_non_empty(_first(row, "candidate_id", "record_id", "id")), identity_group(row)))
    try:
        split_result = split_records(
            (_split_record(row) for row in rows),
            ratios=(1.0 - val_ratio - test_ratio, val_ratio, test_ratio),
            seed=seed,
        )
    except SplitError as exc:
        raise DatasetBuildError(str(exc)) from exc
    candidate_assignments = {
        candidate_id: ("val" if split == "validation" else split)
        for candidate_id, split in split_result.assignments.items()
    }
    assignments: dict[str, str] = {}
    for row in rows:
        candidate_id = _non_empty(_first(row, "candidate_id", "record_id", "id"))
        group = identity_group(row)
        split = candidate_assignments[candidate_id]
        previous = assignments.setdefault(group, split)
        if previous != split:
            raise DatasetBuildError(f"identity group was assigned to multiple splits: {group}")
    manifest = split_manifest(
        rows,
        assignments,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        leakage_guard=split_result.manifest,
    )
    examples = [
        normalise_candidate(
            row,
            split=candidate_assignments[_non_empty(_first(row, "candidate_id", "record_id", "id"))],
            positive_threshold=positive_threshold,
            trust_score_label=trust_score_label,
        )
        for row in rows
    ]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    quarantined_rows: list[dict[str, Any]] = []
    for example in examples:
        row = example.to_row(
            split_algorithm=SPLIT_ALGORITHM,
            seed=seed,
            manifest_sha256=manifest["manifest_sha256"],
        )
        inadmissible = bool(example.provenance_missing) or any(
            failure in {
                "phone_truth_ipa_incomplete",
                "phone_truth_features_incomplete",
            }
            for failure in example.positive_gate_failures
        )
        if inadmissible:
            row["training_admissible"] = False
            row["quarantine_reasons"] = sorted(
                {
                    *example.provenance_missing,
                    *(
                        failure
                        for failure in example.positive_gate_failures
                        if failure.startswith("phone_truth_")
                    ),
                }
            )
            quarantined_rows.append(row)
        else:
            row["training_admissible"] = True
            output_rows[example.split].append(row)
    if not output_rows["train"]:
        raise DatasetBuildError("ranker dataset has no admissible training rows")
    for split, split_rows in output_rows.items():
        _write_jsonl(output / f"{split}.jsonl", split_rows)
    _write_jsonl(output / "quarantined.jsonl", quarantined_rows)
    _write_json(output / "vocab.json", vocabulary([row for rows in output_rows.values() for row in rows]))
    _write_json(output / "split_manifest.json", manifest)
    failure_counts = Counter(example.failure_kind for example in examples)
    provenance_counts = Counter(str(example.label_provenance.get("kind", "unknown")) for example in examples)
    split_counts = {split: len(split_rows) for split, split_rows in output_rows.items()}
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_key": input_key,
        "input_path": str(Path(scored_candidates)),
        "input_row_count": len(examples),
        "row_count": sum(len(rows) for rows in output_rows.values()),
        "quarantined_count": len(quarantined_rows),
        "positive_count": sum(int(row["label"]) for rows in output_rows.values() for row in rows),
        "negative_count": sum(1 - int(row["label"]) for rows in output_rows.values() for row in rows),
        "failure_kind_counts": dict(sorted(failure_counts.items())),
        "label_provenance_counts": dict(sorted(provenance_counts.items())),
        "label_provenance_incomplete_count": sum(bool(example.provenance_missing) for example in examples),
        "truth_ipa_incomplete_count": sum(
            not example.phones or any(not item.get("ipa") for item in example.phones) for example in examples
        ),
        "truth_features_incomplete_count": sum(
            not example.phones or any(not item.get("features") for item in example.phones) for example in examples
        ),
        "split_counts": split_counts,
        "group_count": split_result.manifest["group_count"],
        "positive_threshold": positive_threshold,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "split_algorithm": SPLIT_ALGORITHM,
        "split_manifest_sha256": manifest["manifest_sha256"],
        "trust_score_label": trust_score_label,
        "dataset_sha256": sha256_json({split: output_rows[split] for split in ("train", "val", "test")}),
    }
    _write_json(output / "summary.json", summary)
    return 0, summary


__all__ = [
    "DatasetBuildError",
    "RankerExample",
    "SCHEMA_VERSION",
    "SPLIT_ALGORITHM",
    "VOCAB_SCHEMA_VERSION",
    "build_dataset",
    "canonical_json",
    "grouped_split",
    "identity_group",
    "infer_failure_kind",
    "load_candidates",
    "normalise_candidate",
    "sha256_json",
    "split_manifest",
    "vocabulary",
]
